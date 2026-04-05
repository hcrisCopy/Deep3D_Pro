"""Benchmark Deep3D_Pro on sequential 1080p frame folders.

The benchmark treats each clip folder under ``data_root`` as a video sequence:

    clip_id/
      left/*.png
      right/*.png

It runs temporal inference from left frames, writes a side-by-side stereo video,
collects latency / FPS / memory stats, and saves a few visualization panels.
"""

import argparse
import csv
import datetime as dt
import json
import os
import resource
import re
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.transforms import PreProcess, tensor2im
from tools.file_utils import ensure_dir
from tools.video_utils import create_video_writer


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Deep3D_Pro on sequential frame folders and export stereo videos."
    )
    parser.add_argument("--gpu_id", default=0, type=int, help="GPU device id, use -1 for CPU.")
    parser.add_argument(
        "--model",
        default="../data/pretrained/deep3d_v1.0_1280x720_cuda.pt",
        type=str,
        help="TorchScript model path relative to Deep3D_Pro.",
    )
    parser.add_argument(
        "--data_root",
        default="../data/test_set/speed_test",
        type=str,
        help="Clip root containing clip_id/left and optional clip_id/right.",
    )
    parser.add_argument(
        "--out_root",
        default="../data/test_on_speed",
        type=str,
        help="Parent directory for benchmark runs. Each run is saved in its own timestamped subdirectory.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        type=str,
        help="Optional run folder name. Defaults to a timestamp like 20260405_123456.",
    )
    parser.add_argument(
        "--fps",
        default=25.0,
        type=float,
        help="FPS for exported videos because frame folders do not carry timing metadata.",
    )
    parser.add_argument("--alpha", default=5, type=int, help="Temporal offset for far frames.")
    parser.add_argument(
        "--model_size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Inference resolution expected by the TorchScript model. Defaults to parsing WIDTHxHEIGHT from the model filename.",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Final left/right frame size before video writing. Defaults to the source resolution.",
    )
    parser.add_argument(
        "--warmup_frames",
        default=10,
        type=int,
        help="Ignore the first N frames when reporting steady-state FPS and latency.",
    )
    parser.add_argument(
        "--sample_vis_per_clip",
        default=3,
        type=int,
        help="How many frames to visualize per clip.",
    )
    parser.add_argument(
        "--save_pred_frames",
        action="store_true",
        help="Also save predicted right-eye frames as images.",
    )
    parser.add_argument("--inv", action="store_true", help="Swap left/right order in the stereo video.")
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def infer_model_size(model_path):
    match = re.search(r"_(\d+)x(\d+)_", os.path.basename(str(model_path)))
    if match is None:
        raise ValueError(
            "Cannot infer model input size from filename. Please pass --model_size WIDTH HEIGHT."
        )
    return int(match.group(1)), int(match.group(2))


def to_project_relative(path):
    try:
        return os.path.relpath(path, PROJECT_ROOT)
    except ValueError:
        return str(path)


def natural_key(path):
    name = path.name if hasattr(path, "name") else str(path)
    parts = re.split(r"(\d+)", name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def list_image_files(folder):
    return sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES],
        key=natural_key,
    )


def discover_clips(data_root):
    clips = []
    for clip_dir in sorted([path for path in data_root.iterdir() if path.is_dir()], key=natural_key):
        left_dir = clip_dir / "left"
        right_dir = clip_dir / "right"
        if not left_dir.is_dir():
            continue
        left_files = {path.name: path for path in list_image_files(left_dir)}
        right_files = {}
        if right_dir.is_dir():
            right_files = {path.name: path for path in list_image_files(right_dir)}

        common_names = sorted(
            set(left_files) & set(right_files) if right_files else set(left_files),
            key=natural_key,
        )
        if not common_names:
            continue

        clips.append(
            {
                "name": clip_dir.name,
                "left": [left_files[name] for name in common_names],
                "right": [right_files[name] for name in common_names] if right_files else None,
            }
        )
    if not clips:
        raise RuntimeError(f"No valid clips found in: {data_root}")
    return clips


def get_device(gpu_id):
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def get_process_peak_memory_mb():
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss_kb / (1024 * 1024)
    return rss_kb / 1024.0


def get_gpu_memory_snapshot(device):
    if device.type != "cuda":
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "device_name": torch.cuda.get_device_name(device),
        "allocated_mb": torch.cuda.memory_allocated(device) / (1024 ** 2),
        "reserved_mb": torch.cuda.memory_reserved(device) / (1024 ** 2),
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "max_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 ** 2),
        "free_mb": free_bytes / (1024 ** 2),
        "total_mb": total_bytes / (1024 ** 2),
    }


def get_path_size_mb(path):
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024 ** 2)
    total_bytes = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total_bytes += file_path.stat().st_size
    return total_bytes / (1024 ** 2)


def read_bgr_frame(path, target_size=None):
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Cannot read image: {path}")
    if target_size is not None and (frame.shape[1], frame.shape[0]) != target_size:
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
    return frame


def build_frame_window(frame_paths, alpha, target_size):
    first_frame = torch.from_numpy(read_bgr_frame(frame_paths[0], target_size))
    window = deque([first_frame] * alpha)
    window.append(first_frame)

    last_frame = first_frame
    preload_end = min(len(frame_paths), alpha + 1)
    for frame_path in frame_paths[1:preload_end]:
        next_frame = torch.from_numpy(read_bgr_frame(frame_path, target_size))
        last_frame = next_frame
        window.append(next_frame)

    while len(window) < 2 * alpha + 1:
        window.append(last_frame)

    return window, preload_end, last_frame


def prepare_tensor(frame_tensor, process, device, model_size=None):
    if model_size is not None and (frame_tensor.shape[1], frame_tensor.shape[0]) != model_size:
        frame_np = cv2.resize(frame_tensor.numpy(), model_size, interpolation=cv2.INTER_LANCZOS4)
        frame_tensor = torch.from_numpy(frame_np)
    frame_tensor = frame_tensor.to(device, non_blocking=device.type == "cuda")
    if device.type == "cuda":
        frame_tensor = frame_tensor.half()
    return process(frame_tensor)


def save_visualization(save_path, left, pred, target=None):
    diff_pred_left = cv2.absdiff(pred, left)

    anaglyph = np.zeros_like(pred)
    anaglyph[:, :, 0] = pred[:, :, 0]
    anaglyph[:, :, 1] = pred[:, :, 1]
    anaglyph[:, :, 2] = left[:, :, 2]

    tiles = [left, pred, anaglyph, diff_pred_left]
    labels = ["Left Input", "Pred Right", "Anaglyph", "Pred-Left Diff"]

    if target is not None:
        diff_pred_gt = cv2.absdiff(pred, target)
        tiles.extend([target, diff_pred_gt])
        labels.extend(["GT Right", "Pred-GT Diff"])

    h, w = left.shape[:2]
    cols = 3
    rows = int(np.ceil(len(tiles) / cols))
    canvas = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

    for index, (tile, label) in enumerate(zip(tiles, labels)):
        row = index // cols
        col = index % cols
        canvas[row * h:(row + 1) * h, col * w:(col + 1) * w] = tile
        origin = (col * w + 16, row * h + 36)
        cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(save_path), canvas)


def summarize_timings(seconds_list):
    if not seconds_list:
        return {
            "count": 0,
            "avg_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }
    timings = np.array(seconds_list, dtype=np.float64) * 1000.0
    return {
        "count": int(timings.size),
        "avg_ms": float(np.mean(timings)),
        "p50_ms": float(np.percentile(timings, 50)),
        "p90_ms": float(np.percentile(timings, 90)),
        "p95_ms": float(np.percentile(timings, 95)),
        "p99_ms": float(np.percentile(timings, 99)),
    }


def main():
    args = parse_args()

    model_path = resolve_path(args.model)
    data_root = resolve_path(args.data_root)
    out_parent = resolve_path(args.out_root)
    run_name = args.run_name or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = out_parent / run_name
    video_root = out_root / "videos"
    vis_root = out_root / "visualizations"
    pred_root = out_root / "pred_right_frames"
    ensure_dir(str(out_parent))
    ensure_dir(str(out_root))
    ensure_dir(str(video_root))
    ensure_dir(str(vis_root))
    if args.save_pred_frames:
        ensure_dir(str(pred_root))

    clips = discover_clips(data_root)
    device = get_device(args.gpu_id)
    model_size = tuple(args.model_size) if args.model_size else infer_model_size(model_path)

    net = torch.jit.load(str(model_path), map_location="cpu")
    process = PreProcess()
    if device.type == "cuda":
        net = net.to(device).half()
        process = process.to(device).half()
        torch.cuda.reset_peak_memory_stats(device)
    else:
        net = net.to(device)
    net.eval()

    process_peak_mb_before = get_process_peak_memory_mb()
    total_frames = 0
    total_wall_seconds = 0.0
    total_model_seconds = 0.0
    warm_wall_seconds = 0.0
    warm_model_seconds = 0.0
    warm_frames = 0
    resolved_output_sizes = []
    per_clip_rows = []
    per_frame_rows = []
    run_start = time.perf_counter()

    for clip in clips:
        clip_name = clip["name"]
        frame_count = len(clip["left"])
        sample_count = min(args.sample_vis_per_clip, frame_count)
        sample_indices = set()
        if sample_count > 0:
            sample_indices = {
                int(round(pos)) for pos in np.linspace(0, frame_count - 1, num=sample_count)
            }

        first_frame = read_bgr_frame(clip["left"][0])
        output_size = tuple(args.output_size) if args.output_size else (first_frame.shape[1], first_frame.shape[0])
        resolved_output_sizes.append(list(output_size))

        clip_video_path = video_root / f"{clip_name}.mp4"
        clip_vis_dir = vis_root / clip_name
        ensure_dir(str(clip_vis_dir))
        if args.save_pred_frames:
            clip_pred_dir = pred_root / clip_name
            ensure_dir(str(clip_pred_dir))
        else:
            clip_pred_dir = None

        writer = create_video_writer(str(clip_video_path), args.fps, (output_size[0] * 2, output_size[1]))
        window, next_input_index, last_frame = build_frame_window(clip["left"], args.alpha, output_size)

        x0 = prepare_tensor(window[args.alpha], process, device, model_size=model_size)
        clip_wall_seconds = 0.0
        clip_model_seconds = 0.0
        clip_warm_frames = 0
        clip_warm_wall = 0.0
        clip_warm_model = 0.0
        clip_wall_samples = []
        clip_model_samples = []

        progress = tqdm(range(frame_count), desc=f"Speed {clip_name}", leave=False)
        for frame_idx in progress:
            if args.alpha > 0:
                x1 = window[0]
                x2 = window[args.alpha - 1]
                x3 = window[args.alpha]
                x4 = window[args.alpha + 1]
                x5 = window[-1]
            else:
                x1 = x2 = x3 = x4 = x5 = window[0]

            wall_start = time.perf_counter()
            tensors = [prepare_tensor(frame, process, device, model_size=model_size) for frame in (x1, x2, x3, x4, x5)]
            input_data = torch.cat((tensors[0], tensors[1], x0, tensors[2], tensors[3], tensors[4]), dim=0).unsqueeze(0)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            infer_start = time.perf_counter()
            with torch.no_grad():
                out = net(input_data)
                x0 = out[0].detach()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            infer_seconds = time.perf_counter() - infer_start

            pred_bgr = tensor2im(out[0])
            if (pred_bgr.shape[1], pred_bgr.shape[0]) != output_size:
                pred_bgr = cv2.resize(pred_bgr, output_size, interpolation=cv2.INTER_LANCZOS4)

            left_bgr = x3.cpu().numpy()
            if left_bgr.shape[:2] != (output_size[1], output_size[0]):
                left_bgr = cv2.resize(left_bgr, output_size, interpolation=cv2.INTER_LANCZOS4)

            stereo_pair = np.concatenate((pred_bgr, left_bgr), axis=1) if args.inv else np.concatenate((left_bgr, pred_bgr), axis=1)
            writer.write(stereo_pair)

            target_bgr = None
            if clip["right"] is not None:
                target_bgr = read_bgr_frame(clip["right"][frame_idx], output_size)

            if frame_idx in sample_indices:
                save_visualization(
                    clip_vis_dir / f"{clip['left'][frame_idx].stem}_viz.jpg",
                    left_bgr,
                    pred_bgr,
                    target_bgr,
                )

            if clip_pred_dir is not None:
                cv2.imwrite(str(clip_pred_dir / clip["left"][frame_idx].name), pred_bgr)

            wall_seconds = time.perf_counter() - wall_start
            clip_wall_seconds += wall_seconds
            clip_model_seconds += infer_seconds
            clip_wall_samples.append(wall_seconds)
            clip_model_samples.append(infer_seconds)
            total_frames += 1
            total_wall_seconds += wall_seconds
            total_model_seconds += infer_seconds

            steady = frame_idx >= args.warmup_frames
            if steady:
                warm_frames += 1
                warm_wall_seconds += wall_seconds
                warm_model_seconds += infer_seconds
                clip_warm_frames += 1
                clip_warm_wall += wall_seconds
                clip_warm_model += infer_seconds

            per_frame_rows.append(
                {
                    "clip": clip_name,
                    "frame": clip["left"][frame_idx].name,
                    "wall_ms": wall_seconds * 1000.0,
                    "model_ms": infer_seconds * 1000.0,
                    "steady_state": int(steady),
                }
            )
            progress.set_postfix(
                fps=f"{(frame_idx + 1) / clip_wall_seconds:.2f}",
                model_fps=f"{(frame_idx + 1) / clip_model_seconds:.2f}",
            )

            if args.alpha > 0:
                if next_input_index < frame_count:
                    next_frame = torch.from_numpy(read_bgr_frame(clip["left"][next_input_index], output_size))
                    last_frame = next_frame
                    next_input_index += 1
                else:
                    next_frame = last_frame
                window.popleft()
                window.append(next_frame)

        writer.release()

        per_clip_rows.append(
            {
                "clip": clip_name,
                "frames": frame_count,
                "fps": frame_count / clip_wall_seconds if clip_wall_seconds > 0 else 0.0,
                "model_fps": frame_count / clip_model_seconds if clip_model_seconds > 0 else 0.0,
                "steady_fps": clip_warm_frames / clip_warm_wall if clip_warm_wall > 0 else 0.0,
                "steady_model_fps": clip_warm_frames / clip_warm_model if clip_warm_model > 0 else 0.0,
                "wall_avg_ms": (clip_wall_seconds / frame_count) * 1000.0 if frame_count else 0.0,
                "model_avg_ms": (clip_model_seconds / frame_count) * 1000.0 if frame_count else 0.0,
                "video_path": to_project_relative(clip_video_path),
            }
        )

    elapsed = time.perf_counter() - run_start
    process_peak_mb_after = get_process_peak_memory_mb()
    summary = {
        "model_path": to_project_relative(model_path),
        "data_root": to_project_relative(data_root),
        "out_parent": to_project_relative(out_parent),
        "out_root": to_project_relative(out_root),
        "run_name": run_name,
        "device": str(device),
        "fps_assumption": args.fps,
        "alpha": args.alpha,
        "model_input_size": list(model_size),
        "video_frame_sizes": resolved_output_sizes,
        "clips": len(clips),
        "frames": total_frames,
        "throughput": {
            "pipeline_fps": total_frames / total_wall_seconds if total_wall_seconds > 0 else 0.0,
            "model_fps": total_frames / total_model_seconds if total_model_seconds > 0 else 0.0,
            "steady_pipeline_fps": warm_frames / warm_wall_seconds if warm_wall_seconds > 0 else 0.0,
            "steady_model_fps": warm_frames / warm_model_seconds if warm_model_seconds > 0 else 0.0,
            "elapsed_fps": total_frames / elapsed if elapsed > 0 else 0.0,
        },
        "latency": {
            "pipeline": summarize_timings([row["wall_ms"] / 1000.0 for row in per_frame_rows]),
            "model": summarize_timings([row["model_ms"] / 1000.0 for row in per_frame_rows]),
            "steady_pipeline": summarize_timings(
                [row["wall_ms"] / 1000.0 for row in per_frame_rows if row["steady_state"]]
            ),
            "steady_model": summarize_timings(
                [row["model_ms"] / 1000.0 for row in per_frame_rows if row["steady_state"]]
            ),
        },
        "memory": {
            "process_peak_mb_before": process_peak_mb_before,
            "process_peak_mb_after": process_peak_mb_after,
            "process_peak_delta_mb": process_peak_mb_after - process_peak_mb_before,
            "gpu": get_gpu_memory_snapshot(device),
            "model_file_mb": get_path_size_mb(model_path),
            "output_dir_mb": get_path_size_mb(out_root),
        },
        "deployment_reference": {
            "target_limit_mb": 500,
            "notes": [
                "process_peak_mb_after reflects host RAM peak for the whole benchmark process",
                "gpu.max_reserved_mb is the peak CUDA allocator reservation during inference",
                "model_file_mb is the serialized TorchScript size on disk",
            ],
        },
        "per_clip": per_clip_rows,
    }

    with (out_root / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    with (out_root / "per_clip_speed.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "clip",
                "frames",
                "fps",
                "model_fps",
                "steady_fps",
                "steady_model_fps",
                "wall_avg_ms",
                "model_avg_ms",
                "video_path",
            ],
        )
        writer.writeheader()
        writer.writerows(per_clip_rows)

    with (out_root / "per_frame_speed.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["clip", "frame", "wall_ms", "model_ms", "steady_state"],
        )
        writer.writeheader()
        writer.writerows(per_frame_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
