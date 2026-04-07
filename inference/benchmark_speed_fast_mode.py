"""Benchmark Deep3D_Pro fast mode on sequential frame folders.

Fast mode lowers the model input resolution without retraining, then upsamples
the predicted right-eye frame back to the requested output size. The default is
960x540 model input and 1920x1080 output. For a stronger speed/quality tradeoff,
run with ``--model_size 640 360``.

The output layout and printed JSON summary intentionally follow the baseline
speed benchmark format:

    OUT_ROOT/YYYYMMDD/HHMMSS_fast_960x540_to_1920x1080/
      videos/*.mp4
      visualizations/*/*.jpg
      pred_right_frames/*/*.png        # only with --save_pred_frames
      exported_models/*.pt
      summary.json
      per_clip_speed.csv
      per_frame_speed.csv
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
from models.deep3d_network import Deep3DNet, load_pretrained_jit
from tools.file_utils import ensure_dir
from tools.video_utils import create_video_writer


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
DEFAULT_MODEL_SIZE = (960, 540)
DEFAULT_OUTPUT_SIZE = (1920, 1080)
BASELINE_720P_MODEL_SIZE = (1280, 720)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Deep3D_Pro fast mode: lower model input resolution, "
            "then upsample outputs to 1080p."
        )
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
        help=(
            "Parent directory for benchmark runs. Outputs are saved under "
            "OUT_ROOT/YYYYMMDD/HHMMSS_fast_WIDTHxHEIGHT_to_1920x1080 by default."
        ),
    )
    parser.add_argument(
        "--run_name",
        default=None,
        type=str,
        help=(
            "Optional run folder name under the date directory. Defaults to "
            "HHMMSS_fast_WIDTHxHEIGHT_to_OUTPUT_WIDTHxOUTPUT_HEIGHT in UTC."
        ),
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
        default=DEFAULT_MODEL_SIZE,
        help=(
            "Fast-mode inference resolution. Defaults to 960 540; use 640 360 "
            "for a stronger speed/quality tradeoff."
        ),
    )
    parser.add_argument(
        "--output_size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=DEFAULT_OUTPUT_SIZE,
        help="Final left/right frame size before video writing. Defaults to 1920 1080.",
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
    parser.add_argument(
        "--target_device_tops",
        default=4.0,
        type=float,
        help="Competition target compute budget in TOPS/TOPS-equivalent for rough scaling.",
    )
    parser.add_argument(
        "--current_device_tops",
        default=None,
        type=float,
        help=(
            "Optional TOPS/TOPS-equivalent of the current test device. If provided, "
            "the script estimates 4T-device FPS by linear scaling."
        ),
    )
    parser.add_argument(
        "--target_fps",
        default=50.0,
        type=float,
        help="Competition target FPS. Default: 50.",
    )
    parser.add_argument(
        "--target_memory_mb",
        default=500.0,
        type=float,
        help="Competition target memory limit in MB. Default: 500.",
    )
    parser.add_argument("--inv", action="store_true", help="Swap left/right order in the stereo video.")
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


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


def get_device_profile(device):
    profile = {
        "device": str(device),
        "type": device.type,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        profile.update(
            {
                "device_name": props.name,
                "total_memory_mb": props.total_memory / (1024 ** 2),
                "multi_processor_count": props.multi_processor_count,
                "compute_capability": f"{props.major}.{props.minor}",
            }
        )
    return profile


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


def prepare_tensor(frame_tensor, process, device, model_size):
    if (frame_tensor.shape[1], frame_tensor.shape[0]) != model_size:
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


def safe_div(numerator, denominator):
    if denominator in (0, 0.0, None):
        return None
    return numerator / denominator


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


def build_competition_estimate(
    *,
    args,
    device_profile,
    process_peak_mb_after,
    gpu_memory,
    model_file_mb,
    total_frames,
    total_wall_seconds,
    total_model_seconds,
    warm_frames,
    warm_wall_seconds,
    warm_model_seconds,
):
    pipeline_fps = safe_div(total_frames, total_wall_seconds) or 0.0
    model_fps = safe_div(total_frames, total_model_seconds) or 0.0
    steady_pipeline_fps = safe_div(warm_frames, warm_wall_seconds)
    steady_model_fps = safe_div(warm_frames, warm_model_seconds)

    if steady_pipeline_fps is None:
        steady_pipeline_fps = pipeline_fps
    if steady_model_fps is None:
        steady_model_fps = model_fps

    steady_pipeline_ms = safe_div(1000.0, steady_pipeline_fps) or 0.0
    steady_model_ms = safe_div(1000.0, steady_model_fps) or 0.0
    frame_budget_ms = safe_div(1000.0, args.target_fps) or 0.0

    runtime_device_peak_mb = None
    if gpu_memory is not None:
        runtime_device_peak_mb = gpu_memory["max_reserved_mb"]

    estimate = {
        "target": {
            "device_tops": args.target_device_tops,
            "fps": args.target_fps,
            "frame_budget_ms": frame_budget_ms,
            "memory_mb": args.target_memory_mb,
        },
        "current_device": {
            **device_profile,
            "provided_current_device_tops": args.current_device_tops,
        },
        "current_measurement": {
            "steady_pipeline_fps": steady_pipeline_fps,
            "steady_model_fps": steady_model_fps,
            "steady_pipeline_ms": steady_pipeline_ms,
            "steady_model_ms": steady_model_ms,
            "pipeline_meets_target_fps": bool(steady_pipeline_fps >= args.target_fps),
            "model_meets_target_fps": bool(steady_model_fps >= args.target_fps),
            "host_process_peak_mb": process_peak_mb_after,
            "host_process_meets_target_memory": bool(process_peak_mb_after <= args.target_memory_mb),
            "runtime_device_peak_mb_proxy": runtime_device_peak_mb,
            "runtime_device_peak_meets_target_memory_proxy": None
            if runtime_device_peak_mb is None
            else bool(runtime_device_peak_mb <= args.target_memory_mb),
            "model_file_mb": model_file_mb,
            "model_file_meets_target_memory": bool(model_file_mb <= args.target_memory_mb),
        },
        "tops_linear_estimate": {
            "enabled": args.current_device_tops is not None and args.current_device_tops > 0,
            "assumption": "Assumes FPS scales linearly with effective compute TOPS on the same operator set, precision, bandwidth regime, and software stack.",
            "warning": "This is only a rough estimate. GPU TFLOPS/TOPS and edge NPU TOPS are not directly equivalent across precision and memory systems.",
        },
    }

    if args.current_device_tops is not None and args.current_device_tops > 0:
        scale = args.target_device_tops / args.current_device_tops
        estimate["tops_linear_estimate"].update(
            {
                "scale_factor_target_over_current": scale,
                "estimated_target_pipeline_fps": steady_pipeline_fps * scale,
                "estimated_target_model_fps": steady_model_fps * scale,
                "estimated_target_pipeline_ms": steady_pipeline_ms / scale if scale > 0 else None,
                "estimated_target_model_ms": steady_model_ms / scale if scale > 0 else None,
                "estimated_pipeline_meets_target_fps": bool(steady_pipeline_fps * scale >= args.target_fps),
                "estimated_model_meets_target_fps": bool(steady_model_fps * scale >= args.target_fps),
                "required_tops_for_pipeline_target_fps": safe_div(
                    args.current_device_tops * args.target_fps, steady_pipeline_fps
                ),
                "required_tops_for_model_target_fps": safe_div(
                    args.current_device_tops * args.target_fps, steady_model_fps
                ),
            }
        )

    return estimate


def get_run_name(args, run_time):
    if args.run_name:
        return args.run_name
    model_width, model_height = args.model_size
    output_width, output_height = args.output_size
    return f"{run_time}_fast_{model_width}x{model_height}_to_{output_width}x{output_height}"


def build_fast_mode_summary(model_size, output_size):
    baseline_area = BASELINE_720P_MODEL_SIZE[0] * BASELINE_720P_MODEL_SIZE[1]
    model_area = model_size[0] * model_size[1]
    output_area = output_size[0] * output_size[1]
    return {
        "enabled": True,
        "model_input_size": list(model_size),
        "output_size": list(output_size),
        "baseline_720p_model_size": list(BASELINE_720P_MODEL_SIZE),
        "model_input_area_ratio_vs_1280x720": model_area / baseline_area,
        "model_input_area_percent_vs_1280x720": model_area * 100.0 / baseline_area,
        "upsample_output": model_size != output_size,
        "upsample_scale_x": output_size[0] / model_size[0],
        "upsample_scale_y": output_size[1] / model_size[1],
        "output_area_ratio_vs_model_input": output_area / model_area,
        "note": "Fast mode is training-free. It trades right-eye detail and disparity accuracy for lower model compute.",
    }


def export_fast_torchscript_model(model, export_path, model_size):
    width, height = model_size
    export_path.parent.mkdir(parents=True, exist_ok=True)

    model_cpu = Deep3DNet()
    model_cpu.load_state_dict({key: value.detach().cpu() for key, value in model.state_dict().items()})
    model_cpu.eval()

    example_input = torch.rand(1, 18, height, width, dtype=torch.float32)
    export_start = time.perf_counter()
    with torch.no_grad():
        scripted = torch.jit.trace(model_cpu, example_input, check_trace=False)
    scripted.save(str(export_path))

    return {
        "path": to_project_relative(export_path),
        "input_size": [width, height],
        "format": "torchscript_trace",
        "frozen": False,
        "seconds": time.perf_counter() - export_start,
        "file_mb": get_path_size_mb(export_path),
    }


def main():
    args = parse_args()

    model_path = resolve_path(args.model)
    data_root = resolve_path(args.data_root)
    out_parent = resolve_path(args.out_root)
    model_size = tuple(args.model_size)
    output_size = tuple(args.output_size)

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_date, run_time = timestamp.split("_", 1)
    run_name = get_run_name(args, run_time)
    out_date_root = out_parent / run_date
    out_root = out_date_root / run_name
    video_root = out_root / "videos"
    vis_root = out_root / "visualizations"
    pred_root = out_root / "pred_right_frames"
    export_root = out_root / "exported_models"
    ensure_dir(str(out_parent))
    ensure_dir(str(out_date_root))
    ensure_dir(str(out_root))
    ensure_dir(str(video_root))
    ensure_dir(str(vis_root))
    ensure_dir(str(export_root))
    if args.save_pred_frames:
        ensure_dir(str(pred_root))

    clips = discover_clips(data_root)
    device = get_device(args.gpu_id)

    # The released TorchScript model bakes in its traced 1280x720 grid, so fast
    # mode rebuilds the dynamic PyTorch module, copies the pretrained weights,
    # then exports a new TorchScript model at the requested fast-mode size.
    weight_model = Deep3DNet()
    missing_keys, unexpected_keys = load_pretrained_jit(weight_model, str(model_path), device="cpu")
    exported_model_path = export_root / f"deep3d_fast_{model_size[0]}x{model_size[1]}_torchscript.pt"
    export_info = export_fast_torchscript_model(weight_model, exported_model_path, model_size)

    net = torch.jit.load(str(exported_model_path), map_location="cpu").eval()
    process = PreProcess()
    if device.type == "cuda":
        torch.cuda.set_device(device)
        net = net.to(device).half()
        net = torch.jit.freeze(net.eval())
        process = process.to(device).half()
        torch.cuda.reset_peak_memory_stats(device)
    else:
        net = net.to(device)
        net = torch.jit.freeze(net.eval())
    net.eval()

    process_peak_mb_before = get_process_peak_memory_mb()
    device_profile = get_device_profile(device)
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
        source_size = (first_frame.shape[1], first_frame.shape[0])
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
                "source_width": source_size[0],
                "source_height": source_size[1],
                "benchmark_width": output_size[0],
                "benchmark_height": output_size[1],
                "simulated_1080p_input": int(source_size != output_size),
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
    gpu_memory = get_gpu_memory_snapshot(device)
    model_file_mb = get_path_size_mb(model_path)
    summary = {
        "model_path": to_project_relative(model_path),
        "data_root": to_project_relative(data_root),
        "out_parent": to_project_relative(out_parent),
        "out_date_root": to_project_relative(out_date_root),
        "out_root": to_project_relative(out_root),
        "run_name": run_name,
        "device": str(device),
        "fps_assumption": args.fps,
        "alpha": args.alpha,
        "model_input_size": list(model_size),
        "video_frame_sizes": resolved_output_sizes,
        "benchmark_policy": {
            "simulate_1080p_when_source_is_not_1080p": True,
            "simulated_1080p_size": list(output_size),
            "explicit_output_size_overrides_policy": True,
            "fast_mode_downscale_before_model": True,
            "fast_mode_upsample_after_model": model_size != output_size,
        },
        "fast_mode": build_fast_mode_summary(model_size, output_size),
        "weight_loading": {
            "source": "pretrained_torchscript_state_dict",
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "note": "The source TorchScript file has fixed 1280x720 warp grids, so fast mode rebuilds the dynamic PyTorch model, loads its weights, and exports a new fast-size TorchScript model.",
        },
        "torchscript_export": {
            **export_info,
            "used_for_inference": True,
        },
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
            "gpu": gpu_memory,
            "model_file_mb": model_file_mb,
            "output_dir_mb": get_path_size_mb(out_root),
        },
        "competition_estimate": build_competition_estimate(
            args=args,
            device_profile=device_profile,
            process_peak_mb_after=process_peak_mb_after,
            gpu_memory=gpu_memory,
            model_file_mb=model_file_mb,
            total_frames=total_frames,
            total_wall_seconds=total_wall_seconds,
            total_model_seconds=total_model_seconds,
            warm_frames=warm_frames,
            warm_wall_seconds=warm_wall_seconds,
            warm_model_seconds=warm_model_seconds,
        ),
        "deployment_reference": {
            "target_limit_mb": 500,
            "notes": [
                "process_peak_mb_after reflects host RAM peak for the whole benchmark process",
                "gpu.max_reserved_mb is the peak CUDA allocator reservation during inference",
                "model_file_mb is the serialized TorchScript size on disk",
                "fast_mode reduces model input area without retraining, then upsamples the output",
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
                "source_width",
                "source_height",
                "benchmark_width",
                "benchmark_height",
                "simulated_1080p_input",
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
