"""Benchmark Deep3D_Pro on 1-minute video segments from left-view videos.

Expected input layout:

    ../data/test_set/speed_test_video/
      clip_id/
        left/
          *.mp4

For each input video, the script extracts the segment [1:00, 2:00), runs
Deep3D_Pro on that segment, and writes the following files under:

    ../data/test_on_speed/video/clip_id/
      <video_stem>_left.mp4
      <video_stem>_right.mp4
      <video_stem>_half_sbs.mp4
      <video_stem>_red_blue.mp4
      <video_stem>_summary.json
      <video_stem>_per_frame_speed.csv

It also writes a global ``summary.json`` and CSV reports under
``../data/test_on_speed/video`` and prints the global summary to stdout.
"""

import argparse
import csv
import json
import os
import resource
import re
import shutil
import subprocess
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
from tools.video_utils import create_video_writer, get_video_metadata


VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".mts", ".rmvb")
TARGET_1080P_SIZE = (1920, 1080)
SEGMENT_START_SEC = 60.0
SEGMENT_END_SEC = 120.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Deep3D_Pro on 1-minute [1:00, 2:00) left-view video segments."
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
        default="../data/test_set/speed_test_video",
        type=str,
        help="Clip root containing clip_id/left/*.mp4.",
    )
    parser.add_argument(
        "--out_root",
        default="../data/test_on_speed/video",
        type=str,
        help="Output root. Each clip is saved under OUT_ROOT/clip_id.",
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
        help="Final left/right frame size before video writing. Defaults to 1920x1080.",
    )
    parser.add_argument(
        "--fps",
        default=None,
        type=float,
        help="Optional FPS override for exported videos. Defaults to the input video FPS.",
    )
    parser.add_argument(
        "--warmup_frames",
        default=10,
        type=int,
        help="Ignore the first N frames when reporting steady-state FPS and latency.",
    )
    parser.add_argument(
        "--segment_start_sec",
        default=SEGMENT_START_SEC,
        type=float,
        help="Segment start time in seconds. Default: 60.0.",
    )
    parser.add_argument(
        "--segment_end_sec",
        default=SEGMENT_END_SEC,
        type=float,
        help="Segment end time in seconds. Default: 120.0.",
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
        help="Optional TOPS/TOPS-equivalent of the current test device. If provided, the script estimates 4T-device FPS by linear scaling.",
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
    parser.add_argument("--inv", action="store_true", help="Swap left/right order in the half-SBS video.")
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


def list_video_files(folder):
    return sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES],
        key=natural_key,
    )


def discover_clips(data_root):
    clips = []
    for clip_dir in sorted([path for path in data_root.iterdir() if path.is_dir()], key=natural_key):
        left_dir = clip_dir / "left"
        if not left_dir.is_dir():
            continue
        video_files = list_video_files(left_dir)
        if not video_files:
            continue
        clips.append({"name": clip_dir.name, "videos": video_files})
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
    profile = {"device": str(device), "type": device.type}
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


def read_frame(cap, target_size):
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if target_size is not None and (frame.shape[1], frame.shape[0]) != target_size:
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
    return torch.from_numpy(frame)


def seek_video(cap, start_frame, fallback_msec):
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
    if abs(current_frame - start_frame) > 2:
        cap.set(cv2.CAP_PROP_POS_MSEC, fallback_msec * 1000.0)


def build_initial_window(cap, alpha, target_size):
    first_frame = read_frame(cap, target_size)
    if first_frame is None:
        raise RuntimeError("Cannot read the first frame from the selected video segment.")

    window = deque([first_frame] * alpha)
    window.append(first_frame)

    last_frame = first_frame
    for _ in range(alpha):
        next_frame = read_frame(cap, target_size)
        if next_frame is None:
            next_frame = last_frame
        else:
            last_frame = next_frame
        window.append(next_frame)

    return window, last_frame


def prepare_tensor(frame_tensor, process, device, model_size):
    if (frame_tensor.shape[1], frame_tensor.shape[0]) != model_size:
        frame_np = cv2.resize(frame_tensor.numpy(), model_size, interpolation=cv2.INTER_LANCZOS4)
        frame_tensor = torch.from_numpy(frame_np)
    frame_tensor = frame_tensor.to(device, non_blocking=device.type == "cuda")
    if device.type == "cuda":
        frame_tensor = frame_tensor.half()
    return process(frame_tensor)


def make_red_blue_video_frame(left_bgr, right_bgr):
    anaglyph = np.zeros_like(left_bgr)
    anaglyph[:, :, 0] = right_bgr[:, :, 0]
    anaglyph[:, :, 1] = right_bgr[:, :, 1]
    anaglyph[:, :, 2] = left_bgr[:, :, 2]
    return anaglyph


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


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_segment_range(metadata, start_sec, end_sec):
    fps = metadata["fps"]
    frame_count = metadata["frame_count"]
    if fps <= 0:
        raise RuntimeError("Input video FPS is invalid.")
    start_frame = max(0, int(round(start_sec * fps)))
    end_frame = min(frame_count, int(round(end_sec * fps)))
    if end_frame <= start_frame:
        raise RuntimeError(
            f"Requested segment [{start_sec:.3f}, {end_sec:.3f}) is outside the video. "
            f"frame_count={frame_count}, fps={fps:.6f}"
        )
    return start_frame, end_frame, end_frame - start_frame


def can_read_frame(video_path, start_frame):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    try:
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, frame = cap.read()
        return bool(ok and frame is not None)
    finally:
        cap.release()


def ensure_ffmpeg_available():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "OpenCV cannot decode this video and ffmpeg is not available for fallback transcoding."
        )


def build_fallback_video(video_path, clip_out_dir, start_sec, end_sec, source_size, source_fps):
    ensure_ffmpeg_available()
    fallback_path = clip_out_dir / f"{video_path.stem}_segment_h264_fallback.mp4"
    width, height = source_size
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.6f}",
        "-to",
        f"{end_sec:.6f}",
        "-i",
        str(video_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-vf",
        f"scale={width}:{height}",
        "-r",
        f"{source_fps:.12f}",
        "-pix_fmt",
        "yuv420p",
        str(fallback_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"OpenCV failed to decode {video_path} and ffmpeg fallback transcoding also failed.\n"
            f"ffmpeg stderr:\n{exc.stderr}"
        ) from exc
    return fallback_path


def main():
    args = parse_args()

    if args.segment_end_sec <= args.segment_start_sec:
        raise ValueError("--segment_end_sec must be greater than --segment_start_sec.")

    model_path = resolve_path(args.model)
    data_root = resolve_path(args.data_root)
    out_root = resolve_path(args.out_root)
    ensure_dir(str(out_root))

    clips = discover_clips(data_root)
    device = get_device(args.gpu_id)
    model_size = tuple(args.model_size) if args.model_size else infer_model_size(model_path)
    output_size = tuple(args.output_size) if args.output_size else TARGET_1080P_SIZE

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
    device_profile = get_device_profile(device)
    total_frames = 0
    total_wall_seconds = 0.0
    total_model_seconds = 0.0
    warm_wall_seconds = 0.0
    warm_model_seconds = 0.0
    warm_frames = 0
    per_clip_rows = []
    global_per_frame_rows = []
    run_start = time.perf_counter()

    for clip in clips:
        clip_name = clip["name"]
        clip_out_dir = out_root / clip_name
        ensure_dir(str(clip_out_dir))

        for video_path in clip["videos"]:
            metadata = get_video_metadata(str(video_path))
            input_fps = metadata["fps"]
            export_fps = args.fps if args.fps is not None else input_fps
            source_size = (metadata["width"], metadata["height"])
            start_frame, end_frame, segment_frames = compute_segment_range(
                metadata, args.segment_start_sec, args.segment_end_sec
            )
            actual_start_sec = start_frame / input_fps
            actual_end_sec = end_frame / input_fps

            decode_video_path = video_path
            decode_start_frame = start_frame
            used_ffmpeg_fallback = False
            if not can_read_frame(video_path, start_frame):
                decode_video_path = build_fallback_video(
                    video_path,
                    clip_out_dir,
                    actual_start_sec,
                    actual_end_sec,
                    source_size,
                    input_fps,
                )
                decode_start_frame = 0
                used_ffmpeg_fallback = True

            cap = cv2.VideoCapture(str(decode_video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {decode_video_path}")
            seek_video(cap, decode_start_frame, actual_start_sec if decode_start_frame > 0 else 0.0)

            left_video_path = clip_out_dir / f"{video_path.stem}_left.mp4"
            right_video_path = clip_out_dir / f"{video_path.stem}_right.mp4"
            half_sbs_video_path = clip_out_dir / f"{video_path.stem}_half_sbs.mp4"
            red_blue_video_path = clip_out_dir / f"{video_path.stem}_red_blue.mp4"

            left_writer = create_video_writer(str(left_video_path), export_fps, output_size)
            right_writer = create_video_writer(str(right_video_path), export_fps, output_size)
            half_sbs_writer = create_video_writer(
                str(half_sbs_video_path), export_fps, (output_size[0] * 2, output_size[1])
            )
            red_blue_writer = create_video_writer(str(red_blue_video_path), export_fps, output_size)

            window, last_frame = build_initial_window(cap, args.alpha, output_size)
            x0 = prepare_tensor(window[args.alpha], process, device, model_size=model_size)

            clip_wall_seconds = 0.0
            clip_model_seconds = 0.0
            clip_warm_frames = 0
            clip_warm_wall = 0.0
            clip_warm_model = 0.0
            per_frame_rows = []

            progress = tqdm(
                range(segment_frames),
                desc=f"Speed {clip_name}/{video_path.stem}",
                leave=False,
            )
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
                tensors = [
                    prepare_tensor(frame, process, device, model_size=model_size)
                    for frame in (x1, x2, x3, x4, x5)
                ]
                input_data = torch.cat(
                    (tensors[0], tensors[1], x0, tensors[2], tensors[3], tensors[4]), dim=0
                ).unsqueeze(0)

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

                left_writer.write(left_bgr)
                right_writer.write(pred_bgr)
                stereo_pair = (
                    np.concatenate((pred_bgr, left_bgr), axis=1)
                    if args.inv
                    else np.concatenate((left_bgr, pred_bgr), axis=1)
                )
                half_sbs_writer.write(stereo_pair)
                red_blue_writer.write(make_red_blue_video_frame(left_bgr, pred_bgr))

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

                row = {
                    "clip": clip_name,
                    "video": video_path.name,
                    "frame_index": frame_idx,
                    "source_frame_index": start_frame + frame_idx,
                    "wall_ms": wall_seconds * 1000.0,
                    "model_ms": infer_seconds * 1000.0,
                    "steady_state": int(steady),
                }
                per_frame_rows.append(row)
                global_per_frame_rows.append(row)
                progress.set_postfix(
                    fps=f"{(frame_idx + 1) / clip_wall_seconds:.2f}",
                    model_fps=f"{(frame_idx + 1) / clip_model_seconds:.2f}",
                )

                if args.alpha > 0:
                    next_frame = read_frame(cap, output_size)
                    if next_frame is None:
                        next_frame = last_frame
                    else:
                        last_frame = next_frame
                    window.popleft()
                    window.append(next_frame)

            left_writer.release()
            right_writer.release()
            half_sbs_writer.release()
            red_blue_writer.release()
            cap.release()

            clip_summary = {
                "clip": clip_name,
                "video": video_path.name,
                "frames": segment_frames,
                "segment_start_sec": actual_start_sec,
                "segment_end_sec": actual_end_sec,
                "segment_duration_sec": actual_end_sec - actual_start_sec,
                "input_fps": input_fps,
                "export_fps": export_fps,
                "source_width": source_size[0],
                "source_height": source_size[1],
                "output_width": output_size[0],
                "output_height": output_size[1],
                "is_source_1080p": int(source_size == TARGET_1080P_SIZE),
                "fps": segment_frames / clip_wall_seconds if clip_wall_seconds > 0 else 0.0,
                "model_fps": segment_frames / clip_model_seconds if clip_model_seconds > 0 else 0.0,
                "steady_fps": clip_warm_frames / clip_warm_wall if clip_warm_wall > 0 else 0.0,
                "steady_model_fps": clip_warm_frames / clip_warm_model if clip_warm_model > 0 else 0.0,
                "wall_avg_ms": (clip_wall_seconds / segment_frames) * 1000.0 if segment_frames else 0.0,
                "model_avg_ms": (clip_model_seconds / segment_frames) * 1000.0 if segment_frames else 0.0,
                "left_video_path": to_project_relative(left_video_path),
                "right_video_path": to_project_relative(right_video_path),
                "half_sbs_video_path": to_project_relative(half_sbs_video_path),
                "red_blue_video_path": to_project_relative(red_blue_video_path),
                "used_ffmpeg_fallback": int(used_ffmpeg_fallback),
            }
            per_clip_rows.append(clip_summary)

            clip_report = {
                "clip": clip_name,
                "video": video_path.name,
                "input_video_path": to_project_relative(video_path),
                "decode_video_path": to_project_relative(decode_video_path),
                "left_video_path": to_project_relative(left_video_path),
                "right_video_path": to_project_relative(right_video_path),
                "half_sbs_video_path": to_project_relative(half_sbs_video_path),
                "red_blue_video_path": to_project_relative(red_blue_video_path),
                "used_ffmpeg_fallback": used_ffmpeg_fallback,
                "segment": {
                    "requested_start_sec": args.segment_start_sec,
                    "requested_end_sec": args.segment_end_sec,
                    "actual_start_sec": actual_start_sec,
                    "actual_end_sec": actual_end_sec,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "frames": segment_frames,
                },
                "input_fps": input_fps,
                "export_fps": export_fps,
                "source_size": list(source_size),
                "output_size": list(output_size),
                "throughput": {
                    "pipeline_fps": clip_summary["fps"],
                    "model_fps": clip_summary["model_fps"],
                    "steady_pipeline_fps": clip_summary["steady_fps"],
                    "steady_model_fps": clip_summary["steady_model_fps"],
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
            }

            with (clip_out_dir / f"{video_path.stem}_summary.json").open("w", encoding="utf-8") as file:
                json.dump(clip_report, file, indent=2, ensure_ascii=False)

            write_csv(
                clip_out_dir / f"{video_path.stem}_per_frame_speed.csv",
                ["clip", "video", "frame_index", "source_frame_index", "wall_ms", "model_ms", "steady_state"],
                per_frame_rows,
            )

    elapsed = time.perf_counter() - run_start
    process_peak_mb_after = get_process_peak_memory_mb()
    gpu_memory = get_gpu_memory_snapshot(device)
    model_file_mb = get_path_size_mb(model_path)
    summary = {
        "model_path": to_project_relative(model_path),
        "data_root": to_project_relative(data_root),
        "out_root": to_project_relative(out_root),
        "device": str(device),
        "alpha": args.alpha,
        "model_input_size": list(model_size),
        "video_output_size": list(output_size),
        "segment": {
            "requested_start_sec": args.segment_start_sec,
            "requested_end_sec": args.segment_end_sec,
            "requested_duration_sec": args.segment_end_sec - args.segment_start_sec,
        },
        "clips": len(clips),
        "videos": len(per_clip_rows),
        "frames": total_frames,
        "throughput": {
            "pipeline_fps": total_frames / total_wall_seconds if total_wall_seconds > 0 else 0.0,
            "model_fps": total_frames / total_model_seconds if total_model_seconds > 0 else 0.0,
            "steady_pipeline_fps": warm_frames / warm_wall_seconds if warm_wall_seconds > 0 else 0.0,
            "steady_model_fps": warm_frames / warm_model_seconds if warm_model_seconds > 0 else 0.0,
            "elapsed_fps": total_frames / elapsed if elapsed > 0 else 0.0,
        },
        "latency": {
            "pipeline": summarize_timings([row["wall_ms"] / 1000.0 for row in global_per_frame_rows]),
            "model": summarize_timings([row["model_ms"] / 1000.0 for row in global_per_frame_rows]),
            "steady_pipeline": summarize_timings(
                [row["wall_ms"] / 1000.0 for row in global_per_frame_rows if row["steady_state"]]
            ),
            "steady_model": summarize_timings(
                [row["model_ms"] / 1000.0 for row in global_per_frame_rows if row["steady_state"]]
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
        "per_clip": per_clip_rows,
    }

    with (out_root / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    write_csv(
        out_root / "per_clip_speed.csv",
        [
            "clip",
            "video",
            "frames",
            "segment_start_sec",
            "segment_end_sec",
            "segment_duration_sec",
            "input_fps",
            "export_fps",
            "source_width",
            "source_height",
            "output_width",
            "output_height",
            "is_source_1080p",
            "fps",
            "model_fps",
            "steady_fps",
            "steady_model_fps",
            "wall_avg_ms",
            "model_avg_ms",
            "left_video_path",
            "right_video_path",
            "half_sbs_video_path",
            "red_blue_video_path",
            "used_ffmpeg_fallback",
        ],
        per_clip_rows,
    )

    write_csv(
        out_root / "per_frame_speed.csv",
        ["clip", "video", "frame_index", "source_frame_index", "wall_ms", "model_ms", "steady_state"],
        global_per_frame_rows,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
