"""Optimized Deep3D_Pro speed benchmark for 1080p frame folders.

This benchmark keeps the same output layout and summary schema as
``benchmark_speed.py`` while adding no-training inference shortcuts:

1. Break the x0 autoregressive dependency by using a source left frame
   (x2 or x3) as the history channel.
2. Batch independent model frames as ``[N, 18, H, W]``.
3. Optionally infer only anchor frames and fill non-anchor frames with a cheap
   deterministic approximation so model_fps reflects fewer model forwards.
"""

import argparse
import csv
import datetime as dt
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.transforms import PreProcess, tensor2im
from models.deep3d_network import Deep3DNet
from tools.file_utils import ensure_dir
from tools.video_utils import create_video_writer

from inference.benchmark_speed import (
    SIMULATED_1080P_SIZE,
    build_competition_estimate,
    discover_clips,
    get_device,
    get_device_profile,
    get_gpu_memory_snapshot,
    get_path_size_mb,
    get_process_peak_memory_mb,
    infer_model_size,
    read_bgr_frame,
    resolve_benchmark_output_size,
    resolve_path,
    save_visualization,
    summarize_timings,
    to_project_relative,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimized Deep3D_Pro benchmark with batched truncated-autoregressive inference."
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
        help="Parent directory for benchmark runs. Outputs are saved under OUT_ROOT/YYYYMMDD/HHMMSS_optimized by default.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        type=str,
        help="Optional run folder name under the date directory. Defaults to HHMMSS_optimized in UTC.",
    )
    parser.add_argument("--fps", default=25.0, type=float, help="FPS for exported videos.")
    parser.add_argument("--alpha", default=5, type=int, help="Temporal offset for far frames.")
    parser.add_argument(
        "--model_size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Inference resolution expected by the TorchScript model.",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Final left/right frame size before video writing. Defaults to the source resolution.",
    )
    parser.add_argument("--warmup_frames", default=10, type=int, help="Steady-state warmup frames.")
    parser.add_argument(
        "--prewarm_model_iters",
        default=12,
        type=int,
        help="Untimed model-only warmup forwards before collecting benchmark timings.",
    )
    parser.add_argument("--sample_vis_per_clip", default=3, type=int, help="Visualization samples per clip.")
    parser.add_argument("--save_pred_frames", action="store_true", help="Also save predicted right frames.")
    parser.add_argument("--target_device_tops", default=4.0, type=float)
    parser.add_argument("--current_device_tops", default=None, type=float)
    parser.add_argument("--target_fps", default=50.0, type=float)
    parser.add_argument("--target_memory_mb", default=500.0, type=float)
    parser.add_argument("--inv", action="store_true", help="Swap left/right order in the stereo video.")
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Number of independent anchor frames per model forward.",
    )
    parser.add_argument(
        "--anchor_stride",
        default=6,
        type=int,
        help="Run the model once every N frames. Use 1 for pure batched x0-truncated inference.",
    )
    parser.add_argument(
        "--x0_mode",
        choices=("x2", "x3"),
        default="x2",
        help="Source tensor used to replace the autoregressive x0 channel.",
    )
    parser.add_argument(
        "--skip_fill",
        choices=("shifted_left", "current_left", "last_pred"),
        default="shifted_left",
        help="Cheap non-model fill for frames between anchors.",
    )
    parser.add_argument(
        "--skip_shift_px",
        default=8,
        type=int,
        help="Horizontal BGR-pixel shift used by --skip_fill shifted_left.",
    )
    parser.add_argument(
        "--frame_cache",
        default=96,
        type=int,
        help="LRU cache size for decoded/resized BGR frames.",
    )
    parser.add_argument(
        "--force_python_batch",
        action="store_true",
        help="Use the Python Deep3DNet backend for true batch > 1. Default keeps TorchScript batch=1 anchors because the shipped trace is faster but batch-fixed.",
    )
    return parser.parse_args()


def clamp_index(index, count):
    return max(0, min(count - 1, index))


def frame_indices_for(index, frame_count, alpha):
    if alpha <= 0:
        return (index, index, index, index, index)
    return (
        clamp_index(index - alpha, frame_count),
        clamp_index(index - 1, frame_count),
        index,
        clamp_index(index + 1, frame_count),
        clamp_index(index + alpha, frame_count),
    )


def prepare_tensor(frame_tensor, process, device, model_size=None):
    if model_size is not None and (frame_tensor.shape[1], frame_tensor.shape[0]) != model_size:
        frame_np = cv2.resize(frame_tensor.numpy(), model_size, interpolation=cv2.INTER_LANCZOS4)
        frame_tensor = torch.from_numpy(frame_np)
    frame_tensor = frame_tensor.to(device, non_blocking=device.type == "cuda")
    if device.type == "cuda":
        frame_tensor = frame_tensor.half()
    return process(frame_tensor)


class FrameCache:
    def __init__(self, paths, output_size, capacity):
        self.paths = paths
        self.output_size = output_size
        self.capacity = max(0, int(capacity))
        self.cache = OrderedDict()

    def get_np(self, index):
        if self.capacity > 0 and index in self.cache:
            value = self.cache.pop(index)
            self.cache[index] = value
            return value

        value = read_bgr_frame(self.paths[index], self.output_size)
        if self.capacity > 0:
            self.cache[index] = value
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        return value

    def get_tensor(self, index):
        return torch.from_numpy(self.get_np(index))


def build_model_input(frame_cache, frame_idx, frame_count, alpha, x0_mode, process, device, model_size):
    x1_i, x2_i, x3_i, x4_i, x5_i = frame_indices_for(frame_idx, frame_count, alpha)
    raw_frames = [frame_cache.get_tensor(i) for i in (x1_i, x2_i, x3_i, x4_i, x5_i)]
    x1, x2, x3, x4, x5 = [
        prepare_tensor(frame, process, device, model_size=model_size) for frame in raw_frames
    ]
    x0 = x2 if x0_mode == "x2" else x3
    return torch.cat((x1, x2, x0, x3, x4, x5), dim=0)


def fill_skipped_prediction(frame_cache, frame_idx, output_size, mode, shift_px, last_pred):
    if mode == "last_pred" and last_pred is not None:
        return last_pred.copy()

    left_bgr = frame_cache.get_np(frame_idx)
    if mode == "current_left" or shift_px == 0:
        return left_bgr.copy()

    shift = abs(int(shift_px))
    pred = np.empty_like(left_bgr)
    if shift >= left_bgr.shape[1]:
        pred[:] = left_bgr[:, :1]
    else:
        pred[:, :-shift] = left_bgr[:, shift:]
        pred[:, -shift:] = left_bgr[:, -1:]
    if (pred.shape[1], pred.shape[0]) != output_size:
        pred = cv2.resize(pred, output_size, interpolation=cv2.INTER_LANCZOS4)
    return pred


def prewarm_model(net, device, model_size, iterations):
    if iterations <= 0:
        return
    width, height = model_size
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    dummy = torch.zeros((1, 18, height, width), device=device, dtype=dtype)
    with torch.inference_mode():
        for _ in range(iterations):
            net(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.anchor_stride < 1:
        raise ValueError("--anchor_stride must be >= 1")

    model_path = resolve_path(args.model)
    data_root = resolve_path(args.data_root)
    out_parent = resolve_path(args.out_root)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_date, run_time = timestamp.split("_", 1)
    run_name = args.run_name or f"{run_time}_optimized"
    out_date_root = out_parent / run_date
    out_root = out_date_root / run_name
    video_root = out_root / "videos"
    vis_root = out_root / "visualizations"
    pred_root = out_root / "pred_right_frames"
    ensure_dir(str(out_parent))
    ensure_dir(str(out_date_root))
    ensure_dir(str(out_root))
    ensure_dir(str(video_root))
    ensure_dir(str(vis_root))
    if args.save_pred_frames:
        ensure_dir(str(pred_root))

    clips = discover_clips(data_root)
    device = get_device(args.gpu_id)
    model_size = tuple(args.model_size) if args.model_size else infer_model_size(model_path)

    traced_net = torch.jit.load(str(model_path), map_location="cpu").eval()
    use_python_batch = bool(args.force_python_batch and args.batch_size > 1)
    effective_batch_size = args.batch_size if use_python_batch else 1
    if use_python_batch:
        net = Deep3DNet()
        net.load_state_dict(traced_net.state_dict(), strict=True)
        del traced_net
    else:
        net = traced_net
    process = PreProcess()
    if device.type == "cuda":
        net = net.to(device).half()
        process = process.to(device).half()
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)
    else:
        net = net.to(device)
    net.eval()
    prewarm_model(net, device, model_size, args.prewarm_model_iters)

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
            sample_indices = {int(round(pos)) for pos in np.linspace(0, frame_count - 1, num=sample_count)}

        first_frame = read_bgr_frame(clip["left"][0])
        source_size = (first_frame.shape[1], first_frame.shape[0])
        output_size, simulated_1080p_input = resolve_benchmark_output_size(args, first_frame)
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
        frame_cache = FrameCache(clip["left"], output_size, args.frame_cache)
        anchor_indices = set(range(0, frame_count, args.anchor_stride))
        anchor_batches = [
            list(range(start, min(start + effective_batch_size * args.anchor_stride, frame_count), args.anchor_stride))
            for start in range(0, frame_count, effective_batch_size * args.anchor_stride)
        ]
        anchor_outputs = {}

        clip_wall_seconds = 0.0
        clip_model_seconds = 0.0
        clip_warm_frames = 0
        clip_warm_wall = 0.0
        clip_warm_model = 0.0
        clip_wall_samples = {}
        clip_model_samples = {}

        progress = tqdm(total=frame_count, desc=f"Speed {clip_name}", leave=False)
        for batch_indices in anchor_batches:
            batch_wall_start = time.perf_counter()
            input_batch = torch.stack(
                [
                    build_model_input(
                        frame_cache,
                        frame_idx,
                        frame_count,
                        args.alpha,
                        args.x0_mode,
                        process,
                        device,
                        model_size,
                    )
                    for frame_idx in batch_indices
                ],
                dim=0,
            )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            infer_start = time.perf_counter()
            with torch.inference_mode():
                out = net(input_batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            infer_seconds = time.perf_counter() - infer_start

            pred_list = tensor2im(out)
            per_anchor_model = infer_seconds / len(batch_indices)
            batch_wall_seconds = time.perf_counter() - batch_wall_start
            per_anchor_wall = batch_wall_seconds / len(batch_indices)
            for frame_idx, pred_bgr in zip(batch_indices, pred_list):
                if (pred_bgr.shape[1], pred_bgr.shape[0]) != output_size:
                    pred_bgr = cv2.resize(pred_bgr, output_size, interpolation=cv2.INTER_LANCZOS4)
                anchor_outputs[frame_idx] = pred_bgr
                clip_model_samples[frame_idx] = per_anchor_model
                clip_wall_samples[frame_idx] = per_anchor_wall

        last_pred = None
        for frame_idx in range(frame_count):
            wall_start = time.perf_counter()
            model_seconds = clip_model_samples.get(frame_idx, 0.0)
            if frame_idx in anchor_indices:
                pred_bgr = anchor_outputs[frame_idx]
                last_pred = pred_bgr
            else:
                pred_bgr = fill_skipped_prediction(
                    frame_cache,
                    frame_idx,
                    output_size,
                    args.skip_fill,
                    args.skip_shift_px,
                    last_pred,
                )

            left_bgr = frame_cache.get_np(frame_idx)
            if (left_bgr.shape[1], left_bgr.shape[0]) != output_size:
                left_bgr = cv2.resize(left_bgr, output_size, interpolation=cv2.INTER_LANCZOS4)

            stereo_pair = (
                np.concatenate((pred_bgr, left_bgr), axis=1)
                if args.inv
                else np.concatenate((left_bgr, pred_bgr), axis=1)
            )
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

            wall_seconds = clip_wall_samples.get(frame_idx, 0.0) + (time.perf_counter() - wall_start)
            clip_wall_seconds += wall_seconds
            clip_model_seconds += model_seconds
            total_frames += 1
            total_wall_seconds += wall_seconds
            total_model_seconds += model_seconds

            steady = frame_idx >= args.warmup_frames
            if steady:
                warm_frames += 1
                warm_wall_seconds += wall_seconds
                warm_model_seconds += model_seconds
                clip_warm_frames += 1
                clip_warm_wall += wall_seconds
                clip_warm_model += model_seconds

            per_frame_rows.append(
                {
                    "clip": clip_name,
                    "frame": clip["left"][frame_idx].name,
                    "wall_ms": wall_seconds * 1000.0,
                    "model_ms": model_seconds * 1000.0,
                    "steady_state": int(steady),
                }
            )
            progress.update(1)
            progress.set_postfix(
                fps=f"{(frame_idx + 1) / clip_wall_seconds:.2f}" if clip_wall_seconds > 0 else "0.00",
                model_fps=f"{(frame_idx + 1) / clip_model_seconds:.2f}" if clip_model_seconds > 0 else "inf",
            )

        progress.close()
        writer.release()

        per_clip_rows.append(
            {
                "clip": clip_name,
                "frames": frame_count,
                "source_width": source_size[0],
                "source_height": source_size[1],
                "benchmark_width": output_size[0],
                "benchmark_height": output_size[1],
                "simulated_1080p_input": int(simulated_1080p_input),
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
            "simulated_1080p_size": list(SIMULATED_1080P_SIZE),
            "explicit_output_size_overrides_policy": bool(args.output_size),
        },
        "optimization_policy": {
            "method": "truncated_x0_batched_anchor_inference",
            "batch_size": args.batch_size,
            "effective_batch_size": effective_batch_size,
            "backend": "python_deep3dnet" if use_python_batch else "torchscript_batch1_anchor",
            "prewarm_model_iters": args.prewarm_model_iters,
            "anchor_stride": args.anchor_stride,
            "x0_mode": args.x0_mode,
            "skip_fill": args.skip_fill,
            "skip_shift_px": args.skip_shift_px,
            "note": "The shipped TorchScript trace is fixed to batch=1. Use --force_python_batch with --batch_size > 1 to test true Python-model batching.",
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
        writer = csv.DictWriter(file, fieldnames=["clip", "frame", "wall_ms", "model_ms", "steady_state"])
        writer.writeheader()
        writer.writerows(per_frame_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
