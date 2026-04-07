"""Benchmark Deep3D_Pro with TorchScript inference optimizations.

This script mirrors ``inference/benchmark_speed.py`` output layout and summary
format, but focuses on model-forward FPS optimizations:

* ``torch.jit.freeze``
* ``torch.jit.optimize_for_inference``
"""

import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.transforms import PreProcess, tensor2im
from inference.benchmark_speed import (
    SIMULATED_1080P_SIZE,
    build_competition_estimate,
    build_frame_window,
    create_video_writer,
    discover_clips,
    get_device,
    get_device_profile,
    get_gpu_memory_snapshot,
    get_path_size_mb,
    get_process_peak_memory_mb,
    infer_model_size,
    prepare_tensor,
    read_bgr_frame,
    resolve_benchmark_output_size,
    resolve_path,
    save_visualization,
    summarize_timings,
    to_project_relative,
)
from tools.file_utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Deep3D_Pro with JIT freeze/optimize_for_inference."
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
        help="Parent directory for benchmark runs. Outputs are saved under OUT_ROOT/YYYYMMDD/HHMMSS_jit_optimized by default.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        type=str,
        help="Optional run folder name under the date directory. Defaults to HHMMSS_jit_optimized in UTC.",
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
    parser.add_argument("--inv", action="store_true", help="Swap left/right order in the stereo video.")
    parser.add_argument(
        "--no_jit_freeze",
        action="store_true",
        help="Disable torch.jit.freeze. Enabled by default.",
    )
    parser.add_argument(
        "--no_jit_optimize",
        action="store_true",
        help="Disable torch.jit.optimize_for_inference. Enabled by default.",
    )
    return parser.parse_args()


def load_optimized_jit_model(args, model_path, device):
    net = torch.jit.load(str(model_path), map_location="cpu").eval()
    if device.type == "cuda":
        net = net.to(device).half().eval()
    else:
        net = net.to(device).eval()

    applied = {
        "jit_freeze": False,
        "jit_optimize_for_inference": False,
    }

    if not args.no_jit_freeze:
        net = torch.jit.freeze(net)
        applied["jit_freeze"] = True

    if not args.no_jit_optimize:
        net = torch.jit.optimize_for_inference(net)
        applied["jit_optimize_for_inference"] = True

    return net.eval(), applied


class ModelRunner:
    def __init__(self, net):
        self.net = net

    def __call__(self, input_data):
        with torch.inference_mode():
            return self.net(input_data)


def main():
    args = parse_args()

    model_path = resolve_path(args.model)
    data_root = resolve_path(args.data_root)
    out_parent = resolve_path(args.out_root)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_date, run_time = timestamp.split("_", 1)
    run_name = args.run_name or f"{run_time}_jit_optimized"
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

    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device.index)

    net, applied_optimizations = load_optimized_jit_model(args, model_path, device)
    model_runner = ModelRunner(net)
    applied_optimizations["cudnn_benchmark"] = bool(device.type == "cuda")

    process = PreProcess()
    if device.type == "cuda":
        process = process.to(device).half()

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
            out = model_runner(input_data)
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
        "optimizations": applied_optimizations,
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
        writer = csv.DictWriter(
            file,
            fieldnames=["clip", "frame", "wall_ms", "model_ms", "steady_state"],
        )
        writer.writeheader()
        writer.writerows(per_frame_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
