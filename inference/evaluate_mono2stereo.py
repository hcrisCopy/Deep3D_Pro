"""Evaluate Deep3D_Pro on an image-based mono-to-stereo test set."""

import argparse
import csv
import json
import math
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.transforms import PreProcess, tensor2im
from tools.file_utils import ensure_dir


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Deep3D_Pro on a left/right image-pair benchmark."
    )
    parser.add_argument("--gpu_id", default=0, type=int, help="GPU device id, use -1 for CPU.")
    parser.add_argument(
        "--model",
        default="../data/pretrained/deep3d_v1.0_1280x720_cuda.pt",
        type=str,
        help="Path to the TorchScript model relative to Deep3D_Pro.",
    )
    parser.add_argument(
        "--data_root",
        default="../data/test_set/mono2stereo_test",
        type=str,
        help="Path to the test set root containing clip/left and clip/right.",
    )
    parser.add_argument(
        "--out_root",
        default="../data/test_on_mono",
        type=str,
        help="Directory to save evaluation outputs.",
    )
    parser.add_argument(
        "--alpha",
        default=5,
        type=int,
        help="Kept for interface compatibility. Single-image evaluation mode does not use temporal offsets.",
    )
    parser.add_argument(
        "--sample_vis_per_clip",
        default=2,
        type=int,
        help="How many visualization samples to save per clip.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Optional resize for evaluation. If omitted, metrics are computed at 1280x800 and model inference runs at 1280x720.",
    )
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def get_device(gpu_id):
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def list_image_files(folder):
    return sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    )


def discover_clips(data_root):
    clips = []
    for clip_dir in sorted([path for path in data_root.iterdir() if path.is_dir()]):
        left_dir = clip_dir / "left"
        right_dir = clip_dir / "right"
        if not left_dir.is_dir() or not right_dir.is_dir():
            continue
        left_files = {path.name: path for path in list_image_files(left_dir)}
        right_files = {path.name: path for path in list_image_files(right_dir)}
        common_names = sorted(set(left_files) & set(right_files))
        if not common_names:
            continue
        clips.append(
            {
                "name": clip_dir.name,
                "left": [left_files[name] for name in common_names],
                "right": [right_files[name] for name in common_names],
            }
        )
    if not clips:
        raise RuntimeError(f"No valid left/right clips found in: {data_root}")
    return clips


def read_frame(path, target_size=None):
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Cannot read image: {path}")
    if target_size is not None:
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def detect_edges(image, low, high):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    return edges


def edge_overlap(edge1, edge2):
    intersection = np.logical_and(edge1, edge2).sum()
    union = np.logical_or(edge1, edge2).sum()
    if union == 0:
        return 1.0
    return intersection / union


def compute_siou(pred, target, left):
    left_edges = detect_edges(left, 100, 200)
    pred_edges = detect_edges(pred, 100, 200)
    right_edges = detect_edges(target, 100, 200)

    diff_gl = abs(pred - left)
    diff_rl = abs(target - left)
    diff_gl = cv2.cvtColor(diff_gl, cv2.COLOR_BGR2GRAY)
    diff_rl = cv2.cvtColor(diff_rl, cv2.COLOR_BGR2GRAY)
    diff_gl_ = np.zeros(diff_rl.shape)
    diff_rl_ = np.zeros(diff_rl.shape)
    diff_gl_[diff_gl > 5] = 1
    diff_rl_[diff_rl > 5] = 1

    edge_overlap_gr = edge_overlap(pred_edges, right_edges)
    diff_overlap_grl = edge_overlap(diff_gl_, diff_rl_)

    return 0.75 * edge_overlap_gr + 0.25 * diff_overlap_grl


def eval_stereo(pred, target, left):
    max_pixel = 255.0
    assert pred.shape == target.shape
    diff = pred - target

    mse_err = np.mean(diff ** 2)
    rmse = np.sqrt(mse_err)
    _ = np.mean(np.abs(diff))
    if rmse == 0:
        psnr = 32.0
    else:
        psnr = 20 * np.log10(max_pixel / rmse)

    ssim_value, _ = ssim(pred, target, full=True, multichannel=True, win_size=7, channel_axis=2)
    siou_value = compute_siou(pred, target, left)

    return {
        "rmse": float(rmse),
        "mse": float(mse_err),
        "siou": float(siou_value),
        "psnr": float(psnr),
        "ssim": float(ssim_value),
    }


def save_visualization(save_path, left, pred, target):
    diff_pred_gt = cv2.absdiff(pred, target)
    diff_pred_left = cv2.absdiff(pred, left)

    diff_gray = cv2.cvtColor(diff_pred_gt, cv2.COLOR_RGB2GRAY)
    diff_color = cv2.applyColorMap(diff_gray, cv2.COLORMAP_HOT)
    diff_color = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)

    anaglyph = np.zeros_like(pred)
    anaglyph[:, :, 0] = left[:, :, 0]
    anaglyph[:, :, 1] = pred[:, :, 1]
    anaglyph[:, :, 2] = pred[:, :, 2]

    h, w = left.shape[:2]
    canvas = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)
    tiles = [left, pred, target, diff_color, anaglyph, diff_pred_left]
    labels = [
        "Left Input",
        "Pred Right",
        "GT Right",
        "Pred-GT Diff",
        "Anaglyph",
        "Pred-Left Diff",
    ]

    for index, (tile, label) in enumerate(zip(tiles, labels)):
        row = index // 3
        col = index % 3
        canvas[row * h:(row + 1) * h, col * w:(col + 1) * w] = tile
        origin = (col * w + 16, row * h + 36)
        cv2.putText(
            canvas,
            label,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(save_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


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
        "allocated_mb": torch.cuda.memory_allocated(device) / (1024 ** 2),
        "reserved_mb": torch.cuda.memory_reserved(device) / (1024 ** 2),
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "max_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 ** 2),
        "free_mb": free_bytes / (1024 ** 2),
        "total_mb": total_bytes / (1024 ** 2),
        "device_name": torch.cuda.get_device_name(device),
    }


def main():
    args = parse_args()

    model_path = resolve_path(args.model)
    data_root = resolve_path(args.data_root)
    out_root = resolve_path(args.out_root)
    vis_root = out_root / "visualizations"
    pred_root = out_root / "predictions"
    ensure_dir(str(out_root))
    ensure_dir(str(vis_root))
    ensure_dir(str(pred_root))

    device = get_device(args.gpu_id)
    clips = discover_clips(data_root)
    eval_size = tuple(args.resize) if args.resize else (1280, 800)
    model_size = (1280, 720)

    net = torch.jit.load(str(model_path), map_location="cpu")
    process = PreProcess()
    if device.type == "cuda":
        net = net.to(device).half()
        process = process.to(device).half()
        torch.cuda.reset_peak_memory_stats(device)
    else:
        net = net.to(device)
    net.eval()

    overall_metrics = {key: 0.0 for key in ("mse", "rmse", "psnr", "ssim", "siou")}
    clip_rows = []
    frame_rows = []
    total_frames = 0
    total_model_seconds = 0.0
    total_wall_seconds = 0.0
    eval_start = time.perf_counter()
    process_peak_mb_before = get_process_peak_memory_mb()

    for clip in clips:
        clip_name = clip["name"]
        num_frames = len(clip["left"])
        clip_metric_sums = {key: 0.0 for key in overall_metrics}
        clip_model_seconds = 0.0
        clip_wall_seconds = 0.0

        clip_vis_dir = vis_root / clip_name
        clip_pred_dir = pred_root / clip_name
        ensure_dir(str(clip_vis_dir))
        ensure_dir(str(clip_pred_dir))

        sample_indices = set()
        if args.sample_vis_per_clip > 0:
            sample_indices = {
                int(round(position))
                for position in np.linspace(0, num_frames - 1, num=min(args.sample_vis_per_clip, num_frames))
            }

        left_frames = [read_frame(path, eval_size) for path in clip["left"]]
        right_frames = [read_frame(path, eval_size) for path in clip["right"]]

        progress = tqdm(range(num_frames), desc=f"Eval {clip_name}", leave=False)
        for frame_idx in progress:
            wall_start = time.perf_counter()
            current_left = left_frames[frame_idx]

            # mono2stereo_test is an image benchmark rather than a temporal sequence.
            # We therefore build the 6-frame network input by repeating the same left image
            # in every slot instead of borrowing neighboring samples from unrelated images.
            frame_candidates = [current_left] * 6

            input_tensors = []
            for frame_rgb in frame_candidates:
                model_rgb = cv2.resize(frame_rgb, model_size, interpolation=cv2.INTER_LANCZOS4) if (frame_rgb.shape[1], frame_rgb.shape[0]) != model_size else frame_rgb
                frame_tensor = torch.from_numpy(model_rgb).to(device)
                if device.type == "cuda":
                    frame_tensor = frame_tensor.half()
                frame_tensor = process(frame_tensor)
                input_tensors.append(frame_tensor)

            input_data = torch.cat(input_tensors, dim=0).unsqueeze(0)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            infer_start = time.perf_counter()
            with torch.no_grad():
                out = net(input_data)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            infer_seconds = time.perf_counter() - infer_start

            pred_rgb = tensor2im(out[0])
            if (pred_rgb.shape[1], pred_rgb.shape[0]) != eval_size:
                pred_rgb = cv2.resize(pred_rgb, eval_size, interpolation=cv2.INTER_LANCZOS4)
            left_rgb = left_frames[frame_idx]
            right_rgb = right_frames[frame_idx]
            metrics = eval_stereo(
                cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR),
            )

            for key, value in metrics.items():
                overall_metrics[key] += value
                clip_metric_sums[key] += value

            pred_path = clip_pred_dir / clip["left"][frame_idx].name
            cv2.imwrite(str(pred_path), cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))

            if frame_idx in sample_indices:
                vis_path = clip_vis_dir / f"{clip['left'][frame_idx].stem}_viz.jpg"
                save_visualization(vis_path, left_rgb, pred_rgb, right_rgb)

            wall_seconds = time.perf_counter() - wall_start
            clip_model_seconds += infer_seconds
            clip_wall_seconds += wall_seconds
            total_model_seconds += infer_seconds
            total_wall_seconds += wall_seconds
            total_frames += 1

            frame_rows.append(
                {
                    "clip": clip_name,
                    "frame": clip["left"][frame_idx].name,
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "siou": metrics["siou"],
                    "mse": metrics["mse"],
                    "rmse": metrics["rmse"],
                    "model_ms": infer_seconds * 1000.0,
                    "wall_ms": wall_seconds * 1000.0,
                }
            )
            progress.set_postfix(
                psnr=f"{metrics['psnr']:.2f}",
                ssim=f"{metrics['ssim']:.4f}",
                siou=f"{metrics['siou']:.4f}",
            )

        clip_rows.append(
            {
                "clip": clip_name,
                "frames": num_frames,
                "psnr": clip_metric_sums["psnr"] / num_frames,
                "ssim": clip_metric_sums["ssim"] / num_frames,
                "siou": clip_metric_sums["siou"] / num_frames,
                "mse": clip_metric_sums["mse"] / num_frames,
                "rmse": clip_metric_sums["rmse"] / num_frames,
                "fps": num_frames / clip_wall_seconds if clip_wall_seconds > 0 else 0.0,
                "model_fps": num_frames / clip_model_seconds if clip_model_seconds > 0 else 0.0,
            }
        )

    elapsed = time.perf_counter() - eval_start
    summary = {
        "model_path": str(model_path),
        "data_root": str(data_root),
        "out_root": str(out_root),
        "eval_size": list(eval_size),
        "model_size": list(model_size),
        "evaluation_mode": "single_image_independent",
        "device": str(device),
        "clips": len(clips),
        "frames": total_frames,
        "metrics": {key: value / total_frames for key, value in overall_metrics.items()},
        "fps": total_frames / total_wall_seconds if total_wall_seconds > 0 else 0.0,
        "model_fps": total_frames / total_model_seconds if total_model_seconds > 0 else 0.0,
        "elapsed_fps": total_frames / elapsed if elapsed > 0 else 0.0,
        "time_seconds": {
            "elapsed": elapsed,
            "accumulated_wall": total_wall_seconds,
            "accumulated_model": total_model_seconds,
        },
        "memory": {
            "process_peak_mb_before": process_peak_mb_before,
            "process_peak_mb_after": get_process_peak_memory_mb(),
            "gpu": get_gpu_memory_snapshot(device),
        },
        "per_clip": clip_rows,
    }

    with (out_root / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    with (out_root / "per_clip_metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["clip", "frames", "psnr", "ssim", "siou", "mse", "rmse", "fps", "model_fps"],
        )
        writer.writeheader()
        writer.writerows(clip_rows)

    with (out_root / "per_frame_metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["clip", "frame", "psnr", "ssim", "siou", "mse", "rmse", "model_ms", "wall_ms"],
        )
        writer.writeheader()
        writer.writerows(frame_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
