"""Video inference entrypoint for Deep3D_Pro.

The current pipeline intentionally omits all audio extraction and muxing logic
and produces a silent side-by-side stereo video.
"""

import argparse
import numpy as np
import os
import re
import sys
from collections import deque
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.transforms import PreProcess, tensor2im
from tools.file_utils import ensure_parent_dir
from tools.video_utils import create_video_writer, get_video_metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Run Deep3D_Pro inference on a video.")
    parser.add_argument("--gpu_id", default=0, type=int, help="GPU device id, use -1 for CPU.")
    parser.add_argument("--model", default="./export/deep3d_v1.0.pt", type=str,
                        help="Path to the TorchScript model.")
    parser.add_argument("--video", default="./medias/wood.mp4", type=str, help="Input video path.")
    parser.add_argument("--out", default="./results/wood.mp4", type=str,
                        help="Output side-by-side video path.")
    parser.add_argument("--alpha", default=5, type=int,
                        help="Temporal offset for far-before and far-after frames.")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=None,
                        help="Optional output size. Defaults to the input video resolution.")
    parser.add_argument("--model_size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=None,
                        help="Optional model input size. Defaults to parsing WIDTHxHEIGHT from the model filename.")
    parser.add_argument("--inv", action="store_true",
                        help="Reverse left and right views in the final side-by-side output.")
    return parser.parse_args()


def read_frame(cap, target_size):
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if target_size is not None:
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
    return torch.from_numpy(frame)


def build_initial_window(cap, alpha, target_size):
    first_frame = read_frame(cap, target_size)
    if first_frame is None:
        raise RuntimeError("Cannot read the first frame from the input video.")

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


def get_device(gpu_id):
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def infer_model_size(model_path):
    match = re.search(r'_(\d+)x(\d+)_', os.path.basename(model_path))
    if match is None:
        raise ValueError(
            "Cannot infer model input size from filename. Please pass --model_size WIDTH HEIGHT."
        )
    return int(match.group(1)), int(match.group(2))


def prepare_tensor(frame_tensor, process, device, model_size):
    if (frame_tensor.shape[1], frame_tensor.shape[0]) != model_size:
        frame_np = cv2.resize(frame_tensor.numpy(), model_size, interpolation=cv2.INTER_LANCZOS4)
        frame_tensor = torch.from_numpy(frame_np)
    frame_tensor = frame_tensor.to(device)
    if device.type == "cuda":
        frame_tensor = frame_tensor.half()
    return process(frame_tensor)


def main():
    args = parse_args()

    device = get_device(args.gpu_id)
    metadata = get_video_metadata(args.video)
    output_size = tuple(args.resize) if args.resize else (metadata["width"], metadata["height"])
    model_size = tuple(args.model_size) if args.model_size else infer_model_size(args.model)
    output_width, output_height = output_size

    net = torch.jit.load(args.model, map_location="cpu")
    process = PreProcess()
    if device.type == "cuda":
        net = net.to(device).half()
        process = process.to(device).half()
    else:
        net = net.to(device)
    net.eval()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    ensure_parent_dir(args.out)
    window, last_frame = build_initial_window(cap, args.alpha, output_size)
    writer = create_video_writer(args.out, metadata["fps"], (output_width * 2, output_height))

    x0 = prepare_tensor(window[args.alpha], process, device, model_size)

    for _ in tqdm(range(metadata["frame_count"]), desc="Inference"):
        if args.alpha > 0:
            x1 = window[0]
            x2 = window[args.alpha - 1]
            x3 = window[args.alpha]
            x4 = window[args.alpha + 1]
            x5 = window[-1]
        else:
            x1 = x2 = x3 = x4 = x5 = window[0]

        x1, x2, x3, x4, x5 = [prepare_tensor(frame, process, device, model_size) for frame in (x1, x2, x3, x4, x5)]

        input_data = torch.cat((x1, x2, x0, x3, x4, x5), dim=0).unsqueeze(0)
        with torch.no_grad():
            out = net(input_data)
            x0 = out[0].detach()

        left = tensor2im(x3)
        right = tensor2im(out[0])
        if (right.shape[1], right.shape[0]) != output_size:
            right = cv2.resize(right, output_size, interpolation=cv2.INTER_LANCZOS4)
        stereo_pair = np.concatenate((right, left), axis=1) if args.inv else np.concatenate((left, right), axis=1)
        writer.write(stereo_pair)

        if args.alpha > 0:
            next_frame = read_frame(cap, output_size)
            if next_frame is None:
                next_frame = last_frame
            else:
                last_frame = next_frame
            window.popleft()
            window.append(next_frame)

    writer.release()
    cap.release()


if __name__ == "__main__":
    main()
