"""Video inference entrypoint for Deep3D_Pro.

The current pipeline intentionally omits all audio extraction and muxing logic
and produces a silent side-by-side stereo video.
"""

import argparse
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


def main():
    args = parse_args()

    device = get_device(args.gpu_id)
    metadata = get_video_metadata(args.video)
    target_size = tuple(args.resize) if args.resize else (metadata["width"], metadata["height"])
    target_width, target_height = target_size

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
    window, last_frame = build_initial_window(cap, args.alpha, target_size)
    writer = create_video_writer(args.out, metadata["fps"], (target_width * 2, target_height))

    x0 = window[args.alpha].to(device)
    if device.type == "cuda":
        x0 = x0.half()
    x0 = process(x0)

    for _ in tqdm(range(metadata["frame_count"]), desc="Inference"):
        if args.alpha > 0:
            x1 = window[0]
            x2 = window[args.alpha - 1]
            x3 = window[args.alpha]
            x4 = window[args.alpha + 1]
            x5 = window[-1]
        else:
            x1 = x2 = x3 = x4 = x5 = window[0]

        frames = [x1, x2, x3, x4, x5]
        frames = [frame.to(device) for frame in frames]
        if device.type == "cuda":
            frames = [frame.half() for frame in frames]
        x1, x2, x3, x4, x5 = [process(frame) for frame in frames]

        input_data = torch.cat((x1, x2, x0, x3, x4, x5), dim=0).unsqueeze(0)
        with torch.no_grad():
            out = net(input_data)
            x0 = out[0].detach()

        left = x3
        right = out[0]
        stereo_pair = torch.cat((right, left), dim=2) if args.inv else torch.cat((left, right), dim=2)
        writer.write(tensor2im(stereo_pair))

        if args.alpha > 0:
            next_frame = read_frame(cap, target_size)
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
