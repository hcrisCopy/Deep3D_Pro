"""
Convert a 2D video to 3D using a trained Deep3D model.

Produces anaglyph (red-cyan) and side-by-side 3D output videos.

Usage:
    python convert_movie.py model.pth input.mp4 --output output_3d
"""

import argparse
import logging
import os

import cv2
import numpy as np
import torch

from model import Deep3DNet
from dataset import split_stereo, anaglyph, sbs


def get_args():
    parser = argparse.ArgumentParser(description='Convert 2D video to 3D')
    parser.add_argument('model_path', type=str, help='Path to model checkpoint (.pth)')
    parser.add_argument('input_video', type=str, help='Path to input 2D video')
    parser.add_argument('--output', type=str, default='output_3d',
                        help='Output filename prefix')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id (-1 for CPU)')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for inference')
    parser.add_argument('--data_shape', type=int, nargs=2, default=[384, 160],
                        help='Width and height for model input')
    parser.add_argument('--scale_min', type=int, default=-15)
    parser.add_argument('--scale_max', type=int, default=17)
    parser.add_argument('--fps', type=float, default=24,
                        help='Output video FPS')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Max number of frames to process')
    parser.add_argument('--codec', type=str, default='mp4v',
                        help='FourCC codec for output video')
    return parser.parse_args()


@torch.no_grad()
def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    logging.info(f'Using device: {device}')

    # Model
    scale = (args.scale_min, args.scale_max)
    W, H = args.data_shape
    model = Deep3DNet(scale=scale, input_height=H, input_width=W)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    logging.info(f'Loaded model from {args.model_path}')

    # Open input video
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {args.input_video}')

    in_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f'Input video: {total_frames} frames at {in_fps:.1f} FPS')

    # Output videos
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    out_ana = cv2.VideoWriter(f'{args.output}_anaglyph.mp4', fourcc, in_fps, (W, H))
    out_sbs = cv2.VideoWriter(f'{args.output}_sbs.mp4', fourcc, in_fps, (W, H))
    out_left = cv2.VideoWriter(f'{args.output}_left.mp4', fourcc, in_fps, (W, H))
    out_right = cv2.VideoWriter(f'{args.output}_pred_right.mp4', fourcc, in_fps, (W, H))

    frame_count = 0
    batch_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to model input size
        frame = cv2.resize(frame, (W, H))
        batch_frames.append(frame)

        if len(batch_frames) == args.batch_size:
            _process_batch(batch_frames, model, device, out_ana, out_sbs, out_left, out_right)
            frame_count += len(batch_frames)
            batch_frames = []

            if frame_count % 100 == 0:
                logging.info(f'Processed {frame_count}/{total_frames} frames')

        if args.max_frames is not None and frame_count >= args.max_frames:
            break

    # Process remaining frames
    if batch_frames:
        _process_batch(batch_frames, model, device, out_ana, out_sbs, out_left, out_right)
        frame_count += len(batch_frames)

    cap.release()
    out_ana.release()
    out_sbs.release()
    out_left.release()
    out_right.release()

    logging.info(f'Done. Processed {frame_count} frames. Output prefix: {args.output}')


def _process_batch(frames, model, device, out_ana, out_sbs, out_left, out_right):
    """Process a batch of frames through the model and write to output videos."""
    # Prepare input tensor
    batch = []
    for f in frames:
        img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        batch.append(img)
    batch_tensor = torch.from_numpy(np.array(batch)).to(device)

    # Forward pass
    output = model(batch_tensor)

    # Convert back to numpy BGR uint8
    output = output.cpu().numpy()
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    output = output.transpose(0, 2, 3, 1)  # NCHW -> NHWC

    for i in range(len(frames)):
        left_bgr = frames[i]
        right_rgb = output[i]
        right_bgr = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR)

        out_left.write(left_bgr)
        out_right.write(right_bgr)
        out_ana.write(anaglyph(left_bgr, right_bgr))
        out_sbs.write(sbs(left_bgr, right_bgr))


if __name__ == '__main__':
    main()
