"""
Data preprocessing: extract stereo frame pairs from side-by-side 3D video files
into left/ and right/ image directories for training.

Usage:
    python preprocess.py movie.mkv output_dir/ --sbs3d
    python preprocess.py movie.mkv output_dir/ --no-sbs3d  # for 2D-only video
"""

import argparse
import os
import logging

import cv2
import numpy as np


def get_clip_rect(fname, vert=True):
    """Detect black bars and compute clipping rectangle for a stereo video."""
    cap = cv2.VideoCapture(fname)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {fname}')

    # Skip initial frames (logos, intros)
    for _ in range(24 * 60 * 2):
        ret = cap.read()[0]
        if not ret:
            break

    ret, sample = cap.read()
    if not ret:
        raise RuntimeError('Video too short for analysis')
    shape = sample.shape
    logging.info(f'Original shape: {shape}')

    # Accumulate average frame to find black bars
    acc = np.zeros(shape, dtype=np.float64)
    count = 0
    for _ in range(24 * 60):
        ret, frame = cap.read()
        if not ret:
            break
        acc += frame
        count += 1
    if count > 0:
        acc /= count

    y0 = 0
    while y0 < shape[0] and acc[y0].mean() < 2:
        y0 += 1
    y1 = shape[0] - 1
    while y1 > 0 and acc[y1].mean() < 2:
        y1 -= 1
    y1 += 1

    logging.info(f'Detected content region: y=[{y0}, {y1}), height={y1 - y0}')

    cap.release()

    if vert:
        half_w = shape[1] // 2
        return (0, y0, half_w, y1)
    else:
        return (0, y0, shape[1], y1)


def extract_frames(video_path, output_dir, reshape=(384, 160), vert=True,
                   clip_rect=None, max_frames=None, skip_frames=0,
                   is_2d=False, frame_interval=1):
    """Extract frames from video and save as left/right image pairs."""
    left_dir = os.path.join(output_dir, 'left')
    right_dir = os.path.join(output_dir, 'right')
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')

    for _ in range(skip_frames):
        cap.read()

    count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % frame_interval != 0:
            continue

        if is_2d:
            # Same frame for both views (for testing without stereo source)
            lframe = frame
            rframe = frame
            if clip_rect is not None:
                lframe = lframe[clip_rect[1]:clip_rect[3], clip_rect[0]:clip_rect[2]]
                rframe = rframe[clip_rect[1]:clip_rect[3], clip_rect[0]:clip_rect[2]]
            lframe = cv2.resize(lframe, reshape)
            rframe = cv2.resize(rframe, reshape)
        else:
            if vert:
                half = frame.shape[1] // 2
                lframe = frame[:, :half]
                rframe = frame[:, half:]
            else:
                half = frame.shape[0] // 2
                lframe = frame[:half, :]
                rframe = frame[half:, :]
            if clip_rect is not None:
                lframe = lframe[clip_rect[1]:clip_rect[3], clip_rect[0]:clip_rect[2]]
                rframe = rframe[clip_rect[1]:clip_rect[3], clip_rect[0]:clip_rect[2]]
            lframe = cv2.resize(lframe, reshape)
            rframe = cv2.resize(rframe, reshape)

        fname = f'{saved:06d}.jpg'
        cv2.imwrite(os.path.join(left_dir, fname), lframe, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(right_dir, fname), rframe, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

        if saved % 1000 == 0:
            logging.info(f'Saved {saved} frame pairs')

        if max_frames is not None and saved >= max_frames:
            break

    cap.release()
    logging.info(f'Extracted {saved} frame pairs from {video_path}')
    return saved


def main():
    parser = argparse.ArgumentParser(description='Extract stereo frames from video')
    parser.add_argument('video', type=str, help='Path to input video file')
    parser.add_argument('output', type=str, help='Output directory')
    parser.add_argument('--sbs3d', action='store_true', default=True,
                        help='Video is side-by-side 3D (default)')
    parser.add_argument('--no-sbs3d', dest='sbs3d', action='store_false',
                        help='Video is 2D (not stereo)')
    parser.add_argument('--reshape', type=int, nargs=2, default=[384, 160],
                        help='Output frame size (width height)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames to extract')
    parser.add_argument('--skip_frames', type=int, default=0,
                        help='Initial frames to skip')
    parser.add_argument('--frame_interval', type=int, default=1,
                        help='Extract every N-th frame')
    parser.add_argument('--auto_clip', action='store_true',
                        help='Automatically detect and clip black bars')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    clip_rect = None
    if args.auto_clip:
        clip_rect = get_clip_rect(args.video, vert=args.sbs3d)
        logging.info(f'Auto clip rect: {clip_rect}')

    extract_frames(
        args.video, args.output,
        reshape=tuple(args.reshape),
        vert=args.sbs3d,
        clip_rect=clip_rect,
        max_frames=args.max_frames,
        skip_frames=args.skip_frames,
        is_2d=not args.sbs3d,
        frame_interval=args.frame_interval,
    )


if __name__ == '__main__':
    main()
