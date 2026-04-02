"""
Dataset for Deep3D training: loads left/right stereo image pairs from video files
or image directories.

Replaces the original LMDB-based Mov3dStack MXNet DataIter with a standard
PyTorch Dataset + DataLoader approach.
"""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def crop_img(img, p, shape, margin=0, test=False, grid=1):
    """Crop an image at position p (x, y). If p is None, pick randomly or center."""
    if p is None:
        if test:
            p = ((img.shape[1] - shape[0] - margin) // grid // 2,
                 (img.shape[0] - shape[1]) // grid // 2)
        else:
            p = (random.randint(0, max(0, (img.shape[1] - shape[0] - margin) // grid)),
                 random.randint(0, max(0, (img.shape[0] - shape[1]) // grid)))
    return img[p[1] * grid:p[1] * grid + shape[1],
               p[0] * grid:p[0] * grid + shape[0]], p


def split_stereo(frame, reshape=None, vert=True, clip=None):
    """Split a stereo frame into left and right views."""
    if vert is True:
        half = frame.shape[1] // 2
        lframe = frame[:, :half]
        rframe = frame[:, half:]
    elif vert is False:
        half = frame.shape[0] // 2
        lframe = frame[:half, :]
        rframe = frame[half:, :]
    else:
        lframe = frame
        rframe = frame
    if clip is not None:
        lframe = lframe[clip[1]:clip[3], clip[0]:clip[2]]
        rframe = rframe[clip[1]:clip[3], clip[0]:clip[2]]
    if reshape is not None:
        lframe = cv2.resize(lframe, reshape)
        rframe = cv2.resize(rframe, reshape)
    return lframe, rframe


def anaglyph(lframe, rframe):
    """Create red-cyan anaglyph from left and right frames (BGR format)."""
    frame = np.zeros_like(lframe)
    frame[:, :, :2] = rframe[:, :, :2]
    frame[:, :, 2:] = lframe[:, :, 2:]
    return frame


def sbs(lframe, rframe):
    """Create side-by-side stereo view (left|right squeezed into original width)."""
    frame = np.zeros_like(lframe)
    sep = lframe.shape[1] // 2
    frame[:, sep:] = cv2.resize(rframe, (sep, rframe.shape[0]))
    frame[:, :sep] = cv2.resize(lframe, (sep, lframe.shape[0]))
    return frame


class StereoImageDataset(Dataset):
    """
    Load stereo pairs from a directory structure:

    data_root/
        left/
            000001.jpg
            000002.jpg
            ...
        right/
            000001.jpg
            000002.jpg
            ...

    Each pair has matching filenames. Images are loaded as float32 [0, 1] RGB.
    """

    def __init__(self, data_root, data_shape=(384, 160), test_mode=False):
        """
        Args:
            data_root: root directory containing left/ and right/ subdirectories.
            data_shape: (width, height) to resize images to.
            test_mode: if True, use center crop; otherwise random crop.
        """
        self.data_shape = data_shape  # (W, H)
        self.test_mode = test_mode

        left_dir = os.path.join(data_root, 'left')
        right_dir = os.path.join(data_root, 'right')

        self.left_files = sorted([
            os.path.join(left_dir, f) for f in os.listdir(left_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
        self.right_files = sorted([
            os.path.join(right_dir, f) for f in os.listdir(right_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
        assert len(self.left_files) == len(self.right_files), \
            f"Mismatch: {len(self.left_files)} left vs {len(self.right_files)} right images"

    def __len__(self):
        return len(self.left_files)

    def __getitem__(self, idx):
        left_img = cv2.imread(self.left_files[idx])
        right_img = cv2.imread(self.right_files[idx])

        if left_img is None:
            raise RuntimeError(f"Cannot read image: {self.left_files[idx]}")
        if right_img is None:
            raise RuntimeError(f"Cannot read image: {self.right_files[idx]}")

        # Resize to target shape
        W, H = self.data_shape
        left_img = cv2.resize(left_img, (W, H))
        right_img = cv2.resize(right_img, (W, H))

        # BGR -> RGB, HWC -> CHW, uint8 -> float32 [0, 1]
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        left_tensor = torch.from_numpy(left_img.transpose(2, 0, 1))
        right_tensor = torch.from_numpy(right_img.transpose(2, 0, 1))

        return left_tensor, right_tensor


class StereoVideoDataset(Dataset):
    """
    Load stereo pairs by extracting frames from a side-by-side 3D video file.

    The video should be a side-by-side 3D video where the left half is the
    left view and the right half is the right view.
    """

    def __init__(self, video_path, data_shape=(384, 160), max_frames=None,
                 vert=True, clip_rect=None, test_mode=False, skip_frames=0):
        """
        Args:
            video_path: path to the stereo video file.
            data_shape: (width, height) to resize each view to.
            max_frames: maximum number of frames to load (None = all).
            vert: True for side-by-side, False for top-bottom.
            clip_rect: optional (x1, y1, x2, y2) crop before splitting.
            test_mode: if True, use center crop; otherwise random crop.
            skip_frames: number of initial frames to skip (e.g. to skip logos).
        """
        self.data_shape = data_shape
        self.test_mode = test_mode
        self.vert = vert
        self.clip_rect = clip_rect

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        # Skip initial frames
        for _ in range(skip_frames):
            cap.read()

        self.left_frames = []
        self.right_frames = []
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            lf, rf = split_stereo(frame, reshape=tuple(data_shape), vert=vert, clip=clip_rect)
            self.left_frames.append(lf)
            self.right_frames.append(rf)
            count += 1
            if max_frames is not None and count >= max_frames:
                break
        cap.release()

    def __len__(self):
        return len(self.left_frames)

    def __getitem__(self, idx):
        left_img = self.left_frames[idx]
        right_img = self.right_frames[idx]

        # BGR -> RGB, float32 [0, 1]
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        left_tensor = torch.from_numpy(left_img.transpose(2, 0, 1))
        right_tensor = torch.from_numpy(right_img.transpose(2, 0, 1))

        return left_tensor, right_tensor


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True):
    """Create a DataLoader from a dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
