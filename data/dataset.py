"""
Dataset for Deep3D v1.0 training: loads temporal sequences of stereo image pairs.

Each training sample provides:
- 6 left-view frames at different temporal offsets (concatenated as 18 channels)
- 1 right-view target frame (ground truth)
"""

import os
import random
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _is_image_file(name):
    return name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))


def _natural_key(name):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', name)]


class TemporalStereoDataset(Dataset):
    """Load temporal sequences of stereo pairs for multi-frame Deep3D training.

    Directory structure:
        data_root/
            clip_id/
                left/   (sorted frame images)
                right/  (sorted frame images, matching filenames)

    Each sample returns:
        input_tensor: (18, H, W) — 6 concatenated left/right frames in [0, 1] RGB
            Layout: [left[t-alpha], left[t-1], x0, left[t], left[t+1], left[t+alpha]]
            where x0 = right[t-1] (teacher forcing) or left[t] depending on prev_mode.
        target: (3, H, W) — right view at time t, in [0, 1] RGB.

    Args:
        data_root: root directory containing clip subdirectories.
        data_shape: (width, height) to resize frames to.
        alpha: temporal offset for far-before/after frames (default: 5).
        prev_mode: how to generate x0 (previous prediction channel):
            'right_gt' — use right[t-1] ground truth (teacher forcing, default)
            'left' — use left[t] (no temporal dependency, simpler)
    """

    def __init__(self, data_root, data_shape=(640, 360), alpha=5, prev_mode='right_gt'):
        self.data_shape = data_shape  # (W, H)
        self.alpha = alpha
        self.prev_mode = prev_mode
        self.clips = self._load_clips(data_root)
        self.sample_indices = self._build_index()

        if len(self.sample_indices) == 0:
            raise RuntimeError(f'No stereo pairs found in: {data_root}')

    @classmethod
    def _load_clips(cls, data_root):
        clips = []
        for clip_name in sorted(os.listdir(data_root), key=_natural_key):
            clip_dir = os.path.join(data_root, clip_name)
            if not os.path.isdir(clip_dir):
                continue
            left_dir = os.path.join(clip_dir, 'left')
            right_dir = os.path.join(clip_dir, 'right')
            if not os.path.isdir(left_dir) or not os.path.isdir(right_dir):
                continue

            left_files = sorted([f for f in os.listdir(left_dir) if _is_image_file(f)], key=_natural_key)
            right_files = sorted([f for f in os.listdir(right_dir) if _is_image_file(f)], key=_natural_key)
            common = sorted(set(left_files) & set(right_files), key=_natural_key)

            if len(common) > 0:
                clips.append({
                    'left': [os.path.join(left_dir, f) for f in common],
                    'right': [os.path.join(right_dir, f) for f in common],
                })
        return clips

    def _build_index(self):
        indices = []
        for clip_idx, clip in enumerate(self.clips):
            n_frames = len(clip['left'])
            for frame_idx in range(n_frames):
                indices.append((clip_idx, frame_idx))
        return indices

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        clip_idx, t = self.sample_indices[idx]
        clip = self.clips[clip_idx]
        n = len(clip['left'])
        alpha = self.alpha

        # Temporal indices with boundary clamping
        t_far_before = max(0, t - alpha)
        t_before = max(0, t - 1)
        t_after = min(n - 1, t + 1)
        t_far_after = min(n - 1, t + alpha)

        # Load left frames
        left_far_before = self._load_image(clip['left'][t_far_before])
        left_before = self._load_image(clip['left'][t_before])
        left_current = self._load_image(clip['left'][t])
        left_after = self._load_image(clip['left'][t_after])
        left_far_after = self._load_image(clip['left'][t_far_after])

        # x0: previous prediction channel
        if self.prev_mode == 'right_gt':
            # Teacher forcing: use right[t-1] as previous prediction
            # For first frame (t=0), use left[0] to avoid information leak
            if t > 0:
                x0 = self._load_image(clip['right'][t_before])
            else:
                x0 = left_current.clone()
        else:  # 'left'
            x0 = left_current.clone()

        # Target: right view at current time
        right_current = self._load_image(clip['right'][t])

        # Stack input: [x1, x2, x0, x3, x4, x5] = 18 channels
        input_tensor = torch.cat([
            left_far_before,   # x1: far-before
            left_before,       # x2: just-before
            x0,                # x0: previous prediction
            left_current,      # x3: current
            left_after,        # x4: just-after
            left_far_after,    # x5: far-after
        ], dim=0)  # (18, H, W)

        return input_tensor, right_current

    def _load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f'Cannot read image: {path}')
        W, H = self.data_shape
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img.transpose(2, 0, 1))  # (3, H, W)


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
