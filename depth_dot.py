"""
DepthDot operation: pure PyTorch implementation of the custom MXNet DepthDot CUDA operator.

Reconstructs a right-eye view from a left-eye view using predicted depth probability maps.
For each disparity level, shifts the left image horizontally and weights it by the predicted
depth probability, then sums all shifted versions.
"""

import torch
import torch.nn.functional as F


def depth_dot(prob, left_img, scale, upsample=1):
    """
    Reconstruct right view from left view using depth probability map.

    Args:
        prob: (N, D, H, W) depth probability map (after softmax).
              D = scale[1] - scale[0] + 1.
        left_img: (N, C, H*upsample, W*upsample) left image.
        scale: tuple (min_disp, max_disp), e.g. (-15, 17).
        upsample: upsampling factor for higher resolution output.

    Returns:
        (N, C, H*upsample, W*upsample) reconstructed right image.
    """
    s0, s1 = scale
    N, D, H, W = prob.shape
    _, C, uH, uW = left_img.shape

    if upsample > 1:
        prob = F.interpolate(prob, size=(uH, uW), mode='bilinear', align_corners=False)
        u = upsample
    else:
        u = 1

    output = torch.zeros_like(left_img)
    # Channel 0 of prob is unused (matches original CUDA kernel behavior)
    for j in range(s0, s1):
        d_idx = j - s0 + 1  # depth channel index 1..D-1
        weight = prob[:, d_idx:d_idx + 1, :, :]  # (N, 1, uH, uW)
        shift = j * u
        shifted = _shift_image(left_img, shift)
        output = output + weight * shifted

    return output


def _shift_image(img, shift):
    """Shift image horizontally by `shift` pixels with edge clamping.

    output[..., w] = img[..., w - shift], clamped to image boundaries.
    """
    if shift == 0:
        return img
    W = img.shape[-1]
    if shift > 0:
        # output[w] = img[w - shift]; for w < shift, clamp to img[0]
        return torch.cat([
            img[..., 0:1].expand(*img.shape[:-1], shift),
            img[..., :W - shift]
        ], dim=-1)
    else:
        s = -shift
        # output[w] = img[w + s]; for w >= W - s, clamp to img[W-1]
        return torch.cat([
            img[..., s:],
            img[..., -1:].expand(*img.shape[:-1], s)
        ], dim=-1)
