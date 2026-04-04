"""
Deep3D v1.0 model: multi-frame optical flow network for 2D-to-3D conversion (PyTorch).

Architecture: 4-level coarse-to-fine refinement blocks, each predicting optical flow
fields and blending masks. Uses backward warping + mask blending to reconstruct the
right-eye view from multiple left-eye frames.

Reverse-engineered from the pretrained JIT model: deep3d_v1.0_1280x720_cuda.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Flow scaling constant from the pretrained model
FLOW_SCALE = 0.08235294117647  # ≈ 7/85


class ConvBnPReLU(nn.Sequential):
    """Conv2d (no bias) + BatchNorm2d + PReLU.

    Using nn.Sequential so that parameter names match the JIT model:
    0.weight (Conv2d), 1.weight/1.bias (BN), 2.weight (PReLU).
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch),
        )


class RefinementBlock(nn.Module):
    """One level of iterative refinement.

    Downsamples the input spatially, processes through conv layers with residual,
    then upsamples back. Outputs optical flow residual (8ch) and mask residual (3ch).

    Args:
        in_channels: input channels (19 for block0, 24 for blocks 1-3).
        mid_channels: channels after first conv0 layer.
        out_channels: channels after second conv0 layer and in convblock.
        spatial_scale: downsample factor before processing.
        flow_out_scale: scaling factor applied to raw flow output.
        flow_in_scale: scaling factor applied to flow input for concatenation.
            None for block0 (no flow input).
    """

    def __init__(self, in_channels, mid_channels, out_channels,
                 spatial_scale, flow_out_scale, flow_in_scale=None):
        super().__init__()
        self.spatial_scale = spatial_scale
        self.flow_out_scale = flow_out_scale
        self.flow_in_scale = flow_in_scale

        # conv0: 2 ConvBnPReLU with stride 2 each (total 4x downsample)
        self.conv0 = nn.Sequential(
            ConvBnPReLU(in_channels, mid_channels, 3, stride=2, padding=1),
            ConvBnPReLU(mid_channels, out_channels, 3, stride=2, padding=1),
        )

        # convblock: 8 ConvBnPReLU with stride 1 (same spatial size), used as residual
        self.convblock = nn.Sequential(
            *[ConvBnPReLU(out_channels, out_channels, 3, stride=1, padding=1)
              for _ in range(8)]
        )

        # lastconv: transposed conv to upsample 2x, outputs 11ch (8 flow + 3 mask)
        self.lastconv = nn.ConvTranspose2d(out_channels, 11, 4, stride=2, padding=1)

    def forward(self, x, flow_in=None):
        H, W = x.shape[2], x.shape[3]

        # Spatial downsampling
        x_down = F.interpolate(x, scale_factor=self.spatial_scale,
                               mode='bilinear', align_corners=True)

        # Optionally concatenate scaled flow
        if flow_in is not None and self.flow_in_scale is not None:
            flow_down = F.interpolate(flow_in, scale_factor=self.spatial_scale,
                                      mode='bilinear', align_corners=True)
            flow_down = flow_down * self.flow_in_scale
            x_down = torch.cat([x_down, flow_down], dim=1)

        # Process
        feat = self.conv0(x_down)
        feat = self.convblock(feat) + feat  # residual

        # Upsample back
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, size=(H, W), mode='bilinear', align_corners=True)

        # Split into flow (8ch) and mask (3ch)
        flow = tmp[:, :8] * self.flow_out_scale
        mask = tmp[:, 8:]

        return flow, mask


def backwarp(img, flow):
    """Backward warp image using optical flow via grid_sample.

    Args:
        img: (N, C, H, W) image to warp.
        flow: (N, 2, H, W) optical flow in pixel units.

    Returns:
        (N, C, H, W) warped image.
    """
    N, _, H, W = flow.shape

    # Build the base grid from tensors derived from flow so traced TorchScript
    # keeps device placement dynamic instead of baking in the trace device.
    grid_x = flow.new_ones((W,), dtype=torch.float32).cumsum(0) - 1.0
    grid_x = grid_x / ((W - 1) / 2.0) - 1.0
    grid_x = grid_x.view(1, 1, 1, W).expand(N, -1, H, -1)
    grid_y = flow.new_ones((H,), dtype=torch.float32).cumsum(0) - 1.0
    grid_y = grid_y / ((H - 1) / 2.0) - 1.0
    grid_y = grid_y.view(1, 1, H, 1).expand(N, -1, -1, W)
    base_grid = torch.cat([grid_x, grid_y], dim=1).type_as(flow)  # (N, 2, H, W)

    # Normalize flow to [-1, 1] range for grid_sample
    flow_norm = torch.empty_like(flow)
    flow_norm[:, 0:1] = flow[:, 0:1] / ((W - 1) / 2.0)
    flow_norm[:, 1:2] = flow[:, 1:2] / ((H - 1) / 2.0)

    grid = (base_grid + flow_norm).permute(0, 2, 3, 1)  # (N, H, W, 2)
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='border',
                         align_corners=True)


class Deep3DNet(nn.Module):
    """Deep3D v1.0 network for multi-frame 2D-to-3D video conversion.

    Takes 6 temporal left-eye frames as input and produces the right-eye view
    for the current frame using optical flow warping and mask blending.

    Input layout (18 channels):
        [0:3]   x1  - far-before left frame (t - alpha)
        [3:6]   x2  - just-before left frame (t - 1)
        [6:9]   x0  - previous prediction / right (t - 1)
        [9:12]  x3  - current left frame (t)
        [12:15] x4  - just-after left frame (t + 1)
        [15:18] x5  - far-after left frame (t + alpha)
    """

    def __init__(self):
        super().__init__()
        self.flow_scale = float(FLOW_SCALE)

        # Block0: coarsest level (1/8 scale), input: 19ch (18 imgs + 1 onehot)
        # flow_out_scale = 2/spatial_scale = 16
        self.block0 = RefinementBlock(
            in_channels=19, mid_channels=96, out_channels=192,
            spatial_scale=0.125, flow_out_scale=16.0,
            flow_in_scale=None,
        )

        # Block1: 1/4 scale, input: 24ch (16 warped+mask+onehot + 8 flow_scaled)
        # flow_out_scale = 2/spatial_scale = 8, flow_in_scale = spatial_scale = 0.25
        self.block1 = RefinementBlock(
            in_channels=24, mid_channels=64, out_channels=128,
            spatial_scale=0.25, flow_out_scale=8.0,
            flow_in_scale=0.25,
        )

        # Block2: 1/2 scale
        # flow_out_scale = 2/0.5 = 4, flow_in_scale = 0.5
        self.block2 = RefinementBlock(
            in_channels=24, mid_channels=48, out_channels=96,
            spatial_scale=0.5, flow_out_scale=4.0,
            flow_in_scale=0.5,
        )

        # Block3: full scale (1x)
        # flow_out_scale = 2/1.0 = 2, flow_in_scale = 1.0
        self.block3 = RefinementBlock(
            in_channels=24, mid_channels=32, out_channels=64,
            spatial_scale=1.0, flow_out_scale=2.0,
            flow_in_scale=1.0,
        )

    def forward(self, imgs):
        """
        Args:
            imgs: (N, 18, H, W) — 6 concatenated RGB frames, float [0, 1].

        Returns:
            output: (N, 3, H, W) — predicted right-eye view, float [0, 1].
        """
        N, _, H, W = imgs.shape

        # Extract reference frames for warping
        bef = imgs[:, 0:3]     # x1: far-before
        pred = imgs[:, 6:9]    # x0: previous prediction
        cur = imgs[:, 9:12]    # x3: current
        aft = imgs[:, 15:18]   # x5: far-after

        # Onehot indicator channel
        onehot = imgs.new_full((N, 1, H, W), self.flow_scale)

        # ---- Block 0: coarsest estimation ----
        x = torch.cat([imgs, onehot], dim=1)  # (N, 19, H, W)
        flow, mask = self.block0(x)

        # Warp reference frames
        w_bef = backwarp(bef, flow[:, 0:2])
        w_pred = backwarp(pred, flow[:, 2:4])
        w_cur = backwarp(cur, flow[:, 4:6])
        w_aft = backwarp(aft, flow[:, 6:8])

        # ---- Block 1: first refinement ----
        x1_in = torch.cat([w_bef, w_pred, w_cur, w_aft, mask, onehot], dim=1)  # 16ch
        flow_resid, mask_resid = self.block1(x1_in, flow)
        flow = flow + flow_resid
        mask = mask + mask_resid

        # Re-warp with refined flow
        w_bef = backwarp(bef, flow[:, 0:2])
        w_pred = backwarp(pred, flow[:, 2:4])
        w_cur = backwarp(cur, flow[:, 4:6])
        w_aft = backwarp(aft, flow[:, 6:8])

        # ---- Block 2: second refinement ----
        x2_in = torch.cat([w_bef, w_pred, w_cur, w_aft, mask, onehot], dim=1)
        flow_resid, mask_resid = self.block2(x2_in, flow)
        flow = flow + flow_resid
        mask = mask + mask_resid

        # Re-warp
        w_bef = backwarp(bef, flow[:, 0:2])
        w_pred = backwarp(pred, flow[:, 2:4])
        w_cur = backwarp(cur, flow[:, 4:6])
        w_aft = backwarp(aft, flow[:, 6:8])

        # ---- Block 3: finest refinement ----
        x3_in = torch.cat([w_bef, w_pred, w_cur, w_aft, mask, onehot], dim=1)
        flow_resid, mask_resid = self.block3(x3_in, flow)
        flow = flow + flow_resid
        mask = mask + mask_resid

        # ---- Final warp and blend ----
        w_bef = backwarp(bef, flow[:, 0:2])
        w_pred = backwarp(pred, flow[:, 2:4])
        w_cur = backwarp(cur, flow[:, 4:6])
        w_aft = backwarp(aft, flow[:, 6:8])

        # Blend using sigmoid-activated mask channels
        mask = torch.sigmoid(mask)
        mask_bef = mask[:, 0:1]
        mask_aft = mask[:, 1:2]
        mask_pred = mask[:, 2:3]

        output = w_bef * mask_bef + w_cur * (1 - mask_bef)
        output = w_aft * mask_aft + output * (1 - mask_aft)
        output = w_pred * mask_pred + output * (1 - mask_pred)

        return output


def load_pretrained_jit(model, jit_path, device='cpu'):
    """Load weights from a pretrained JIT model into the Deep3DNet.

    Parameter names match because both use nn.Sequential indexing.
    """
    jit_model = torch.jit.load(jit_path, map_location=device)

    jit_state = {}
    for name, param in jit_model.named_parameters():
        jit_state[name] = param.data
    for name, buf in jit_model.named_buffers():
        jit_state[name] = buf.data

    missing, unexpected = model.load_state_dict(jit_state, strict=False)
    return missing, unexpected
