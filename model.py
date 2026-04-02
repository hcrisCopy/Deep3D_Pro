"""
Deep3D model: automatic 2D-to-3D conversion network (PyTorch).

Architecture: Simplified VGG16 backbone with multi-scale depth prediction branches,
fused via deconvolution upsampling, followed by softmax + DepthDot reconstruction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_dot import depth_dot


class Deep3DNet(nn.Module):
    """Deep3D network for automatic 2D-to-3D video conversion.

    Takes a left-eye image as input and produces a reconstructed right-eye image
    using learned depth probability maps.

    Args:
        scale: tuple (min_disp, max_disp), e.g. (-15, 17). Number of depth
               channels = max_disp - min_disp + 1.
        input_height: expected input image height (default 160).
        input_width: expected input image width (default 384).
    """

    def __init__(self, scale=(-15, 17), input_height=160, input_width=384):
        super().__init__()
        self.scale = scale
        self.num_depth = scale[1] - scale[0] + 1  # 33

        # Spatial size after 5 max-pool layers (stride 2 each)
        self.h5 = input_height // 32
        self.w5 = input_width // 32

        # ImageNet normalization buffers (RGB, [0, 1] range)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        D = self.num_depth

        # ---- VGG backbone ----
        # Group 1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        # Group 2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        # Group 3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        # Group 4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        # Group 5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2)

        # ---- FC layers ----
        self.fc6 = nn.Linear(512 * self.h5 * self.w5, 512)
        self.drop6 = nn.Dropout(0.5)
        self.fc7 = nn.Linear(512, 512)
        self.drop7 = nn.Dropout(0.5)
        self.fc8 = nn.Linear(512, D * self.h5 * self.w5)

        # ---- Prediction branches ----
        self.bn_pool1 = nn.BatchNorm2d(64)
        self.pred1 = nn.Conv2d(64, D, 3, padding=1)
        self.bn_pool2 = nn.BatchNorm2d(128)
        self.pred2 = nn.Conv2d(128, D, 3, padding=1)
        self.bn_pool3 = nn.BatchNorm2d(256)
        self.pred3 = nn.Conv2d(256, D, 3, padding=1)
        self.bn_pool4 = nn.BatchNorm2d(512)
        self.pred4 = nn.Conv2d(512, D, 3, padding=1)

        # ---- Deconvolution upsampling (to pool1 resolution) ----
        self.deconv_pred1 = nn.ConvTranspose2d(D, D, 1, stride=1, padding=0)
        self.deconv_pred2 = nn.ConvTranspose2d(D, D, 4, stride=2, padding=1)
        self.deconv_pred3 = nn.ConvTranspose2d(D, D, 8, stride=4, padding=2)
        self.deconv_pred4 = nn.ConvTranspose2d(D, D, 16, stride=8, padding=4)
        self.deconv_pred5 = nn.ConvTranspose2d(D, D, 32, stride=16, padding=8)

        # ---- Final upsampling (pool1 resolution -> full resolution) ----
        self.deconv_up = nn.ConvTranspose2d(D, D, 4, stride=2, padding=1)
        self.final_conv = nn.Conv2d(D, D, 3, padding=1)

    def forward(self, left_img):
        """
        Args:
            left_img: (N, 3, H, W) left image, RGB, float [0, 1].

        Returns:
            output: (N, 3, H, W) reconstructed right image, float [0, 1].
        """
        # Normalize for backbone feature extraction
        x = (left_img - self.mean) / self.std

        # ---- Backbone ----
        x = F.relu(self.conv1_1(x))
        p1 = self.pool1(x)

        x = F.relu(self.conv2_1(p1))
        p2 = self.pool2(x)

        x = F.relu(self.conv3_1(p2))
        x = F.relu(self.conv3_2(x))
        p3 = self.pool3(x)

        x = F.relu(self.conv4_1(p3))
        x = F.relu(self.conv4_2(x))
        p4 = self.pool4(x)

        x = F.relu(self.conv5_1(p4))
        x = F.relu(self.conv5_2(x))
        p5 = self.pool5(x)

        # ---- FC layers ----
        x = p5.flatten(1)
        x = self.drop6(F.relu(self.fc6(x)))
        x = self.drop7(F.relu(self.fc7(x)))
        x = self.fc8(x)

        # pred5 from FC output
        pred5 = x.view(-1, self.num_depth, self.h5, self.w5)

        # ---- Multi-scale prediction branches ----
        pred4 = self.pred4(self.bn_pool4(p4))
        pred3 = self.pred3(self.bn_pool3(p3))
        pred2 = self.pred2(self.bn_pool2(p2))
        pred1 = self.pred1(self.bn_pool1(p1))

        # ---- Deconv upsampling (ReLU applied BEFORE deconv, matching original) ----
        pred1 = self.deconv_pred1(F.relu(pred1))
        pred2 = self.deconv_pred2(F.relu(pred2))
        pred3 = self.deconv_pred3(F.relu(pred3))
        pred4 = self.deconv_pred4(F.relu(pred4))
        pred5 = self.deconv_pred5(F.relu(pred5))

        # ---- Fuse ----
        feat = pred1 + pred2 + pred3 + pred4 + pred5
        feat = F.relu(feat)

        # ---- Final upsampling ----
        up = self.deconv_up(feat)
        up = F.relu(up)
        up = self.final_conv(up)

        # ---- Softmax over depth channels + DepthDot reconstruction ----
        softmax_out = F.softmax(up, dim=1)
        output = depth_dot(softmax_out, left_img, self.scale)

        return output

    def get_depth_probs(self, left_img):
        """Return the depth probability map (softmax output) for visualization."""
        x = (left_img - self.mean) / self.std

        x = F.relu(self.conv1_1(x))
        p1 = self.pool1(x)
        x = F.relu(self.conv2_1(p1))
        p2 = self.pool2(x)
        x = F.relu(self.conv3_1(p2))
        x = F.relu(self.conv3_2(x))
        p3 = self.pool3(x)
        x = F.relu(self.conv4_1(p3))
        x = F.relu(self.conv4_2(x))
        p4 = self.pool4(x)
        x = F.relu(self.conv5_1(p4))
        x = F.relu(self.conv5_2(x))
        p5 = self.pool5(x)

        x = p5.flatten(1)
        x = self.drop6(F.relu(self.fc6(x)))
        x = self.drop7(F.relu(self.fc7(x)))
        x = self.fc8(x)

        pred5 = x.view(-1, self.num_depth, self.h5, self.w5)
        pred4 = self.pred4(self.bn_pool4(p4))
        pred3 = self.pred3(self.bn_pool3(p3))
        pred2 = self.pred2(self.bn_pool2(p2))
        pred1 = self.pred1(self.bn_pool1(p1))

        pred1 = self.deconv_pred1(F.relu(pred1))
        pred2 = self.deconv_pred2(F.relu(pred2))
        pred3 = self.deconv_pred3(F.relu(pred3))
        pred4 = self.deconv_pred4(F.relu(pred4))
        pred5 = self.deconv_pred5(F.relu(pred5))

        feat = pred1 + pred2 + pred3 + pred4 + pred5
        feat = F.relu(feat)
        up = self.deconv_up(feat)
        up = F.relu(up)
        up = self.final_conv(up)

        return F.softmax(up, dim=1)

    def load_vgg_pretrained(self):
        """Load pretrained VGG16 weights from torchvision into backbone layers."""
        import torchvision.models as models
        vgg16 = models.vgg16(weights='IMAGENET1K_V1')

        # Mapping: torchvision VGG16 features index -> our layer
        mapping = {
            0: self.conv1_1,    # Conv2d(3, 64, 3, 1, 1)
            5: self.conv2_1,    # Conv2d(64, 128, 3, 1, 1)
            10: self.conv3_1,   # Conv2d(128, 256, 3, 1, 1)
            12: self.conv3_2,   # Conv2d(256, 256, 3, 1, 1)
            17: self.conv4_1,   # Conv2d(256, 512, 3, 1, 1)
            19: self.conv4_2,   # Conv2d(512, 512, 3, 1, 1)
            24: self.conv5_1,   # Conv2d(512, 512, 3, 1, 1)
            26: self.conv5_2,   # Conv2d(512, 512, 3, 1, 1)
        }

        for idx, layer in mapping.items():
            src = vgg16.features[idx]
            layer.weight.data.copy_(src.weight.data)
            layer.bias.data.copy_(src.bias.data)

    def init_weights(self):
        """Initialize model weights: VGG pretrained + bilinear deconv + uniform for rest."""
        # Non-pretrained layers: small uniform init
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if 'conv' not in name or 'deconv' in name or 'pred' in name or 'fc' in name or 'final' in name:
                    nn.init.uniform_(param, -0.01, 0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)

        # BatchNorm defaults (weight=1, bias=0) are fine

        # Load VGG pretrained weights for backbone
        self.load_vgg_pretrained()

        # Bilinear init for deconvolution layers
        for name, module in self.named_modules():
            if isinstance(module, nn.ConvTranspose2d) and 'deconv' in name:
                _bilinear_init(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def _bilinear_init(tensor):
    """Initialize ConvTranspose2d weight with bilinear interpolation kernel."""
    shape = tensor.shape
    if shape[2] == 1 and shape[3] == 1:
        # 1x1 kernel: use identity-like init
        nn.init.eye_(tensor.view(shape[0], shape[1]))
        tensor.data = tensor.data.view(shape)
        return

    f_h = np.ceil(shape[2] / 2.0)
    f_w = np.ceil(shape[3] / 2.0)
    c_h = (2 * f_h - 1 - f_h % 2) / (2.0 * f_h)
    c_w = (2 * f_w - 1 - f_w % 2) / (2.0 * f_w)

    kernel = np.zeros(shape[2:], dtype=np.float32)
    for y in range(shape[2]):
        for x in range(shape[3]):
            kernel[y, x] = (1 - abs(y / f_h - c_h)) * (1 - abs(x / f_w - c_w))

    weight = np.zeros(shape, dtype=np.float32)
    for i in range(min(shape[0], shape[1])):
        weight[i, i] = kernel
    tensor.data.copy_(torch.from_numpy(weight))
