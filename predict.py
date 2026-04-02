"""
Predict and visualize depth maps from single images using a trained Deep3D model.

Usage:
    python predict.py model.pth image.jpg --output result.png
    python predict.py model.pth image_dir/ --output output_dir/
"""

import argparse
import logging
import os

import cv2
import numpy as np
import torch

from model import Deep3DNet
from dataset import anaglyph, sbs


def get_args():
    parser = argparse.ArgumentParser(description='Deep3D single image prediction')
    parser.add_argument('model_path', type=str, help='Path to model checkpoint (.pth)')
    parser.add_argument('input', type=str, help='Input image or directory of images')
    parser.add_argument('--output', type=str, default='output',
                        help='Output path (image or directory)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id (-1 for CPU)')
    parser.add_argument('--data_shape', type=int, nargs=2, default=[384, 160],
                        help='Width and height for model input')
    parser.add_argument('--scale_min', type=int, default=-15)
    parser.add_argument('--scale_max', type=int, default=17)
    parser.add_argument('--save_depth', action='store_true',
                        help='Also save depth probability visualization')
    parser.add_argument('--save_anaglyph', action='store_true', default=True,
                        help='Save anaglyph (red-cyan) output')
    parser.add_argument('--save_sbs', action='store_true', default=True,
                        help='Save side-by-side output')
    return parser.parse_args()


@torch.no_grad()
def predict_image(model, img_bgr, data_shape, device):
    """Run model on a single image. Returns predicted right view (BGR, uint8)."""
    W, H = data_shape
    img_resized = cv2.resize(img_bgr, (W, H))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

    output = model(tensor)
    output = output.cpu().numpy()[0]
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    output = output.transpose(1, 2, 0)  # CHW -> HWC, RGB
    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def predict_depth(model, img_bgr, data_shape, device):
    """Get depth probability map visualization."""
    W, H = data_shape
    img_resized = cv2.resize(img_bgr, (W, H))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

    probs = model.get_depth_probs(tensor)
    probs = probs.cpu().numpy()[0]  # (D, H, W)

    # Create a 4x(D//4+1) grid visualization
    D = probs.shape[0]
    cols = 4
    rows = (D + cols - 1) // cols
    cell_h, cell_w = probs.shape[1], probs.shape[2]
    grid = np.zeros((rows * cell_h, cols * cell_w), dtype=np.float32)

    for d in range(D):
        r, c = divmod(d, cols)
        grid[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = probs[d]

    # Normalize to 0-255
    grid = (grid / (grid.max() + 1e-8) * 255).astype(np.uint8)
    grid = cv2.applyColorMap(grid, cv2.COLORMAP_JET)
    return grid


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

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

    data_shape = tuple(args.data_shape)

    if os.path.isfile(args.input):
        # Single image
        img = cv2.imread(args.input)
        if img is None:
            raise RuntimeError(f'Cannot read image: {args.input}')

        right = predict_image(model, img, data_shape, device)
        left = cv2.resize(img, (W, H))
        
        base, ext = os.path.splitext(args.output)
        if not ext:
            ext = '.png'
        cv2.imwrite(f'{base}_right{ext}', right)
        if args.save_anaglyph:
            cv2.imwrite(f'{base}_anaglyph{ext}', anaglyph(left, right))
        if args.save_sbs:
            cv2.imwrite(f'{base}_sbs{ext}', sbs(left, right))
        if args.save_depth:
            depth_vis = predict_depth(model, img, data_shape, device)
            cv2.imwrite(f'{base}_depth{ext}', depth_vis)
        logging.info(f'Saved results to {base}_*{ext}')

    elif os.path.isdir(args.input):
        # Directory of images
        os.makedirs(args.output, exist_ok=True)
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        files = sorted([f for f in os.listdir(args.input)
                        if os.path.splitext(f)[1].lower() in exts])

        for fname in files:
            img = cv2.imread(os.path.join(args.input, fname))
            if img is None:
                logging.warning(f'Skip unreadable: {fname}')
                continue

            right = predict_image(model, img, data_shape, device)
            left = cv2.resize(img, (W, H))
            base, ext = os.path.splitext(fname)

            cv2.imwrite(os.path.join(args.output, f'{base}_right{ext}'), right)
            if args.save_anaglyph:
                cv2.imwrite(os.path.join(args.output, f'{base}_anaglyph{ext}'),
                            anaglyph(left, right))
            if args.save_sbs:
                cv2.imwrite(os.path.join(args.output, f'{base}_sbs{ext}'),
                            sbs(left, right))
            if args.save_depth:
                depth_vis = predict_depth(model, img, data_shape, device)
                cv2.imwrite(os.path.join(args.output, f'{base}_depth{ext}'), depth_vis)

        logging.info(f'Processed {len(files)} images -> {args.output}/')
    else:
        raise RuntimeError(f'Input not found: {args.input}')


if __name__ == '__main__':
    main()
