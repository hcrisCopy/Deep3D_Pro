"""
Training script for Deep3D PyTorch version.

Usage:
    python train.py --data_root data/frames --batch_size 16 --epochs 100
    python train.py --video_path data/raw/movie.mkv --batch_size 16 --epochs 100
"""

import os
import sys
import argparse
import logging
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import Deep3DNet
from dataset import StereoImageDataset, StereoVideoDataset, create_dataloader, anaglyph, sbs


OUTPUT_ROOT = '/root/autodl-tmp/deep3d/data/exp'


def get_args():
    parser = argparse.ArgumentParser(description='Train Deep3D model')
    default_data_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'train_set')
    )

    # Data
    parser.add_argument('--data_root', type=str, default=default_data_root,
                        help='Root directory for stereo dataset (expects clip_id/left,right layout)')
    parser.add_argument('--video_path', type=str, default=None,
                        help='Path to a side-by-side 3D video file')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Fraction of data used for validation')
    parser.add_argument('--data_shape', type=int, nargs=2, default=[384, 160],
                        help='Width and height of input images')

    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--lr_step', type=int, default=20,
                        help='Number of epochs between LR decay steps')
    parser.add_argument('--lr_factor', type=float, default=0.1,
                        help='LR decay factor')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_workers', type=int, default=4)

    # Model
    parser.add_argument('--scale_min', type=int, default=-15)
    parser.add_argument('--scale_max', type=int, default=17)
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Do not load pretrained VGG16 weights')

    # Output
    parser.add_argument('--exp_dir', type=str, default=OUTPUT_ROOT,
                        help='Experiment output directory')
    parser.add_argument('--prefix', type=str, default='deep3d')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Print training stats every N batches')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')

    # Hardware
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id (-1 for CPU)')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    return parser.parse_args()


def setup_logging(exp_dir, prefix):
    os.makedirs(exp_dir, exist_ok=True)
    log_file = os.path.join(
        exp_dir, f"{prefix}_{datetime.now().strftime('%Y_%m_%d-%H_%M')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ]
    )


def build_run_dir(base_dir, prefix):
    """Create per-run directory: base_dir/YYYYMMDD/prefix_HHMMSS."""
    date_dir = datetime.now().strftime('%Y%m%d')
    run_subdir = f"{prefix}_{datetime.now().strftime('%H%M%S')}"
    run_dir = os.path.join(base_dir, date_dir, run_subdir)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def tensor_rgb_to_bgr_uint8(tensor):
    """Convert CHW RGB tensor in [0,1] to HWC BGR uint8 image for OpenCV saving."""
    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    arr = arr.transpose(1, 2, 0)  # CHW -> HWC (RGB)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def draw_panel_label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (img.shape[1], 28), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


@torch.no_grad()
def save_pretrain_visualizations(model, dataset, device, output_dir, num_samples=5):
    """Save first N visualizations before training starts for quick sanity checks."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    n = min(num_samples, len(dataset))
    if n == 0:
        logging.warning('Dataset is empty. Skip pre-training visualization export.')
        return

    for i in range(n):
        left, right = dataset[i]
        pred = model(left.unsqueeze(0).to(device)).cpu()[0]

        left_bgr = tensor_rgb_to_bgr_uint8(left)
        right_bgr = tensor_rgb_to_bgr_uint8(right)
        pred_bgr = tensor_rgb_to_bgr_uint8(pred)

        ana_bgr = anaglyph(left_bgr, pred_bgr)
        sbs_bgr = sbs(left_bgr, pred_bgr)

        panels = [
            draw_panel_label(left_bgr, 'left_input'),
            draw_panel_label(right_bgr, 'right_gt'),
            draw_panel_label(pred_bgr, 'right_pred_before_train'),
            draw_panel_label(ana_bgr, 'anaglyph(left,pred)'),
            draw_panel_label(sbs_bgr, 'sbs(left,pred)'),
        ]
        vis = np.concatenate(panels, axis=1)
        out_path = os.path.join(output_dir, f'pretrain_vis_{i + 1:02d}.png')
        cv2.imwrite(out_path, vis)

    logging.info(f'Saved {n} pre-training visualizations to: {output_dir}')


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, log_interval):
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (left, right) in enumerate(dataloader):
        left = left.to(device)
        right = right.to(device)

        optimizer.zero_grad()
        output = model(left)
        loss = criterion(output, right)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            logging.info(
                f'Epoch [{epoch}] Batch [{batch_idx + 1}/{len(dataloader)}] '
                f'Loss: {avg_loss:.6f}'
            )
            running_loss = 0.0

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for left, right in dataloader:
        left = left.to(device)
        right = right.to(device)

        output = model(left)
        loss = criterion(output, right)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    args = get_args()
    base_output_dir = args.exp_dir if args.exp_dir else OUTPUT_ROOT
    args.exp_dir = build_run_dir(base_output_dir, args.prefix)
    setup_logging(args.exp_dir, args.prefix)
    logging.info(f'Run output directory: {args.exp_dir}')
    logging.info(f'Arguments: {args}')

    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    logging.info(f'Using device: {device}')

    # Data
    data_shape = tuple(args.data_shape)
    if args.data_root is not None:
        full_dataset = StereoImageDataset(args.data_root, data_shape=data_shape)
    elif args.video_path is not None:
        full_dataset = StereoVideoDataset(args.video_path, data_shape=data_shape)
    else:
        logging.error('Must specify either --data_root or --video_path')
        sys.exit(1)

    # Train/val split
    n = len(full_dataset)
    n_val = max(1, int(n * args.val_ratio))
    n_train = n - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    logging.info(f'Dataset: {n} total, {n_train} train, {n_val} val')

    train_loader = create_dataloader(train_dataset, args.batch_size,
                                     shuffle=True, num_workers=args.num_workers)
    val_loader = create_dataloader(val_dataset, args.batch_size,
                                   shuffle=False, num_workers=args.num_workers)

    # Model
    scale = (args.scale_min, args.scale_max)
    model = Deep3DNet(scale=scale, input_height=data_shape[1], input_width=data_shape[0])
    if not args.no_pretrained:
        logging.info('Initializing with VGG16 pretrained weights...')
        model.init_weights()
    model = model.to(device)

    # Save first 5 visualizations before training starts.
    pretrain_vis_dir = os.path.join(args.exp_dir, 'pretrain_vis')
    save_pretrain_visualizations(model, full_dataset, device, pretrain_vis_dir, num_samples=5)

    # Loss (L1 / MAE, matching original)
    criterion = nn.L1Loss()

    # Optimizer (SGD with momentum, matching original)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # LR scheduler (step decay, matching original)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_factor
    )

    # Resume
    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f'Resumed from epoch {start_epoch}')

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.exp_dir, 'tb_logs'))

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(start_epoch, args.epochs):
        logging.info(f'Epoch {epoch + 1}/{args.epochs}, LR: {scheduler.get_last_lr()[0]:.6f}')

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1, args.log_interval
        )
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        logging.info(f'Epoch {epoch + 1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
        writer.add_scalar('Loss/train', train_loss, epoch + 1)
        writer.add_scalar('Loss/val', val_loss, epoch + 1)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch + 1)

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.exp_dir, f'{args.prefix}-{epoch + 1:04d}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, ckpt_path)
            logging.info(f'Saved checkpoint: {ckpt_path}')

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.exp_dir, f'{args.prefix}-best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, best_path)
            logging.info(f'Saved best model: {best_path} (val_loss={val_loss:.6f})')

    writer.close()
    logging.info('Training finished.')


if __name__ == '__main__':
    main()
