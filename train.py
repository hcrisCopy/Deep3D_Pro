"""
Training script for Deep3D v1.0 PyTorch model.

Usage:
    python train.py --data_root ../data/train_set --batch_size 4 --epochs 100
    python train.py --data_root ../data/train_set --pretrained ../data/pretrained/deep3d_v1.0_1280x720_cuda.pt --lr 0.0001
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

from model import Deep3DNet, load_pretrained_jit
from data.dataset import TemporalStereoDataset, create_dataloader


OUTPUT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data', 'exp')
)


def get_args():
    parser = argparse.ArgumentParser(description='Train Deep3D v1.0 model')
    default_data_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'train_set')
    )

    # Data
    parser.add_argument('--data_root', type=str, default=default_data_root,
                        help='Root directory for stereo dataset (clip_id/left,right layout)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Fraction of data used for validation')
    parser.add_argument('--data_shape', type=int, nargs=2, default=[640, 360],
                        help='Width and height of input images')
    parser.add_argument('--alpha', type=int, default=5,
                        help='Temporal offset for far-before/after frames')
    parser.add_argument('--prev_mode', type=str, default='right_gt',
                        choices=['right_gt', 'left'],
                        help='How to generate x0 (previous prediction): '
                             'right_gt=teacher forcing, left=use current left')

    # Training
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_step', type=int, default=30,
                        help='Number of epochs between LR decay steps')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='LR decay factor')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--num_workers', type=int, default=4)

    # Model
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained JIT model (.pt) for weight init')

    # Output
    parser.add_argument('--exp_dir', type=str, default=OUTPUT_ROOT,
                        help='Experiment output directory')
    parser.add_argument('--prefix', type=str, default='deep3d')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Print training stats every N batches')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--vis_interval', type=int, default=5,
                        help='Save visualizations every N epochs')

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
    date_dir = datetime.now().strftime('%Y%m%d')
    run_subdir = f"{prefix}_{datetime.now().strftime('%H%M%S')}"
    run_dir = os.path.join(base_dir, date_dir, run_subdir)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def tensor_to_bgr_uint8(tensor):
    """Convert CHW RGB tensor [0,1] to HWC BGR uint8 for OpenCV."""
    arr = tensor.detach().cpu().clamp(0, 1).numpy()
    arr = (arr * 255).astype(np.uint8)
    arr = arr.transpose(1, 2, 0)  # CHW -> HWC
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def make_anaglyph(left_bgr, right_bgr):
    """Red-cyan anaglyph from BGR images."""
    ana = np.zeros_like(left_bgr)
    ana[:, :, :2] = right_bgr[:, :, :2]  # G, R from right
    ana[:, :, 2:] = left_bgr[:, :, 2:]   # B from left
    return ana


@torch.no_grad()
def save_visualizations(model, dataset, device, output_dir, epoch, num_samples=4):
    """Save side-by-side visualizations: left | right_gt | right_pred | anaglyph."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    n = min(num_samples, len(dataset))
    for i in range(n):
        input_tensor, right_gt = dataset[i]
        pred = model(input_tensor.unsqueeze(0).to(device)).cpu()[0]

        # Extract current left frame from input (channels 9:12)
        left = input_tensor[9:12]

        left_bgr = tensor_to_bgr_uint8(left)
        gt_bgr = tensor_to_bgr_uint8(right_gt)
        pred_bgr = tensor_to_bgr_uint8(pred)
        ana_bgr = make_anaglyph(left_bgr, pred_bgr)

        # Concatenate panels
        vis = np.concatenate([left_bgr, gt_bgr, pred_bgr, ana_bgr], axis=1)
        out_path = os.path.join(output_dir, f'epoch{epoch:03d}_sample{i + 1:02d}.png')
        cv2.imwrite(out_path, vis)

    model.train()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, log_interval):
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (input_tensor, right_gt) in enumerate(dataloader):
        input_tensor = input_tensor.to(device)
        right_gt = right_gt.to(device)

        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, right_gt)
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

    for input_tensor, right_gt in dataloader:
        input_tensor = input_tensor.to(device)
        right_gt = right_gt.to(device)

        output = model(input_tensor)
        loss = criterion(output, right_gt)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    args = get_args()

    # Build experiment directory
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

    # Dataset
    data_shape = tuple(args.data_shape)
    full_dataset = TemporalStereoDataset(
        args.data_root,
        data_shape=data_shape,
        alpha=args.alpha,
        prev_mode=args.prev_mode,
    )

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
    model = Deep3DNet()

    # Load weights
    if args.pretrained:
        logging.info(f'Loading pretrained weights from: {args.pretrained}')
        missing, unexpected = load_pretrained_jit(model, args.pretrained, device='cpu')
        if missing:
            logging.info(f'Missing keys: {missing}')
        if unexpected:
            logging.info(f'Unexpected keys: {unexpected}')
        logging.info('Pretrained weights loaded.')

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Model params: {total_params:,} total, {trainable_params:,} trainable')

    # Save initial visualizations
    vis_dir = os.path.join(args.exp_dir, 'vis')
    save_visualizations(model, full_dataset, device, vis_dir, epoch=0, num_samples=4)
    logging.info(f'Saved initial visualizations to: {vis_dir}')

    # Loss
    criterion = nn.L1Loss()

    # Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr,
            momentum=0.9, weight_decay=args.weight_decay
        )

    # LR scheduler
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
        lr = scheduler.get_last_lr()[0]
        logging.info(f'Epoch {epoch + 1}/{args.epochs}, LR: {lr:.6f}')

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1, args.log_interval
        )
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        logging.info(f'Epoch {epoch + 1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
        writer.add_scalar('Loss/train', train_loss, epoch + 1)
        writer.add_scalar('Loss/val', val_loss, epoch + 1)
        writer.add_scalar('LR', lr, epoch + 1)

        # Save visualizations
        if (epoch + 1) % args.vis_interval == 0:
            save_visualizations(model, full_dataset, device, vis_dir,
                                epoch=epoch + 1, num_samples=4)

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
            logging.info(f'New best model: {best_path} (val_loss={val_loss:.6f})')

    writer.close()
    logging.info('Training finished.')


if __name__ == '__main__':
    main()
