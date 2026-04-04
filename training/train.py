"""Training entrypoint for the Deep3D_Pro PyTorch model."""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import TemporalStereoDataset, create_dataloader
from models.deep3d_network import Deep3DNet, load_pretrained_jit


OUTPUT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, 'data', 'exp'))


def get_args():
    parser = argparse.ArgumentParser(description='Train Deep3D_Pro model')
    default_data_root = os.path.abspath(os.path.join(PROJECT_ROOT, 'data', 'train_set'))

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
                        help='How to generate x0: right_gt=teacher forcing, left=use current left')

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_step', type=int, default=30,
                        help='Number of epochs between LR decay steps')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='LR decay factor')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'], help='Optimizer type')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained JIT model (.pt) for weight init')

    parser.add_argument('--exp_dir', type=str, default=OUTPUT_ROOT,
                        help='Experiment output directory')
    parser.add_argument('--prefix', type=str, default='deep3d')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Print training stats every N batches')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--vis_interval', type=int, default=5,
                        help='Save visualizations every N epochs')

    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id (-1 for CPU)')
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
    arr = tensor.detach().cpu().clamp(0, 1).numpy()
    arr = (arr * 255).astype(np.uint8)
    arr = arr.transpose(1, 2, 0)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def make_anaglyph(left_bgr, right_bgr):
    ana = np.zeros_like(left_bgr)
    ana[:, :, :2] = right_bgr[:, :, :2]
    ana[:, :, 2:] = left_bgr[:, :, 2:]
    return ana


def build_export_model_name(data_shape, device):
    width, height = data_shape
    device_tag = 'cuda' if device.type == 'cuda' else 'cpu'
    return f'deep3d_v1.0_{width}x{height}_{device_tag}.pt'


def export_torchscript_model(model, output_path, data_shape, device):
    width, height = data_shape
    model_cpu = Deep3DNet()
    model_cpu.load_state_dict({key: value.detach().cpu() for key, value in model.state_dict().items()})
    model_cpu.eval()

    example_input = torch.rand(1, 18, height, width, dtype=torch.float32)
    with torch.no_grad():
        scripted = torch.jit.trace(model_cpu, example_input)
    scripted.save(output_path)
    logging.info(f'Exported TorchScript model for {device.type}: {output_path}')


@torch.no_grad()
def select_low_loss_samples(model, dataset, device, criterion, num_samples) -> List[Tuple[int, float]]:
    scored_samples = []
    model.eval()

    for idx in range(len(dataset)):
        input_tensor, right_gt = dataset[idx]
        pred = model(input_tensor.unsqueeze(0).to(device)).cpu()[0]
        sample_loss = criterion(pred, right_gt).item()
        scored_samples.append((idx, sample_loss))

    scored_samples.sort(key=lambda item: item[1])
    return scored_samples[:min(num_samples, len(scored_samples))]


@torch.no_grad()
def save_visualizations(model, dataset, device, output_dir, epoch, criterion, num_samples=4):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    selected_samples = select_low_loss_samples(model, dataset, device, criterion, num_samples)
    for rank, (sample_idx, sample_loss) in enumerate(selected_samples, start=1):
        input_tensor, right_gt = dataset[sample_idx]
        pred = model(input_tensor.unsqueeze(0).to(device)).cpu()[0]

        left = input_tensor[9:12]

        left_bgr = tensor_to_bgr_uint8(left)
        gt_bgr = tensor_to_bgr_uint8(right_gt)
        pred_bgr = tensor_to_bgr_uint8(pred)
        ana_bgr = make_anaglyph(left_bgr, pred_bgr)

        vis = np.concatenate([left_bgr, gt_bgr, pred_bgr, ana_bgr], axis=1)
        out_path = os.path.join(
            output_dir,
            f'epoch{epoch:03d}_rank{rank:02d}_idx{sample_idx:05d}_loss{sample_loss:.6f}.png',
        )
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
                f'Epoch [{epoch}] Batch [{batch_idx + 1}/{len(dataloader)}] Loss: {avg_loss:.6f}'
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

    base_output_dir = args.exp_dir if args.exp_dir else OUTPUT_ROOT
    args.exp_dir = build_run_dir(base_output_dir, args.prefix)
    setup_logging(args.exp_dir, args.prefix)
    logging.info(f'Run output directory: {args.exp_dir}')
    logging.info(f'Arguments: {args}')

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    logging.info(f'Using device: {device}')

    data_shape = tuple(args.data_shape)
    full_dataset = TemporalStereoDataset(
        args.data_root,
        data_shape=data_shape,
        alpha=args.alpha,
        prev_mode=args.prev_mode,
    )

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

    model = Deep3DNet()

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

    criterion = nn.L1Loss()

    vis_dir = os.path.join(args.exp_dir, 'vis')
    save_visualizations(model, val_dataset, device, vis_dir, epoch=0,
                        criterion=criterion, num_samples=4)
    logging.info(f'Saved initial visualizations to: {vis_dir}')

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                              weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_factor
    )

    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f'Resumed from epoch {start_epoch}')

    writer = SummaryWriter(log_dir=os.path.join(args.exp_dir, 'tb_logs'))

    best_val_loss = float('inf')
    best_checkpoint_path = os.path.join(args.exp_dir, f'{args.prefix}-best.pth')
    best_export_name = build_export_model_name(data_shape, device)
    best_export_path = os.path.join(args.exp_dir, best_export_name)
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

        if (epoch + 1) % args.vis_interval == 0:
            save_visualizations(model, val_dataset, device, vis_dir,
                                epoch=epoch + 1, criterion=criterion, num_samples=4)

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, best_checkpoint_path)
            export_torchscript_model(model, best_export_path, data_shape, device)
            logging.info(
                f'New best model: {best_checkpoint_path} and {best_export_path} '
                f'(val_loss={val_loss:.6f})'
            )

    writer.close()
    logging.info('Training finished.')


if __name__ == '__main__':
    main()
