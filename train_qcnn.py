"""Train script for Test III: Hybrid QCNN with CUDA-Q.

Two modes:
  1. precompute: Generate quantum features for all images (slow, once)
  2. train: Train classical layers on precomputed features (fast, many epochs)
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import cudaq

from dataset import get_dataloaders, CLASS_NAMES
from model_qcnn import HybridQCNN, N_QUBITS
from evaluate import evaluate_model


# ── Precomputed feature classifier ─────────────────────────────────────────
class QFeatureClassifier(nn.Module):
    """Classical classifier on top of precomputed quantum features.

    Input: (B, 16, 2, 2) from 16-qubit quantum conv on 8x8 images
    """
    def __init__(self, in_channels=N_QUBITS, spatial=3, num_classes=3):
        super().__init__()
        flat_dim = in_channels * spatial * spatial  # 16 * 3 * 3 = 144
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(flat_dim),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_on_precomputed(args, device):
    """Train classical classifier on precomputed quantum features."""
    feat_dir = args.qfeatures_dir

    train_X = torch.from_numpy(np.load(os.path.join(feat_dir, 'train_features.npy')))
    train_y = torch.from_numpy(np.load(os.path.join(feat_dir, 'train_labels.npy')))
    val_X = torch.from_numpy(np.load(os.path.join(feat_dir, 'val_features.npy')))
    val_y = torch.from_numpy(np.load(os.path.join(feat_dir, 'val_labels.npy')))

    print(f'Train features: {train_X.shape}, Val features: {val_X.shape}')

    train_loader = DataLoader(TensorDataset(train_X, train_y),
                              batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_X, val_y),
                            batch_size=64, shuffle=False)

    model = QFeatureClassifier(in_channels=N_QUBITS, num_classes=3).to(device)
    print(f'Classifier params: {sum(p.numel() for p in model.parameters()):,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        rl, c, t = 0., 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            rl += loss.item() * X.size(0)
            c += (out.argmax(1) == y).sum().item()
            t += X.size(0)

        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                vc += (out.argmax(1) == y).sum().item()
                vt += X.size(0)

        scheduler.step()
        val_acc = vc / vt
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_qcnn.pt')

        print(f'Epoch {epoch+1:02d}/{args.epochs} | '
              f'Train Loss: {rl/t:.4f} Acc: {c/t:.4f} | Val Acc: {val_acc:.4f}')

    print(f'\nBest Val Accuracy: {best_val_acc:.4f}')

    # Final evaluation
    model.load_state_dict(torch.load('best_qcnn.pt'))
    evaluate_model(model, val_loader, device, save_path='roc_qcnn.png')


def train_end_to_end(args, device):
    """End-to-end training with quantum conv in the loop.

    Supports resume from checkpoint: saves model/optimizer/scheduler/epoch
    to {prefix}_ckpt.pt every epoch.
    """
    prefix = args.save_prefix
    subset = args.train_subset if args.train_subset > 0 else None
    train_loader, val_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size,
        downsample_size=args.downsample,
        num_workers=0, train_subset=subset)

    model = HybridQCNN(downsample_size=args.downsample, num_classes=3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    q_params = model.qconv.params.numel()
    print(f'Total params: {total_params:,} '
          f'(Quantum: {q_params}, Classical: {total_params - q_params:,})')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    start_epoch = 0

    # Resume from checkpoint if exists
    ckpt_path = f'{prefix}_ckpt.pt'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        best_val_acc = ckpt['best_val_acc']
        print(f'Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}')

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        rl, c, t = 0., 0, 0
        for bi, (imgs, labels) in enumerate(train_loader):
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            rl += loss.item() * imgs.size(0)
            c += (outputs.argmax(1) == labels).sum().item()
            t += imgs.size(0)
            if (bi + 1) % 10 == 0:
                print(f'  Batch {bi+1}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f}', flush=True)

        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                labels = labels.to(device)
                outputs = model(imgs)
                vc += (outputs.argmax(1) == labels).sum().item()
                vt += imgs.size(0)

        scheduler.step()
        val_acc = vc / vt
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_{prefix}.pt')

        # Save checkpoint for resume
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
        }, ckpt_path)

        print(f'Epoch {epoch+1:02d}/{args.epochs} ({time.time()-t0:.0f}s) | '
              f'Train Loss: {rl/t:.4f} Acc: {c/t:.4f} | Val Acc: {val_acc:.4f}')

    print(f'\nBest Val Accuracy: {best_val_acc:.4f}')
    model.load_state_dict(torch.load(f'best_{prefix}.pt'))
    evaluate_model(model, val_loader, device, save_path=f'roc_{prefix}.png')


def main():
    parser = argparse.ArgumentParser(description='Test III: Hybrid QCNN')
    parser.add_argument('--mode', type=str, default='precomputed',
                        choices=['precomputed', 'e2e'],
                        help='precomputed: train on precomputed features; '
                             'e2e: end-to-end with quantum in loop')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--qfeatures-dir', type=str, default='qfeatures')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--downsample', type=int, default=14)
    parser.add_argument('--train-subset', type=int, default=3000)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--target', type=str, default='nvidia')
    parser.add_argument('--save-prefix', type=str, default='qcnn',
                        help='Prefix for saved model/roc files')
    args = parser.parse_args()

    if args.target == 'nvidia':
        try:
            cudaq.set_target('nvidia')
            print('Using CUDA-Q nvidia (GPU) target')
        except:
            print('Using default CPU target')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'PyTorch device: {device}')

    if args.mode == 'precomputed':
        train_on_precomputed(args, device)
    else:
        train_end_to_end(args, device)


if __name__ == '__main__':
    main()
