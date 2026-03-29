"""Train script for PennyLane VQC."""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(__file__))
from dataset import LensingDataset, CLASS_NAMES
from evaluate import evaluate_model
from model_vqc_pennylane import HybridVQC


def load_and_pca(data_dir, n_components, train_subset=None):
    """Load data, flatten, PCA, normalize to [0,1]."""
    # Load raw
    X_train, y_train = [], []
    X_val, y_val = [], []
    for ci, cn in enumerate(CLASS_NAMES):
        for split, X_list, y_list in [('train', X_train, y_train),
                                       ('val', X_val, y_val)]:
            d = os.path.join(data_dir, split, cn)
            for f in sorted(os.listdir(d)):
                if f.endswith('.npy'):
                    img = np.load(os.path.join(d, f)).astype(np.float32).flatten()
                    X_list.append(img)
                    y_list.append(ci)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    # PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    print(f'PCA: {X_train.shape[1]} → {n_components}, '
          f'explained variance: {pca.explained_variance_ratio_.sum():.4f}')

    # Normalize to [0, 1]
    pca_min = X_train_pca.min(axis=0)
    pca_max = X_train_pca.max(axis=0)
    X_train_norm = (X_train_pca - pca_min) / (pca_max - pca_min + 1e-8)
    X_val_norm = np.clip((X_val_pca - pca_min) / (pca_max - pca_min + 1e-8), 0, 1)

    # Subsample training
    if train_subset is not None and train_subset < len(X_train_norm):
        per_class = train_subset // 3
        idx = []
        for c in range(3):
            ci = np.where(y_train == c)[0]
            idx.extend(np.random.choice(ci, per_class, replace=False).tolist())
        np.random.shuffle(idx)
        X_train_norm = X_train_norm[idx]
        y_train = y_train[idx]

    return (torch.tensor(X_train_norm, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
            torch.tensor(X_val_norm, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long))


def main():
    parser = argparse.ArgumentParser(description='Train PennyLane VQC')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--n-qubits', type=int, default=8)
    parser.add_argument('--n-layers', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--train-subset', type=int, default=3000)
    parser.add_argument('--save-prefix', type=str, default='vqc_pennylane')
    args = parser.parse_args()

    X_tr, y_tr, X_vl, y_vl = load_and_pca(
        args.data_dir, args.n_qubits, args.train_subset)
    print(f'Train: {len(X_tr)}, Val: {len(X_vl)}')

    train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_vl, y_vl),
                            batch_size=args.batch_size, shuffle=False)

    model = HybridVQC(n_qubits=args.n_qubits, n_layers=args.n_layers)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    ckpt_path = f'{args.save_prefix}_ckpt.pt'
    start_epoch = 0

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        best_val_acc = ckpt['best_val_acc']
        print(f'Resumed from epoch {start_epoch}')

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        rl, c, t = 0., 0, 0
        for bi, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            rl += loss.item() * X.size(0)
            c += (out.argmax(1) == y).sum().item()
            t += X.size(0)
            if (bi + 1) % 10 == 0:
                print(f'  Batch {bi+1}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f}', flush=True)

        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                out = model(X)
                vc += (out.argmax(1) == y).sum().item()
                vt += X.size(0)

        scheduler.step()
        val_acc = vc / vt
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'weights/best_{args.save_prefix}.pt')

        torch.save({
            'epoch': epoch + 1, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
        }, ckpt_path)

        print(f'Epoch {epoch+1:02d}/{args.epochs} ({time.time()-t0:.0f}s) | '
              f'Train Loss: {rl/t:.4f} Acc: {c/t:.4f} | '
              f'Val Acc: {val_acc:.4f} | Best: {best_val_acc:.4f}')

    print(f'\nBest Val Accuracy: {best_val_acc:.4f}')


if __name__ == '__main__':
    main()
