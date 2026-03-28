"""Train script for Test I: Classical CNN (ResNet-18)."""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model_cnn import build_resnet18
from evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description='Test I: Classical CNN')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Data
    train_loader, val_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers)

    # Model
    model = build_resnet18(num_classes=3).to(device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                      T_max=args.epochs)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        # Train
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # Validate
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * imgs.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += imgs.size(0)
        val_loss = running_loss / total
        val_acc = correct / total

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_cnn.pt')

        print(f'Epoch {epoch+1:02d}/{args.epochs} | '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

    print(f'\nBest Val Accuracy: {best_val_acc:.4f}')

    # Final evaluation with ROC/AUC
    model.load_state_dict(torch.load('best_cnn.pt'))
    evaluate_model(model, val_loader, device, save_path='roc_cnn.png')


if __name__ == '__main__':
    main()
