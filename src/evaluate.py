"""Evaluation utilities: ROC curve and AUC score."""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

from dataset import CLASS_NAMES


def evaluate_model(model, val_loader, device, save_path='roc_curve.png'):
    """Run evaluation and produce ROC curve + AUC scores.

    Returns:
        macro_auc: macro-averaged AUC score
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device) if hasattr(model, '_classical_device') is False else imgs
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2])

    # Accuracy
    preds = all_probs.argmax(axis=1)
    acc = (preds == all_labels).mean()

    # Per-class AUC
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    class_aucs = {}

    for i, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
        roc_auc_val = auc(fpr, tpr)
        class_aucs[class_name] = roc_auc_val
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'{class_name} (AUC = {roc_auc_val:.4f})')

    macro_auc = roc_auc_score(all_labels_bin, all_probs,
                               multi_class='ovr', average='macro')

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves (Macro AUC = {macro_auc:.4f})')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f'Accuracy: {acc:.4f}')
    for name, a in class_aucs.items():
        print(f'  {name} AUC: {a:.4f}')
    print(f'Macro AUC: {macro_auc:.4f}')
    print(f'ROC curve saved to {save_path}')

    return macro_auc
