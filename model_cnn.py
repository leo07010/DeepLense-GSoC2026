"""Test I: Classical CNN model for 3-class lensing classification."""

import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


def build_resnet18(num_classes=3):
    """ResNet-18 style model for single-channel 150x150 input."""
    return nn.Sequential(
        # Stem: (1, 150, 150) → (64, 75, 75)
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=2, padding=1),  # (64, 38, 38)

        # Layer 1: (64, 38, 38)
        ResBlock(64, 64),
        ResBlock(64, 64),

        # Layer 2: (128, 19, 19)
        ResBlock(64, 128, stride=2),
        ResBlock(128, 128),

        # Layer 3: (256, 10, 10)
        ResBlock(128, 256, stride=2),
        ResBlock(256, 256),

        # Layer 4: (512, 5, 5)
        ResBlock(256, 512, stride=2),
        ResBlock(512, 512),

        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(512, num_classes),
    )
