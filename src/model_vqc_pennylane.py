"""Variational Quantum Classifier using PennyLane + PyTorch.

Pipeline:
  Image → PCA (n_qubits dims) → Angle Encoding → VQC (data re-uploading) → 3 classes

Uses parameter-shift rule for exact gradients via PennyLane autograd.
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml


class HybridVQC(nn.Module):
    """PCA features → VQC → Linear → class logits.

    Input:  (B, n_qubits) — PCA features in [0, 1]
    Output: (B, n_classes) — class logits
    """

    def __init__(self, n_qubits=8, n_layers=3, n_classes=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes

        # Trainable quantum parameters
        n_params = n_layers * 2 * n_qubits
        self.q_weights = nn.Parameter(torch.randn(n_params) * 0.1)

        # Classical post-processing
        self.post = nn.Linear(n_classes, n_classes)

        # Build QNode
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev,
                               interface="torch", diff_method="parameter-shift")

    def _circuit(self, inputs, weights):
        """VQC circuit for a single sample."""
        for layer in range(self.n_layers):
            base = layer * 2 * self.n_qubits
            for i in range(self.n_qubits):
                qml.RY(inputs[i] * np.pi + weights[base + i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            if self.n_qubits > 1:
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            for i in range(self.n_qubits):
                qml.RZ(weights[base + self.n_qubits + i], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_classes)]

    def forward(self, x):
        """Forward pass: loop over batch, run QNode per sample."""
        B = x.shape[0]
        q_outputs = []
        for b in range(B):
            out = self.qnode(x[b], self.q_weights)
            q_outputs.append(torch.stack(out))
        q_out = torch.stack(q_outputs).float()  # (B, n_classes)
        return self.post(q_out)


class DressedVQC(nn.Module):
    """Dressed quantum circuit: FC → VQC → FC.

    Input:  (B, input_dim) — e.g. 512-dim CNN features
    Output: (B, n_classes) — class logits
    """

    def __init__(self, input_dim=512, n_qubits=8, n_layers=3, n_classes=3):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(input_dim, n_qubits),
            nn.Sigmoid(),
        )
        self.vqc = HybridVQC(n_qubits, n_layers, n_classes)

    def forward(self, x):
        x = self.pre(x)
        return self.vqc(x)
