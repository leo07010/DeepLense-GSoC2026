"""Test III: Variational Quantum Classifier (VQC) with CUDA-Q.

Pipeline:
  1. PCA: (1, 150, 150) → 16 features
  2. Angle encoding: Ry(feature_i * pi) on qubit_i
  3. Variational layers: Ry/Rz + CNOT ring (data re-uploading)
  4. Measure: <Z> on 3 qubits → 3 class logits

Framework: NVIDIA CUDA-Q + PyTorch
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
import cudaq
from cudaq import spin

# ── Config ──────────────────────────────────────────────────────────────────
N_QUBITS = 16
N_LAYERS = 3
# Per layer: N_QUBITS Ry (data re-upload) + N_QUBITS Rz = 2 * N_QUBITS
N_PARAMS = N_LAYERS * 2 * N_QUBITS  # 96
N_CLASSES = 3


# ── VQC Kernel ──────────────────────────────────────────────────────────────
@cudaq.kernel
def vqc_kernel(data: list[float], params: list[float],
               n_qubits: int, n_layers: int):
    """Variational Quantum Classifier.

    For each layer:
      1. Data re-uploading: Ry(data[i] * pi + theta)
      2. Ring CNOT entanglement
      3. Trainable Rz rotations
    """
    q = cudaq.qvector(n_qubits)

    for layer in range(n_layers):
        base = layer * 2 * n_qubits

        # Data encoding + trainable Ry
        for i in range(n_qubits):
            ry(data[i] * 3.14159265 + params[base + i], q[i])

        # Ring CNOT entanglement
        for i in range(n_qubits - 1):
            x.ctrl(q[i], q[i + 1])
        x.ctrl(q[n_qubits - 1], q[0])

        # Trainable Rz
        for i in range(n_qubits):
            rz(params[base + n_qubits + i], q[i])


# Observables: Z on first 3 qubits → 3 class logits
CLASS_OBS = [spin.z(i) for i in range(N_CLASSES)]
COMBINED_OBS = sum(CLASS_OBS)


def run_vqc(data_list, params_list):
    """Run VQC on one sample. Returns 3 expectation values."""
    result = cudaq.observe(vqc_kernel, COMBINED_OBS,
                           data_list, params_list, N_QUBITS, N_LAYERS)
    return [result.expectation(obs) for obs in CLASS_OBS]


# ── PyTorch autograd (SPSA) ────────────────────────────────────────────────
class VQCFunction(Function):
    """Differentiable VQC via SPSA gradient."""

    @staticmethod
    def forward(ctx, features, params):
        """
        Args:
            features: (B, N_QUBITS) tensor — PCA features
            params: (N_PARAMS,) tensor
        Returns:
            logits: (B, N_CLASSES) tensor
        """
        B = features.shape[0]
        params_list = params.detach().cpu().numpy().tolist()
        feat_np = features.detach().cpu().numpy()

        logits = np.zeros((B, N_CLASSES))
        for b in range(B):
            exp_vals = run_vqc(feat_np[b].tolist(), params_list)
            for c in range(N_CLASSES):
                logits[b, c] = exp_vals[c]

        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        ctx.save_for_backward(features.cpu(), params.cpu())
        ctx.param_device = params.device
        return logits_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """SPSA gradient: 2 forward passes for all params."""
        features, params = ctx.saved_tensors
        param_device = ctx.param_device
        epsilon = 0.1

        B = features.shape[0]
        feat_np = features.detach().cpu().numpy()
        params_np = params.detach().cpu().numpy()
        grad_out_np = grad_output.detach().cpu().numpy()

        # Random perturbation direction
        delta = np.random.choice([-1.0, 1.0], size=params_np.shape)
        params_plus = (params_np + epsilon * delta).tolist()
        params_minus = (params_np - epsilon * delta).tolist()

        grad_params = np.zeros_like(params_np)

        for b in range(B):
            data = feat_np[b].tolist()
            exp_plus = run_vqc(data, params_plus)
            exp_minus = run_vqc(data, params_minus)

            for c in range(N_CLASSES):
                diff = exp_plus[c] - exp_minus[c]
                grad_params += (grad_out_np[b, c]
                                * diff / (2.0 * epsilon) / delta)

        grad_tensor = torch.tensor(grad_params, dtype=torch.float32)
        return None, grad_tensor.to(param_device)


# ── VQC Module ──────────────────────────────────────────────────────────────
class VQCLayer(nn.Module):
    """Variational Quantum Classifier as a PyTorch module.

    Input:  (B, N_QUBITS) — PCA features
    Output: (B, N_CLASSES) — class logits
    """

    def __init__(self, n_params=N_PARAMS):
        super().__init__()
        self.params = nn.Parameter(torch.randn(n_params) * 0.1)

    def forward(self, x):
        return VQCFunction.apply(x, self.params)


# ── Full model: PCA (pre-computed) → VQC ────────────────────────────────────
class HybridVQC(nn.Module):
    """Hybrid PCA + VQC classifier.

    Input:  (B, N_QUBITS) — PCA-reduced features
    Output: (B, N_CLASSES) — class logits
    """

    def __init__(self, num_classes=3):
        super().__init__()
        self.vqc = VQCLayer()

    def forward(self, x):
        return self.vqc(x)
