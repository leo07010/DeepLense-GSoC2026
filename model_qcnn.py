"""Test III: Hybrid QCNN — Quantum Convolutional Layer + Classical CNN.

The first Conv2d layer is replaced by a parameterized quantum circuit (PQC)
acting as a 4x4 convolutional filter. The PQC slides over the (downsampled 8x8)
image with stride 4, encoding each 4x4 patch into 16 qubits via angle encoding,
applying variational rotations + CNOT entanglement, and reading out <Z> on
each qubit to produce 16 output channels.

Config: 8x8 image → 4x4 kernel, stride 4 → 2x2 = 4 patches → 4 circuit calls/image

Framework: NVIDIA CUDA-Q for quantum simulation, PyTorch for classical layers.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
import cudaq
from cudaq import spin

# ── Quantum circuit config ──────────────────────────────────────────────────
N_QUBITS = 16  # 4x4 kernel → 16 qubits
N_LAYERS = 2   # variational layers
N_PARAMS = N_LAYERS * 2 * N_QUBITS  # 64 trainable quantum parameters


# ── Quantum kernel ──────────────────────────────────────────────────────────
@cudaq.kernel
def qconv_kernel(data: list[float], params: list[float],
                 n_qubits: int, n_layers: int):
    """Parameterized quantum circuit acting as a 4x4 conv filter.

    For each variational layer:
      1. Angle encoding with data re-uploading: Ry(pixel*pi + theta)
      2. Ring CNOT entanglement
      3. Trainable Rz rotations

    No measurement — cudaq.observe() handles readout.
    """
    q = cudaq.qvector(n_qubits)

    for layer in range(n_layers):
        base = layer * 2 * n_qubits

        # Data encoding + trainable rotation
        for i in range(n_qubits):
            ry(data[i] * 3.14159265 + params[base + i], q[i])

        # Entanglement: ring of CNOTs
        for i in range(n_qubits - 1):
            x.ctrl(q[i], q[i + 1])
        x.ctrl(q[n_qubits - 1], q[0])

        # Trainable Rz
        for i in range(n_qubits):
            rz(params[base + n_qubits + i], q[i])


# Combined observable: sum of Z on all qubits (single observe call)
INDIVIDUAL_OBS = [spin.z(i) for i in range(N_QUBITS)]
COMBINED_OBS = sum(INDIVIDUAL_OBS)


def run_qconv_patch(patch_data, params_list):
    """Run quantum conv on a single 4x4 patch.

    Uses combined observable for speedup.

    Args:
        patch_data: list of 16 floats (pixel values in [0, 1])
        params_list: list of N_PARAMS floats

    Returns:
        list of N_QUBITS expectation values in [-1, 1]
    """
    result = cudaq.observe(qconv_kernel, COMBINED_OBS,
                           patch_data, params_list, N_QUBITS, N_LAYERS)
    return [result.expectation(obs) for obs in INDIVIDUAL_OBS]


# ── PyTorch autograd integration ────────────────────────────────────────────
class QuantumConvFunction(Function):
    """Differentiable quantum convolution via parameter-shift rule."""

    @staticmethod
    def forward(ctx, input_img, params, kernel_size, stride):
        B, C, H, W = input_img.shape
        H_out = (H - kernel_size) // stride + 1
        W_out = (W - kernel_size) // stride + 1

        params_list = params.detach().cpu().numpy().tolist()
        input_np = input_img.detach().cpu().numpy()

        output = np.zeros((B, N_QUBITS, H_out, W_out))

        for b in range(B):
            for i in range(H_out):
                for j in range(W_out):
                    r, c = i * stride, j * stride
                    patch = input_np[b, 0, r:r+kernel_size,
                                     c:c+kernel_size].flatten().tolist()
                    exp_vals = run_qconv_patch(patch, params_list)
                    for q_idx in range(N_QUBITS):
                        output[b, q_idx, i, j] = exp_vals[q_idx]

        output_tensor = torch.tensor(output, dtype=torch.float32)
        ctx.save_for_backward(input_img.cpu(), params.cpu())
        ctx.param_device = params.device
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """Gradient via SPSA: only 2 forward passes regardless of param count.

        SPSA approximates the gradient by simultaneously perturbing ALL parameters
        with random ±epsilon, then computing the finite difference. This gives an
        unbiased gradient estimate with O(1) circuit calls instead of O(N_params).
        """
        input_img, params = ctx.saved_tensors
        param_device = ctx.param_device
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        epsilon = 0.1  # perturbation size

        B, C, H, W = input_img.shape
        H_out = (H - kernel_size) // stride + 1
        W_out = (W - kernel_size) // stride + 1

        input_np = input_img.detach().cpu().numpy()
        params_np = params.detach().cpu().numpy()
        grad_out_np = grad_output.detach().cpu().numpy()

        # Random perturbation direction: each element ±1
        delta = np.random.choice([-1.0, 1.0], size=params_np.shape)

        params_plus = (params_np + epsilon * delta).tolist()
        params_minus = (params_np - epsilon * delta).tolist()

        # Compute f(θ+εΔ) and f(θ-εΔ) for all patches
        grad_params = np.zeros_like(params_np)

        for b in range(B):
            for i in range(H_out):
                for j in range(W_out):
                    r, c = i * stride, j * stride
                    patch = input_np[b, 0, r:r+kernel_size,
                                     c:c+kernel_size].flatten().tolist()

                    exp_plus = run_qconv_patch(patch, params_plus)
                    exp_minus = run_qconv_patch(patch, params_minus)

                    for q_idx in range(N_QUBITS):
                        diff = exp_plus[q_idx] - exp_minus[q_idx]
                        # SPSA gradient estimate: (f+ - f-) / (2ε) * (1/Δ_k)
                        # Chain rule with upstream gradient
                        grad_params += (grad_out_np[b, q_idx, i, j]
                                        * diff / (2.0 * epsilon) / delta)

        grad_tensor = torch.tensor(grad_params, dtype=torch.float32)
        return None, grad_tensor.to(param_device), None, None


# ── Quantum Conv2d module ───────────────────────────────────────────────────
class QuantumConv2d(nn.Module):
    """Drop-in replacement for nn.Conv2d using a 16-qubit PQC.

    Input:  (B, 1, 8, 8)
    Output: (B, 16, 2, 2)  — 4x4 kernel, stride 4
    """

    def __init__(self, kernel_size=4, stride=4, n_params=N_PARAMS):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.params = nn.Parameter(torch.randn(n_params) * 0.1)

    def forward(self, x):
        return QuantumConvFunction.apply(
            x, self.params, self.kernel_size, self.stride)


# ── Hybrid QCNN model ──────────────────────────────────────────────────────
class HybridQCNN(nn.Module):
    """Hybrid Quantum-Classical CNN.

    Architecture:
        Input (1, H, H) where H = downsample size (8 or 14)
        → QuantumConv2d 4x4/s4  (16, H_out, H_out)  ← quantum layer (16 qubits)
        → Flatten → FC layers → 3 classes
    """

    def __init__(self, downsample_size=8, num_classes=3):
        super().__init__()
        self.qconv = QuantumConv2d(kernel_size=4, stride=4)

        h_out = (downsample_size - 4) // 4 + 1  # 8→2, 14→3
        flat_dim = N_QUBITS * h_out * h_out      # 8→64, 14→144

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(N_QUBITS),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.qconv(x)  # quantum conv (returns CPU tensor)
        x = x.to(next(self.classifier.parameters()).device)
        x = self.classifier(x)
        return x
