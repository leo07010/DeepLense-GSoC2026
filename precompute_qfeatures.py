"""Precompute quantum convolutional features for all images.

Config: 16 qubits, 4x4 kernel, stride 4, downsample to 8x8
  → 2x2 = 4 patches per image, 1 observe call per patch
  → 4 circuit calls per image (very fast)
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import cudaq
from cudaq import spin

# ── Quantum circuit config ──────────────────────────────────────────────────
N_QUBITS = 16
N_LAYERS = 2
N_PARAMS = N_LAYERS * 2 * N_QUBITS  # 64

@cudaq.kernel
def qconv_kernel(data: list[float], params: list[float],
                 n_qubits: int, n_layers: int):
    q = cudaq.qvector(n_qubits)
    for layer in range(n_layers):
        base = layer * 2 * n_qubits
        for i in range(n_qubits):
            ry(data[i] * 3.14159265 + params[base + i], q[i])
        for i in range(n_qubits - 1):
            x.ctrl(q[i], q[i + 1])
        x.ctrl(q[n_qubits - 1], q[0])
        for i in range(n_qubits):
            rz(params[base + n_qubits + i], q[i])

INDIVIDUAL_OBS = [spin.z(i) for i in range(N_QUBITS)]
COMBINED_OBS = sum(INDIVIDUAL_OBS)

def run_qconv_patch(patch_data, params_list):
    result = cudaq.observe(qconv_kernel, COMBINED_OBS,
                           patch_data, params_list, N_QUBITS, N_LAYERS)
    return [result.expectation(obs) for obs in INDIVIDUAL_OBS]

def qconv_image(img_2d, params_list, kernel_size=4, stride=4):
    """Apply quantum conv to a single 2D image. Returns (N_QUBITS, H_out, W_out)."""
    H, W = img_2d.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1
    output = np.zeros((N_QUBITS, H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            r, c = i * stride, j * stride
            patch = img_2d[r:r+kernel_size, c:c+kernel_size].flatten().tolist()
            exp_vals = run_qconv_patch(patch, params_list)
            for q in range(N_QUBITS):
                output[q, i, j] = exp_vals[q]
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--output-dir', type=str, default='qfeatures')
    parser.add_argument('--downsample', type=int, default=8,
                        help='Downsample images to this size')
    parser.add_argument('--target', type=str, default='nvidia')
    args = parser.parse_args()

    if args.target == 'nvidia':
        try:
            cudaq.set_target('nvidia')
            print('Using nvidia GPU target')
        except:
            print('Using default CPU target')

    os.makedirs(args.output_dir, exist_ok=True)

    # Fixed random quantum params for feature extraction
    np.random.seed(42)
    params = (np.random.randn(N_PARAMS) * 0.5).tolist()
    np.save(os.path.join(args.output_dir, 'qparams.npy'), np.array(params))
    print(f'Quantum: {N_QUBITS} qubits, {N_LAYERS} layers, {N_PARAMS} params')
    print(f'Image: {args.downsample}x{args.downsample}, kernel 4x4, stride 4')
    H_out = (args.downsample - 4) // 4 + 1
    print(f'Output: ({N_QUBITS}, {H_out}, {H_out}) per image, {H_out*H_out} patches')

    CLASS_NAMES = ['no', 'sphere', 'vort']

    for split in ['train', 'val']:
        all_features = []
        all_labels = []

        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            cls_dir = os.path.join(args.data_dir, split, cls_name)
            files = sorted([f for f in os.listdir(cls_dir) if f.endswith('.npy')])
            print(f'\n{split}/{cls_name}: {len(files)} files')

            t0 = time.time()
            for fi, fname in enumerate(files):
                img = np.load(os.path.join(cls_dir, fname)).astype(np.float32)
                img_t = torch.from_numpy(img).unsqueeze(0)
                img_t = F.adaptive_avg_pool2d(img_t, args.downsample)
                img_2d = img_t.squeeze().numpy()

                features = qconv_image(img_2d, params)
                all_features.append(features)
                all_labels.append(cls_idx)

                if (fi + 1) % 500 == 0:
                    elapsed = time.time() - t0
                    rate = (fi + 1) / elapsed
                    print(f'  {fi+1}/{len(files)} ({rate:.1f} img/s)', flush=True)

            elapsed = time.time() - t0
            print(f'  Done: {len(files)} images in {elapsed:.1f}s '
                  f'({len(files)/elapsed:.1f} img/s)')

        all_features = np.array(all_features, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int64)

        np.save(os.path.join(args.output_dir, f'{split}_features.npy'), all_features)
        np.save(os.path.join(args.output_dir, f'{split}_labels.npy'), all_labels)
        print(f'\nSaved {split}: features {all_features.shape}, labels {all_labels.shape}')


if __name__ == '__main__':
    main()
