# DeepLense GSoC 2026 — Quantum Machine Learning

Gravitational lensing image classification using classical and quantum machine learning approaches, built with **PyTorch** and **NVIDIA CUDA-Q**.

## Task

Classify strong gravitational lensing images into 3 classes:
- `no` — no substructure
- `sphere` — subhalo substructure
- `vort` — vortex substructure

## Results

| Model | Val Accuracy | Macro AUC | Framework |
|-------|-------------|-----------|-----------|
| **Classical CNN (ResNet-18)** | **91.67%** | **0.9838** | PyTorch |
| CNN + VQC (16 qubits) | TBD | TBD | CUDA-Q + PyTorch |
| VQC with PCA (16 qubits) | ~40% | ~0.50 | CUDA-Q |
| QCNN 8×8 (e2e, SPSA) | ~34% | ~0.53 | CUDA-Q + PyTorch |

## Project Structure

```
├── Test_I_Classical_CNN.ipynb          # Test I: Classical CNN notebook
├── Test_III_VQC.ipynb                  # Test III: PCA + VQC (original)
├── Test_III_VQC_tuned.ipynb            # Test III: PCA + VQC (tuned hyperparams)
├── Test_III_CNN_VQC.ipynb              # Test III: CNN features + VQC
├── Test_III_Quantum_QCNN.ipynb         # Test III: Quantum Conv layer (QCNN)
│
├── model_cnn.py                        # ResNet-18 model
├── model_qcnn.py                       # Quantum Conv2d + Hybrid QCNN
├── model_vqc.py                        # Variational Quantum Classifier
├── dataset.py                          # Dataset loader
├── evaluate.py                         # ROC/AUC evaluation
├── train_cnn.py                        # CNN training script
├── train_qcnn.py                       # QCNN training script (e2e + precompute)
├── precompute_qfeatures.py             # Precompute quantum features
│
├── best_cnn.pt                         # Best CNN model weights
├── roc_cnn.png                         # CNN ROC curve
├── roc_vqc.png                         # VQC ROC curve
│
├── run_all.sh                          # SLURM job script
└── data/                               # Dataset (not included, see below)
```

## Setup

### Requirements
- Python 3.9+
- PyTorch 2.x
- NVIDIA CUDA-Q 0.8+
- scikit-learn, matplotlib, numpy

### Dataset
Download from [Google Drive](https://drive.google.com/file/d/1ZEyNMEO43u3qhJAwJeBZxFBEYc_pVYZQ/view) and extract to `data/`:
```bash
gdown --id 1ZEyNMEO43u3qhJAwJeBZxFBEYc_pVYZQ -O dataset.zip
unzip dataset.zip -d data_tmp && mv data_tmp/dataset data && rm -rf data_tmp dataset.zip
```

## Approach

### Test I: Classical CNN
ResNet-18 adapted for single-channel 150×150 images. Trained with SGD (lr=0.01, momentum=0.9, cosine annealing, 50 epochs).

### Test III: Quantum ML

**Method 1 — VQC (Variational Quantum Classifier):**
```
Image → PCA (16 dims) → Angle Encoding (16 qubits) → VQC (3 layers) → ⟨Z⟩ → 3 classes
```
- Data re-uploading for higher expressivity
- SPSA gradient (2 circuit calls per backward, vs 192 for parameter-shift)
- Ring CNOT entanglement topology

**Method 2 — CNN + VQC:**
```
Image → Pre-trained CNN → 512-dim → Projection → 16-dim → VQC → 3 classes
```
- Uses CNN features (discriminative) instead of PCA (variance-based)
- VQC operates on task-relevant feature space

**Method 3 — QCNN (Quantum Convolution):**
```
Image → Downsample (8×8) → Quantum Conv2d (4×4 kernel, 16 qubits) → Classical FC → 3 classes
```
- Quantum circuit acts as a convolutional filter
- End-to-end training with SPSA gradient

### Key Technical Choices
- **CUDA-Q**: GPU-accelerated quantum simulation (nvidia target)
- **SPSA gradient**: O(1) circuit evaluations regardless of parameter count
- **Combined observable**: `Z₀ + Z₁ + ... + Z₁₅` in single `cudaq.observe()` call (~4× speedup)
- **Data re-uploading**: Re-encode data at each variational layer for higher expressivity

## Running

### With SLURM (HPC)
```bash
sbatch run_all.sh
```

### Local
```bash
# Test I
python train_cnn.py --data-dir data --epochs 50 --batch-size 128 --lr 0.01

# Test III (VQC)
jupyter nbconvert --execute --to notebook Test_III_VQC.ipynb
```

## References
- Henderson et al., "Quanvolutional Neural Networks" (2020)
- Pérez-Salinas et al., "Data re-uploading for a universal quantum classifier" (2020)
- NVIDIA CUDA-Q: https://nvidia.github.io/cuda-quantum/
