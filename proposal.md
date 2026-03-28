# GSoC 2026 Proposal: Hybrid Quantum-Classical Representation Learning for Dark Matter Substructure Classification

**Organization:** ML4SCI / DeepLense
**Project:** Hybrid Quantum-Classical Representation Learning for Dark Matter Substructure Classification
**Duration:** 350 hours (Large)
**Applicant:** [Your Name]
**Email:** [Your Email]
**GitHub:** https://github.com/leo07010
**Test Solution:** https://github.com/leo07010/DeepLense-GSoC2026

---

## 1. Synopsis

I propose developing a hybrid quantum-classical framework for classifying dark matter substructure in strong gravitational lensing images. The project will systematically explore three quantum integration strategies — Variational Quantum Classifier (VQC), Quantum Vision Transformer (QViT), and Quanvolutional Neural Network (QCNN) — benchmarked against a classical ResNet baseline. Through my preliminary experiments with CUDA-Q, I have identified the core challenges (barren plateaus, gradient noise, information bottleneck) and propose concrete solutions grounded in recent literature, including data re-uploading, dressed quantum circuits, and hardware-efficient ansätze. The final deliverable is a modular PennyLane-based library with trained models, reproducible benchmarks, and a hardware feasibility assessment for IBM/IonQ backends.

---

## 2. Motivation

Strong gravitational lensing provides a unique probe of dark matter substructure. The DeepLense project has demonstrated that deep learning can classify lensing images into distinct dark matter scenarios (no substructure, CDM subhalos, vortex/axion substructure) with high accuracy [Alexander et al., 2020]. However, all existing DeepLense models are purely classical.

Quantum machine learning offers a fundamentally different computational paradigm. Parameterized quantum circuits can encode information in exponentially large Hilbert spaces, potentially capturing correlations that classical models miss. Recent results show:

- **Quanvolutional layers** outperform classical CNNs on MNIST at small data regimes [Henderson et al., 2019]
- **Quantum Vision Transformers** achieve competitive accuracy on particle physics classification with fewer parameters [Cherrat et al., 2024]
- **Data re-uploading** provides universal approximation guarantees for quantum classifiers [Pérez-Salinas et al., 2020]

Despite this progress, **no prior work has applied quantum ML to gravitational lensing classification** — this project fills that gap.

---

## 3. Preliminary Results

I have completed the evaluation tests using NVIDIA CUDA-Q, exploring multiple quantum architectures:

| Model | Val Accuracy | Macro AUC | Key Finding |
|-------|-------------|-----------|-------------|
| Classical CNN (ResNet-18) | **91.67%** | **0.9838** | Strong baseline; requires careful optimization |
| VQC (PCA → 16 qubits) | 47.87% | 0.5112 | SPSA gradient too noisy; PCA loses discriminative info |
| CNN + VQC | — | 0.5004 | CNN features help but VQC optimization still struggles |
| QCNN (16q, e2e) | 34.47% | ~0.53 | Quanvolutional approach with SPSA does not converge |

**Key lessons from preliminary experiments:**

1. **The dataset is challenging** — three classes have nearly identical pixel statistics (mean ~0.06, std ~0.12). Classical CNN needs SGD with high momentum and ~15 epochs to break through random accuracy.

2. **SPSA gradient is insufficient** — while computationally efficient (O(1) circuit calls), SPSA provides gradient estimates too noisy for this subtle classification task. Parameter-shift rule with shot-based execution may be more reliable despite higher cost.

3. **PCA bottleneck** — linear dimensionality reduction to 16 dimensions preserves only 85% variance and loses task-discriminative features. A learnable quantum-classical feature extractor is needed.

4. **Combined observable optimization** — using `Z₀ + Z₁ + ... + Z_n` in a single `observe()` call provides ~4× speedup over separate measurements, crucial for practical training.

These insights directly inform my proposed approach.

---

## 4. Proposed Approach

### Phase 1: Foundation (Weeks 1-4)

**4.1 Classical Baseline & Feature Extraction**

Reproduce and solidify the ResNet-18 baseline (91.67% accuracy). Extract intermediate feature representations at multiple depths for quantum processing:

```
ResNet-18 → Layer 2 output (128-dim) → PCA/Autoencoder → n_qubits features
```

Key improvement over my preliminary work: use a **trainable bottleneck** (autoencoder or learned linear projection) instead of raw PCA, preserving discriminative information.

**4.2 PennyLane Framework Setup**

Port the quantum circuits from CUDA-Q to PennyLane, which provides:
- Native PyTorch integration via `qml.qnn.TorchLayer`
- Automatic differentiation (parameter-shift rule built-in)
- Hardware backend support (IBM Qiskit, IonQ, Amazon Braket)
- Rich circuit analysis tools (entanglement entropy, expressibility metrics)

```python
import pennylane as qml

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def vqc(features, weights):
    # Angle encoding with data re-uploading
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RY(features[i] * np.pi + weights[layer, i, 0], wires=i)
        # Entanglement
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        qml.CNOT(wires=[n_qubits-1, 0])
        for i in range(n_qubits):
            qml.RZ(weights[layer, i, 1], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(3)]
```

### Phase 2: Hybrid Architectures (Weeks 5-9)

**4.3 Dressed Quantum Circuit (Primary Approach)**

Based on the "dressed quantum circuit" paradigm [Mari et al., 2020], which has shown the best performance in hybrid models:

```
Image → ResNet (frozen) → 512-dim → FC → n_qubits → VQC → n_qubits → FC → 3 classes
        ╰── classical ──╯            ╰── quantum ──╯          ╰── classical ──╯
```

The classical pre- and post-processing layers "dress" the quantum circuit, allowing:
- Classical layers to learn the optimal encoding for the quantum circuit
- Quantum circuit to focus on the hardest part of the classification
- Classical output layer to decode quantum measurements into class probabilities

**Training strategy:**
1. Pre-train classical layers (freeze quantum, train FC layers only) — 5 epochs
2. Joint training (all parameters) with parameter-shift gradients — 50 epochs
3. Use Adam optimizer with separate learning rates: classical (1e-3) and quantum (0.01)

**4.4 Quantum Vision Transformer (QViT)**

Inspired by [Cherrat et al., 2024], replace the self-attention mechanism in a Vision Transformer with quantum self-attention:

```
Image → Patch Embedding → [Quantum Self-Attention + Classical FFN] × L → Classification
```

Quantum self-attention computes attention scores via quantum kernel:
- K(x_i, x_j) = |⟨0| U†(x_j) U(x_i) |0⟩|²

This naturally captures non-local correlations between image patches — important for detecting extended dark matter substructure.

**4.5 Quanvolutional Layer (Improved)**

Improve upon my preliminary QCNN by:
- Using **trainable** quantum parameters (not random/fixed)
- Employing **parameter-shift rule** instead of SPSA for reliable gradients
- Adding **multiple quanvolutional filters** (different random initializations → diverse features)
- Implementing **quantum pooling** via conditional measurements

### Phase 3: Optimization & Analysis (Weeks 10-13)

**4.6 Addressing Barren Plateaus**

My preliminary experiments showed loss stuck at log(3) ≈ 1.099, consistent with barren plateau behavior. Mitigation strategies:

| Strategy | Description | Reference |
|----------|-------------|-----------|
| **Layer-wise training** | Train one variational layer at a time | [Skolik et al., 2021] |
| **Hardware-efficient ansatz** | Reduce circuit depth, use local entanglement | [Kandala et al., 2017] |
| **Identity initialization** | Initialize parameters near identity transform | [Grant et al., 2019] |
| **Entanglement monitoring** | Track entanglement entropy during training | [Marrero et al., 2021] |

**4.7 Qubit Scaling Study**

Systematically evaluate performance vs. qubit count:
- 4, 8, 12, 16, 20 qubits
- Measure: accuracy, AUC, training time, gradient variance
- Determine the practical sweet spot for this task

**4.8 Hardware Feasibility Assessment**

Run best models on real quantum hardware:
- **IBM Eagle (127 qubits)** via Qiskit Runtime
- **IonQ Harmony (11 qubits)** via Amazon Braket
- Compare simulator vs. hardware performance
- Analyze noise impact and error mitigation strategies

### Phase 4: Documentation & Delivery (Weeks 14-16)

**4.9 Deliverables**
1. PennyLane-based library with modular quantum circuit components
2. Trained models for all architectures (VQC, QViT, QCNN)
3. Comprehensive benchmark: quantum vs. classical, simulator vs. hardware
4. Jupyter notebooks with reproducible experiments
5. Technical report / blog post suitable for ML4SCI website
6. Pull request to DeepLense repository

---

## 5. Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1-2 | Community bonding | Setup PennyLane environment, reproduce classical baseline, literature review |
| 3-4 | Phase 1 | Classical feature extractor, PennyLane VQC implementation |
| 5-6 | Phase 2a | Dressed quantum circuit — training & tuning |
| 7-8 | Phase 2b | QViT implementation & training |
| 9 | Phase 2c | Improved QCNN, architecture comparison |
| 10 | Midterm evaluation | Report on all architectures, identify best approach |
| 11-12 | Phase 3a | Barren plateau mitigation, qubit scaling study |
| 13 | Phase 3b | Hardware experiments (IBM/IonQ) |
| 14-15 | Phase 4 | Documentation, notebooks, blog post |
| 16 | Final delivery | PR to DeepLense, final report |

---

## 6. About Me

### Background
[Briefly describe your academic background, university, year of study]

### Relevant Experience

**Quantum Computing Projects:**
- **QKA-Cardiac** — Quantum Kernel Alignment for cardiac data classification using CUDA-Q
- **GQCBM-VAE** — Generalized Quantum Circuit Born Machine with Variational Autoencoder
- **GQE-MTS** — Quantum kernels for multivariate time series
- **VQE-NQS** — Variational Quantum Eigensolver with Neural Quantum States

All implemented with NVIDIA CUDA-Q, demonstrating proficiency in:
- Parameterized quantum circuit design
- Hybrid quantum-classical optimization
- GPU-accelerated quantum simulation
- Integration with PyTorch for end-to-end training

**DeepLense Test Results:**
- Completed Test I (CNN) and Test III (Quantum ML) with multiple approaches
- Repository: https://github.com/leo07010/DeepLense-GSoC2026

### Technical Skills
- **Quantum:** CUDA-Q, PennyLane, Qiskit (will expand during project)
- **Classical ML:** PyTorch, ResNet, Vision Transformers
- **HPC:** SLURM, multi-GPU training, NVIDIA H200
- **Languages:** Python, C++

### Availability
I can dedicate [X] hours per week to this project during the GSoC period (June-August 2026). I have no conflicting commitments during this time.

---

## 7. References

1. Alexander, S., Gleyzer, S., McDonough, E., Toomey, M.W., Usai, E. "Deep Learning the Morphology of Dark Matter Substructure." ApJ 893, 15 (2020). arXiv:1909.07346
2. Henderson, M., Shakya, S., Pradhan, S., Cook, T. "Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits." Quantum Mach. Intell. 2, 2 (2020). arXiv:1904.04767
3. Pérez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E., Latorre, J.I. "Data re-uploading for a universal quantum classifier." Quantum 4, 226 (2020).
4. Cherrat, E.A., et al. "Quantum Vision Transformers." Quantum 8, 1265 (2024).
5. Mari, A., Bromley, T.R., Killoran, N. "Transfer learning in hybrid classical-quantum neural networks." Quantum 4, 340 (2020).
6. Skolik, A., et al. "Layerwise learning for quantum neural networks." Quantum Mach. Intell. 3, 5 (2021).
7. Spall, J.C. "Multivariate Stochastic Approximation Using a Simultaneous Perturbation Gradient Approximation." IEEE TAC 37(3), 332-341 (1992).
8. NVIDIA CUDA-Q: https://nvidia.github.io/cuda-quantum/
