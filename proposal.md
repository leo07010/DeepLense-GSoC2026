# GSoC 2026 Proposal: Hybrid Quantum-Classical Representation Learning for Dark Matter Substructure Classification

**Organization:** ML4SCI / DeepLense
**Project:** Hybrid Quantum-Classical Representation Learning for Dark Matter Substructure Classification
**Duration:** 350 hours (Large)
**Applicant:** [Your Name]
**Email:** [Your Email]
**GitHub:** https://github.com/leo07010
**Timezone:** [Your Timezone]

---

## Abstract

Strong gravitational lensing images encode subtle signatures of dark matter substructure — but extracting these signatures from high-dimensional pixel data remains challenging. This project proposes leveraging quantum computing to learn representations in exponentially large Hilbert spaces, where superposition and entanglement may capture correlations that classical models cannot efficiently represent. I bring direct experience building hybrid quantum-classical systems with CUDA-Q across multiple scientific domains (quantum kernel alignment, variational quantum eigensolvers, quantum circuit Born machines), and I aim to apply this expertise to develop PennyLane-based variational quantum circuits and quantum kernel methods tailored for the DeepLense classification pipeline. The final deliverable is a benchmarked, NISQ-compatible hybrid model with a hardware feasibility assessment on IBM quantum backends.

---

## 1. Benefits to the Community

The DeepLense project has established state-of-the-art classical deep learning pipelines for gravitational lensing analysis. However, **no quantum approach has been integrated into DeepLense to date**. This project would:

- **Open a new research direction** for ML4SCI by establishing the first quantum ML baseline for lensing classification, enabling future quantum-advantage studies as hardware matures
- **Provide a reusable quantum module** compatible with the existing DeepLense pipeline — other ML4SCI projects (QMLHEP) could adapt it for particle physics tasks
- **Produce a comprehensive benchmark** of quantum vs. classical performance with noise analysis, serving as a reference for the quantum ML for science community
- **Attract the quantum computing community** to gravitational lensing — a domain where quantum approaches are unexplored

---

## 2. Project Goals

Based on the [project description](https://ml4sci.org/gsoc/2026/proposal_DEEPLENSE2.html), the core goals are:

1. **Develop hybrid quantum-classical architectures** using variational quantum circuits (VQC) and quantum kernels for multi-class lensing image classification
2. **Train a NISQ-compatible model** that can run on near-term quantum hardware
3. **Benchmark quantum vs. classical performance** with analysis of noise robustness and hardware feasibility

---

## 3. Proposed Approach

### 3.1 Dressed Variational Quantum Classifier (Primary)

The "dressed quantum circuit" paradigm [Mari et al., 2020] is the most mature hybrid architecture, where classical layers handle dimensionality reduction and the quantum circuit focuses on the classification boundary:

```
Image (150×150)
  → Classical CNN backbone (frozen, pre-trained) → 512-dim features
  → Trainable FC projection → n_qubits dims → Sigmoid → [0, 1]
  → Angle Encoding + VQC (data re-uploading, L layers)
  → ⟨Z⟩ measurements → n_classes expectations
  → Classical FC → class logits
```

**Why this architecture:**
- Classical layers handle the high-dimensional image → low-dimensional embedding (where CNNs excel)
- Quantum circuit operates on a compact, task-relevant feature space
- Data re-uploading provides universal approximation guarantees [Pérez-Salinas et al., 2020]
- The classical "dressing" layers learn the optimal encoding for the quantum circuit

**Implementation:** PennyLane with `parameter-shift` gradient for exact differentiation, integrated with PyTorch via QNode.

### 3.2 Quantum Kernel Method

As an alternative to variational training, quantum kernel methods compute a kernel matrix $K(x_i, x_j) = |\langle 0 | U^\dagger(x_j) U(x_i) | 0 \rangle|^2$ and feed it to a classical SVM or kernel ridge regression:

- No barren plateau issue (no variational parameters to optimize)
- Provably captures feature maps in exponential-dimensional space
- I have direct experience implementing this approach in my **QKA-Cardiac** project using CUDA-Q

**Comparison plan:** Evaluate both VQC and quantum kernel approaches to determine which is more effective for the lensing classification task.

### 3.3 Noise Robustness & Hardware Feasibility

- Simulate realistic noise models (depolarizing, amplitude damping) using PennyLane's `default.mixed` device
- Evaluate performance degradation vs. noise strength
- Run the best model on **IBM quantum hardware** via Qiskit Runtime
- Compare simulator vs. hardware results and identify the minimum circuit depth/qubit count needed for meaningful classification

---

## 4. Technical Details

### Circuit Design

| Parameter | Choice | Rationale |
|-----------|--------|-----------|
| Qubits | 4–16 (scaling study) | Balance expressivity vs. trainability |
| Encoding | Angle encoding (Ry) | Hardware-efficient, 1 gate per feature |
| Ansatz | Ry → ring CNOT → Rz per layer | Creates full entanglement with minimal depth |
| Layers | 2–4 (with data re-uploading) | Re-uploading proven to increase expressivity |
| Readout | ⟨Z⟩ on n_classes qubits | Direct mapping to class logits |
| Gradient | Parameter-shift rule | Exact gradients (PennyLane native) |

### Addressing Barren Plateaus

The key optimization challenge for variational quantum circuits at scale:

| Strategy | Description |
|----------|-------------|
| **Layer-wise training** | Train one layer at a time, freeze previous layers [Skolik et al., 2021] |
| **Identity initialization** | Initialize near identity to avoid exponentially flat landscape [Grant et al., 2019] |
| **Local entanglement** | Start with nearest-neighbor CNOT, add long-range only if needed |
| **Qubit scaling study** | Systematically measure gradient variance at 4, 8, 12, 16 qubits |

---

## 5. Timeline

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1–2 | **Community bonding** | Set up PennyLane environment, reproduce classical baseline, align with mentors on priorities |
| 3–4 | **Dressed VQC implementation** | Working VQC with PennyLane + PyTorch, initial training results |
| 5–6 | **Optimization & tuning** | Layer-wise training, qubit scaling study (4→16), best hyperparams |
| 7 | **Quantum kernel method** | Kernel matrix computation, SVM classifier, comparison with VQC |
| 8 | **Midterm evaluation** | Report: VQC vs. kernel vs. classical, identify best approach |
| 9–10 | **Noise analysis** | Simulate noise models, measure robustness, error mitigation |
| 11 | **Hardware experiments** | Run best model on IBM quantum backend |
| 12–13 | **Integration & documentation** | Clean code, Jupyter notebooks, PR to DeepLense repo |
| 14 | **Final delivery** | Blog post, final report, all deliverables |

---

## 6. About Me

### Background

[University, department, year — please fill in]

### Relevant Quantum Computing Experience

I have built and trained hybrid quantum-classical models across multiple scientific domains, all implemented with NVIDIA CUDA-Q and integrated with PyTorch:

| Project | Domain | Quantum Technique |
|---------|--------|-------------------|
| **QKA-Cardiac** | Cardiac data classification | Quantum Kernel Alignment — directly relevant to the quantum kernel approach proposed here |
| **GQCBM-VAE** | Generative modeling | Quantum Circuit Born Machine with `exp_pauli` gates and transformer-guided circuit selection |
| **GQE-MTS** | Multivariate time series | Quantum kernel methods with counteradiabatic rotations |
| **VQE-NQS** | Quantum chemistry | Variational Quantum Eigensolver with `cudaq.observe()` and UCCSD ansatz |

This experience covers the core skills required by this project:
- **Parameterized quantum circuit design** — variational ansätze, encoding strategies, entanglement topologies
- **Hybrid optimization** — quantum-classical training loops, gradient methods (parameter-shift, SPSA)
- **GPU-accelerated simulation** — essential for training quantum models at practical scale
- **Framework proficiency** — CUDA-Q (primary), PennyLane (will be primary for this project), familiar with Qiskit

### Technical Skills

- **Quantum:** CUDA-Q, PennyLane, Qiskit
- **Classical ML:** PyTorch, CNNs, Vision Transformers
- **HPC:** SLURM, multi-GPU training (NVIDIA H200)
- **Languages:** Python, C++

---

## 7. Communication Plan

- **Weekly sync** with mentors (30 min video call or async written update)
- **Bi-weekly progress report** with results, blockers, and next steps
- **Code review:** Submit PR-ready code every 2 weeks for mentor feedback
- **Channel:** Email (ml4-sci@cern.ch) + Slack/Mattermost (whichever ML4SCI uses)

---

## 8. Availability

I can dedicate **[X] hours per week** to this project during the GSoC period (June–August 2026). I have no conflicting internships, courses, or other commitments during this time.

---

## 9. References

1. Alexander, S., Gleyzer, S., McDonough, E., Toomey, M.W., Usai, E. "Deep Learning the Morphology of Dark Matter Substructure." ApJ 893, 15 (2020). arXiv:1909.07346
2. Pérez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E., Latorre, J.I. "Data re-uploading for a universal quantum classifier." Quantum 4, 226 (2020).
3. Mari, A., Bromley, T.R., Killoran, N. "Transfer learning in hybrid classical-quantum neural networks." Quantum 4, 340 (2020).
4. Cherrat, E.A., et al. "Quantum Vision Transformers." Quantum 8, 1265 (2024).
5. Skolik, A., et al. "Layerwise learning for quantum neural networks." Quantum Mach. Intell. 3, 5 (2021).
6. Grant, E., et al. "An initialization strategy for addressing barren plateaus in parametrized quantum circuits." Quantum 3, 214 (2019).
7. Havlíček, V., et al. "Supervised learning with quantum-enhanced feature maps." Nature 567, 209–212 (2019).
