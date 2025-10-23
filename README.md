# QINNs: Quantum Informed Neural Networks for Jet Tomography

**Author:** [Aritra Bal](https://etpwww.etp.kit.edu/~abal/), ETP, KIT  
**Contact:** [Email](mailto:aritra.bal@do-not-spam.kit.edu)

[![arXiv](https://img.shields.io/badge/arXiv-2502.17301-b31b1b.svg)](https://arxiv.org/abs/2510.17984)

## Overview

This study aims to use quantum information and geometry to improve the convergence and performance of graph neural networks used for the classification of jet flavor in particle physics. In this particular case, we try to distinguish between jets arising from light quarks/gluons (QCD jets) and those arising from the hadronic decay of a top quark (TTbar jets). The difference in substructure of these jets - such as the multiplicity, angular distribution, and momentum sharing of constituent particles - is what a machine learning algorithm would be expected to pick up on. By incorporating quantum-derived correlations into the graph structure, we enhance the classifier's ability to distinguish between these jet types.

## Theoretical Foundation

This study builds on the quantum machine learning framework **1P1Q** (One Particle One Qubit), published in [Physical Review D](https://journals.aps.org/prd/accepted/10.1103/l8y2-87vq) and available on [arXiv:2502.17301](https://arxiv.org/abs/2502.17301).

### Quantum Fisher Information Matrix (QFI)

We extract the Quantum Fisher Information (QFI) matrix from the 1P1Q framework to create enhanced graph structures. The QFI is mathematically defined as:

$$Q_{ij} = 4 \text{Re}\left[\langle \partial_i \psi | \partial_j \psi \rangle - \langle \partial_i \psi | \psi \rangle \langle \psi | \partial_j \psi \rangle\right]$$

where $\psi(\theta)$ is the parameterized quantum state, in this case a tensor product of Hilbert spaces, $\theta = \{\theta_i\}$ are the trainable parameters, and therefore $\partial_i = \dfrac{\partial}{\partial \theta_i}$.

### Quantum-Enhanced Graph Construction

The quantum-enhanced graph structure is constructed by placing the QFI matrix on the edges of a bidirectional graph, whose nodes are populated by kinematic features of particles (transverse momentum $p_T$, pseudorapidity $\eta$, azimuthal angle $\phi$). Since the QFI encodes the sensitivity of the quantum classifier to its trainable parameters (3 parameters per qubit in our parameterization), it serves as an indirect measure of the correlations between particle features, as a consequence of the one-particle one-qubit mapping. This quantum-enhanced graph structure can be expected to have better performance or faster convergence compared to a classifier that does not contain these quantum-derived correlations.

## Technical Setup

### Environment Installation

Run the setup script to create and install a conda environment named `qGNN` with all necessary packages:

```
source setup_env.sh
```

**Note:** This installation uses CUDA 11.8 to maintain compatibility with the ETP Deepthought GPU machine.

### Configuration System

This repository uses YAML configuration files to control all aspects of training and inference. Configuration files are located in the `configs/` directory and allow users to easily modify:

- **Model architecture**: Number of layers, hidden dimensions, pooling strategies
- **Training parameters**: Learning rate, batch size, number of epochs
- **Data paths**: Training, validation, and test file locations
- **Hardware settings**: Device selection (CPU/GPU)

**Example configuration structure:**

```yaml
experiment:
  name: "quantum_convGNN"
  seed: "0007"

model:
  type: "conv1d"
  num_mp_layers: 6
  mp_hidden_layers: [64, 32, 16, 12, 3]
  classifier_hidden_layers: [16, 12, 8, 4]
  pooling: "max"
  activation: "elu"

training:
  num_epochs: 50
  learning_rate: 0.005
  batch_size: 256

data:
  use_qfi_correlations: true
  train_files: ["/path/to/train/data.h5"]
  val_files: ["/path/to/val/data.h5"]
```

## Classification Approaches

This repository implements two distinct classification approaches:

### 1. Untrained (Fixed-Operation) Classifier

The untrained classifier uses physics-informed features on graph nodes and QFI matrices on edges to compute mathematical transformations that serve as graph messages, without any learnable parameters. Classification is achieved using statistical distance metrics.

**Mahalanobis Distance Metric:**
$$d_M^2(\mathbf{x}, \boldsymbol{\mu}) = (\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$$

where $\mathbf{x}$ is the feature vector, $\boldsymbol{\mu}$ is the class mean, and $\mathbf{\Sigma}^{-1}$ is the inverse covariance matrix.

The distance of a single jet sample from each class mean serves as a measure of the probability it belongs to that class, with classification assigned to the class with minimum Mahalanobis distance.

### 2. Trained (Learnable) Classifier

The trained classifier uses learnable graph neural network architectures with quantum-enhanced edge features derived from the QFI matrix, allowing the model to learn optimal representations through gradient-based optimization.

## Script Usage

### Trained Classifier Pipeline

**1. Training:**
```
python train.py --config configs/base_conv1d.yaml
```

**2. Testing:**
```
python test.py --experiment-dir ./experiments/quantum_convGNN_0007/
```

**3. Model comparison:**
```
python graph_comparisons.py --npz-files experiments/model1/results/metrics.npz experiments/model2/results/metrics.npz --labels "Conv1D GNN" "GAT Model"
```

### Untrained Classifier Pipeline

**1. Compute class statistics:**
```
python fetch_statistics.py --config configs/base_notrain.yaml
```

**2. Run inference with statistical classification:**
```
python graph_inference.py --config configs/base_notrain.yaml
```

**3. Compare classical vs quantum-enhanced features:**
```
python graph_inference.py --config configs/base_notrain.yaml --classical
```

### Data Processing

**Convert ROOT files to HDF5:**
```
python data_utils/h5_maker_multi.py --type ttbar --purpose train --input-dir /path/to/root/files --output-dir /path/to/h5/output
```

**Merge TTbar and QCD datasets:**
```
python data_utils/h5_merger.py --input-dir /path/to/individual/h5/files --purpose train --output-base /path/to/merged/output
```

## Repository Structure

```
packable/
├── configs/                    # YAML configuration files
├── data_utils/                 # Data processing and loading utilities
├── src/                        # Core model implementations
│   ├── gnn.py                 # Trainable GNN architectures
│   ├── graph_classifier.py   # Untrained statistical classifiers
│   ├── layers.py              # Custom neural network layers
│   ├── trainer.py             # Training loop implementation
│   └── logs.py                # Logging utilities
├── train.py                   # Main training script
├── test.py                    # Model evaluation script
├── graph_inference.py         # Untrained classifier inference
├── fetch_statistics.py        # Compute class statistics
└── graph_comparisons.py       # Performance comparison plots
```

## Key Features

- **Quantum-Enhanced Edges**: QFI matrices provide quantum-derived correlations between particles
- **Multiple Architectures**: Support for various GNN types (GAT, Conv1D, Correlation-based MP)
- **Statistical Classification**: Parameter-free classification using Mahalanobis distance
- **Comprehensive Evaluation**: ROC curves, SIC analysis, and performance comparisons
- **Flexible Configuration**: YAML-based parameter management
- **Memory Efficient**: Streaming data loaders for large datasets
- **Reproducible**: Deterministic training with configurable random seeds

## Performance Metrics

The repository provides comprehensive evaluation metrics including:
- **ROC Curves**: True Positive Rate vs False Positive Rate analysis
- **SIC Curves**: Significance Improvement Characteristic for signal/background separation
- **Classification Accuracy**: Standard accuracy metrics and confusion matrices
- **AUC Scores**: Area Under the Curve for model comparison

## Citation

If you use this code in your research, please cite the original 1P1Q paper:
- [Physical Review D publication](https://journals.aps.org/prd/accepted/10.1103/l8y2-87vq)
- [arXiv:2502.17301](https://arxiv.org/abs/2502.17301)

and also the QINN [preprint](https://arxiv.org/abs/2510.17984).

## Contact
 
See above. 