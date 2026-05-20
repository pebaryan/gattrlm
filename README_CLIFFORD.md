# Geometric Clifford/Rotor-based Attractor Model

A **Geometric Algebra (Clifford) extension** of the official [Attractor](https://github.com/jacobfa/Attractor) framework. This project replaces standard transformer layers with **Clifford/Geometric algebra layers** — rotors, geometric products, and grade-preserving linear maps — inside a deep equilibrium (DEQ) attractor to achieve **built-in rotation/reflection equivariance** and **efficient spatial reasoning**.

## Why Geometric Algebra?

| Feature | Standard Neural Net | Clifford Attractor |
|---|---|---|
| **Rotation equivariance** | Learned via data augmentation | Built-in via **RotorLayer** (sandwich product R x R~) |
| **Spatial representation** | Scalar embeddings | **Multivector embeddings** in Cl(3,0) or Cl(4,1) |
| **Geometric reasoning** | Implicit | Explicit via **geometric product**, wedge, and grade projection |
| **Parameter efficiency** | 4× intermediate expansion | **GeometricProductLayer** creates nonlinear interactions without extra parameters |
| **Memory** | Grows with depth | **Constant** (DEQ fixed-point, same as original Attractor) |

## Architecture

### Cl(p,q) Multivector Representation
Each token is embedded as a **multivector** in Cl(p,q) with **channels** independent feature dimensions:
- **Cl(3,0) Euclidean**: 8 basis blades (scalar + 3 vectors + 3 bivectors + pseudoscalar)
- **Cl(4,1) Conformal**: 32 basis blades (includes null-cone for spheres/planes)

### CliffordAttractorBlock
```
x → LayerNorm → RotorLayer(RxR~) + skip → CliffordLinear → GeometricProduct(x⊗x) → GeometricGELU → CliffordLinear → BladeSelector → + skip
```

### Fixed-Point Iteration
The attractor solves `X* = f(X*)` using:
- **Anderson acceleration** (default, m=5) for fast convergence
- **Simple fixed-point** iteration fallback
- **Jacobian regularization** via power iteration

## Installation

```bash
# From project root
pip install -e .

# Dependencies
pip install torch numpy
```

## Quick Start: Toy Example

```bash
python experiments/clifford_toy_example.py
```

This trains a CliffordAttractor on a synthetic vector field fixed-point task. Output shows convergence and loss curves.

## Key Files

| File | Description |
|---|---|
| `attractor/models/clifford_attractor.py` | Core implementation: Clifford algebra operations, layers, config, and fixed-point solver |
| `experiments/clifford_toy_example.py` | Minimal working example on synthetic data |
| `README_CLIFFORD.md` | This file |

## CliffordAlgebra Module

The `CliffordAlgebra` module implements:

- **Geometric Product** (`gp(a, b)`): Fundamental GA operation via precomputed Cayley table
- **Sandwich Product** (`R * x * R~`): Rotor/versor action on multivectors
- **Bivector Exponential** (`exp(B)`): Closed-form for elliptic/hyperbolic signatures
- **Grade Projection**: Extract scalar, vector, bivector, pseudoscalar components
- **Reversion & Conjugation**: Grade-dependent sign flips

## Layers

### RotorLayer
```python
RotorLayer(algebra, channels)
```
Learns bivector coefficients → exponentiate to rotors → apply `R * x * R~`. Provides **built-in rotation equivariance** in Cl(3,0).

### CliffordLinear
```python
CliffordLinear(algebra, in_channels, out_channels)
```
Channel-mixing linear layer: `out[b, o, k] = W[o, i] * x[b, i, k]`. Preserves blade structure across channels.

### CliffordLayerNorm
```python
CliffordLayerNorm(algebra, channels)
```
Geometric RMSNorm: normalizes multivector norm per channel, preserves direction, recovers scale in grade-0.

### GeometricProductLayer
```python
GeometricProductLayer(algebra, channels)
```
Computes `GP(x, x)` — quadratic geometric self-interaction that creates blade cross-terms.

### GeometricGELU
```python
GeometricGELU(algebra, channels)
```
`x' = x * GELU(||x|| + b) / ||x||` — preserves direction while nonlinearly scaling magnitude.

### BladeSelector
```python
BladeSelector(algebra, channels)
```
`2 * sigmoid(logit_g)` per grade — learns which geometric grades to amplify/suppress.

## Expected Advantages

1. **Equivariance**: RotorLayer provides exact SO(3)/Pin(3) equivariance, crucial for spatial puzzle tasks (Sudoku, Maze, ARC).
2. **Parameter Efficiency**: Geometric product mixing creates rich interactions without large hidden dimensions.
3. **Spatial Reasoning**: Multivector representations naturally encode geometric relationships (points, lines, planes in Cl(3,0); spheres, circles, point-pairs in Cl(4,1)).
4. **Constant Memory**: DEQ fixed-point iteration decouples effective depth from memory usage.
5. **Compatibility**: Drop-in replacement for existing Attractor models via `CliffordAttractorConfig`.

## References

- [Attractor: Meta-learning the Answer not the Solution](https://github.com/jacobfa/Attractor)
- [Versor: Universal Geometric Algebra Neural Network](https://github.com/Concode0/Versor)
- [GATr: Geometric Algebra Transformer](https://github.com/Qualcomm-AI-research/geometric-algebra-transformer)
- [Deep Equilibrium Models (Bai et al. 2019)](https://arxiv.org/abs/1909.01377)
