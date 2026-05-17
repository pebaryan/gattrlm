<div align="center">

# Solve the Loop: Attractor Models for Language and Reasoning

**Jacob Fein-Ashley &nbsp;&middot;&nbsp; Paria Rashidinejad**

University of Southern California

[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=github)](https://attractor-models.github.io)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2605.12466)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[![Attractor-140M](https://img.shields.io/badge/%F0%9F%A4%97-Attractor--140M-yellow?style=flat-square)](https://huggingface.co/jacobfa1/attractor-140m)
[![Attractor-370M](https://img.shields.io/badge/%F0%9F%A4%97-Attractor--370M-yellow?style=flat-square)](https://huggingface.co/jacobfa1/attractor-370m)
[![Attractor-770M](https://img.shields.io/badge/%F0%9F%A4%97-Attractor--770M-yellow?style=flat-square)](https://huggingface.co/jacobfa1/attractor-770m)

</div>

---

## Table of Contents

- [Installation](#installation)
- [What is an Attractor Model?](#what-is-an-attractor-model)
- [Quick Start: Standard Attractor](#quick-start-standard-attractor)
- [Geometric (Clifford) Algebra Extension](#geometric-clifford-algebra-extension)
  - [Why Geometric Algebra?](#why-geometric-algebra)
  - [Architecture Overview](#architecture-overview)
  - [Quick Start: Clifford Toy Example](#quick-start-clifford-toy-example)
- [Cl(3,0) Euclidean Algebra](#cl30-euclidean-algebra)
  - [Multivector Basics](#multivector-basics)
  - [Usage Example](#cl30-usage-example)
- [Cl(4,1) Conformal Geometric Algebra](#cl41-conformal-geometric-algebra)
  - [CGA Fundamentals](#cga-fundamentals)
  - [CGA Formulas](#cga-formulas)
  - [Usage Example](#cl41-usage-example)
- [Layer Reference](#layer-reference)
- [Training](#training)
- [Project Structure](#project-structure)
- [Pretrained Models](#pretrained-models)
- [Citation](#citation)
- [References](#references)

---

## Installation

Requires Python 3.11+ and PyTorch 2.4+. Install PyTorch first following [pytorch.org](https://pytorch.org/get-started/locally/), then:

```bash
pip install -e .
```

### Test the Clifford module

```bash
# Run all Clifford algebra and CGA tests
python -m pytest tests/test_clifford_attractor.py -v        # 65 Cl(3,0) tests
python -m pytest tests/test_clifford_cga.py -v              # 47 Cl(4,1) CGA tests
```

---

## What is an Attractor Model?

Attractor models replace deep transformer stacks with a **deep equilibrium (DEQ) fixed-point solver**. Instead of stacking L layers, an attractor learns a single block `f` and iterates:

```
x_{t+1} = f(x_t)   for t = 0..T
```

until convergence to a fixed point `x* = f(x*)`. This decouples **effective depth from memory** — you can iterate for hundreds of steps while only storing the final state (via implicit differentiation).

The paper ["Solve the Loop"](https://arxiv.org/abs/2605.12466) shows attractor models match or exceed standard transformers at language modeling, reasoning (Sudoku, ARC-AGI), and in-context learning, while using constant memory.

---

## Quick Start: Standard Attractor

```python
from attractor.models.attractor import Attractor, AttractorConfig

config = AttractorConfig.from_name("attractor-small-140m")
model = config.construct_model()
```

---

## Geometric (Clifford) Algebra Extension

This repository extends the Attractor framework with **Geometric (Clifford) algebra layers** — rotors, geometric products, and grade-preserving linear maps — inside the DEQ attractor to achieve **built-in rotation/reflection equivariance**.

### Why Geometric Algebra?

| Feature | Standard Neural Net | Clifford Attractor |
|---|---|---|
| **Rotation equivariance** | Learned via data augmentation | Built-in via **RotorLayer** (sandwich product `R x R̃`) |
| **Spatial representation** | Scalar embeddings | **Multivector embeddings** in Cl(3,0) or Cl(4,1) |
| **Geometric reasoning** | Implicit | Explicit via **geometric product**, wedge, and grade projection |
| **Parameter efficiency** | 4× intermediate expansion | **GeometricProductLayer** creates nonlinear interactions without extra parameters |
| **Memory** | Grows with depth | **Constant** (DEQ fixed-point) |

### Architecture Overview

```
             ┌─────────────────────────────────────────────┐
x → Embed →  │ LayerNorm → Rotor(RxR̃) → + → Linear        │
             │ GeometricProduct(x⊗x) → GeometricGELU      │  ← DEQ block f
             │ Linear → BladeSelector → +                 │
             └─────────────────────────────────────────────┘
                            ↓ iterate to fixed point
                         x* = f(x*)
```

### Quick Start: Clifford Toy Example

```bash
python experiments/clifford_toy_example.py
```

This trains a CliffordAttractor on a synthetic vector field fixed-point task. Output shows convergence and loss curves.

---

## Cl(3,0) Euclidean Algebra

### Multivector Basics

In Cl(3,0) with basis vectors `{e₁, e₂, e₃}` satisfying `eᵢ² = +1`, a multivector has **8 blades**:

| Grade | Basis Blades | Count |
|-------|-------------|-------|
| 0 (scalar) | `1` | 1 |
| 1 (vectors) | `e₁, e₂, e₃` | 3 |
| 2 (bivectors) | `e₁₂, e₁₃, e₂₃` | 3 |
| 3 (pseudoscalar) | `e₁₂₃` | 1 |

A multivector is represented as an 8-element vector `[s, e₁, e₂, e₃, e₁₂, e₁₃, e₂₃, e₁₂₃]`, and a batch has shape `(batch, channels, 8)`.

### Cl(3,0) Usage Example

```python
import torch
from attractor.models.clifford_attractor import (
    CliffordAlgebra, CliffordAttractorConfig, CliffordAttractor
)

# Create a Cl(3,0) algebra (Euclidean 3D)
algebra = CliffordAlgebra(p=3, q=0)
print(f"Algebra dimension: {algebra.dim}")  # 8 blades

# ── RotorLayer: built-in rotation equivariance ──
from attractor.models.clifford_attractor import RotorLayer
rotor = RotorLayer(algebra, channels=4)
x = torch.randn(2, 4, 8)         # (batch, channels, 8 blades)
x_rotated = rotor(x)              # R * x * R̃

# ── CliffordLinear: channel mixing with blade structure ──
from attractor.models.clifford_attractor import CliffordLinear
linear = CliffordLinear(algebra, in_channels=4, out_channels=8)
y = linear(x)                     # (2, 8, 8)

# ── GeometricProductLayer: quadratic self-interaction ──
from attractor.models.clifford_attractor import CliffordAttractorBlock
block = CliffordAttractorBlock(algebra, channels=4)
z = block(x)                      # full block: norm → rotor → linear → gp → gelu → linear → blade-select

# ── Full CliffordAttractor model with DEQ solver ──
cfg = CliffordAttractorConfig(
    p=3, q=0,
    channels=16,
    num_blocks=1,
    max_iter=10,                   # DEQ fixed-point iterations
    vocab_size=11,                 # for token embedding
    d_model=64,
)
model = CliffordAttractor(cfg, vocab_size=11)
tokens = torch.randint(0, 10, (2, 9))
logits = model(tokens)            # forward pass with DEQ solve
loss = logits.sum()
loss.backward()                   # implicit differentiation through fixed point
```

---

## Cl(4,1) Conformal Geometric Algebra

### CGA Fundamentals

CGA embeds 3D Euclidean space into a 5D Minkowski-like space Cl(4,1) where:

- `e₁² = e₂² = e₃² = e₄² = +1` (4 positive-norm vectors)
- `e₅² = -1` (1 negative-norm vector)

Two **null vectors** are formed from `e₄` and `e₅`:

- **Origin**: `e₀ = ½(e₅ - e₄)` &nbsp;&nbsp; `e₀² = 0`
- **Infinity**: `e∞ = e₄ + e₅` &nbsp;&nbsp; `e∞² = 0`
- **Inner product**: `e₀ · e∞ = -1`

These null vectors allow **spheres, planes, circles, and lines** to be represented as **grade-1 multivectors** — the same algebraic object as a point. Operations like intersection become simple products.

### CGA Formulas

#### Point Embedding

A 3D point `x = x₁e₁ + x₂e₂ + x₃e₃` is embedded as a null vector:

```
P(x) = e₀ + x + ½|x|² e∞
```

**Key property**: `P(x)² = 0` (null condition — confirms the point lies on the conformal null cone).

```python
from attractor.models.clifford_cga import embed_point
import torch

algebra = CliffordAlgebra(p=4, q=1)
pt = embed_point(algebra, torch.tensor([1.0, 2.0, 3.0]))
pt_sq = algebra.geometric_product(pt, pt)[0]
print(f"P(x)² = {pt_sq.item():.6f}")  # ≈ 0 (float32 precision)
```

#### Sphere

A sphere with center `c` and radius `r`:

```
S(c, r) = P(c) - ½r² e∞
```

```python
from attractor.models.clifford_cga import embed_sphere
sphere = embed_sphere(algebra, torch.tensor([0.0, 0.0, 0.0]), radius=2.0)
```

A point is a sphere of radius 0: `P(x) = S(x, 0)`.

#### Plane

A plane with unit normal `n` and signed distance `d` (from origin):

```
π(n, d) = n + d e∞
```

A plane can also be seen as a sphere through infinity.

```python
from attractor.models.clifford_cga import embed_plane
plane = embed_plane(algebra, torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0]))
```

#### Circle (Meet of Two Spheres)

```
Circle = ⟨S₁ * S₂ * I⁻¹⟩₃
```

where `I⁻¹` is the inverse pseudoscalar and `⟨·⟩₃` projects to grade 3.

```python
from attractor.models.clifford_cga import embed_sphere, meet

s1 = embed_sphere(algebra, torch.tensor([0.0, 0.0, 0.0]), 2.0)
s2 = embed_sphere(algebra, torch.tensor([1.0, 0.0, 0.0]), 2.0)
circle = meet(algebra, s1, s2)
```

#### Line (Meet of Two Planes)

```
Line = ⟨π₁ * π₂ * I⁻¹⟩₃
```

#### Translation Rotor

Translates by vector `t`:

```
T(t) = exp(-½ t e∞) = 1 - ½ t e∞
```

Since `e∞² = 0`, this simplifies to a linear transformation. Applying `T` to a point:

```
T(x) P(x) T̃(x) = P(x + t)
```

```python
from attractor.models.clifford_cga import translation_rotor
T = translation_rotor(algebra, torch.tensor([1.0, 0.0, 0.0]))
```

#### Rotation Rotor

Standard SO(3) rotation by angle `θ` around axis `n` (unit bivector):

```
R(θ, n) = exp(½θ n) = cos(θ/2) + n sin(θ/2)
```

```python
from attractor.models.clifford_cga import rotation_rotor
R = rotation_rotor(algebra, torch.tensor([0.0, 0.0, 1.0]), angle=torch.tensor(0.5))
```

#### Screw Rotor (Rigid Motion)

Combines rotation and translation into a single rotor:

```
M(t, θ, n) = T(t) R(θ, n)
```

```python
from attractor.models.clifford_cga import screw_rotor
M = screw_rotor(algebra, translation=torch.tensor([1.0, 0.0, 0.0]),
                       axis=torch.tensor([0.0, 0.0, 1.0]),
                       angle=torch.tensor(0.5))
```

#### Dual

The CGA dual maps between geometric primitives. For example, the dual of a sphere is a point-pair, and vice versa:

```
A* = A * I⁻¹
```

where `I⁻¹ = -I` because for Cl(4,1), `I² = -1`.

```python
from attractor.models.clifford_cga import dual
sphere_dual = dual(algebra, sphere)  # a point-pair
```

#### Outer Product (Wedge)

The outer product grades up: `grade(a ∧ b) = grade(a) + grade(b)`. Used for constructing higher-grade objects.

```python
from attractor.models.clifford_cga import outer_product
circle = outer_product(algebra, point1, point2)  # circle through two points (plus at infinity)
```

### Cl(4,1) Usage Example

```python
import torch
from attractor.models.clifford_attractor import CliffordAlgebra, CliffordAttractorConfig, CliffordAttractor
from attractor.models.clifford_cga import (
    embed_point, embed_sphere, embed_plane,
    translation_rotor, rotation_rotor, screw_rotor,
    dual, meet, outer_product, create_cga_attractor
)

algebra = CliffordAlgebra(p=4, q=1)

# ── Geometric primitives ──
pt = embed_point(algebra, torch.tensor([1.0, 2.0, 3.0]))
sphere = embed_sphere(algebra, torch.tensor([0.0, 0.0, 0.0]), radius=2.0)
plane = embed_plane(algebra, torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0]))

# ── Intersections ──
s1 = embed_sphere(algebra, torch.tensor([0.0, 0.0, 0.0]), 2.0)
s2 = embed_sphere(algebra, torch.tensor([1.0, 0.0, 0.0]), 2.0)
circle = meet(algebra, s1, s2)     # circle from intersecting spheres

p1 = embed_plane(algebra, torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0]))
p2 = embed_plane(algebra, torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0]))
line = meet(algebra, p1, p2)       # line from intersecting planes

# ── Rigid motions via CGA rotors ──
T = translation_rotor(algebra, torch.tensor([1.0, 0.0, 0.0]))
R = rotation_rotor(algebra, torch.tensor([0.0, 0.0, 1.0]), angle=torch.tensor(0.5))
M = screw_rotor(algebra, translation=torch.tensor([1.0, 0.0, 0.0]),
                       axis=torch.tensor([0.0, 0.0, 1.0]),
                       angle=torch.tensor(0.5))

# Apply rotors: R x R̃ (sandwich product)
point_moved = algebra.geometric_product(
    algebra.geometric_product(M, pt), algebra.reverse(M)
)

# ── Full CGA attractor model ──
model = create_cga_attractor(
    channels=16,
    num_blocks=1,
    max_iter=10,
    vocab_size=11,
    d_model=64,
)
tokens = torch.randint(0, 10, (2, 9))
logits = model(tokens)
```

For a complete working example, see [`experiments/clifford_toy_example.py`](experiments/clifford_toy_example.py).

---

## Layer Reference

All layers are in [`attractor/models/clifford_attractor.py`](attractor/models/clifford_attractor.py).

### CliffordAlgebra

Core geometric algebra engine for arbitrary `Cl(p,q,r)` signatures. Precomputes:
- **Geometric product (Cayley table)**: `gp(a, b)` — fundamental GA multiplication
- **Grade index map**: which grade each blade belongs to
- **Metric signs**: blade squared norms (±1)

Key methods:
| Method | Description |
|--------|-------------|
| `geometric_product(x, y)` | GA multiplication via einsum over precomputed table |
| `norm_sq(x)` | Euclidean squared norm `∑xᵢ²` (uses absolute signs) |
| `reverse(x)` | Grade-dependent sign flips for reversion |
| `involution(x)` | Grade-dependent sign flips for grade involution |
| `grade_projection(x, k)` | Extract grade-k components |
| `_grade_index` (buffer) | Precomputed grade of each blade index |
| `exp_bivector(B)` | Closed-form bivector exponential (handles elliptic/hyperbolic) |

### RotorLayer

```python
RotorLayer(algebra, channels)
```

Learns bivector coefficients → exponentiate to rotors → `R * x * R̃`. Provides **built-in rotation equivariance**.

### CliffordLinear

```python
CliffordLinear(algebra, in_channels, out_channels)
```

Channel-mixing linear layer: `out[b, o, k] = W[o, i] * x[b, i, k]`. Preserves blade structure across channels.

### CliffordLayerNorm

```python
CliffordLayerNorm(algebra, channels)
```

Geometric RMSNorm: normalizes multivector norm per channel, preserves direction, recovers scale via learned scalar weight.

### GeometricProductLayer

```python
GeometricProductLayer(algebra, channels)
```

Computes `GP(x, x)` — quadratic geometric self-interaction that creates blade cross-terms without extra parameters.

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

### CGA Functions

All in [`attractor/models/clifford_cga.py`](attractor/models/clifford_cga.py):

| Function | Description |
|----------|-------------|
| `cga_basis(algebra)` | Build null basis vectors `e₀`, `e∞` as multivectors |
| `embed_point(algebra, x)` | Embed 3D point `P(x) = e₀ + x + ½|x|²e∞` |
| `extract_euclidean(algebra, pt)` | Recover 3D coordinates from embedded point |
| `squared_distance(algebra, p1, p2)` | `P₁·P₂ = -½d²` |
| `embed_sphere(algebra, center, radius)` | `S(c,r) = P(c) - ½r²e∞` |
| `embed_plane(algebra, normal, distance)` | `π(n,d) = n + d e∞` |
| `translation_rotor(algebra, t)` | `T = 1 - ½t e∞` |
| `rotation_rotor(algebra, axis, angle)` | `R = exp(½θn)` |
| `screw_rotor(algebra, translation, axis, angle)` | `M = T R` |
| `dual(algebra, A)` | `A* = A * I⁻¹` |
| `outer_product(algebra, A, B)` | Grade-increasing wedge product |
| `meet(algebra, A, B)` | `⟨A * B * I⁻¹⟩₃` for grade-1 intersection |
| `create_cga_attractor(...)` | Factory for end-to-end CGA Attractor model |

---

## Training

### Language Modeling

Training is configured via YAML files in `launch_configs/`.

| Config | Architecture | Parameters |
|--------|-------------|------------|
| `attractor-small-140m.yaml` | Attractor | 140M |
| `attractor-medium-370m.yaml` | Attractor | 370M |
| `attractor-large-770m.yaml` | Attractor | 770M |
| `attractor-xlarge-1_3b.yaml` | Attractor | 1.3B |
| `parcae-small-140m.yaml` | Parcae (baseline) | 140M |
| `parcae-medium-370m.yaml` | Parcae (baseline) | 370M |
| `gpt-small-140m.yaml` | GPT (baseline) | 140M |
| `gpt-medium-370m.yaml` | GPT (baseline) | 370M |

Launch with:

```bash
bash runs/run_training.sh launch_configs/attractor-small-140m.yaml attractor-small 2
```

Current benchmark note: for the Clifford LM path in this repo, the default benchmark setting is `CliffordAttractor-LM-24ch-LightSolve`. It has been the best speed/quality tradeoff we have measured so far, while the other Clifford entries remain as ablations for mixer, width, and solver depth.

Benchmark summary on Wikitext-2 in `scabi`:

| Model | Final eval loss | Time | Note |
|---|---:|---:|---|
| `RepoNativeGPT-Small` | `8.6983` | `47.5s` | fastest, weak validation |
| `RepoNativeGPT-Medium` | `8.0046` | `67.8s` | slightly better, still weak validation |
| `MiniCliffordAttractor` | `6.7621` | `35.9s` | simple Clifford proxy |
| `CausalSequenceProxyBaseline` | `6.9677` | `73.8s` | non-Clifford causal stand-in |
| `CliffordAttractor-LM-24ch-LightSolve` | `5.9553` | `210.2s` | best validation among the Clifford runs |
| `Parcae-Small-Original` | `7.8179` | `408.0s` | best of the recovered original baselines |
| `Attractor-Small-Original` | `10.1806` | `351.8s` | overfits hard |
| `EQLM-Small-Original` | `10.4783` | `353.7s` | overfits hard |

### Sudoku & Maze Reasoning

```bash
torchrun --standalone --nproc_per_node=2 \
    -m experiments.attractor_puzzles.trm.train_trm_deq \
    --data_dir /path/to/sudoku-data \
    --out_dir /path/to/output
```

### Clifford Sudoku

```bash
torchrun --nnodes=1 --nproc-per-node=1 \
    experiments/attractor_puzzles/trm/train_clifford_trm.py \
    --dataset-paths /path/to/sudoku-extreme \
    --output-dir ./outputs/clifford-sudoku \
    --p 3 --q 0 --channels 16 --num-blocks 3 \
    --lr 1e-3 --deq-max-iter 30 --jacobian-reg-lambda 0.01
```

### ARC-AGI Puzzles

```bash
bash experiments/attractor_puzzles/launch_arc_deq.sh
```

### Evaluation

```bash
python scripts/eval.py --out_dir /path/to/checkpoint --eval_tasks core
```

---

## Project Structure

```
gattrlm/
├── attractor/
│   ├── models/
│   │   ├── clifford_attractor.py    # Core: CliffordAlgebra, layers, config, DEQ solver
│   │   └── clifford_cga.py          # Cl(4,1) CGA: point/sphere/plane, rotors, meet
│   └── __init__.py                  # Lazy exports for all public APIs
├── experiments/
│   ├── clifford_toy_example.py      # Minimal working example
│   ├── attractor_puzzles/           # Sudoku, Maze, ARC-AGI with TRM-DEQ
│   └── eqlm_sudoku/                 # Sudoku via EQLM
├── tests/
│   ├── test_clifford_attractor.py   # 65 tests: algebra, layers, DEQ, backward
│   └── test_clifford_cga.py         # 47 tests: null basis, embedding, rotors, meet, gradients
├── recpre/                          # Training infrastructure
├── receval/                         # Evaluation infrastructure
├── scripts/                         # Training, eval, generation entry points
├── README.md                        # This file
└── README_CLIFFORD.md               # Original Clifford documentation
```

---

## Pretrained Models

| Model | Parameters | HuggingFace |
|-------|-----------|-------------|
| Attractor-140M | 140M | [jacobfa1/attractor-140m](https://huggingface.co/jacobfa1/attractor-140m) |
| Attractor-370M | 370M | [jacobfa1/attractor-370m](https://huggingface.co/jacobfa1/attractor-370m) |
| Attractor-770M | 770M | [jacobfa1/attractor-770m](https://huggingface.co/jacobfa1/attractor-770m) |

---

## Citation

```bibtex
@article{feinashley2026attractor,
  title={Solve the Loop: Attractor Models for Language and Reasoning},
  author={Fein-Ashley, Jacob and Rashidinejad, Paria},
  year={2026},
  url={https://arxiv.org/abs/2605.12466}
}
```

---

## References

- [Attractor: Meta-learning the Answer not the Solution](https://github.com/jacobfa/Attractor) — Original Attractor repository
- [Deep Equilibrium Models (Bai et al. 2019)](https://arxiv.org/abs/1909.01377) — DEQ fixed-point solving and implicit differentiation
- [Versor: Universal Geometric Algebra Neural Network](https://github.com/Concode0/Versor) — GA neural network framework
- [GATr: Geometric Algebra Transformer (Qualcomm AI Research)](https://github.com/Qualcomm-AI-research/geometric-algebra-transformer) — GA transformers for equivariant learning
- [A Guided Tour to the Plane-Based Geometric Algebra pga3d (Gunn)](https://bivector.org/PGA.pdf) — Reference for conformal GA formulas
- [Geometric Algebra for Physicists (Doran & Lasenby)](https://www.cambridge.org/us/academic/subjects/physics/theoretical-physics-and-mathematical-physics/geometric-algebra-physicists) — Comprehensive GA textbook
- [Parcae](https://github.com/sandyresearch/parcae) — Base training infrastructure
