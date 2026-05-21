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
- [Empirical Results](#empirical-results)
  - [The variants tested](#the-variants-tested)
  - [Language modeling on wikitext-103](#language-modeling-on-wikitext-103)
  - [Rotation equivariance (rotor regression)](#rotation-equivariance-rotor-regression)
  - [Findings](#findings)
  - [Reproducing the runs](#reproducing-the-runs)
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

## Configuration: Three-Layer Architecture

The project uses three independent config layers, each with its own directory and format:

| Layer | Directory | Format | Purpose |
|-------|-----------|--------|---------|
| **1 — Model** | `attractor/configs/` | Python | Architecture params (`n_embd`, `n_layers`, heads, vocab size) |
| **2 — Training** | `launch_configs/` | YAML | Trainer settings (LR, batch size, data paths, optimizer) |
| **3 — Evaluation** | `eval_configs/` | YAML | Task definitions (benchmarks, precision, solver overrides) |

### Layer 1: Model Architecture (`attractor/configs/`)

Python files defining architecture parameters. Loaded automatically by `attractor.registry` at import time. Each file exports a `config` dict with a unique `name`.

```python
from attractor.models.config import Config
cfg = Config.from_name("gpt-small-140m")  # loads from attractor/configs/gpt/gpt-small-140m.py
model = cfg.construct_model()
```

### Layer 2: Training Run (`launch_configs/`)

YAML files defining training recipes. Passed to `scripts/train.py` via `jsonargparse.CLI`. Each YAML references a `model_name` that maps to Layer 1, and can override specific architecture params via `model_overwrite`.

```bash
python scripts/train.py launch_configs/gpt-small-140m.yaml
```

### Layer 3: Evaluation Task (`eval_configs/`)

YAML files defining evaluation workloads. Passed to `scripts/eval.py`. Can include `eval_solver` overrides for Attractor/EQLM DEQ solvers (more iterations, tighter tolerance at eval time).

```bash
python scripts/eval.py eval_configs/eval-core.yaml --out_dir /path/to/checkpoint
```

### Key design notes

- **Layers are independent** — you can mix any model config with any training YAML (as long as `model_name` matches)
- **Model configs are portable** — architecture definitions live in the package, while training/eval YAMLs contain environment-specific paths
- **No duplication** — model architecture params live only in Layer 1; the YAML layers reference them by name

---

## Empirical Results

The Clifford extension introduces several architectural knobs — which sublayer carries the Clifford structure, where in the stack to place it, what positional encoding to use. To map the design space, we ran a 7-way LM A/B on wikitext-103 and a 4-arm rotation-equivariance benchmark on a single RTX 5060 Ti (Blackwell, 17 GB VRAM, Windows). All LM runs use the same 140M-parameter recipe (`block_size=512`, `batch=8`, MuonAdamW lr 4e-3, bf16-mixed, 500 steps).

### The variants tested

| Variant | FP attn | FP MLP | Prelude attn | Positional encoding |
|---|---|---|---|---|
| **Attractor** | std | std | std | RoPE |
| **CliffordLM** | std | Clifford | std | RoPE |
| **AttnOnlyCliffordLM** | Clifford | std | std | RoPE (prelude) |
| **NativeCliffordLM** | Clifford | Clifford | std | RoPE (prelude) |
| PreludeOnlyClifford | std | std | Clifford | learned `wpe` (no RoPE) |
| PreludeAndFPClifford | Clifford | std | Clifford | learned `wpe` (no RoPE) |
| MV-RoPE prelude | std | std | Clifford | **multivector RoPE** |
| RoPE-control | std | std | std | learned `wpe` (no RoPE) |

These map onto `CliffordLMConfig` flags `clifford_attention`, `clifford_mlp`, `clifford_attention_prelude`, `multivector_rope`, and `disable_rope_in_prelude`.

### Language modeling on wikitext-103

| Variant | val (step 499) | wall-clock | Δ vs Attractor |
|---|---:|---:|---:|
| **AttnOnlyCliffordLM** | **6.0156** | 5:17 | **−0.022** (best) |
| Attractor | 6.0376 | 4:59 | — |
| CliffordLM | 6.0619 | 9:04 | +0.024 |
| MV-RoPE prelude | 6.1640 | 4:23 | +0.126 |
| RoPE-control | 6.1955 | 5:17 | +0.158 |
| PreludeAndFPClifford | 6.2196 | 4:04 | +0.182 |
| PreludeOnlyClifford | 6.2286 | 4:33 | +0.191 |

The top three are tied within noise; the bottom four trail by ~0.13–0.19 val loss.

**Decomposing the prelude penalty.** PreludeOnly (no RoPE, Clifford attn) lands +0.191 from baseline. The RoPE-control (no RoPE, _standard_ attn) lands +0.158, isolating the RoPE-loss cost. Multivector RoPE (which preserves the relative-position property — see test `test_multivector_rope_relative_position`) recovers ~0.032 of that, leaving ~0.126 as the intrinsic Clifford-attention expressivity gap vs std attention at this scale.

### Rotation equivariance (rotor regression)

Synthetic task: predict `y = R · v · R̃` from a 3D vector `v` and rotor `R` (both encoded as Cl(3,0) multivectors). Two extrapolation probes:

- **Angle**: train on rotations of angle `[0, π/2]`, test on `[π/2, π]`. Axes uniform on the sphere both times.
- **Axis**: train with rotation axes in a 30° cone around `+z`, test on the full sphere. Angles uniform `[0, π]` both times.

Lower **extrapolation gap** (out-of-dist MSE ÷ in-dist MSE) means better generalization.

| Arm | Params | Angle gap | Axis gap | Wall-clock |
|---|---:|---:|---:|---:|
| MLP (no Clifford) | 21k | 109× | 215× | ~10s |
| CliffordMLP (DEQ + Clifford block) | 11k | 17.6× | 17.8× | ~230s |
| **CliffordAttn** (Clifford attn + std FF) | 18k | **4.2×** | **2.5×** | ~37s |
| CliffordBoth (Clifford attn + Clifford FF) | 19k | 4.2× | 2.5× | ~150s |

CliffordAttn and CliffordBoth are identical to 2 decimal places — once attention is Clifford, the MLP algebra is irrelevant for equivariance, and Clifford-MLP adds 4× wall-clock for zero quality.

### Findings

1. **Clifford attention** is the operative component for rotation equivariance; the Clifford MLP earns nothing on top.
2. **On plain text, all Clifford variants are quality-neutral or slightly worse** than the standard Attractor baseline. Geometric structure pays no rent on tasks without geometric structure.
3. **Where to place Clifford attention**: in the FP head (`AttnOnlyCliffordLM`), not the prelude. Prelude placement costs ~0.13 val loss even with multivector RoPE substituting for lost RoPE.
4. **Multivector RoPE works mathematically** — the relative-position property `<R_m q R̃_m, R_n k R̃_n>` reduces to a function of `m−n` (verified to 1e-4 in `tests/test_models.py`). It only recovers ~20% of the prelude penalty empirically; the residual is intrinsic Clifford-attention expressivity loss.

**Pareto recommendation:** `AttnOnlyCliffordLM` dominates the design space — matches Attractor on text within noise (+6% wall-clock), and inherits the full 4–48× equivariance extrapolation advantage of Clifford attention on geometric tasks. For pure-text workloads with no geometric structure expected anywhere downstream, plain `Attractor` is still the right default.

### Reproducing the runs

All recipes live in `launch_configs/`:

| YAML | Variant |
|---|---|
| `attractor-small-140m-wikitext-long.yaml` | Attractor (baseline) |
| `clifford-small-140m-wikitext-long.yaml` | CliffordLM |
| `attn-only-clifford-small-140m-wikitext-long.yaml` | **AttnOnlyCliffordLM** |
| `prelude-only-clifford-small-140m-wikitext-long.yaml` | PreludeOnly Clifford |
| `prelude-and-fp-clifford-small-140m-wikitext-long.yaml` | PreludeAndFP Clifford |
| `rope-control-small-140m-wikitext-long.yaml` | RoPE-control (std attn, no RoPE) |
| `mvrope-prelude-clifford-small-140m-wikitext-long.yaml` | Clifford prelude + multivector RoPE |

For data, the recipes read wikitext-103 parquet shards from the local HuggingFace cache (`Salesforce/wikitext`) and the tokenizer from `SandyResearch/parcae-tokenizer`. Launch any of them with:

```bash
WANDB_MODE=disabled LOCAL_WORLD_SIZE=1 HF_HUB_OFFLINE=1 \
    python scripts/train.py --config launch_configs/attn-only-clifford-small-140m-wikitext-long.yaml
```

The 4-arm rotation-equivariance benchmark is a single self-contained script:

```bash
python experiments/benchmark_equivariance.py --device cuda --epochs 80
```

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
| `clifford-small-140m.yaml` | CliffordLM | 140M |
| `parcae-small-140m.yaml` | Parcae (baseline) | 140M |
| `parcae-medium-370m.yaml` | Parcae (baseline) | 370M |
| `gpt-small-140m.yaml` | GPT (baseline) | 140M |
| `gpt-medium-370m.yaml` | GPT (baseline) | 370M |

Launch with:

```bash
bash runs/run_training.sh launch_configs/attractor-small-140m.yaml attractor-small 2
```

For the head-to-head Clifford-variant comparison on a single GPU using the local HuggingFace cache, see the [Empirical Results](#empirical-results) section above and the `*-wikitext-long.yaml` recipes.

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
│   ├── configs/                     # Model architecture definitions (Python)
│   │   ├── gpt/                     # GPT baseline configs (small → xlarge)
│   │   ├── attractor/               # Attractor configs (small → xlarge)
│   │   ├── eqlm/                    # EQLM configs
│   │   └── parcae/                  # Parcae configs
│   └── __init__.py                  # Lazy exports for all public APIs
├── launch_configs/                 # Training run configs (YAML) — hyperparameters, data paths
├── eval_configs/                   # Evaluation task configs (YAML) — benchmarks, solver overrides
├── experiments/
│   ├── clifford_toy_example.py      # Minimal working example
├── tests/
│   ├── test_clifford_attractor.py   # 65 tests: algebra, layers, DEQ, backward
│   ├── test_clifford_cga.py         # 47 tests: null basis, embedding, rotors, meet, gradients
│   └── test_models.py               # 65 tests: GPT, Parcae, EQLM, Attractor
├── recpre/                          # Training infrastructure, optimizers & schedulers
├── receval/                         # Evaluation infrastructure, tasks & metrics
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
