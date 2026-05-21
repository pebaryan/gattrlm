# Evaluation Task Configs (Layer 3)

These YAML files define **evaluation task parameters** — which benchmarks to run, precision, batch size, solver overrides for DEQ models, etc. They are consumed by `scripts/eval.py` via `receval.settings.CLISettings`.

## How they work

Each YAML is passed as a command-line argument to `eval.py`:

```bash
python scripts/eval.py eval_configs/eval-core.yaml --out_dir /path/to/checkpoint
```

Key fields:

| Field | Purpose |
|-------|---------|
| `eval_tasks` | Comma-separated list of tasks (`core`, `core_extended`, `lm_eval`, `bpb`, `sample`) |
| `precision` | Inference precision (`bf16`, `fp16`, `fp32`) |
| `tasks` | Per-task settings (seeds, max samples, few-shot config) |
| `eval_solver` | Attractor/EQLM forward-solver overrides (max_iter, tol, etc.) |

## Available configurations

```
eval-core.yaml        → CORE benchmark (reasoning tasks)
eval-core-extended.yaml → CORE extended benchmark
eval-lambada.yaml     → LAMBADA language modeling
eval-attractor.yaml   → CORE + attractor solver overrides
eval-eqlm.yaml        → CORE + EQLM solver overrides
eval-val-loss.yaml    → Validation loss / bits-per-byte
```

## Relationship to other configs

| Layer | Directory | Format | Purpose |
|-------|-----------|--------|---------|
| **1 — Model** | `attractor/configs/` | Python | Architecture params (n_embd, layers, heads) |
| **2 — Training** | `launch_configs/` | YAML | Trainer settings (LR, batch size, data paths, optimizer) |
| **3 — Evaluation** | `eval_configs/` | YAML | Task definitions (benchmarks, precision, solver overrides) |

## About eval_solver

For Attractor and EQLM models, the `eval_solver` field overrides the DEQ solver at evaluation time (after checkpoint load). This lets you evaluate with more iterations and tighter tolerance than training:

```yaml
eval_solver:
  tol: 1.0e-4
  max_iter: 64
  min_iter: 8
```
