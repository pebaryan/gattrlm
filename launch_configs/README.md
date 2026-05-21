# Training Run Configs (Layer 2)

These YAML files define **training recipe parameters** — optimizer settings, learning rate schedule, batch size, data paths, etc. They are consumed by `scripts/train.py` via `jsonargparse.CLI`.

## How they work

Each YAML is passed as a command-line argument to `train.py`:

```bash
python scripts/train.py launch_configs/gpt-small-140m.yaml
```

`jsonargparse.CLI` auto-parses the YAML into a `CLISettings` dataclass. Key fields:

| Field | Purpose |
|-------|---------|
| `model_name` | References a model config from `attractor/configs/` |
| `model_overwrite` | Override specific architecture params (e.g., `vocab_size`) |
| `optimizer` / `optim_config` | Optimizer type and hyperparameters |
| `data_config` | Training and validation data sources |
| `block_size`, `world_batch_size`, etc. | Training loop parameters |

## Relationship to other configs

| Layer | Directory | Format | Purpose |
|-------|-----------|--------|---------|
| **1 — Model** | `attractor/configs/` | Python | Architecture params (n_embd, layers, heads) |
| **2 — Training** | `launch_configs/` | YAML | Trainer settings (LR, batch size, data paths, optimizer) |
| **3 — Evaluation** | `eval_configs/` | YAML | Task definitions (benchmarks, precision, solver overrides) |

## Available configurations

```
attractor-small-140m.yaml    → attractor-small-140m
attractor-medium-370m.yaml   → attractor-medium-370m
attractor-large-770m.yaml    → attractor-large-770m
attractor-xlarge-1_3b.yaml   → attractor-xlarge-1_3b
clifford-small-140m.yaml     → clifford-small-140m (hybrid: attention + Clifford MLP)
eqlm-small-140m.yaml         → eqlm-small-140m (identical to attractor config)
parcae-small-140m.yaml       → parcae-small-140m
gpt-small-140m.yaml          → gpt-small-140m
```

## Notes

- Data paths point to cluster-specific locations (`/resource/data/`, `/scratch1/feinashl/`, etc.)
- Training YAMLs are environment-specific; the model configs in `attractor/configs/` are portable
