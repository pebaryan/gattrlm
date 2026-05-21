# Model Architecture Configs (Layer 1)

These Python files define **model architecture parameters** — `n_embd`, `n_layers`, `vocab_size`, etc. They are the canonical source of truth for what each model variant looks like.

## How they work

Each file defines a `config` dict (or a list of configs) that gets loaded by `attractor/registry.py`:

```python
# attractor/registry.py scans this directory at import time
name_to_config = {config["name"]: config for config in configs}
```

Configs are then retrieved by name via `Config.from_name()`:

```python
from attractor.models.config import Config
cfg = Config.from_name("gpt-small-140m")
model = cfg.construct_model()
```

## Directory layout

```
attractor/configs/
├── gpt/          # GPT baseline: small-140m → xlarge-1_3b
├── attractor/    # Attractor: small-140m → xlarge-1_3b
├── eqlm/         # EQLM: small-140m → xlarge-1_3b
├── parcae/       # Parcae: small-140m → xlarge-1_3b
└── __init__.py
```

## Relationship to other configs

| Layer | Directory | Format | Purpose |
|-------|-----------|--------|---------|
| **1 — Model** | `attractor/configs/` | Python | Architecture params (n_embd, layers, heads) |
| **2 — Training** | `launch_configs/` | YAML | Trainer settings (LR, batch size, data paths, optimizer) |
| **3 — Evaluation** | `eval_configs/` | YAML | Task definitions (benchmarks, precision, solver overrides) |

Training YAMLs reference a `model_name` (e.g., `gpt-small-140m`) that maps to a config here. The `model_overwrite` field in the YAML can override specific architecture params.
