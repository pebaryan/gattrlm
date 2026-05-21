"""Eval-time wrapper for CliffordLM.

CliffordLM subclasses Attractor, so ModelingCliffordLM multi-inherits from
ModelingAttractor (which adds save/load, generate, KV cache, eval-solver
overrides, solver stats) and from CliffordLM. The C3 MRO puts CliffordLM
ahead of Attractor in the chain, so the Clifford ._make_fp_block is picked
up by Attractor.__init__ as expected.

We only override from_pretrained to use CliffordLMConfig instead of
AttractorConfig.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import torch

from attractor.models.clifford_lm import CliffordLM, CliffordLMConfig
from receval.models.attractor import ModelingAttractor


class ModelingCliffordLM(ModelingAttractor, CliffordLM):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path],
                        device=None, dtype=None, **kwargs):
        path = Path(pretrained_model_name_or_path)
        if not path.exists():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(
                repo_id=str(pretrained_model_name_or_path),
                allow_patterns=["*.json", "*.bin", "*.safetensors", "*.pt"]))
        from attractor.models.config import RoPESettings
        with open(path / "config.json") as f:
            config_dict = json.load(f)
        if "rope_settings" in config_dict and isinstance(config_dict["rope_settings"], dict):
            config_dict["rope_settings"] = RoPESettings(**config_dict["rope_settings"])
        config_dict.update(kwargs)
        for key in ["_class_name", "init"]:
            config_dict.pop(key, None)
        model = cls(CliffordLMConfig(**config_dict))

        weights_path = None
        for name in ["pytorch_model.bin", "model.safetensors", "model.bin", "model.pt"]:
            if (path / name).exists():
                weights_path = path / name
                break
        if weights_path is None:
            raise FileNotFoundError(f"No weights found in {path}")
        if weights_path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        cleaned = {}
        for k, v in state_dict.items():
            for prefix in ["module.", "_orig_mod.", "model.", "_forward_module."]:
                if k.startswith(prefix):
                    k = k[len(prefix):]
            cleaned[k] = v
        model.load_state_dict(cleaned, strict=False)
        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device=device)
        model.eval()
        return model
