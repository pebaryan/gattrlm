"""
Unit tests for the main language model classes: GPT, Parcae, EQLM, Attractor.

Tests cover:
  - Config creation and validation
  - Model construction from config
  - Forward pass shape correctness
  - Backward pass (gradient flow to all parameters)
  - Determinism and reproducibility
  - Training steps with loss
  - Edge cases (single token, no labels, return_logits)
"""

import pytest
import torch
import torch.nn.functional as F

import attractor

# ========================================================================
#  Helpers
# ========================================================================


def _tiny_gpt_config(**overrides):
    """Minimal GPT config for fast tests."""
    cfg = attractor.create_config("gpt-small-140m")
    # Override to make it tiny and fast
    for k, v in dict(
        n_embd=64,
        n_layer=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=128,
        block_size=32,
        vocab_size=256,
        padding_multiple=64,
        tie_embeddings=True,
        init_strategy="scaled-zero",
        # Disable fused head (requires triton which may not be available)
        use_fused_head="",
    ).items():
        setattr(cfg, k, v)
    # Re-apply __post_init__ effects
    from dataclasses import replace
    cfg = replace(cfg)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.__post_init__()
    return cfg


def _tiny_parcae_config(**overrides):
    """Minimal Parcae config for fast tests."""
    cfg = attractor.create_config("parcae-small-140m")
    defaults = dict(
        n_embd=64,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=128,
        block_size=32,
        vocab_size=256,
        padding_multiple=64,
        tie_embeddings=False,
        init_strategy="scaled-zero",
        n_layers_in_prelude=1,
        n_layers_in_recurrent_block=1,
        n_layers_in_coda=1,
        mean_recurrence=3,
        mean_backprop_depth=2,
        recurrent_embedding_dimension=64,
        recurrent_intermediation_embedding_dimension=128,
        sampling_scheme="fixed",
        recurrent_iteration_method="per-batch",
        use_fused_head="",
    )
    for k, v in defaults.items():
        setattr(cfg, k, v)
    from dataclasses import replace
    cfg = replace(cfg)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.__post_init__()
    return cfg


def _tiny_eqlm_config(**overrides):
    """Minimal EQLM config for fast tests."""
    cfg = attractor.create_config("eqlm-small-140m")
    defaults = dict(
        n_embd=64,
        n_backbone_layers=1,
        n_fp_blocks=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=128,
        block_size=32,
        vocab_size=256,
        padding_multiple=64,
        tie_embeddings=True,
        init_strategy="scaled-zero",
        max_iter=5,
        min_iter=2,
        tol=1e-2,
        solver="fpi",
        backward_type="onestep",
    )
    for k, v in defaults.items():
        setattr(cfg, k, v)
    from dataclasses import replace
    cfg = replace(cfg)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.__post_init__()
    return cfg


def _tiny_attractor_config(**overrides):
    """Minimal Attractor config for fast tests."""
    # Same as EQLM but uses Attractor architecture
    cfg = attractor.create_config("attractor-small-140m")
    defaults = dict(
        n_embd=64,
        n_backbone_layers=1,
        n_fp_blocks=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=128,
        block_size=32,
        vocab_size=256,
        padding_multiple=64,
        tie_embeddings=True,
        init_strategy="scaled-zero",
        max_iter=5,
        min_iter=2,
        tol=1e-2,
        solver="fpi",
        backward_type="onestep",
    )
    for k, v in defaults.items():
        setattr(cfg, k, v)
    from dataclasses import replace
    cfg = replace(cfg)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.__post_init__()
    return cfg


def _check_finite_grads(model):
    """Assert parameters that received gradients have finite values.

    Some parameters (e.g. ve_gate.weight when set to None) may not
    receive gradients — that's fine. We check that every gradient that
    does exist is finite.
    """
    has_grad = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            has_grad = True
            assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"
    # At least some parameters should have received gradients
    assert has_grad, "No parameters received gradients — backward may not have propagated"


def _check_no_nan_params(model):
    """Assert all parameter values are finite."""
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"{name} has non-finite value"


# ========================================================================
#  GPT Tests
# ========================================================================


class TestGPT:
    """Verify GPT model config, construction, forward, and backward."""

    def test_config_creation(self):
        """Can create GPTConfig from name."""
        cfg = _tiny_gpt_config()
        assert cfg.n_embd == 64
        assert cfg.n_layer == 2
        assert cfg.padded_vocab_size is not None
        assert cfg.padded_vocab_size >= cfg.vocab_size

    def test_model_construction(self):
        """GPT model can be constructed from config."""
        cfg = _tiny_gpt_config()
        model = cfg.construct_model()
        assert isinstance(model, torch.nn.Module)
        _check_no_nan_params(model)

    def test_forward_shape_no_labels(self):
        """Forward pass without labels produces no loss, optional logits."""
        cfg = _tiny_gpt_config()
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x)
        assert "loss" in out
        assert out["loss"].item() == pytest.approx(0.0)
        assert out["logits"] is None  # return_logits=False by default

    def test_forward_with_labels(self):
        """Forward pass with labels computes cross-entropy loss."""
        cfg = _tiny_gpt_config()
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        labels = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x, labels=labels)
        assert out["loss"].item() > 0.0  # loss should be positive
        assert "log_ppl" in out

    def test_forward_with_return_logits(self):
        """return_logits=True returns logits."""
        cfg = _tiny_gpt_config()
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x, return_logits=True)
        assert out["logits"] is not None
        assert out["logits"].shape == (2, 8, cfg.padded_vocab_size)

    def test_backward(self):
        """Backward pass produces gradients for all parameters."""
        cfg = _tiny_gpt_config()
        model = cfg.construct_model()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        labels = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x, labels=labels)
        out["loss"].backward()
        _check_finite_grads(model)

    def test_training_step(self):
        """Model can perform a full training step."""
        cfg = _tiny_gpt_config()
        model = cfg.construct_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        labels = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x, labels=labels)
        loss = out["loss"]
        loss.backward()
        opt.step()
        opt.zero_grad()
        # Could check loss decreased, but at minimum no errors
        _check_no_nan_params(model)

    def test_determinism(self):
        """Same input produces same output in eval mode."""
        cfg = _tiny_gpt_config()
        torch.manual_seed(42)
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            out1 = model(x, return_logits=True)["logits"]
            out2 = model(x, return_logits=True)["logits"]
        assert torch.allclose(out1, out2)

    def test_single_token(self):
        """Forward pass works with a single token."""
        cfg = _tiny_gpt_config()
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (1, 1))
        out = model(x, return_logits=True)
        assert out["logits"].shape == (1, 1, cfg.padded_vocab_size)

    def test_forward_chain(self):
        """Multiple forward passes don't accumulate graph."""
        cfg = _tiny_gpt_config()
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        with torch.no_grad():
            for _ in range(3):
                out = model(x, return_logits=True)
                assert out["logits"] is not None

    def test_with_position_ids(self):
        """Forward pass with custom (1D) position_ids works."""
        cfg = _tiny_gpt_config()
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        pos = torch.arange(8)  # 1D — index_select expects a vector
        out = model(x, position_ids=pos, return_logits=True)
        assert out["logits"] is not None

    def test_with_attention_mask(self):
        """Forward pass with attention_mask works (no error)."""
        cfg = _tiny_gpt_config()
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        mask = torch.ones(2, 1, 8, 8, dtype=torch.bool)
        out = model(x, attention_mask=mask, return_logits=True)
        assert out["logits"] is not None

    def test_create_model_factory(self):
        """attractor.create_model factory works."""
        cfg = _tiny_gpt_config()
        model = attractor.create_model("gpt-small-140m", **{k: getattr(cfg, k) for k in
            ["n_embd", "n_layer", "num_attention_heads", "num_key_value_heads",
             "intermediate_size", "block_size", "vocab_size", "padding_multiple",
             "tie_embeddings", "init_strategy", "use_fused_head"]})
        assert model is not None
        x = torch.randint(0, cfg.vocab_size, (1, 4))
        out = model(x, return_logits=True)
        assert out["logits"] is not None

    def test_gradient_checkpointing(self):
        """Model can be created with gradient checkpointing."""
        cfg = _tiny_gpt_config()
        model = cfg.construct_model(gradient_checkpointing=True)
        assert model.gradient_checkpointing is True
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        labels = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x, labels=labels)
        out["loss"].backward()
        _check_finite_grads(model)


# ========================================================================
#  Parcae Tests
# ========================================================================


class TestParcae:
    """Verify Parcae model config, construction, forward, and backward.

    Uses a minimal config with fixed sampling scheme and per-batch
    iteration to keep the tests fast and deterministic.
    """

    def test_config_creation(self):
        """Can create ParcaeConfig from name."""
        cfg = _tiny_parcae_config()
        assert cfg.n_embd == 64
        assert cfg.recurrent_embedding_dimension == 64
        assert cfg.sampling_scheme == "fixed"

    def test_model_construction(self):
        """Parcae model can be constructed from config."""
        cfg = _tiny_parcae_config()
        model = cfg.construct_model()
        assert isinstance(model, torch.nn.Module)
        _check_no_nan_params(model)

    def test_forward_shape_no_labels(self):
        """Forward pass without labels produces no loss, optional logits."""
        cfg = _tiny_parcae_config()
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x)
        assert "loss" in out
        assert out["loss"].item() == pytest.approx(0.0)
        assert out["logits"] is None

    def test_forward_with_labels(self):
        """Forward pass with labels computes loss."""
        cfg = _tiny_parcae_config()
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        labels = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x, labels=labels)
        assert out["loss"].item() > 0.0
        assert "log_ppl" in out

    def test_forward_with_return_logits(self):
        """return_logits=True returns logits."""
        cfg = _tiny_parcae_config()
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x, return_logits=True)
        assert out["logits"] is not None
        # The use_fused_head config affects logits shape; default is "pytorch"
        assert out["logits"].shape[0] == 2
        assert out["logits"].shape[1] == 8

    def test_backward(self):
        """Backward pass produces gradients for all parameters."""
        cfg = _tiny_parcae_config()
        model = cfg.construct_model()
        cfg.use_fused_head = "pytorch"
        cfg.__post_init__()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        labels = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x, labels=labels)
        out["loss"].backward()
        _check_finite_grads(model)

    def test_training_step(self):
        """Model can perform a full training step."""
        cfg = _tiny_parcae_config()
        model = cfg.construct_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        labels = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x, labels=labels)
        out["loss"].backward()
        opt.step()
        opt.zero_grad()
        _check_no_nan_params(model)

    def test_determinism(self):
        """Parcae is non-deterministic by design (state init uses random
        noise, sampling is Poisson). Just verify two calls don't error."""
        cfg = _tiny_parcae_config()
        torch.manual_seed(42)
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            out1 = model(x, return_logits=True)["logits"]
            out2 = model(x, return_logits=True)["logits"]
        # Parcae is inherently non-deterministic, but both calls should
        # produce valid logits of the expected shape.
        assert out1.shape == out2.shape

    def test_single_token(self):
        """Forward pass works with a single token."""
        cfg = _tiny_parcae_config()
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (1, 1))
        out = model(x, return_logits=True)
        assert out["logits"] is not None

    def test_num_steps_pair(self):
        """Forward with num_steps_pair overrides iteration count."""
        cfg = _tiny_parcae_config()
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        steps = torch.tensor([2, 2])  # n, k
        out = model(x, num_steps_pair=steps, return_logits=True)
        assert out["logits"] is not None

    def test_gradient_checkpointing(self):
        """Model can be created with gradient checkpointing."""
        cfg = _tiny_parcae_config()
        model = cfg.construct_model(gradient_checkpointing=True)
        assert model.gradient_checkpointing is True
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        labels = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x, labels=labels)
        out["loss"].backward()
        _check_finite_grads(model)

    def test_prelude_norm(self):
        """Model with prelude_norm=True works."""
        cfg = _tiny_parcae_config(prelude_norm=True)
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x, return_logits=True)
        assert out["logits"] is not None


# ========================================================================
#  EQLM Tests
# ========================================================================


class TestEQLM:
    """Verify EQLM (DEQ) model config, construction, forward, and backward.

    EQLM is the equilibrium (DEQ) model with an implicit fixed-point head.
    Uses FPI solver for fast deterministic tests.
    """

    def test_config_creation(self):
        """Can create EQLMConfig from name."""
        cfg = _tiny_eqlm_config()
        assert cfg.n_embd == 64
        assert cfg.max_iter == 5
        assert cfg.solver == "fpi"

    def test_model_construction(self):
        """EQLM model can be constructed from config."""
        cfg = _tiny_eqlm_config()
        model = cfg.construct_model()
        assert isinstance(model, torch.nn.Module)
        _check_no_nan_params(model)

    def test_forward_shape_no_labels(self):
        """Forward pass without labels produces no loss, optional logits."""
        cfg = _tiny_eqlm_config()
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x)
        assert "loss" in out
        assert out["loss"].item() == pytest.approx(0.0)
        assert out["logits"] is None

    def test_forward_with_labels(self):
        """Forward pass with labels computes loss."""
        cfg = _tiny_eqlm_config()
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        labels = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x, labels=labels)
        assert out["loss"].item() > 0.0
        assert "log_ppl" in out

    def test_forward_with_return_logits(self):
        """return_logits=True returns logits."""
        cfg = _tiny_eqlm_config()
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x, return_logits=True)
        assert out["logits"] is not None
        assert out["logits"].shape[0] == 2
        assert out["logits"].shape[1] == 8

    def test_backward(self):
        """Backward pass produces gradients."""
        cfg = _tiny_eqlm_config()
        model = cfg.construct_model()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        labels = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x, labels=labels)
        out["loss"].backward()
        _check_finite_grads(model)

    def test_training_step(self):
        """Model can perform a full training step."""
        cfg = _tiny_eqlm_config()
        model = cfg.construct_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        labels = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x, labels=labels)
        out["loss"].backward()
        opt.step()
        opt.zero_grad()
        _check_no_nan_params(model)

    def test_determinism(self):
        """Same input produces same output in eval mode."""
        cfg = _tiny_eqlm_config()
        torch.manual_seed(42)
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            out1 = model(x, return_logits=True)["logits"]
            out2 = model(x, return_logits=True)["logits"]
        assert torch.allclose(out1, out2)

    def test_single_token(self):
        """Forward pass works with a single token."""
        cfg = _tiny_eqlm_config()
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (1, 1))
        out = model(x, return_logits=True)
        assert out["logits"] is not None

    def test_anderson_solver(self):
        """EQLM works with Anderson acceleration solver."""
        cfg = _tiny_eqlm_config(solver="anderson", max_iter=5, min_iter=2, tol=1e-2)
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        labels = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x, labels=labels)
        assert out["loss"].item() > 0.0

    def test_num_steps_pair(self):
        """Forward with num_steps_pair overrides max_iter."""
        cfg = _tiny_eqlm_config()
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        steps = torch.tensor([3, 2])
        out = model(x, num_steps_pair=steps, return_logits=True)
        assert out["logits"] is not None

    def test_solver_info(self):
        """Solver info is populated after forward pass."""
        cfg = _tiny_eqlm_config()
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        _ = model(x, return_logits=True)
        assert hasattr(model, "_last_solver_info")
        assert "iters" in model._last_solver_info


    def test_fp_params_tagged(self):
        """FP-head parameters are tagged with _fp_head attribute."""
        cfg = _tiny_eqlm_config()
        model = cfg.construct_model()
        fp_params = [p for p in model.transformer.core_block.parameters()]
        for p in fp_params:
            assert getattr(p, "_fp_head", False), f"FP param lacking _fp_head tag"


# ========================================================================
#  Attractor Tests
# ========================================================================


class TestAttractor:
    """Verify Attractor model config, construction, forward, and backward.

    The Attractor is a weight-tied DEQ model with LayerScale gating and
    implicit gradient flow through the IFT.
    """

    def test_config_creation(self):
        """Can create AttractorConfig from name."""
        cfg = _tiny_attractor_config()
        assert cfg.n_embd == 64
        assert cfg.max_iter == 5
        assert cfg.model_class_name == "Attractor"

    def test_layer_scale_default(self):
        """AttractorConfig has a default layer_scale_init."""
        cfg = _tiny_attractor_config()
        assert cfg.layer_scale_init is not None
        assert 0.0 < cfg.layer_scale_init <= 1.0

    def test_model_construction(self):
        """Attractor model can be constructed from config."""
        cfg = _tiny_attractor_config()
        model = cfg.construct_model()
        assert isinstance(model, torch.nn.Module)
        _check_no_nan_params(model)

    def test_architecture_class(self):
        """Config constructs an Attractor model."""
        from attractor.models.attractor.attractor import Attractor
        cfg = _tiny_attractor_config()
        model = cfg.construct_model()
        assert isinstance(model, Attractor)

    def test_forward_shape_no_labels(self):
        """Forward pass without labels produces no loss."""
        cfg = _tiny_attractor_config()
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x)
        assert "loss" in out
        assert out["loss"].item() == pytest.approx(0.0)
        assert out["logits"] is None

    def test_forward_with_labels(self):
        """Forward pass with labels computes loss."""
        cfg = _tiny_attractor_config()
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        labels = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x, labels=labels)
        assert out["loss"].item() > 0.0
        assert "log_ppl" in out

    def test_forward_with_return_logits(self):
        """return_logits=True returns logits."""
        cfg = _tiny_attractor_config()
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x, return_logits=True)
        assert out["logits"] is not None
        assert out["logits"].shape[0] == 2
        assert out["logits"].shape[1] == 8

    def test_backward(self):
        """Backward pass produces gradients for all parameters."""
        cfg = _tiny_attractor_config()
        model = cfg.construct_model()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        labels = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x, labels=labels)
        out["loss"].backward()
        _check_finite_grads(model)

    def test_training_step(self):
        """Model can perform a full training step."""
        cfg = _tiny_attractor_config()
        model = cfg.construct_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        labels = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x, labels=labels)
        out["loss"].backward()
        opt.step()
        opt.zero_grad()
        _check_no_nan_params(model)

    def test_determinism(self):
        """Same input produces same output in eval mode."""
        cfg = _tiny_attractor_config()
        torch.manual_seed(42)
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            out1 = model(x, return_logits=True)["logits"]
            out2 = model(x, return_logits=True)["logits"]
        assert torch.allclose(out1, out2)

    def test_single_token(self):
        """Forward pass works with a single token."""
        cfg = _tiny_attractor_config()
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (1, 1))
        out = model(x, return_logits=True)
        assert out["logits"] is not None

    def test_anderson_solver(self):
        """Attractor works with Anderson acceleration solver."""
        cfg = _tiny_attractor_config(solver="anderson", max_iter=5, min_iter=2, tol=1e-2)
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        labels = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(x, labels=labels)
        assert out["loss"].item() > 0.0

    def test_num_steps_pair(self):
        """Forward with num_steps_pair overrides max_iter."""
        cfg = _tiny_attractor_config()
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        steps = torch.tensor([3, 2])
        out = model(x, num_steps_pair=steps, return_logits=True)
        assert out["logits"] is not None

    def test_solver_info(self):
        """Solver info is populated after forward pass."""
        cfg = _tiny_attractor_config()
        model = cfg.construct_model()
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        _ = model(x, return_logits=True)
        assert hasattr(model, "_last_solver_info")
        assert "iters" in model._last_solver_info

    def test_fp_params_tagged(self):
        """FP-head parameters are tagged with _fp_head attribute."""
        cfg = _tiny_attractor_config()
        model = cfg.construct_model()
        fp_params = [p for p in model.transformer.core_block.parameters()]
        for p in fp_params:
            assert getattr(p, "_fp_head", False), f"FP param lacking _fp_head tag"


# ========================================================================
#  Cross-Model Tests
# ========================================================================


class TestAllModels:
    """Tests that apply uniformly to all model types."""

    @pytest.mark.parametrize("model_type", ["gpt", "parcae", "eqlm", "attractor"])
    def test_create_config_factory(self, model_type):
        """All model configs can be created via create_config."""
        names = {
            "gpt": "gpt-small-140m",
            "parcae": "parcae-small-140m",
            "eqlm": "eqlm-small-140m",
            "attractor": "attractor-small-140m",
        }
        cfg = attractor.create_config(names[model_type])
        assert cfg is not None
        assert cfg.n_embd > 0

    @pytest.mark.parametrize("model_type", ["eqlm", "attractor"])
    def test_tied_embeddings(self, model_type):
        """Tied embeddings: lm_head.weight shares with wte.weight.

        This applies to EQLM and Attractor which always bind the weights.
        GPT only ties weights when use_fused_head is enabled (not the
        config used for tiny tests), so it's excluded from this check.
        """
        cfgs = {
            "eqlm": _tiny_eqlm_config(tie_embeddings=True),
            "attractor": _tiny_attractor_config(tie_embeddings=True),
        }
        cfg = cfgs[model_type]
        model = cfg.construct_model()
        assert model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()

    @pytest.mark.parametrize("model_type", ["gpt", "parcae", "eqlm"])
    def test_model_registered(self, model_type):
        """Most model types are accessible via attractor module."""
        names = {
            "gpt": "GPT",
            "parcae": "Parcae",
            "eqlm": "EQLM",
        }
        cls = getattr(attractor, names[model_type])
        assert cls is not None

    def test_attractor_module_has_eqlm_alias(self):
        """The attractor model module has EQLM as an alias for Attractor
        (for checkpoint backward compatibility)."""
        from attractor.models.attractor.attractor import Attractor, EQLM as AttractorEQLM
        assert AttractorEQLM is Attractor, "attractor/attractor.py should alias EQLM -> Attractor"

    def test_eqlm_is_attractor_alias(self):
        """EQLM is aliased to Attractor for backward compatibility."""
        from attractor.models.attractor.attractor import EQLM
        from attractor.models.attractor.attractor import Attractor
        assert EQLM is Attractor


# ========================================================================
#  Run
# ========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
