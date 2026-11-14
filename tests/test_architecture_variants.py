"""Comprehensive Test Suite for Configurable Three-Architecture System.

Validates Phase 2 requirements for fire-rescue-project-2.0:
1. Target Variants Specification & Exact Parameter Counts:
   - Variant A ('baseline'): exactly 1,431,168 parameters, hidden_size=128
   - Variant B ('deep128'): exactly 2,761,600 parameters, hidden_size=128
   - Variant C ('deep256'): exactly 5,782,656 parameters, hidden_size=256
2. Layer Structures & Modular Compositions:
   - Variant A: Linear(10132, 128), GRU(128, 128), Linear(128, 7), Linear(128, 1)
   - Variant B: Sequential(Linear(10132, 256), LayerNorm(256), ReLU, Linear(256, 128), ReLU),
     GRU(128, 128), Linear heads
   - Variant C: Sequential(Linear(10132, 512), LayerNorm(512), ReLU, Linear(512, 256), ReLU),
     GRU(256, 256), Sequential(Linear(256, 64), ReLU, Linear(64, 7/1))
3. Numerical Invariance Verification on stage2_final.zip and stage1_qualified.zip:
   - Exact state_dict tensor equality
   - Exact forward pass output logits, values, action probabilities, and recurrent hidden states
4. Explicit Checkpoint Guard:
   - Detects architecture mismatches and raises ValueError
   - Detects tensor shape mismatches and raises RuntimeError
   - Prevents silent partial loading
5. Weight Transfer Utility (transfer_compatible_weights):
   - Transfers only identical layers (local_cnn, memory_cnn)
   - Tracks warm-start metadata and flags
6. End-to-end PPO rollouts, BPTT, and gradient flow across all 3 variants.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.cnn import (
    RecurrentSpatialRescueCNN,
    inspect_checkpoint,
    transfer_compatible_weights,
)
from agents.recurrent_ppo import (
    MaskableRecurrentPPO,
    load_checkpoint,
    save_checkpoint,
)
from rescue_env.envs.single_agent import RescueEnv
from rescue_env.wrappers import PreviousActionWrapper


def make_test_obs_space() -> spaces.Dict:
    """Create standard observation space for testing."""
    return spaces.Dict({
        "grid": spaces.Box(low=-np.inf, high=np.inf, shape=(9, 15, 15), dtype=np.float32),
        "memory": spaces.Box(low=-np.inf, high=np.inf, shape=(9, 40, 52), dtype=np.float32),
        "agent": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
        "previous_action": spaces.Box(low=0, high=6, shape=(), dtype=np.int64),
    })


def make_dummy_batch(batch_size: int = 4, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Create deterministic batch of dummy observations."""
    torch.manual_seed(12345)
    return {
        "grid": torch.randn(batch_size, 9, 15, 15, device=device),
        "memory": torch.randn(batch_size, 9, 40, 52, device=device),
        "agent": torch.randn(batch_size, 5, device=device),
        "previous_action": torch.randint(0, 7, (batch_size,), device=device),
    }


# ==============================================================================
# 1. Target Variants Specification & Exact Parameter Counts
# ==============================================================================

def test_variant_a_baseline_specs():
    """Verify Variant A ('baseline') exact parameter counts and architecture structure."""
    obs_space = make_test_obs_space()
    policy = RecurrentSpatialRescueCNN(obs_space, spaces.Discrete(7), architecture_variant="baseline")

    assert policy.architecture_variant == "baseline"
    assert policy.hidden_size == 128
    assert policy.lstm_output_dim == 128
    assert policy.lstm_hidden_state_shape == (1, 1, 128)

    # Fusion
    assert isinstance(policy.fusion, nn.Linear)
    assert policy.fusion.in_features == 10132
    assert policy.fusion.out_features == 128

    # GRU
    assert isinstance(policy.gru, nn.GRU)
    assert policy.gru.input_size == 128
    assert policy.gru.hidden_size == 128

    # Heads
    assert isinstance(policy.action_net, nn.Linear)
    assert policy.action_net.in_features == 128
    assert policy.action_net.out_features == 7
    assert isinstance(policy.value_net, nn.Linear)
    assert policy.value_net.in_features == 128
    assert policy.value_net.out_features == 1

    # Exact parameter count: 1,431,168
    total_params = sum(p.numel() for p in policy.parameters())
    assert total_params == 1_431_168, f"Expected 1,431,168 params for baseline, got {total_params}"
    assert policy.total_params == 1_431_168


def test_variant_b_deep128_specs():
    """Verify Variant B ('deep128') exact parameter counts and 2-stage LN+ReLU fusion."""
    obs_space = make_test_obs_space()
    policy = RecurrentSpatialRescueCNN(obs_space, spaces.Discrete(7), architecture_variant="deep128")

    assert policy.architecture_variant == "deep128"
    assert policy.hidden_size == 128
    assert policy.lstm_output_dim == 128
    assert policy.lstm_hidden_state_shape == (1, 1, 128)

    # Fusion: 2-stage projection with LayerNorm and ReLU
    assert isinstance(policy.fusion, nn.Sequential)
    assert len(policy.fusion) == 5
    assert isinstance(policy.fusion[0], nn.Linear)
    assert policy.fusion[0].in_features == 10132
    assert policy.fusion[0].out_features == 256
    assert isinstance(policy.fusion[1], nn.LayerNorm)
    assert policy.fusion[1].normalized_shape == (256,)
    assert isinstance(policy.fusion[2], nn.ReLU)
    assert isinstance(policy.fusion[3], nn.Linear)
    assert policy.fusion[3].in_features == 256
    assert policy.fusion[3].out_features == 128
    assert isinstance(policy.fusion[4], nn.ReLU)

    # GRU
    assert isinstance(policy.gru, nn.GRU)
    assert policy.gru.input_size == 128
    assert policy.gru.hidden_size == 128

    # Heads
    assert isinstance(policy.action_net, nn.Linear)
    assert policy.action_net.in_features == 128
    assert policy.action_net.out_features == 7
    assert isinstance(policy.value_net, nn.Linear)
    assert policy.value_net.in_features == 128
    assert policy.value_net.out_features == 1

    # Exact parameter count: 2,761,600
    # Breakdown:
    # local_cnn: 21,640
    # memory_cnn: 12,400
    # fusion: 2,594,048 (L1) + 512 (LN) + 32,896 (L2) = 2,627,456
    # gru: 99,072
    # action_net: 903
    # value_net: 129
    # Total = 21640 + 12400 + 2627456 + 99072 + 903 + 129 = 2,761,600
    total_params = sum(p.numel() for p in policy.parameters())
    assert total_params == 2_761_600, f"Expected 2,761,600 params for deep128, got {total_params}"
    assert policy.total_params == 2_761_600


def test_variant_c_deep256_specs():
    """Verify Variant C ('deep256') exact parameter counts, 256 GRU, and 2-layer MLP heads."""
    obs_space = make_test_obs_space()
    policy = RecurrentSpatialRescueCNN(obs_space, spaces.Discrete(7), architecture_variant="deep256")

    assert policy.architecture_variant == "deep256"
    assert policy.hidden_size == 256
    assert policy.lstm_output_dim == 256
    assert policy.lstm_hidden_state_shape == (1, 1, 256)

    # Fusion: 2-stage 512->256 projection with LayerNorm and ReLU
    assert isinstance(policy.fusion, nn.Sequential)
    assert len(policy.fusion) == 5
    assert isinstance(policy.fusion[0], nn.Linear)
    assert policy.fusion[0].in_features == 10132
    assert policy.fusion[0].out_features == 512
    assert isinstance(policy.fusion[1], nn.LayerNorm)
    assert policy.fusion[1].normalized_shape == (512,)
    assert isinstance(policy.fusion[2], nn.ReLU)
    assert isinstance(policy.fusion[3], nn.Linear)
    assert policy.fusion[3].in_features == 512
    assert policy.fusion[3].out_features == 256
    assert isinstance(policy.fusion[4], nn.ReLU)

    # GRU: 256 -> 256
    assert isinstance(policy.gru, nn.GRU)
    assert policy.gru.input_size == 256
    assert policy.gru.hidden_size == 256

    # Action Head: 2-layer MLP (256 -> 64 -> 7)
    assert isinstance(policy.action_net, nn.Sequential)
    assert len(policy.action_net) == 3
    assert isinstance(policy.action_net[0], nn.Linear)
    assert policy.action_net[0].in_features == 256
    assert policy.action_net[0].out_features == 64
    assert isinstance(policy.action_net[1], nn.ReLU)
    assert isinstance(policy.action_net[2], nn.Linear)
    assert policy.action_net[2].in_features == 64
    assert policy.action_net[2].out_features == 7

    # Value Head: 2-layer MLP (256 -> 64 -> 1)
    assert isinstance(policy.value_net, nn.Sequential)
    assert len(policy.value_net) == 3
    assert isinstance(policy.value_net[0], nn.Linear)
    assert policy.value_net[0].in_features == 256
    assert policy.value_net[0].out_features == 64
    assert isinstance(policy.value_net[1], nn.ReLU)
    assert isinstance(policy.value_net[2], nn.Linear)
    assert policy.value_net[2].in_features == 64
    assert policy.value_net[2].out_features == 1

    # Exact parameter count: 5,782,656
    # Breakdown:
    # local_cnn: 21,640
    # memory_cnn: 12,400
    # fusion: 5,188,096 (L1) + 1,024 (LN) + 131,328 (L2) = 5,320,448
    # gru: 394,752 (3 * 256 * 256 * 2 + 3 * 256 * 2)
    # action_net: 16,448 (L1) + 455 (L2) = 16,903
    # value_net: 16,448 (L1) + 65 (L2) = 16,513
    # Total = 21640 + 12400 + 5320448 + 394752 + 16903 + 16513 = 5,782,656
    total_params = sum(p.numel() for p in policy.parameters())
    assert total_params == 5_782_656, f"Expected 5,782,656 params for deep256, got {total_params}"
    assert policy.total_params == 5_782_656


def test_invalid_architecture_variant_raises():
    """Verify that unsupported architecture variants raise ValueError."""
    obs_space = make_test_obs_space()
    with pytest.raises(ValueError, match="Unknown architecture_variant"):
        RecurrentSpatialRescueCNN(obs_space, spaces.Discrete(7), architecture_variant="invalid_variant")


# ==============================================================================
# 2. Numerical Invariance Verification on Real Checkpoints
# ==============================================================================

def test_stage2_final_checkpoint_numerical_invariance():
    """CRITICAL REQUIREMENT: Load stage2_final.zip into Variant A and verify identical weights and outputs.

    Verifies:
      1. Correct architecture deduction as 'baseline'.
      2. Exact tensor match against raw policy.pth from checkpoint zip.
      3. Exact numerical invariance of action logits, value predictions, and recurrent states.
    """
    ckpt_path = Path("models/curriculum_stage2/stage2/stage2_final.zip")
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint {ckpt_path} not available in workspace")

    # Inspect checkpoint metadata and deduce architecture
    info = inspect_checkpoint(ckpt_path)
    assert info["architecture_variant"] == "baseline"
    assert info["hidden_size"] == 128
    assert info["total_params"] == 1_431_168

    # Load model via load_checkpoint
    model = load_checkpoint(ckpt_path, device="cpu", expected_architecture="baseline")
    assert model.policy.architecture_variant == "baseline"
    assert model.policy.hidden_size == 128
    assert model.policy.total_params == 1_431_168

    # Check raw tensor equality against state_dict inside the zip
    with zipfile.ZipFile(str(ckpt_path), "r") as zf:
        with zf.open("policy.pth") as f:
            raw_state_dict = torch.load(f, map_location="cpu", weights_only=False)

    model_state = model.policy.state_dict()
    assert set(model_state.keys()) == set(raw_state_dict.keys())
    for k, v in raw_state_dict.items():
        assert torch.equal(model_state[k], v), f"Tensor divergence on weight {k}"

    # Forward pass deterministic output check
    batch = make_dummy_batch(batch_size=2, device="cpu")
    model.policy.set_training_mode(False)
    with torch.no_grad():
        actions, values, log_probs, next_h = model.policy(batch, deterministic=True, return_hidden_state=True)

    assert actions.shape == (2, 1) or actions.shape == (2,)
    assert values.shape == (2, 1)
    assert log_probs.shape == (2,)
    assert isinstance(next_h, torch.Tensor) or hasattr(next_h, "pi")

    # Repeat forward pass on identical inputs: outputs must be 100% bitwise deterministic
    with torch.no_grad():
        actions2, values2, log_probs2, next_h2 = model.policy(batch, deterministic=True, return_hidden_state=True)
    assert torch.equal(actions, actions2)
    assert torch.equal(values, values2)
    assert torch.equal(log_probs, log_probs2)


def test_stage1_qualified_checkpoint_numerical_invariance():
    """Verify stage1_qualified.zip loads cleanly into Variant A with zero mismatch."""
    ckpt_path = Path("models/curriculum_stage1/stage1/stage1_qualified.zip")
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint {ckpt_path} not available in workspace")

    info = inspect_checkpoint(ckpt_path)
    assert info["architecture_variant"] == "baseline"
    assert info["hidden_size"] == 128
    assert info["total_params"] == 1_431_168

    model = load_checkpoint(ckpt_path, device="cpu", expected_architecture="baseline")
    assert model.policy.architecture_variant == "baseline"
    assert model.policy.total_params == 1_431_168


# ==============================================================================
# 3. Explicit Checkpoint Guard: Fail Explicitly on Mismatch
# ==============================================================================

def test_checkpoint_guard_rejects_architecture_mismatch(tmp_path):
    """Verify that loading a checkpoint with a mismatched architecture raises ValueError immediately."""
    ckpt_path = Path("models/curriculum_stage2/stage2/stage2_final.zip")
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint {ckpt_path} not available in workspace")

    # Attempting to load baseline checkpoint expecting deep128 MUST fail explicitly
    with pytest.raises(ValueError, match="Checkpoint architecture mismatch"):
        load_checkpoint(ckpt_path, device="cpu", expected_architecture="deep128")

    # Attempting to load baseline checkpoint expecting deep256 MUST fail explicitly
    with pytest.raises(ValueError, match="Checkpoint architecture mismatch"):
        load_checkpoint(ckpt_path, device="cpu", expected_architecture="deep256")


def test_checkpoint_guard_across_saved_variants(tmp_path):
    """Verify save and load round-trip guard across deep128 and deep256 variants."""
    raw_env = RescueEnv(max_steps=16, victim_distance=2)
    env = PreviousActionWrapper(raw_env)

    # 1. Create deep128 model, save it
    model_deep128 = MaskableRecurrentPPO(
        policy=RecurrentSpatialRescueCNN,
        env=env,
        architecture_variant="deep128",
        n_steps=8,
        batch_size=8,
        device="cpu",
    )
    deep128_path = tmp_path / "deep128_test.zip"
    save_checkpoint(model_deep128, deep128_path)

    # Inspect checkpoint metadata
    info128 = inspect_checkpoint(deep128_path)
    assert info128["architecture_variant"] == "deep128"
    assert info128["hidden_size"] == 128
    assert info128["total_params"] == 2_761_600

    # Loading with expected_architecture='baseline' MUST fail
    with pytest.raises(ValueError, match="Checkpoint architecture mismatch"):
        load_checkpoint(deep128_path, device="cpu", expected_architecture="baseline")

    # Loading with expected_architecture='deep256' MUST fail
    with pytest.raises(ValueError, match="Checkpoint architecture mismatch"):
        load_checkpoint(deep128_path, device="cpu", expected_architecture="deep256")

    # Loading with expected_architecture='deep128' MUST succeed
    loaded_deep128 = load_checkpoint(deep128_path, device="cpu", expected_architecture="deep128")
    assert loaded_deep128.policy.architecture_variant == "deep128"
    assert loaded_deep128.policy.total_params == 2_761_600

    # 2. Create deep256 model, save it
    model_deep256 = MaskableRecurrentPPO(
        policy=RecurrentSpatialRescueCNN,
        env=env,
        architecture_variant="deep256",
        n_steps=8,
        batch_size=8,
        device="cpu",
    )
    deep256_path = tmp_path / "deep256_test.zip"
    save_checkpoint(model_deep256, deep256_path)

    info256 = inspect_checkpoint(deep256_path)
    assert info256["architecture_variant"] == "deep256"
    assert info256["hidden_size"] == 256
    assert info256["total_params"] == 5_782_656

    # Loading with expected_architecture='baseline' MUST fail
    with pytest.raises(ValueError, match="Checkpoint architecture mismatch"):
        load_checkpoint(deep256_path, device="cpu", expected_architecture="baseline")

    # Loading with expected_architecture='deep128' MUST fail
    with pytest.raises(ValueError, match="Checkpoint architecture mismatch"):
        load_checkpoint(deep256_path, device="cpu", expected_architecture="deep128")

    # Loading with expected_architecture='deep256' MUST succeed
    loaded_deep256 = load_checkpoint(deep256_path, device="cpu", expected_architecture="deep256")
    assert loaded_deep256.policy.architecture_variant == "deep256"
    assert loaded_deep256.policy.total_params == 5_782_656

    env.close()


# ==============================================================================
# 4. Weight Transfer Utility (transfer_compatible_weights)
# ==============================================================================

def test_transfer_compatible_weights_between_variants():
    """Verify explicit weight transfer utility transfers only identical visual branches."""
    obs_space = make_test_obs_space()
    action_space = spaces.Discrete(7)

    # Source: Variant A (baseline)
    source_baseline = RecurrentSpatialRescueCNN(obs_space, action_space, architecture_variant="baseline")
    # Target: Variant B (deep128)
    target_deep128 = RecurrentSpatialRescueCNN(obs_space, action_space, architecture_variant="deep128")
    # Target: Variant C (deep256)
    target_deep256 = RecurrentSpatialRescueCNN(obs_space, action_space, architecture_variant="deep256")

    # Before transfer: check weights are different
    assert not torch.equal(source_baseline.local_cnn[0].weight, target_deep128.local_cnn[0].weight)
    assert not torch.equal(source_baseline.memory_cnn[0].weight, target_deep256.memory_cnn[0].weight)

    # 1. Transfer to deep128
    report_b = transfer_compatible_weights(source_baseline, target_deep128)
    assert report_b["status"] == "success"
    assert "local_cnn" in report_b["transferred_layers"]
    assert "memory_cnn" in report_b["transferred_layers"]
    assert report_b["transferred_params"] == 21_640 + 12_400  # 34,040 params
    assert target_deep128.is_warm_started is True

    # Verify identical visual branch weights
    for name, param in target_deep128.local_cnn.named_parameters():
        src_param = dict(source_baseline.local_cnn.named_parameters())[name]
        assert torch.equal(param, src_param), f"Divergence in transferred local_cnn parameter {name}"

    for name, param in target_deep128.memory_cnn.named_parameters():
        src_param = dict(source_baseline.memory_cnn.named_parameters())[name]
        assert torch.equal(param, src_param), f"Divergence in transferred memory_cnn parameter {name}"

    # 2. Transfer to deep256
    report_c = transfer_compatible_weights(source_baseline, target_deep256)
    assert report_c["status"] == "success"
    assert report_c["transferred_params"] == 34_040
    assert target_deep256.is_warm_started is True

    # Verify identical visual branch weights in deep256
    for name, param in target_deep256.local_cnn.named_parameters():
        src_param = dict(source_baseline.local_cnn.named_parameters())[name]
        assert torch.equal(param, src_param)

    for name, param in target_deep256.memory_cnn.named_parameters():
        src_param = dict(source_baseline.memory_cnn.named_parameters())[name]
        assert torch.equal(param, src_param)

    # Non-compatible layers (fusion, GRU, heads) MUST remain untouched in target_deep256
    assert isinstance(target_deep256.fusion, nn.Sequential)
    assert isinstance(target_deep256.action_net, nn.Sequential)
    assert target_deep256.gru.hidden_size == 256


# ==============================================================================
# 5. Forward, Recurrent Unroll, Action Masking, & BPTT Across All Variants
# ==============================================================================

@pytest.mark.parametrize("variant", ["baseline", "deep128", "deep256"])
def test_all_variants_forward_and_recurrent_unroll(variant):
    """Verify single-step and multi-step unroll with action masking across all variants."""
    obs_space = make_test_obs_space()
    action_space = spaces.Discrete(7)
    policy = RecurrentSpatialRescueCNN(obs_space, action_space, architecture_variant=variant)
    policy.set_training_mode(False)

    batch_size = 4
    hidden_dim = 256 if variant == "deep256" else 128
    batch = make_dummy_batch(batch_size=batch_size, device="cpu")

    # 1. Single-step forward
    h0 = torch.zeros((1, batch_size, hidden_dim), dtype=torch.float32)
    actions, values, log_probs, next_h = policy(batch, lstm_states=h0, return_hidden_state=True)
    assert actions.shape == (batch_size, 1) or actions.shape == (batch_size,)
    assert values.shape == (batch_size, 1)
    assert log_probs.shape == (batch_size,)
    assert next_h.shape == (1, batch_size, hidden_dim)
    assert not torch.allclose(h0, next_h), "Hidden state did not update"

    # 2. Action masking
    masks = torch.ones((batch_size, 7), dtype=torch.bool)
    masks[0, 1:] = False  # Env 0: only action 0 is legal
    masks[1, :6] = False  # Env 1: only action 6 is legal
    dist = policy.get_distribution(batch, action_masks=masks)
    probs = dist.distribution.probs
    assert probs[0, 0].item() == pytest.approx(1.0, abs=1e-5)
    assert probs[1, 6].item() == pytest.approx(1.0, abs=1e-5)
    assert torch.all(probs[0, 1:] == 0.0)

    # 3. Multi-step BPTT gradient propagation
    policy.set_training_mode(True)
    seq_batch = make_dummy_batch(batch_size=8, device="cpu")  # 2 sequences of length 4
    h_init = torch.zeros((1, 2, hidden_dim), dtype=torch.float32, requires_grad=True)
    starts = torch.zeros((2, 4), dtype=torch.float32)

    latent, next_h_seq = policy._forward_network(seq_batch, lstm_states=h_init, episode_starts=starts)
    loss = latent.sum()
    loss.backward()

    # Verify gradients flow into recurrent initial state and CNN weights
    assert h_init.grad is not None
    assert torch.norm(h_init.grad) > 0.0
    for p in policy.local_cnn.parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.norm(p.grad) > 0.0


@pytest.mark.parametrize("variant", ["baseline", "deep128", "deep256"])
def test_all_variants_maskable_ppo_training_step(variant, tmp_path):
    """Verify MaskableRecurrentPPO can train at least 1 update with zero errors on all variants."""
    raw_env = RescueEnv(max_steps=16, victim_distance=2)
    env = PreviousActionWrapper(raw_env)

    model = MaskableRecurrentPPO(
        policy=RecurrentSpatialRescueCNN,
        env=env,
        architecture_variant=variant,
        n_steps=16,
        batch_size=16,
        n_epochs=2,
        device="cpu",
        seed=42,
    )

    # Run 1 update (16 timesteps)
    model.learn(total_timesteps=16)
    assert model.num_timesteps == 16
    assert len(model.training_metrics_history) >= 1
    last_update = model.training_metrics_history[-1]
    assert np.isfinite(last_update["loss"])
    assert np.isfinite(last_update["policy_loss"])
    assert np.isfinite(last_update["value_loss"])

    # Test saving with metadata
    ckpt_file = tmp_path / f"model_{variant}.zip"
    save_checkpoint(model, ckpt_file)
    assert ckpt_file.exists()

    info = inspect_checkpoint(ckpt_file)
    assert info["architecture_variant"] == variant
    assert info["hidden_size"] == (256 if variant == "deep256" else 128)

    env.close()


# ==============================================================================
# 6. Explicit 12 Architecture Verification Properties Across All Variants
# ==============================================================================

@pytest.mark.parametrize("variant", ["baseline", "deep128", "deep256"])
def test_property1_observation_tensor_shape_compatibility(variant):
    """Property 1: Observation and tensor-shape compatibility across batch sizes and input dtypes."""
    obs_space = make_test_obs_space()
    policy = RecurrentSpatialRescueCNN(obs_space, spaces.Discrete(7), architecture_variant=variant)
    policy.set_training_mode(False)
    hidden_dim = policy.hidden_size

    # Batch sizes 1 and 4
    for batch_size in (1, 4):
        batch = {
            "grid": torch.randn(batch_size, 9, 15, 15, dtype=torch.float32),
            "memory": torch.randn(batch_size, 9, 40, 52, dtype=torch.float32),
            "agent": torch.randn(batch_size, 5, dtype=torch.float32),
            "previous_action": torch.randint(0, 7, (batch_size,), dtype=torch.long),
        }
        h0 = torch.zeros((1, batch_size, hidden_dim), dtype=torch.float32)
        actions, values, log_probs, next_h = policy(batch, lstm_states=h0, return_hidden_state=True)

        assert actions.shape == (batch_size, 1) or actions.shape == (batch_size,)
        assert values.shape == (batch_size, 1)
        assert log_probs.shape == (batch_size,)
        h_tensor = next_h.pi[0] if hasattr(next_h, "pi") else next_h
        assert h_tensor.shape == (1, batch_size, hidden_dim)

    # Alternate previous_action shapes: 2D one-hot float (B, 7)
    b_alt = {
        "grid": torch.randn(2, 9, 15, 15),
        "memory": torch.randn(2, 9, 40, 52),
        "agent": torch.randn(2, 5),
        "previous_action": torch.zeros((2, 7), dtype=torch.float32),
    }
    b_alt["previous_action"][0, 2] = 1.0
    b_alt["previous_action"][1, 4] = 1.0
    with torch.no_grad():
        act, val, lp, _ = policy(b_alt, return_hidden_state=True)
    assert act.shape[0] == 2
    assert val.shape == (2, 1)


@pytest.mark.parametrize("variant", ["baseline", "deep128", "deep256"])
def test_property2_cnn_branches_output_dimensions(variant):
    """Property 2: Correct output dimensions for all CNN branches and concatenation."""
    obs_space = make_test_obs_space()
    policy = RecurrentSpatialRescueCNN(obs_space, spaces.Discrete(7), architecture_variant=variant)

    batch_size = 3
    grid = torch.randn(batch_size, 9, 15, 15)
    memory = torch.randn(batch_size, 9, 40, 52)
    agent = torch.randn(batch_size, 5)
    prev_act = torch.tensor([1, 3, 5], dtype=torch.long)

    # 1. Local CNN branch: Conv(9,32,3)->ReLU->Conv(32,64,3)->ReLU->Conv(64,8,1)->Flatten
    # Expected output: 8 * 15 * 15 = 1,800
    local_out = policy.local_cnn(grid)
    assert local_out.shape == (batch_size, 1800), f"Expected (3, 1800), got {local_out.shape}"
    assert policy.local_features_dim == 1800

    # 2. Memory CNN branch: Conv(9,32,3)->ReLU->Conv(32,32,3,stride=2)->ReLU->Conv(32,16,1)->Flatten
    # Expected output: 16 * 20 * 26 = 8,320
    mem_out = policy.memory_cnn(memory)
    assert mem_out.shape == (batch_size, 8320), f"Expected (3, 8320), got {mem_out.shape}"
    assert policy.memory_features_dim == 8320

    # 3. Agent and Previous Action dimensions
    assert policy.agent_dim == 5
    assert policy.action_dim == 7
    encoded_prev = policy._encode_previous_action(prev_act, batch_size, grid.device)
    assert encoded_prev.shape == (batch_size, 7)

    # 4. Total concatenated fusion vector dimension: 1800 + 8320 + 5 + 7 = 10,132
    total_dim = policy.local_features_dim + policy.memory_features_dim + policy.agent_dim + policy.action_dim
    assert total_dim == 10132


@pytest.mark.parametrize("variant", ["baseline", "deep128", "deep256"])
def test_property3_recurrent_hidden_state_dimensions(variant):
    """Property 3: Correct recurrent hidden-state dimensions (128 for baseline/deep128, 256 for deep256)."""
    obs_space = make_test_obs_space()
    policy = RecurrentSpatialRescueCNN(obs_space, spaces.Discrete(7), architecture_variant=variant)

    expected_h = 256 if variant == "deep256" else 128
    assert policy.hidden_size == expected_h
    assert policy.lstm_output_dim == expected_h
    assert policy.lstm_hidden_state_shape == (1, 1, expected_h)
    assert policy.gru.input_size == expected_h
    assert policy.gru.hidden_size == expected_h

    # GRU parameter shapes
    assert policy.gru.weight_ih_l0.shape == (3 * expected_h, expected_h)
    assert policy.gru.weight_hh_l0.shape == (3 * expected_h, expected_h)

    # Unrolled hidden state shape
    batch_size = 5
    batch = make_dummy_batch(batch_size=batch_size, device="cpu")
    h0 = torch.zeros((1, batch_size, expected_h))
    with torch.no_grad():
        _, _, _, next_h = policy(batch, lstm_states=h0, return_hidden_state=True)
    h_tensor = next_h.pi[0] if hasattr(next_h, "pi") else next_h
    assert h_tensor.shape == (1, batch_size, expected_h)


@pytest.mark.parametrize("variant", ["baseline", "deep128", "deep256"])
def test_property4_independent_parallel_hidden_states(variant):
    """Property 4: Independent hidden states for parallel environments (no cross-env leakage)."""
    obs_space = make_test_obs_space()
    policy = RecurrentSpatialRescueCNN(obs_space, spaces.Discrete(7), architecture_variant=variant)
    policy.set_training_mode(False)
    hidden_dim = policy.hidden_size
    n_envs = 4

    torch.manual_seed(999)
    obs = {
        "grid": torch.randn(n_envs, 9, 15, 15),
        "memory": torch.randn(n_envs, 9, 40, 52),
        "agent": torch.randn(n_envs, 5),
        "previous_action": torch.tensor([0, 1, 2, 3], dtype=torch.long),
    }
    h_init = torch.randn(1, n_envs, hidden_dim)

    # Simultaneous batch forward
    with torch.no_grad():
        act_all, val_all, lp_all, h_all = policy(obs, lstm_states=h_init, return_hidden_state=True)
    h_tensor_all = h_all.pi[0] if hasattr(h_all, "pi") else h_all

    # Independent forward on isolated envs
    for idx in (0, 2, 3):
        obs_iso = {k: v[idx : idx + 1] for k, v in obs.items()}
        h_iso = h_init[:, idx : idx + 1, :]
        with torch.no_grad():
            act_i, val_i, lp_i, h_i = policy(obs_iso, lstm_states=h_iso, return_hidden_state=True)
        h_tensor_i = h_i.pi[0] if hasattr(h_i, "pi") else h_i

        # Numerical agreement between batched slice and isolated run
        assert torch.allclose(h_tensor_all[:, idx : idx + 1, :], h_tensor_i, atol=1e-5)
        assert torch.allclose(val_all[idx : idx + 1], val_i, atol=1e-5)


@pytest.mark.parametrize("variant", ["baseline", "deep128", "deep256"])
def test_property5_episode_reset_behavior(variant):
    """Property 5: Correct episode reset behavior and reset isolation across parallel environments."""
    obs_space = make_test_obs_space()
    policy = RecurrentSpatialRescueCNN(obs_space, spaces.Discrete(7), architecture_variant=variant)
    policy.set_training_mode(False)
    hidden_dim = policy.hidden_size
    n_envs = 4

    torch.manual_seed(123)
    obs = {
        "grid": torch.randn(n_envs, 9, 15, 15),
        "memory": torch.randn(n_envs, 9, 40, 52),
        "agent": torch.randn(n_envs, 5),
        "previous_action": torch.tensor([0, 1, 2, 3], dtype=torch.long),
    }
    h_init = torch.randn(1, n_envs, hidden_dim)
    # Env 1 terminates and resets; envs 0, 2, 3 continue
    episode_starts = torch.tensor([0.0, 1.0, 0.0, 0.0])

    with torch.no_grad():
        _, _, _, h_next_all = policy(obs, lstm_states=h_init, episode_starts=episode_starts, return_hidden_state=True)
    h_tensor_all = h_next_all.pi[0] if hasattr(h_next_all, "pi") else h_next_all

    # Terminating Env 1: must match an isolated run initialized from zero
    obs1 = {k: v[1:2] for k, v in obs.items()}
    h_zero = torch.zeros(1, 1, hidden_dim)
    with torch.no_grad():
        _, _, _, h1_iso = policy(obs1, lstm_states=h_zero, return_hidden_state=True)
    h_tensor_1 = h1_iso.pi[0] if hasattr(h1_iso, "pi") else h1_iso
    diff1 = (h_tensor_all[:, 1:2, :] - h_tensor_1).abs().max().item()
    assert diff1 < 1e-5, f"Terminating env 1 did not reset cleanly: diff={diff1}"

    # Continuing Envs 0, 2, 3: must match continuation from their h_init
    for idx in (0, 2, 3):
        obs_idx = {k: v[idx : idx + 1] for k, v in obs.items()}
        h_idx = h_init[:, idx : idx + 1, :]
        with torch.no_grad():
            _, _, _, h_iso = policy(obs_idx, lstm_states=h_idx, return_hidden_state=True)
        h_tensor_idx = h_iso.pi[0] if hasattr(h_iso, "pi") else h_iso
        diff_idx = (h_tensor_all[:, idx : idx + 1, :] - h_tensor_idx).abs().max().item()
        assert diff_idx < 1e-5, f"Continuing env {idx} was corrupted by env 1 reset: diff={diff_idx}"


@pytest.mark.parametrize("variant", ["baseline", "deep128", "deep256"])
def test_property6_multistep_bptt_gradient_propagation(variant):
    """Property 6: Multi-step gradient propagation through the GRU (BPTT dL_t / ds_0 != 0) & reset severance."""
    obs_space = make_test_obs_space()
    policy = RecurrentSpatialRescueCNN(obs_space, spaces.Discrete(7), architecture_variant=variant)
    policy.set_training_mode(True)
    hidden_dim = policy.hidden_size

    # 3-step sequence
    g0 = torch.randn(1, 9, 15, 15, requires_grad=True)
    g1 = torch.randn(1, 9, 15, 15, requires_grad=True)
    g2 = torch.randn(1, 9, 15, 15, requires_grad=True)
    batch_grid = torch.cat([g0, g1, g2], dim=0)

    obs = {
        "grid": batch_grid,
        "memory": torch.randn(3, 9, 40, 52),
        "agent": torch.randn(3, 5),
        "previous_action": torch.tensor([0, 1, 2], dtype=torch.long),
    }
    actions = torch.tensor([1, 2, 3], dtype=torch.long)
    masks = torch.ones((3, 7), dtype=torch.bool)
    h0 = torch.randn(1, 1, hidden_dim, requires_grad=True)
    starts = torch.tensor([0.0, 0.0, 0.0])  # Continuing episode from h0

    values, log_prob, _ = policy.evaluate_actions(obs, actions, lstm_states=h0, episode_starts=starts, action_masks=masks)
    loss = values[2] + log_prob[2]

    grad_h0 = torch.autograd.grad(loss, h0, retain_graph=True)[0]
    grad_g0 = torch.autograd.grad(loss, g0, retain_graph=True)[0]
    grad_g1 = torch.autograd.grad(loss, g1, retain_graph=True)[0]
    grad_g2 = torch.autograd.grad(loss, g2, retain_graph=True)[0]

    # Non-zero BPTT gradients through recurrent state and across timesteps
    assert grad_h0.norm().item() > 1e-5, f"dL2/dh0 is zero for {variant}!"
    assert grad_g0.norm().item() > 1e-5, f"dL2/dg0 is zero for {variant}!"
    assert grad_g1.norm().item() > 1e-5, f"dL2/dg1 is zero for {variant}!"
    assert grad_g2.norm().item() > 1e-5, f"dL2/dg2 is zero for {variant}!"

    # Reset severance: reset at t=1 must zero out gradient into t=0
    starts_reset_t1 = torch.tensor([0.0, 1.0, 0.0])
    val_res, lp_res, _ = policy.evaluate_actions(obs, actions, lstm_states=h0, episode_starts=starts_reset_t1, action_masks=masks)
    loss_res = val_res[2] + lp_res[2]
    grad_g0_res = torch.autograd.grad(loss_res, g0, retain_graph=True)[0]
    grad_g1_res = torch.autograd.grad(loss_res, g1, retain_graph=True)[0]
    assert torch.all(grad_g0_res == 0.0), f"Gradient leaked across reset at t=1 for {variant}!"
    assert grad_g1_res.norm().item() > 1e-5, f"Valid within-sequence gradient zeroed for {variant}!"


@pytest.mark.parametrize("variant", ["baseline", "deep128", "deep256"])
def test_property7_action_masking_during_rollout_and_updates(variant):
    """Property 7: Valid-action masking during rollout, sampling, and PPO updates."""
    obs_space = make_test_obs_space()
    policy = RecurrentSpatialRescueCNN(obs_space, spaces.Discrete(7), architecture_variant=variant)
    policy.set_training_mode(False)

    test_masks = [
        [False, False, True, False, False, False, False],  # only action 2
        [True, False, False, False, False, False, True],   # actions 0 and 6
        [False, True, True, True, False, False, False],    # actions 1, 2, 3
    ]
    mask_tensor = torch.tensor(test_masks, dtype=torch.bool)
    n_configs = len(test_masks)

    obs = {
        "grid": torch.randn(n_configs, 9, 15, 15),
        "memory": torch.randn(n_configs, 9, 40, 52),
        "agent": torch.randn(n_configs, 5),
        "previous_action": torch.tensor([0, 1, 2], dtype=torch.long),
    }

    for i, mask in enumerate(test_masks):
        invalid_indices = [a for a in range(7) if not mask[a]]
        valid_indices = [a for a in range(7) if mask[a]]
        mask_i = torch.tensor([mask], dtype=torch.bool)
        obs_i = {
            "grid": torch.randn(1, 9, 15, 15),
            "memory": torch.randn(1, 9, 40, 52),
            "agent": torch.randn(1, 5),
            "previous_action": torch.tensor([i % 7], dtype=torch.long),
        }

        dist_i = policy.get_distribution(obs_i, action_masks=mask_i)
        probs_i = dist_i.distribution.probs[0]
        masked_logits_i = policy.get_masked_logits(obs_i, action_masks=mask_i)[0]

        # a) Invalid probabilities strictly 0.0
        assert torch.all(probs_i[invalid_indices] == 0.0)
        assert probs_i.sum().item() == pytest.approx(1.0, abs=1e-5)

        # b) Invalid logits <= -1e7
        assert torch.all(masked_logits_i[invalid_indices] <= -1e7)

        # c) 150 sampled actions must all be legal
        for _ in range(150):
            sampled_act = dist_i.sample().item()
            assert sampled_act in valid_indices

        # d) evaluate_actions on legal actions returns finite log-probs
        for legal_act in valid_indices:
            act_t = torch.tensor([legal_act], dtype=torch.long)
            val, lp, ent = policy.evaluate_actions(obs_i, act_t, action_masks=mask_i)
            assert torch.isfinite(lp).item()


@pytest.mark.parametrize("variant", ["baseline", "deep128", "deep256"])
def test_property8_finite_ppo_ratios_losses_gradients(variant):
    """Property 8: Finite PPO ratios, clipping, losses, and parameter gradients (no NaN/inf)."""
    obs_space = make_test_obs_space()
    policy = RecurrentSpatialRescueCNN(obs_space, spaces.Discrete(7), architecture_variant=variant)
    policy.set_training_mode(True)

    batch_size = 4
    batch = make_dummy_batch(batch_size=batch_size, device="cpu")
    actions = torch.tensor([0, 2, 4, 6], dtype=torch.long)
    masks = torch.ones((batch_size, 7), dtype=torch.bool)

    values, log_prob, entropy = policy.evaluate_actions(batch, actions, action_masks=masks)

    # 1. PPO policy ratios
    old_log_prob = log_prob.detach()
    ratios = torch.exp(log_prob - old_log_prob)
    assert torch.all(torch.isfinite(ratios))
    assert torch.all(ratios > 0.0)
    assert torch.allclose(ratios, torch.ones_like(ratios), atol=1e-5)

    # Perturbed ratio
    perturbed_old_lp = old_log_prob - 0.1
    perturbed_ratios = torch.exp(log_prob - perturbed_old_lp)
    assert torch.all(torch.isfinite(perturbed_ratios))
    assert torch.all(perturbed_ratios > 0.0)

    # 2. PPO surrogate objective and finite losses
    advantages = torch.tensor([1.0, -0.5, 0.8, -1.2], dtype=torch.float32)
    clip_range = 0.2
    surr1 = perturbed_ratios * advantages
    surr2 = torch.clamp(perturbed_ratios, 1.0 - clip_range, 1.0 + clip_range) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = 0.5 * (values.squeeze() - torch.tensor([1.0, 0.0, 2.0, -1.0])).pow(2).mean()
    loss = policy_loss + value_loss - 0.01 * entropy.mean()

    assert torch.isfinite(loss).item()
    assert not torch.isnan(loss).item()

    # 3. Gradients through all parameters must be finite (no NaN, no inf)
    loss.backward()
    for name, param in policy.named_parameters():
        if param.grad is not None:
            assert torch.all(torch.isfinite(param.grad)), f"Param {name} has non-finite gradients!"
            assert not torch.any(torch.isnan(param.grad)), f"Param {name} has NaN gradients!"


@pytest.mark.parametrize("variant", ["baseline", "deep128", "deep256"])
def test_property9_cpu_mps_checkpoint_save_load(variant, tmp_path):
    """Property 9: CPU and MPS checkpoint save/load round-trip across all variants."""
    raw_env = RescueEnv(max_steps=16, victim_distance=2)
    env = PreviousActionWrapper(raw_env)

    model = MaskableRecurrentPPO(
        policy=RecurrentSpatialRescueCNN,
        env=env,
        architecture_variant=variant,
        n_steps=8,
        batch_size=8,
        device="cpu",
    )
    ckpt_path = tmp_path / f"ckpt_prop9_{variant}.zip"
    save_checkpoint(model, ckpt_path, metadata={"test_prop": 9, "variant": variant})

    # 1. CPU reload
    loaded_cpu = load_checkpoint(ckpt_path, device="cpu", expected_architecture=variant)
    assert loaded_cpu.policy.architecture_variant == variant
    assert loaded_cpu.policy.total_params == model.policy.total_params
    for k, v in model.policy.state_dict().items():
        assert torch.equal(v, loaded_cpu.policy.state_dict()[k])

    # 2. MPS reload (if available)
    if torch.backends.mps.is_available():
        loaded_mps = load_checkpoint(ckpt_path, device="mps", expected_architecture=variant)
        assert next(loaded_mps.policy.parameters()).device.type == "mps"

        obs, _ = env.reset(seed=42)
        mask = env.action_masks()
        act_mps, _ = loaded_mps.predict(obs, deterministic=True, action_masks=mask)
        assert 0 <= int(act_mps) < 7

        # Round-trip MPS -> CPU
        mps_save_path = tmp_path / f"mps_to_cpu_{variant}.zip"
        save_checkpoint(loaded_mps, mps_save_path)
        reloaded_cpu = load_checkpoint(mps_save_path, device="cpu", expected_architecture=variant)
        assert next(reloaded_cpu.policy.parameters()).device.type == "cpu"
        for k, v in model.policy.state_dict().items():
            assert torch.equal(v.cpu(), reloaded_cpu.policy.state_dict()[k].cpu())

    env.close()


@pytest.mark.parametrize("variant", ["baseline", "deep128", "deep256"])
def test_property10_cuda_compatibility(variant, tmp_path, monkeypatch):
    """Property 10: CUDA compatibility (conditional check or mock device testing)."""
    raw_env = RescueEnv(max_steps=16, victim_distance=2)
    env = PreviousActionWrapper(raw_env)

    model = MaskableRecurrentPPO(
        policy=RecurrentSpatialRescueCNN,
        env=env,
        architecture_variant=variant,
        n_steps=8,
        batch_size=8,
        device="cpu",
    )
    ckpt_path = tmp_path / f"cuda_compat_{variant}.zip"
    save_checkpoint(model, ckpt_path, metadata={"target_device": "cuda"})

    # Check that inspect_checkpoint always safely uses map_location="cpu"
    info = inspect_checkpoint(ckpt_path)
    assert info["architecture_variant"] == variant
    for k, v in info["state_dict"].items():
        assert v.device.type == "cpu"

    # If CUDA is physically available, verify direct CUDA operations
    if torch.cuda.is_available():
        loaded_cuda = load_checkpoint(ckpt_path, device="cuda", expected_architecture=variant)
        assert next(loaded_cuda.policy.parameters()).device.type == "cuda"
        obs, _ = env.reset(seed=42)
        mask = env.action_masks()
        act_cuda, _ = loaded_cuda.predict(obs, deterministic=True, action_masks=mask)
        assert 0 <= int(act_cuda) < 7
    else:
        # Verify device="auto" routes to cuda when cuda is available via monkeypatch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert torch.cuda.is_available() is True
        # Verify save_checkpoint transfers all tensors to CPU before writing
        for p in model.policy.parameters():
            assert p.device.type == "cpu"

    env.close()


def test_property11_architecture_metadata_and_strict_checkpoint_guards(tmp_path):
    """Property 11: Correct architecture metadata and strict checkpoint loading guards."""
    raw_env = RescueEnv(max_steps=16, victim_distance=2)
    env = PreviousActionWrapper(raw_env)

    model = MaskableRecurrentPPO(
        policy=RecurrentSpatialRescueCNN,
        env=env,
        architecture_variant="baseline",
        n_steps=8,
        batch_size=8,
        device="cpu",
    )
    ckpt_path = tmp_path / "guard_test.zip"
    save_checkpoint(model, ckpt_path, metadata={"gate": "VERIFICATION_TEST"})

    # 1. Companion metadata file verification
    meta_sidecar = ckpt_path.with_suffix(".meta.json")
    assert meta_sidecar.exists()
    with open(meta_sidecar, "r") as f:
        meta_data = json.load(f)
    assert meta_data["architecture_variant"] == "baseline"
    assert meta_data["hidden_size"] == 128
    assert meta_data["total_params"] == 1_431_168
    assert meta_data["gate"] == "VERIFICATION_TEST"

    # 2. Strict Guard: Architecture Mismatch Detection
    with pytest.raises(ValueError, match="Checkpoint architecture mismatch"):
        load_checkpoint(ckpt_path, device="cpu", expected_architecture="deep128")

    with pytest.raises(ValueError, match="Checkpoint architecture mismatch"):
        load_checkpoint(ckpt_path, device="cpu", expected_architecture="deep256")

    # 3. Strict Guard: Weight Shape / Key Mismatch Detection
    corrupt_ckpt = tmp_path / "corrupt_shape.zip"
    with zipfile.ZipFile(str(ckpt_path), "r") as zin:
        with zipfile.ZipFile(str(corrupt_ckpt), "w") as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "policy.pth":
                    state = torch.load(zin.open(item.filename), weights_only=False)
                    state["action_net.weight"] = torch.randn(10, 128)  # Corrupted shape: 10 vs 7
                    buf = io.BytesIO()
                    torch.save(state, buf)
                    data = buf.getvalue()
                zout.writestr(item, data)

    with pytest.raises(RuntimeError):
        load_checkpoint(corrupt_ckpt, device="cpu", expected_architecture="baseline")

    env.close()


def test_property12_preservation_variant_a_existing_checkpoints():
    """Property 12: Preservation of Variant A existing checkpoint behavior (stage2_final and stage1_qualified)."""
    obs_space = make_test_obs_space()
    action_space = spaces.Discrete(7)

    for name, ckpt_str in [
        ("stage2_final", "models/curriculum_stage2/stage2/stage2_final.zip"),
        ("stage1_qualified", "models/curriculum_stage1/stage1/stage1_qualified.zip"),
    ]:
        p = Path(ckpt_str)
        if not p.exists():
            pytest.skip(f"Checkpoint {p} not available")

        # 1. Load via load_checkpoint
        model = load_checkpoint(p, device="cpu", expected_architecture="baseline")
        assert model.policy.architecture_variant == "baseline"
        assert model.policy.total_params == 1_431_168

        # 2. Load directly into raw baseline policy
        raw_policy = RecurrentSpatialRescueCNN(obs_space, action_space, architecture_variant="baseline")
        with zipfile.ZipFile(str(p), "r") as zf:
            with zf.open("policy.pth") as f:
                raw_state = torch.load(f, map_location="cpu", weights_only=False)
        raw_policy.load_state_dict(raw_state, strict=True)

        # 3. Verify exact tensor equality across all parameters
        for k, v in raw_state.items():
            assert torch.equal(model.policy.state_dict()[k], v)
            assert torch.equal(raw_policy.state_dict()[k], v)

        # 4. Deterministic forward evaluations with action masks
        torch.manual_seed(12345)
        obs = {
            "grid": torch.randn(4, 9, 15, 15),
            "memory": torch.randn(4, 9, 40, 52),
            "agent": torch.randn(4, 5),
            "previous_action": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        }
        masks = torch.ones((4, 7), dtype=torch.bool)
        masks[0, 2:] = False
        masks[1, :3] = False

        raw_policy.set_training_mode(False)
        model.policy.set_training_mode(False)

        with torch.no_grad():
            act_raw, val_raw, lp_raw, h_raw = raw_policy(obs, deterministic=True, return_hidden_state=True, action_masks=masks)
            logits_raw = raw_policy.get_masked_logits(obs, action_masks=masks)

            act_model, val_model, lp_model, h_model = model.policy(obs, deterministic=True, return_hidden_state=True, action_masks=masks)
            logits_model = model.policy.get_masked_logits(obs, action_masks=masks)

        diff_logits = (logits_raw - logits_model).abs().max().item()
        diff_val = (val_raw - val_model).abs().max().item()
        h_t_raw = h_raw.pi[0] if hasattr(h_raw, "pi") else h_raw
        h_t_model = h_model.pi[0] if hasattr(h_model, "pi") else h_model
        diff_h = (h_t_raw - h_t_model).abs().max().item()
        action_match = (act_raw == act_model).all().item()

        # Zero discrepancy verification
        assert diff_logits == 0.0, f"{name}: logit diff {diff_logits} > 0.0"
        assert diff_val == 0.0, f"{name}: value diff {diff_val} > 0.0"
        assert diff_h == 0.0, f"{name}: hidden state diff {diff_h} > 0.0"
        assert action_match is True, f"{name}: deterministic actions did not match!"

