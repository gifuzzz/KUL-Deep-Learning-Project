"""Unit and regression tests for Phase 11 Architecture Ablation:
- Variant A (Current): Local CNN (9, 15, 15) + Memory CNN (9, 40, 52) -> Fusion (10132->128) -> GRU128
- Variant B (Memory Only): Memory CNN (9, 40, 52) -> Fusion (8332->128) -> GRU128
- Variant C (Memory Crop): Memory CNN (9, 40, 52) + Memory Crop (9, 15, 15) -> Fusion (10132->128) -> GRU128
"""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from gymnasium import spaces

from agents.cnn import (
    RecurrentSpatialRescueCNN,
    inspect_checkpoint,
    transfer_phase11_weights,
)
from agents.recurrent_ppo import MaskableRecurrentPPO, load_checkpoint, save_checkpoint
from rescue_env.envs.single_agent import RescueEnv
from rescue_env.wrappers import PreviousActionWrapper


@pytest.fixture
def obs_act_spaces():
    obs_space = spaces.Dict({
        "grid": spaces.Box(0.0, 1.0, shape=(9, 15, 15), dtype=np.float32),
        "memory": spaces.Box(0.0, 1.0, shape=(9, 40, 52), dtype=np.float32),
        "agent": spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32),
        "previous_action": spaces.Discrete(7),
    })
    act_space = spaces.Discrete(7)
    return obs_space, act_space


@pytest.fixture
def sample_batch():
    batch_size = 4
    obs = {
        "grid": torch.rand(batch_size, 9, 15, 15),
        "memory": torch.rand(batch_size, 9, 40, 52),
        "agent": torch.rand(batch_size, 5),
        "previous_action": torch.randint(0, 7, (batch_size,)),
    }
    # Set agent position marker in memory (channel 8)
    obs["memory"][:, 8] = 0.0
    for b in range(batch_size):
        r, c = np.random.randint(5, 35), np.random.randint(5, 45)
        obs["memory"][b, 8, r, c] = 1.0
    masks = torch.ones(batch_size, 7, dtype=torch.bool)
    masks[:, 6] = False  # Action 6 masked out
    return obs, masks


def test_parameter_counts(obs_act_spaces):
    obs_space, act_space = obs_act_spaces
    alias_prefixes = ("lstm_actor.", "policy_head.", "value_head.")

    # Variant A (baseline / current)
    model_a = RecurrentSpatialRescueCNN(obs_space, act_space, architecture_variant="current")
    params_a = sum(p.numel() for k, p in model_a.state_dict().items() if not any(k.startswith(pfx) for pfx in alias_prefixes))
    assert params_a == 1_431_168, f"Expected 1,431,168 for Variant A, got {params_a}"
    assert model_a.local_cnn is not None
    assert model_a.local_features_dim == 1800

    # Variant B (memory_only)
    model_b = RecurrentSpatialRescueCNN(obs_space, act_space, architecture_variant="memory_only")
    params_b = sum(p.numel() for k, p in model_b.state_dict().items() if not any(k.startswith(pfx) for pfx in alias_prefixes))
    assert params_b == 1_179_128, f"Expected 1,179,128 for Variant B, got {params_b}"
    assert model_b.local_cnn is None
    assert model_b.local_features_dim == 0
    assert model_b.fusion.in_features == 8332

    # Variant C (memory_crop)
    model_c = RecurrentSpatialRescueCNN(obs_space, act_space, architecture_variant="memory_crop")
    params_c = sum(p.numel() for k, p in model_c.state_dict().items() if not any(k.startswith(pfx) for pfx in alias_prefixes))
    assert params_c == 1_431_168, f"Expected 1,431,168 for Variant C, got {params_c}"
    assert model_c.local_cnn is not None
    assert model_c.local_features_dim == 1800
    assert model_c.fusion.in_features == 10132


def test_forward_pass_and_masking(obs_act_spaces, sample_batch):
    obs_space, act_space = obs_act_spaces
    obs, masks = sample_batch
    batch_size = obs["memory"].shape[0]

    for variant in ["current", "memory_only", "memory_crop"]:
        model = RecurrentSpatialRescueCNN(obs_space, act_space, architecture_variant=variant)
        model.eval()

        with torch.no_grad():
            actions, values, log_prob = model.forward(
                obs, deterministic=True, action_masks=masks
            )
            assert actions.shape == (batch_size,)
            assert values.shape == (batch_size, 1)
            assert log_prob.shape == (batch_size,)
            # Assert action 6 (water hose) was never selected
            assert (actions != 6).all(), f"Variant {variant} selected masked action 6!"

            logits, vals = model.get_action_logits_and_values(obs, action_masks=masks)
            assert logits[:, 6].max().item() <= -1e7, f"Variant {variant} did not mask logit 6"


def test_memory_crop_extraction(obs_act_spaces):
    obs_space, act_space = obs_act_spaces
    model = RecurrentSpatialRescueCNN(obs_space, act_space, architecture_variant="memory_crop")

    B, C, H, W = 2, 9, 40, 52
    memory = torch.zeros(B, C, H, W)
    # Agent at (10, 20) in batch 0, (30, 40) in batch 1
    memory[0, 8, 10, 20] = 1.0
    memory[0, 0, 10, 20] = 0.75  # North passability at agent
    memory[1, 8, 30, 40] = 1.0
    memory[1, 1, 30, 40] = 0.50  # East passability at agent

    crop = model._extract_memory_crop(memory, crop_radius=7)
    assert crop.shape == (2, 9, 15, 15)
    # The center of 15x15 is index (7, 7)
    assert crop[0, 8, 7, 7].item() == 1.0
    assert crop[0, 0, 7, 7].item() == 0.75
    assert crop[1, 8, 7, 7].item() == 1.0
    assert crop[1, 1, 7, 7].item() == 0.50


def test_weight_transfer_phase11(obs_act_spaces):
    obs_space, act_space = obs_act_spaces
    source_path = "models/phase10_variant_b/final_model.zip"
    if not Path(source_path).exists():
        pytest.skip(f"Source checkpoint {source_path} not found")

    for variant in ["current", "memory_only", "memory_crop"]:
        model = RecurrentSpatialRescueCNN(obs_space, act_space, architecture_variant=variant)
        res = transfer_phase11_weights(source_path, model)

        assert res["status"] == "success"
        assert res["transfer_coverage_pct"] == 100.0, f"Expected 100% coverage, got {res['transfer_coverage_pct']}%"

        if variant == "memory_only":
            assert res["transferred_params"] == 1_179_128
            assert len(res["transferred_keys"]) == 24
        else:
            assert res["transferred_params"] == 1_431_168
            assert len(res["transferred_keys"]) == 30


def test_initial_policy_differences(obs_act_spaces):
    """Test initial logit distance and action agreement on 100 diverse states."""
    source_path = "models/phase10_variant_b/final_model.zip"
    if not Path(source_path).exists():
        pytest.skip(f"Source checkpoint {source_path} not found")

    env = PreviousActionWrapper(RescueEnv(reward_shaping=False, geometry_mode="edge"))
    obs, info = env.reset(seed=12345)

    model_a = RecurrentSpatialRescueCNN(env.observation_space, env.action_space, architecture_variant="current")
    model_b = RecurrentSpatialRescueCNN(env.observation_space, env.action_space, architecture_variant="memory_only")
    model_c = RecurrentSpatialRescueCNN(env.observation_space, env.action_space, architecture_variant="memory_crop")

    transfer_phase11_weights(source_path, model_a)
    transfer_phase11_weights(source_path, model_b)
    transfer_phase11_weights(source_path, model_c)

    model_a.eval()
    model_b.eval()
    model_c.eval()

    n_states = 100
    a_acts, b_acts, c_acts = [], [], []
    ab_l1, ac_l1 = [], []

    for step in range(n_states):
        masks = torch.as_tensor(env.action_masks(), dtype=torch.bool).unsqueeze(0)
        t_obs = {
            "grid": torch.as_tensor(obs["grid"], dtype=torch.float32).unsqueeze(0),
            "memory": torch.as_tensor(obs["memory"], dtype=torch.float32).unsqueeze(0),
            "agent": torch.as_tensor(obs["agent"], dtype=torch.float32).unsqueeze(0),
            "previous_action": torch.as_tensor([obs["previous_action"]], dtype=torch.int64),
        }

        with torch.no_grad():
            la, _ = model_a.get_action_logits_and_values(t_obs, action_masks=masks)
            lb, _ = model_b.get_action_logits_and_values(t_obs, action_masks=masks)
            lc, _ = model_c.get_action_logits_and_values(t_obs, action_masks=masks)

        act_a = la.argmax(dim=-1).item()
        act_b = lb.argmax(dim=-1).item()
        act_c = lc.argmax(dim=-1).item()

        a_acts.append(act_a)
        b_acts.append(act_b)
        c_acts.append(act_c)

        # Measure L1 distance over valid actions (exclude masked action 6)
        valid = masks[0].nonzero().squeeze(-1)
        ab_l1.append((la[0, valid] - lb[0, valid]).abs().mean().item())
        ac_l1.append((la[0, valid] - lc[0, valid]).abs().mean().item())

        action = act_a
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    ab_agreement = 100.0 * np.mean(np.array(a_acts) == np.array(b_acts))
    ac_agreement = 100.0 * np.mean(np.array(a_acts) == np.array(c_acts))

    print(f"\nInitial Policy Comparison (100 states):")
    print(f"  Variant A vs B: Argmax Agreement = {ab_agreement:.1f}%, Mean Valid L1 = {np.mean(ab_l1):.4f}")
    print(f"  Variant A vs C: Argmax Agreement = {ac_agreement:.1f}%, Mean Valid L1 = {np.mean(ac_l1):.4f}")

    # Variant C should be very close to Variant A because memory crop is nearly identical to local grid
    assert ac_agreement >= 70.0, f"Variant C agreement too low: {ac_agreement}%"
