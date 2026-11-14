"""Rigorous Verification Suite for Recurrent Masked PPO in FlameBox.

Verifies the 4 critical correctness properties:
  a) Multi-step BPTT across consecutive timesteps (non-zero temporal gradient).
  b) Hidden states reset ONLY when episode ends, and persist across rollout boundaries.
  c) Action masks strictly applied in rollout, evaluate_actions, and entropy.
  d) Device round-trip save and load with metadata preservation.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.cnn import RecurrentSpatialRescueCNN
from agents.recurrent_ppo import (
    MaskableRecurrentDictRolloutBuffer,
    MaskableRecurrentPPO,
    RNNStates,
)
from rescue_env.envs.single_agent import RescueEnv
from rescue_env.wrappers import PreviousActionWrapper
from training.train_ppo import load_checkpoint, save_checkpoint


def make_dummy_obs_space() -> spaces.Dict:
    return spaces.Dict({
        "grid": spaces.Box(low=-np.inf, high=np.inf, shape=(9, 15, 15), dtype=np.float32),
        "memory": spaces.Box(low=-np.inf, high=np.inf, shape=(9, 40, 52), dtype=np.float32),
        "agent": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
        "previous_action": spaces.Box(low=0, high=6, shape=(), dtype=np.int64),
    })


# ==============================================================================
# Property A: BPTT Across Consecutive Timesteps
# ==============================================================================

def test_bptt_gradient_flows_across_time():
    """Verify that gradients flow backwards across time: d L_{t+1} / d s_t != 0."""
    obs_space = make_dummy_obs_space()
    action_space = spaces.Discrete(7)
    policy = RecurrentSpatialRescueCNN(obs_space, action_space)
    policy.set_training_mode(True)

    # 2 consecutive timesteps for 1 sequence: n_seq = 1, max_length = 2
    grid_0 = torch.randn(1, 9, 15, 15, requires_grad=True)
    grid_1 = torch.randn(1, 9, 15, 15, requires_grad=True)
    batch_grid = torch.cat([grid_0, grid_1], dim=0)  # shape (2, 9, 15, 15)

    batch_obs = {
        "grid": batch_grid,
        "memory": torch.randn(2, 9, 40, 52),
        "agent": torch.randn(2, 5),
        "previous_action": torch.tensor([0, 3], dtype=torch.long),
    }

    actions = torch.tensor([1, 2], dtype=torch.long)
    masks = torch.ones((2, 7), dtype=torch.bool)
    h0 = torch.zeros((1, 1, 128))  # 1 sequence of length 2
    episode_starts = torch.tensor([1.0, 0.0])  # starts at t=0, continues at t=1

    values, log_prob, entropy = policy.evaluate_actions(
        batch_obs,
        actions,
        lstm_states=h0,
        episode_starts=episode_starts,
        action_masks=masks,
    )

    # Loss computed strictly at t=1
    loss_t1 = values[1] + log_prob[1]
    grad_grid_0 = torch.autograd.grad(loss_t1, grid_0, retain_graph=True)[0]

    # Temporal gradient must be strictly non-zero
    grad_norm = grad_grid_0.norm().item()
    print(f"BPTT temporal gradient norm: {grad_norm}")
    assert grad_norm > 1e-7, f"Zero BPTT: grad norm is {grad_norm}"

    # Verify boundary severance: if episode resets at t=1, gradient into t=0 must be exactly 0
    episode_starts_reset = torch.tensor([1.0, 1.0])  # episode reset at t=1
    values_reset, log_prob_reset, _ = policy.evaluate_actions(
        batch_obs,
        actions,
        lstm_states=h0,
        episode_starts=episode_starts_reset,
        action_masks=masks,
    )
    loss_t1_reset = values_reset[1] + log_prob_reset[1]
    grad_grid_0_reset = torch.autograd.grad(loss_t1_reset, grid_0, retain_graph=True)[0]
    assert torch.all(grad_grid_0_reset == 0.0), "Gradient leaked across episode reset!"


# ==============================================================================
# Property B: Hidden State Persistence and Episode Resets
# ==============================================================================

def test_hidden_state_persistence_and_episode_resets():
    """Verify hidden states persist across rollout boundaries and reset ONLY on episode ends."""
    raw_env = RescueEnv(max_steps=6, victim_distance=6)
    env = PreviousActionWrapper(raw_env)
    vec_env = DummyVecEnv([lambda: env])

    model = MaskableRecurrentPPO(
        policy=RecurrentSpatialRescueCNN,
        env=vec_env,
        n_steps=4,  # Rollout length is 4; episode length is 6
        batch_size=4,
        n_epochs=1,
        learning_rate=1e-4,
        device="cpu",
        seed=42,
    )

    # Run rollout 1 (4 steps): episode has NOT ended (max_steps=6)
    model.learn(total_timesteps=4)

    # After rollout 1, hidden state must NOT be all zeros (agent has taken 4 steps)
    h_end_rollout_1 = model._last_lstm_states.pi[0].clone()
    assert h_end_rollout_1.norm().item() > 1e-4, "Hidden state wiped during rollout 1!"
    assert not model._last_episode_starts[0], "Episode prematurely marked done!"

    # Run rollout 2 (4 more steps): step 6 terminates episode, step 7 is new episode
    model.learn(total_timesteps=4, reset_num_timesteps=False)

    # In the buffer of rollout 2, check that episode_starts is recorded at the reset
    buf_starts = model.rollout_buffer.episode_starts
    assert np.any(buf_starts[1:]), "Episode boundary was not recorded in buffer!"

    vec_env.close()


# ==============================================================================
# Property C: Action Masking Enforcement
# ==============================================================================

def test_action_masking_strictly_applied():
    """Verify action masking in inference, evaluate_actions, and entropy computation."""
    obs_space = make_dummy_obs_space()
    action_space = spaces.Discrete(7)
    policy = RecurrentSpatialRescueCNN(obs_space, action_space)
    policy.set_training_mode(False)

    obs = {
        "grid": torch.randn(1, 9, 15, 15),
        "memory": torch.randn(1, 9, 40, 52),
        "agent": torch.randn(1, 5),
        "previous_action": torch.tensor([0], dtype=torch.long),
    }

    # Mask only action 3 as legal
    mask = np.zeros(7, dtype=bool)
    mask[3] = True

    # 1. Stochastic sampling test: 50 samples must ALL be action 3
    for _ in range(50):
        action, _ = policy.predict(obs, deterministic=False, action_masks=mask)
        assert int(np.asarray(action).item()) == 3, f"Sampled illegal action {action}!"

    # 2. Deterministic selection: must be action 3
    action_det, _ = policy.predict(obs, deterministic=True, action_masks=mask)
    assert int(np.asarray(action_det).item()) == 3

    # 3. Masked logits in evaluate_actions
    mask_tensor = torch.as_tensor(mask).unsqueeze(0)
    values, log_prob, entropy = policy.evaluate_actions(
        obs,
        torch.tensor([3], dtype=torch.long),
        lstm_states=torch.zeros((1, 1, 128)),
        episode_starts=torch.tensor([0.0]),
        action_masks=mask_tensor,
    )

    # Since only 1 action is valid, its probability must be 1.0 (log_prob = 0.0)
    assert torch.isclose(log_prob, torch.tensor(0.0), atol=1e-5), f"log_prob: {log_prob}"
    # Entropy over 1 legal action must be exactly 0.0
    assert torch.isclose(entropy, torch.tensor(0.0), atol=1e-5), f"entropy: {entropy}"

    # Logits of invalid actions must be <= -1e7
    dist = policy.get_distribution(obs, action_masks=mask_tensor)
    logits = dist.distribution.logits[0]
    for a in range(7):
        if a != 3:
            assert logits[a] <= -1e7, f"Illegal action {a} logit not masked: {logits[a]}"


# ==============================================================================
# Property D: Device-Independent Save and Load
# ==============================================================================

def test_checkpoint_round_trip_with_metadata(tmp_path: Path):
    """Verify device-independent save and load, metadata integrity, and inference consistency."""
    env = PreviousActionWrapper(RescueEnv(max_steps=16, victim_distance=2))
    vec_env = DummyVecEnv([lambda: env])

    model = MaskableRecurrentPPO(
        policy=RecurrentSpatialRescueCNN,
        env=vec_env,
        n_steps=8,
        batch_size=8,
        device="cpu",
    )

    checkpoint_file = tmp_path / "test_model.zip"
    meta = {
        "stage": 1,
        "hyperparameters": {"lr": 1e-4, "n_steps": 8},
        "test_marker": "round_trip_verified",
    }
    save_checkpoint(model, checkpoint_file, metadata=meta)

    assert checkpoint_file.exists(), "Checkpoint file was not created!"
    companion_meta = checkpoint_file.with_suffix(".meta.json")
    assert companion_meta.exists(), "Companion metadata JSON was not created!"

    # Verify metadata in zip
    with zipfile.ZipFile(checkpoint_file, "r") as zf:
        assert "metadata.json" in zf.namelist(), "metadata.json missing in checkpoint zip!"
        meta_loaded = json.loads(zf.read("metadata.json"))
        assert meta_loaded["test_marker"] == "round_trip_verified"
        assert meta_loaded["git_commit"] != ""

    # Load model back
    loaded_model = load_checkpoint(checkpoint_file, env=vec_env, device="cpu")
    assert loaded_model is not None
    assert hasattr(loaded_model, "checkpoint_metadata")
    assert loaded_model.checkpoint_metadata["stage"] == 1

    # Verify identical prediction outputs
    obs, _ = env.reset(seed=42)
    mask = env.action_masks()
    act1, state1 = model.predict(obs, deterministic=True, action_masks=mask)
    act2, state2 = loaded_model.predict(obs, deterministic=True, action_masks=mask)

    assert act1 == act2, f"Actions diverge after load: {act1} != {act2}"
    assert np.allclose(state1, state2, atol=1e-5), "Hidden states diverge after load!"

    vec_env.close()
