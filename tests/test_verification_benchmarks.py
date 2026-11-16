"""Comprehensive Unit Tests for Verification & Benchmarks.

Covers Task 1 specifications:
1. 9-channel (5 wall layers + 4 semantic layers) full-cell shapes:
   - Local grid: (9, 15, 15)
   - Memory canvas: (9, 40, 52)
   - Agent vector: (5,)
2. Legal action masks:
   - 7-dimensional boolean mask (np.bool_)
   - Masking invalid movements, pickups, and drops
   - Policy masking verification (illegal actions receive -1e8 logits and 0.0 prob)
3. GRU episode resets:
   - Hidden state propagates within episodes
   - Hidden state resets cleanly to zeros when episode_starts == 1.0 (dones)
   - Multi-environment isolation during asynchronous resets
4. CPU/MPS checkpoint round-trips:
   - Saving and loading models across CPU and MPS runtimes without device mismatch errors
   - Forward prediction on restored checkpoints without device assertion errors
"""

import numpy as np
import pytest
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO

from agents.cnn import RecurrentSpatialRescueCNN
from rescue_env.envs.single_agent import RescueEnv
from training.train_ppo import save_checkpoint, load_checkpoint
from training.evaluate import load_model


def test_observation_shapes_and_9_channel_semantics():
    """Verify observation space shapes and exact 9-channel full-cell semantics.

    Channels:
      - 5 Wall / Passability layers:
        Ch 0: Passability North (1.0 traversable, else 0.0)
        Ch 1: Passability East (1.0 traversable, else 0.0)
        Ch 2: Passability South (1.0 traversable, else 0.0)
        Ch 3: Passability West (1.0 traversable, else 0.0)
        Ch 4: Debris / Obstacle (1.0 blocked, else 0.0)
      - 4 Active / Semantic layers:
        Ch 5: Explored Mask (1.0 seen in line-of-sight, else 0.0)
        Ch 6: Victim (category / presence)
        Ch 7: Exit (1.0 exit portal cell, else 0.0)
        Ch 8: Agent Position Marker (1.0 at agent position)
      - Agent Vector:
        (5,) = [row_norm, col_norm, health, role_norm, carrying_victim]
    """
    env = RescueEnv(max_steps=64, victim_distance=2, visibility_radius=7)
    try:
        # Check space specifications
        assert env.NUM_CHANNELS == 9
        assert env.observation_space["grid"].shape == (9, 15, 15)
        assert env.observation_space["memory"].shape == (9, 40, 52)
        assert env.observation_space["agent"].shape == (5,)

        obs, info = env.reset(seed=42)
        assert env.observation_space.contains(obs)

        # 1. Local grid shape and dtype
        grid = obs["grid"]
        assert grid.shape == (9, 15, 15)
        assert grid.dtype == np.float32

        # Verify local grid agent position marker is at center (7, 7)
        assert grid[8, 7, 7] == 1.0
        grid_marker_copy = grid[8].copy()
        grid_marker_copy[7, 7] = 0.0
        assert np.all(grid_marker_copy == 0.0)

        # 2. Memory canvas shape and dtype
        memory = obs["memory"]
        assert memory.shape == (9, 40, 52)
        assert memory.dtype == np.float32

        # Verify memory agent position marker matches agent position
        person = env.sim.agents[0]
        pad_top = (env.MEMORY_HEIGHT - env.sim.h) // 2
        pad_left = (env.MEMORY_WIDTH - env.sim.w) // 2
        expected_r = pad_top + person["r"]
        expected_c = pad_left + person["c"]
        assert memory[8, expected_r, expected_c] == 1.0
        mem_marker_copy = memory[8].copy()
        mem_marker_copy[expected_r, expected_c] = 0.0
        assert np.all(mem_marker_copy == 0.0)

        # 3. Agent scalar vector shape and contents
        agent = obs["agent"]
        assert agent.shape == (5,)
        assert agent.dtype == np.float32
        assert 0.0 <= agent[0] <= 1.0  # row_norm
        assert 0.0 <= agent[1] <= 1.0  # col_norm
        assert agent[2] == 1.0          # initial full health
        assert agent[3] == 0.0          # firefighter role
        assert agent[4] == 0.0          # not carrying victim initially

        # Step and verify shape invariance
        step_obs, reward, terminated, truncated, step_info = env.step(0)
        assert env.observation_space.contains(step_obs)
        assert step_obs["grid"].shape == (9, 15, 15)
        assert step_obs["memory"].shape == (9, 40, 52)
        assert step_obs["agent"].shape == (5,)
    finally:
        env.close()


def test_legal_action_masks_7dim_bool():
    """Verify legal action masks are 7-dim bool arrays and correctly block illegal actions."""
    env = RescueEnv(max_steps=64, victim_distance=2)
    try:
        env.reset(seed=42)
        mask = env.action_masks()

        # Check dimension, shape, and boolean dtype
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (7,)
        assert mask.dtype == np.bool_ or mask.dtype == bool

        # Action 0 (Wait) is always legal for living firefighter
        assert bool(mask[0]) is True

        # Action 6 (Water Hose) is masked out in fireless rescue
        assert bool(mask[6]) is False

        # Action 5 (Pickup) is initially False when no victim is adjacent
        assert bool(mask[5]) is False

        # Step closer to victim (Right -> (13, 1) adjacent to victim at (13, 2))
        env.step(2)
        mask_after_move = env.action_masks()
        assert bool(mask_after_move[5]) is True, "Pickup should become legal when adjacent to unrescued victim"

        # Execute pickup
        env.step(5)
        mask_after_pickup = env.action_masks()
        assert bool(mask_after_pickup[5]) is False, "Pickup should be illegal when already carrying victim"

        # Verify mask enforcement via RecurrentSpatialRescueCNN policy
        obs_space = env.observation_space
        policy = RecurrentSpatialRescueCNN(obs_space, env.action_space)

        # Force a mask where only action 0 is legal
        strict_mask = torch.tensor([[True, False, False, False, False, False, False]], dtype=torch.bool)
        sample_obs = {
            "grid": torch.randn(1, 9, 15, 15),
            "memory": torch.randn(1, 9, 40, 52),
            "agent": torch.randn(1, 5),
            "previous_action": torch.tensor([0]),
        }

        # Check logits and probabilities
        masked_logits = policy.get_masked_logits(sample_obs, action_masks=strict_mask)
        assert torch.all(masked_logits[0, 1:] <= -1e7)

        dist = policy.get_distribution(sample_obs, action_masks=strict_mask)
        probs = dist.distribution.probs[0]
        assert probs[0].item() == pytest.approx(1.0, abs=1e-5)
        assert torch.all(probs[1:] == 0.0)

        # Sample 50 times, verify illegal actions are NEVER chosen
        for _ in range(50):
            action, _, _ = policy(sample_obs, action_masks=strict_mask)
            assert int(action.item()) == 0
    finally:
        env.close()


def test_gru_episode_resets_hidden_state_cleanly():
    """Verify GRU hidden state updates across transitions and resets cleanly on episode boundaries."""
    obs_space = spaces.Dict({
        "grid": spaces.Box(low=-np.inf, high=np.inf, shape=(9, 15, 15), dtype=np.float32),
        "memory": spaces.Box(low=-np.inf, high=np.inf, shape=(9, 40, 52), dtype=np.float32),
        "agent": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
    })
    policy = RecurrentSpatialRescueCNN(obs_space, spaces.Discrete(7))

    # Single-environment sequence test
    obs_t0 = {
        "grid": torch.randn(1, 9, 15, 15),
        "memory": torch.randn(1, 9, 40, 52),
        "agent": torch.randn(1, 5),
    }
    h_init = torch.zeros(1, 1, 128)

    # Step 1: h_init -> h_step1
    _, _, _, h_step1 = policy(obs_t0, lstm_states=h_init, return_hidden_state=True)
    assert not torch.allclose(h_init, h_step1), "Hidden state must update from initial zeros"

    # Step 2: Next step within same episode (episode_starts = 0)
    obs_t1 = {
        "grid": torch.randn(1, 9, 15, 15),
        "memory": torch.randn(1, 9, 40, 52),
        "agent": torch.randn(1, 5),
    }
    _, _, _, h_step2 = policy(obs_t1, lstm_states=h_step1, episode_starts=torch.tensor([0.0]), return_hidden_state=True)
    assert not torch.allclose(h_step1, h_step2)

    # Step 3: Start of new episode (episode_starts = 1.0)
    # When episode_starts == 1.0, hidden state MUST be wiped to zero before processing obs_t1
    _, val_reset, _, h_reset = policy(
        obs_t1,
        lstm_states=h_step2,
        episode_starts=torch.tensor([1.0]),
        return_hidden_state=True,
    )

    # Forwarding obs_t1 with h_init (clean zeros) must produce exactly the same result as reset
    _, val_clean, _, h_clean = policy(obs_t1, lstm_states=h_init, return_hidden_state=True)
    assert torch.allclose(h_reset, h_clean, atol=1e-5), "Hidden state did not reset cleanly on episode_starts=1.0"
    assert torch.allclose(val_reset, val_clean, atol=1e-5), "Value prediction differs after episode reset"

    # Batched multi-environment asynchronous reset isolation test
    batch_size = 4
    batched_obs = {
        "grid": torch.randn(batch_size, 9, 15, 15),
        "memory": torch.randn(batch_size, 9, 40, 52),
        "agent": torch.randn(batch_size, 5),
    }
    h_batch_prev = torch.randn(1, batch_size, 128)
    # Envs 0 and 2 are new episodes (reset = 1.0), Envs 1 and 3 continue (reset = 0.0)
    starts = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)

    _, _, _, h_batch_next = policy(
        batched_obs,
        lstm_states=h_batch_prev,
        episode_starts=starts,
        return_hidden_state=True,
    )

    # Compare env 0 with clean step from zeros
    obs_0 = {k: v[0:1] for k, v in batched_obs.items()}
    _, _, _, h_0_clean = policy(obs_0, lstm_states=torch.zeros(1, 1, 128), return_hidden_state=True)
    assert torch.allclose(h_batch_next[:, 0:1], h_0_clean, atol=1e-5)

    # Compare env 1 with continuation from h_batch_prev[:, 1:2]
    obs_1 = {k: v[1:2] for k, v in batched_obs.items()}
    _, _, _, h_1_cont = policy(obs_1, lstm_states=h_batch_prev[:, 1:2], return_hidden_state=True)
    assert torch.allclose(h_batch_next[:, 1:2], h_1_cont, atol=1e-5)


def test_cpu_mps_checkpoint_round_trips(tmp_path):
    """Verify model checkpoints can be saved on CPU/MPS and loaded across devices without mismatch errors."""
    env = RescueEnv(max_steps=32, victim_distance=2)
    try:
        obs, _ = env.reset(seed=42)

        # 1. Create and train a small CPU model
        model_cpu = MaskablePPO(
            policy=RecurrentSpatialRescueCNN,
            env=env,
            n_steps=8,
            batch_size=8,
            device="cpu",
            verbose=0,
        )
        model_cpu.learn(total_timesteps=16)

        cpu_checkpoint = tmp_path / "model_cpu.zip"
        save_checkpoint(model_cpu, cpu_checkpoint)
        assert cpu_checkpoint.exists()

        # 2. Reload on CPU
        loaded_cpu = load_checkpoint(cpu_checkpoint, env=env, device="cpu")
        assert loaded_cpu.device == torch.device("cpu")
        act, state = loaded_cpu.predict(obs, action_masks=env.action_masks())
        assert env.action_space.contains(int(act))
        assert state is not None

        # 3. MPS Round-Trip (if hardware supports MPS)
        if torch.backends.mps.is_available():
            mps_device = torch.device("mps")

            # Load CPU checkpoint onto MPS
            loaded_mps = load_checkpoint(cpu_checkpoint, env=env, device="mps")
            assert loaded_mps.device == mps_device

            # Predict on MPS
            act_mps, state_mps = loaded_mps.predict(obs, action_masks=env.action_masks())
            assert env.action_space.contains(int(act_mps))

            # Save from MPS
            mps_checkpoint = tmp_path / "model_mps.zip"
            save_checkpoint(loaded_mps, mps_checkpoint)
            assert mps_checkpoint.exists()

            # Load MPS checkpoint back onto CPU
            reloaded_cpu = load_checkpoint(mps_checkpoint, env=env, device="cpu")
            assert reloaded_cpu.device == torch.device("cpu")
            act_back, state_back = reloaded_cpu.predict(obs, action_masks=env.action_masks())
            assert env.action_space.contains(int(act_back))

            # Test evaluate CLI loader
            eval_model, eval_dev = load_model(mps_checkpoint, env=env, device="cpu")
            assert eval_dev == torch.device("cpu")
            act_eval, _ = eval_model.predict(obs, action_masks=env.action_masks())
            assert env.action_space.contains(int(act_eval))
    finally:
        env.close()
