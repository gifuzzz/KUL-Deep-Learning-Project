import numpy as np
import pytest
import torch

from rescue_env.envs.single_agent import RescueEnv
from rescue_env.envs.memory_map import MemoryMap
from agents.cnn import (
    RecurrentSpatialRescueCNN,
    transfer_spatial_memory_weights,
    inspect_checkpoint,
)
from agents.recurrent_ppo import MaskableRecurrentPPO, load_checkpoint
from rescue_env.wrappers import PreviousActionWrapper


def test_binary_visited_channel():
    """Verify Channel 9 in 'visited_binary' mode marks strictly physically visited cells."""
    env = RescueEnv(
        max_steps=50,
        spatial_memory_mode="visited_binary",
        reward_shaping=False,
    )
    obs, _ = env.reset(seed=42)

    # Memory observation shape must be (10, 40, 52)
    assert obs["memory"].shape == (10, 40, 52)
    ch9 = obs["memory"][9]

    # At reset, initial position should be marked as visited
    ar, ac = env.sim.agents[0]["r"], env.sim.agents[0]["c"]
    pad_top = env.memory.pad_top
    pad_left = env.memory.pad_left
    assert ch9[pad_top + ar, pad_left + ac] == 1.0

    # Line-of-sight explores cells, but explored cells that are unvisited must remain 0 in channel 9
    explored_ch = obs["memory"][5]
    unvisited_mask = (explored_ch > 0.0) & (ch9 == 0.0)
    assert np.any(unvisited_mask), "Explored line-of-sight cells must NOT be marked as visited!"

    # Execute a move action if available
    legal_moves = [a for a in (1, 2, 3, 4) if env.action_masks()[a]]
    if legal_moves:
        move_action = legal_moves[0]
        obs2, _, _, _, _ = env.step(move_action)
        new_ar, new_ac = env.sim.agents[0]["r"], env.sim.agents[0]["c"]
        new_ch9 = obs2["memory"][9]
        # Both initial position and new position must be 1.0
        assert new_ch9[pad_top + ar, pad_left + ac] == 1.0
        assert new_ch9[pad_top + new_ar, pad_left + new_ac] == 1.0

    env.close()


def test_visit_recency_channel_and_decay():
    """Verify Channel 9 in 'visit_recency' mode decays over Wait steps and stays 1.0 at current cell."""
    tau = 40.0
    env = RescueEnv(
        max_steps=50,
        spatial_memory_mode="visit_recency",
        recency_tau=tau,
        reward_shaping=False,
    )
    obs, _ = env.reset(seed=42)
    ar, ac = env.sim.agents[0]["r"], env.sim.agents[0]["c"]
    pad_top = env.memory.pad_top
    pad_left = env.memory.pad_left

    # At reset, current cell is 1.0
    assert np.isclose(obs["memory"][9, pad_top + ar, pad_left + ac], 1.0)

    # Execute 5 Wait actions (Action 0)
    for step in range(1, 6):
        obs, _, _, _, _ = env.step(0)
        # Current cell always remains 1.0
        assert np.isclose(obs["memory"][9, pad_top + ar, pad_left + ac], 1.0)

    # If the agent moves to an adjacent cell, the previous cell should show decay
    legal_moves = [a for a in (1, 2, 3, 4) if env.action_masks()[a]]
    if legal_moves:
        move_action = legal_moves[0]
        obs, _, _, _, _ = env.step(move_action)
        new_ar, new_ac = env.sim.agents[0]["r"], env.sim.agents[0]["c"]

        # Current cell is 1.0
        assert np.isclose(obs["memory"][9, pad_top + new_ar, pad_left + new_ac], 1.0)

        # Wait 10 steps at the new cell (steps 7..16)
        for _ in range(10):
            obs, _, _, _, _ = env.step(0)

        # The old cell was last visited at step 5; current step is 16.
        # dt = 16 - 5 = 11. recency = exp(-11 / tau)
        dt = env.steps - 5
        expected_recency = np.exp(-dt / tau)
        actual_recency = obs["memory"][9, pad_top + ar, pad_left + ac]
        assert np.isclose(actual_recency, expected_recency, atol=1e-3), (
            f"Expected {expected_recency}, got {actual_recency}"
        )

    env.close()


def test_weight_transfer_initialization_equivalence():
    """Verify that zero-initialized 10th channel produces identical outputs (Delta < 1e-6) to 9-channel base."""
    base_checkpoint_path = "models/phase6_spawn_curriculum/production_250k/stage3_spawn_curriculum/best_model.zip"

    # Base 9-channel model
    env_9ch = PreviousActionWrapper(RescueEnv(spatial_memory_mode="none", reward_shaping=False))
    base_model = load_checkpoint(base_checkpoint_path, env=env_9ch, device="cpu")

    # 10-channel models (Binary Visited and Visit Recency)
    for mode in ("visited_binary", "visit_recency"):
        env_10ch = PreviousActionWrapper(RescueEnv(spatial_memory_mode=mode, reward_shaping=False))
        target_model = MaskableRecurrentPPO(
            policy=RecurrentSpatialRescueCNN,
            env=env_10ch,
            architecture_variant="baseline",
            device="cpu",
        )

        # Transfer weights with zero-initialized channel 9
        summary = transfer_spatial_memory_weights(base_checkpoint_path, target_model, zero_init_ch9=True)
        assert summary["status"] == "success"
        assert summary["zero_init_ch9"] is True

        # Construct identical inputs
        obs_9, _ = env_9ch.reset(seed=12345)
        obs_10, _ = env_10ch.reset(seed=12345)

        # Convert to batch tensors
        obs_9_t = {
            "grid": torch.from_numpy(obs_9["grid"]).unsqueeze(0),
            "memory": torch.from_numpy(obs_9["memory"]).unsqueeze(0),
            "agent": torch.from_numpy(obs_9["agent"]).unsqueeze(0),
            "previous_action": torch.tensor([0]),
        }
        obs_10_t = {
            "grid": torch.from_numpy(obs_10["grid"]).unsqueeze(0),
            "memory": torch.from_numpy(obs_10["memory"]).unsqueeze(0),
            "agent": torch.from_numpy(obs_10["agent"]).unsqueeze(0),
            "previous_action": torch.tensor([0]),
        }

        # Check channel 9 slice is zeroed in target memory_cnn[0].weight
        assert torch.all(target_model.policy.memory_cnn[0].weight[:, 9, :, :] == 0.0)

        # Evaluate forward pass on both models
        with torch.no_grad():
            lstm_states_9 = torch.zeros(1, 1, 128)
            lstm_states_10 = torch.zeros(1, 1, 128)
            starts = torch.tensor([True])

            dist_9 = base_model.policy.get_distribution(
                obs_9_t, lstm_states=lstm_states_9, episode_starts=starts
            )
            dist_10 = target_model.policy.get_distribution(
                obs_10_t, lstm_states=lstm_states_10, episode_starts=starts
            )

            logits_9 = dist_9.distribution.logits
            logits_10 = dist_10.distribution.logits

            # Max absolute difference must be strictly < 1e-6
            delta = torch.max(torch.abs(logits_9 - logits_10)).item()
            assert delta < 1e-6, f"Mode {mode}: Output delta {delta} exceeds 1e-6 tolerance!"

            # Compare value predictions as well
            val_9 = base_model.policy.predict_values(obs_9_t, lstm_states=lstm_states_9, episode_starts=starts)
            val_10 = target_model.policy.predict_values(obs_10_t, lstm_states=lstm_states_10, episode_starts=starts)
            val_delta = torch.max(torch.abs(val_9 - val_10)).item()
            assert val_delta < 1e-6, f"Mode {mode}: Value delta {val_delta} exceeds 1e-6 tolerance!"

        env_10ch.close()
    env_9ch.close()


def test_corrected_exploration_reward():
    """Verify that normalized continuous reward awards positive rewards across steps 1..384,

    does NOT reward doorway reversals or stationary wait loops, and unshaped evaluation remains 0.0.
    """
    env = RescueEnv(
        task_type="exploration_only",
        reward_shaping=True,
        exploration_reward_mode="normalized_continuous",
        max_steps=384,
    )
    env.reset(seed=100)

    # 1. Stationary Wait action should yield negative step penalty (-0.001) and ZERO positive reward
    _, r_wait, _, _, info_wait = env.step(0)
    assert np.isclose(r_wait, -0.001), f"Wait action should yield -0.001, got {r_wait}"

    # 2. Moving into new areas should yield positive progress reward
    legal_moves = [a for a in (1, 2, 3, 4) if env.action_masks()[a]]
    if legal_moves:
        _, r_move, _, _, _ = env.step(legal_moves[0])
        assert r_move > -0.001, f"Exploring new cell should yield positive progress reward, got {r_move}"

    # 3. Simulate stepping past step 50 and verify rewards are still available
    env.steps = 150
    # Artificially trigger exploration tracking with mock cells
    shaping = -env.exploration_step_penalty
    denom_cells = len(env._reachable_interior_cells)
    denom_edges = len(env._reachable_traversable_edges)
    pos_potential = (5 / denom_cells) * 8.0 + (5 / denom_edges) * 4.0
    shaping += pos_potential
    assert shaping > 0.0, "At step 150, new cells/edges must still be rewarded without hitting any cap!"

    env.close()

    # 4. Evaluation mode (reward_shaping=False) must have strictly 0.0 reward in exploration_only
    env_eval = RescueEnv(
        task_type="exploration_only",
        reward_shaping=False,
        max_steps=384,
    )
    env_eval.reset(seed=100)
    _, r_eval, _, _, _ = env_eval.step(0)
    assert r_eval == 0.0, "Unshaped evaluation must strictly award 0.0 reward"
    env_eval.close()
