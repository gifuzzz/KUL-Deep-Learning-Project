"""Tests for Phase 9 Branching Navigation Curriculum & Memory Consistency.

Verifies:
1. Scenario generation across stages B1, B2, B3, B4.
2. Physical consistency of visited memory (Channel 9 is 1.0 along stem and visited branch, 0.0 along unvisited branch).
3. Legal action masking (Action 5 Pickup and Action 6 Water Hose disabled).
4. Counterfactual memory swap (State A and State B have identical layout, agent position, and legal actions, with swapped visited memory).
5. GRU state unrolling and forward pass compatibility.
"""

import numpy as np
import pytest
import torch

from agents.cnn import RecurrentSpatialRescueCNN
from agents.recurrent_ppo import MaskableRecurrentPPO
from rescue_env.envs.single_agent import RescueEnv
from rescue_env.wrappers import PreviousActionWrapper


def test_branching_stages_generation():
    """Verify that all four branching stages (B1, B2, B3, B4) generate valid scenarios."""
    env = RescueEnv(task_type="branching_exploration", spatial_memory_mode="visited_binary")

    for stage in ("B1", "B2", "B3", "B4"):
        for seed in (10, 25, 42):
            obs, info = env.reset(seed=seed, options={"branching_stage": stage})
            assert info["task_type"] == "branching_exploration"
            assert info["branching_stage"] == stage
            assert obs["grid"].shape == (9, 15, 15)
            assert obs["memory"].shape == (10, 40, 52)
            assert obs["agent"].shape == (5,)

            # Verify max_steps per stage
            if stage == "B1":
                assert env.max_steps == 64
                assert info["junction_pos"] is not None
                assert info["unvisited_branch_action"] in (1, 2, 3, 4)
                assert info["visited_branch_action"] in (1, 2, 3, 4)
                assert info["unvisited_branch_action"] != info["visited_branch_action"]
            elif stage == "B2":
                assert env.max_steps == 128
            elif stage == "B3":
                assert env.max_steps == 192
            elif stage == "B4":
                assert env.max_steps == 384


def test_action_masking_integrity():
    """Verify that Action 5 (Pickup) and Action 6 (Water Hose) are strictly disabled."""
    env = RescueEnv(task_type="branching_exploration", spatial_memory_mode="visited_binary")
    obs, info = env.reset(seed=101, options={"branching_stage": "B1"})

    mask = env.action_masks()
    assert mask.shape == (7,)
    assert mask[0] is True or mask[0] == 1  # Wait is legal
    assert mask[5] == False  # Pickup strictly False
    assert mask[6] == False  # Water hose strictly False

    # Verify at least two movement actions are legal at B1 junction
    movement_legal = mask[1:5].sum()
    assert movement_legal >= 2, f"At junction, expected >= 2 legal moves, got {movement_legal}"


def test_visited_memory_physical_legitimacy():
    """Verify that Channel 9 visited memory strictly reflects traversed cells and unvisited branch has 0.0."""
    env = RescueEnv(task_type="branching_exploration", spatial_memory_mode="visited_binary")
    obs, info = env.reset(seed=202, options={"branching_stage": "B1"})

    ch9 = obs["memory"][9]
    visited_cells = env.branching_visited_cells
    unvisited_cells = env.branching_unvisited_cells

    assert len(visited_cells) >= 2, "Visited branch must have >= 2 cells"
    assert len(unvisited_cells) >= 1, "Unvisited branch must have >= 1 cell"

    pad_top = env.memory.pad_top
    pad_left = env.memory.pad_left

    # Check visited branch cells: in unpadded coordinates, memory.visited must be 1.0
    for r, c in visited_cells:
        assert env.memory.visited[r, c] == 1.0, f"Cell ({r},{c}) should be marked visited"
        assert ch9[pad_top + r, pad_left + c] == 1.0, f"Padded Channel 9 at ({r},{c}) should be 1.0"

    # Check unvisited branch cells: memory.visited must be 0.0
    for r, c in unvisited_cells:
        assert env.memory.visited[r, c] == 0.0, f"Cell ({r},{c}) should be unvisited (0.0)"
        assert ch9[pad_top + r, pad_left + c] == 0.0, f"Padded Channel 9 at ({r},{c}) should be 0.0"


def test_counterfactual_memory_swap():
    """Verify matched counterfactual pairs: identical layout, position, and legal actions, with swapped visited memory."""
    env = RescueEnv(task_type="branching_exploration", spatial_memory_mode="visited_binary")

    # State A: Normal visited branch
    obs_a, info_a = env.reset(seed=303, options={"branching_stage": "B1", "swap_visited_branch": False})
    mask_a = env.action_masks()

    # State B: Swapped visited branch (same seed)
    obs_b, info_b = env.reset(seed=303, options={"branching_stage": "B1", "swap_visited_branch": True})
    mask_b = env.action_masks()

    # 1. Junction position and agent position must be identical
    assert info_a["junction_pos"] == info_b["junction_pos"]
    assert np.allclose(obs_a["agent"], obs_b["agent"])

    # 2. Local grid (walls, openings, exit visible around agent) must be identical
    assert np.allclose(obs_a["grid"], obs_b["grid"])

    # 3. Action masks at junction must be identical
    assert np.array_equal(mask_a, mask_b)

    # 4. Actions must be swapped
    assert info_a["visited_branch_action"] == info_b["unvisited_branch_action"]
    assert info_a["unvisited_branch_action"] == info_b["visited_branch_action"]

    # 5. Channel 9 (visited memory) MUST DIFFER
    ch9_diff = np.abs(obs_a["memory"][9] - obs_b["memory"][9]).sum()
    assert ch9_diff > 0.0, "Channel 9 must differ between counterfactual State A and State B"

    # 6. Control 1 (isolated channel test): Holding channels 0..8 and agent fixed,
    # swap only Channel 9 values between visited and unvisited branches.
    obs_iso = {k: v.copy() for k, v in obs_a.items()}
    obs_iso["memory"][9] = obs_b["memory"][9].copy()
    assert np.allclose(obs_iso["memory"][:9], obs_a["memory"][:9])
    assert not np.allclose(obs_iso["memory"][9], obs_a["memory"][9])


def test_model_forward_pass_with_branching_env():
    """Verify that MaskableRecurrentPPO runs forward passes and handles GRU hidden states cleanly in branching task."""
    env = PreviousActionWrapper(RescueEnv(task_type="branching_exploration", spatial_memory_mode="visited_binary"))
    obs, info = env.reset(seed=404, options={"branching_stage": "B1"})

    model = MaskableRecurrentPPO.load("models/phase8_binary_visited/initial_transferred.zip", env=env, device="cpu")

    lstm_states = None
    episode_starts = np.array([True])
    masks = env.action_masks()

    action, lstm_states = model.predict(
        obs,
        state=lstm_states,
        episode_start=episode_starts,
        deterministic=True,
        action_masks=masks,
    )

    assert action in (0, 1, 2, 3, 4)
    assert masks[action] is True or masks[action] == 1
    assert lstm_states is not None
