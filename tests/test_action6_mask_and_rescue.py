"""Unit tests for Action 6 (Water Hose) masking, rescue mechanics, and observation integrity.

Verifies:
1. Action 6 is masked out (False) in fireless rescue tasks.
2. Action 6 can be re-enabled via fire_suppression_enabled=True or active fire.
3. Executing Action 6 in fireless environment produces zero physical/observational change.
4. Victim rescue occurs automatically via movement onto exterior grass while carrying, yielding +10 reward.
5. Action 0 (Wait) remains available and audit behavior.
6. 7-action space and observation shape integrity are maintained.
"""

import numpy as np
import pytest

from rescue_env.envs.single_agent import RescueEnv
from rescue_env.wrappers import PreviousActionWrapper


def test_action6_masked_in_fireless_environment():
    """Verify action 6 is False across seeds and stages when fire is absent."""
    for seed in (42, 100, 50001, 100047):
        env = RescueEnv(geometry_mode="edge")
        try:
            env.reset(seed=seed)
            mask = env.action_masks()
            assert len(mask) == 7
            assert mask[6] is False or mask[6] == np.False_, f"Seed {seed} had mask[6] True"
            assert mask[0] is True or mask[0] == np.True_, "Wait (action 0) should remain True"
        finally:
            env.close()


def test_action6_unmasked_when_fire_enabled():
    """Verify action 6 can be unmasked for fire suppression experiments."""
    env = RescueEnv(geometry_mode="edge", fire_suppression_enabled=True)
    try:
        env.reset(seed=42)
        mask = env.action_masks()
        assert mask[6] is True or mask[6] == np.True_

        # Also verify reset options can toggle it
        env.reset(seed=42, options={"fire_suppression_enabled": False})
        assert not env.action_masks()[6]
    finally:
        env.close()


def test_action6_zero_effect_in_fireless():
    """Verify that forced execution of action 6 produces no state, physics, or observation change."""
    env = RescueEnv(geometry_mode="edge")
    try:
        obs_before, _ = env.reset(seed=42)
        agent_pos_before = (env.sim.agents[0]["r"], env.sim.agents[0]["c"])
        victim_pos_before = (env.sim.victims[0]["r"], env.sim.victims[0]["c"])
        victim_rescued_before = env.sim.victims[0]["rescued"]

        # Directly execute action 6 (FlameBox water command 5)
        obs_after, reward, terminated, truncated, info = env.step(6)

        agent_pos_after = (env.sim.agents[0]["r"], env.sim.agents[0]["c"])
        victim_pos_after = (env.sim.victims[0]["r"], env.sim.victims[0]["c"])
        victim_rescued_after = env.sim.victims[0]["rescued"]

        assert agent_pos_after == agent_pos_before
        assert victim_pos_after == victim_pos_before
        assert victim_rescued_after == victim_rescued_before
        assert reward == 0.0
        assert not terminated
        assert not truncated

        # Check local grid and memory observations are identical
        np.testing.assert_array_equal(obs_after["grid"], obs_before["grid"])
        # Memory channels (except agent marker if any) should match
        np.testing.assert_array_equal(obs_after["memory"][:7], obs_before["memory"][:7])
    finally:
        env.close()


def test_rescue_through_movement_onto_grass():
    """Verify that victim rescue occurs via movement onto exterior grass, NOT action 6."""
    env = RescueEnv(geometry_mode="edge", victim_distance=2, reward_shaping=False)
    try:
        env.reset(seed=42)
        # Seed 42: Agent at (13, 0), victim at (14, 1), exit opening at (13, 0) to col -1.
        # Step Down (action 3) to (14, 0), which is Manhattan-adjacent to victim at (14, 1):
        obs, reward, terminated, truncated, _ = env.step(3)
        assert env.action_masks()[5], "Pickup should be valid when adjacent to victim"

        # Pick up victim (action 5)
        obs, reward, terminated, truncated, _ = env.step(5)
        assert env.sim.victims[0]["carried_by"] == 0

        # Step Up (action 1) back to exit portal cell (13, 0)
        obs, reward, terminated, truncated, _ = env.step(1)
        assert (env.sim.agents[0]["r"], env.sim.agents[0]["c"]) == (13, 0)

        # When carrying victim, moving onto exterior grass (Left, action 4) is unmasked
        mask = env.action_masks()
        assert mask[4], "Moving onto exterior grass carrying victim should be legal"
        assert not mask[6], "Action 6 is NOT the rescue action and should be False"

        # Step onto exterior grass (action 4)
        obs, reward, terminated, truncated, info = env.step(4)

        # Verify rescue succeeded!
        assert env.sim.victims[0]["rescued"] == 1
        assert env.sim.victims[0]["carried_by"] == -1
        assert reward == 10.0, f"Expected +10 rescue reward, got {reward}"
        assert terminated is True, "Episode must terminate upon successful rescue"
    finally:
        env.close()


def test_wait_action_audit():
    """Verify action 0 (Wait) behavior: valid, causes no movement, advances step."""
    env = RescueEnv(geometry_mode="edge")
    try:
        env.reset(seed=42)
        pos_before = (env.sim.agents[0]["r"], env.sim.agents[0]["c"])
        assert env.action_masks()[0] is True or env.action_masks()[0] == np.True_

        obs, reward, terminated, truncated, info = env.step(0)
        pos_after = (env.sim.agents[0]["r"], env.sim.agents[0]["c"])

        assert pos_after == pos_before
        assert env.steps == 1
    finally:
        env.close()


def test_7_action_space_and_wrapper_integrity():
    """Verify action space has 7 dimensions and PreviousActionWrapper handles it cleanly."""
    raw_env = RescueEnv(geometry_mode="edge")
    env = PreviousActionWrapper(raw_env)
    try:
        obs, _ = env.reset(seed=42)
        assert env.action_space.n == 7
        assert "previous_action" in obs
        assert obs["previous_action"] == 0

        # Step with action 2 (Move Right)
        obs, reward, terminated, truncated, _ = env.step(2)
        assert obs["previous_action"] == 2

        # Step with action 0 (Wait)
        obs, reward, terminated, truncated, _ = env.step(0)
        assert obs["previous_action"] == 0

        # Action mask length must be 7
        mask = env.action_masks()
        assert len(mask) == 7
        assert not mask[6]
    finally:
        env.close()
