"""Unit tests for Phase 5 post-evaluation corrections.

Verifies:
1. Distance semantics: distance_metric='spawn' guarantees true spawn-relative distances.
2. Distance semantics: distance_metric='exit' preserves legacy exit-relative placement.
3. Exterior movement masking: moving to exterior grass is masked when not carrying.
4. Exterior movement allowance: moving to exterior grass is legal when carrying.
5. Anti-oscillation training reward: 2-cell limit cycles receive -0.02 penalty during training.
6. Unshaped evaluation invariance: evaluation (reward_shaping=False) has zero shaping or penalties.
"""

from collections import deque
import numpy as np
import pytest

from rescue_env.envs.single_agent import RescueEnv
from rescue_env.envs.reward_shaping import RewardShaper


def compute_bfs_distance(sim, start, target):
    """Compute shortest traversable grid BFS distance between start and target."""
    sr, sc = start
    tr, tc = target
    if (sr, sc) == (tr, tc):
        return 0

    seen = {start}
    queue = deque([(sr, sc, 0)])
    while queue:
        r, c, d = queue.popleft()
        if (r, c) == (tr, tc):
            return d

        neighbors = (
            (r - 1, c, sim.south[r - 1, c] >= 0.5 if r > 0 else False),
            (r, c + 1, sim.east[r, c] >= 0.5 if c + 1 < sim.w else False),
            (r + 1, c, sim.south[r, c] >= 0.5 if r + 1 < sim.h else False),
            (r, c - 1, sim.east[r, c - 1] >= 0.5 if c > 0 else False),
        )
        for nr, nc, passable in neighbors:
            if (
                passable
                and (nr, nc) not in seen
                and not sim.debris[nr, nc]
            ):
                seen.add((nr, nc))
                queue.append((nr, nc, d + 1))
    return None


def test_spawn_relative_distance_guarantee():
    """Verify distance_metric='spawn' places victims strictly within [min_d, max_d] from firefighter spawn."""
    for seed in (50000, 50001, 50002, 50003, 50004):
        env = RescueEnv(
            victim_distance=6,
            min_victim_distance=3,
            distance_metric="spawn",
            reward_shaping=False,
        )
        try:
            env.reset(seed=seed)
            agent_pos = (env.sim.agents[0]["r"], env.sim.agents[0]["c"])
            victim_pos = (env.sim.victims[0]["r"], env.sim.victims[0]["c"])
            bfs_dist = compute_bfs_distance(env.sim, agent_pos, victim_pos)
            assert bfs_dist is not None, f"Seed {seed}: victim unreachable from spawn"
            assert 3 <= bfs_dist <= 6, f"Seed {seed}: spawn BFS distance {bfs_dist} not in [3, 6]"
        finally:
            env.close()


def test_legacy_exit_distance_preservation():
    """Verify distance_metric='exit' preserves legacy multi-exit BFS placement."""
    env_legacy = RescueEnv(
        victim_distance=6,
        min_victim_distance=3,
        distance_metric="exit",
        reward_shaping=False,
    )
    env_default = RescueEnv(
        victim_distance=6,
        min_victim_distance=3,
        reward_shaping=False,
    )
    try:
        for seed in (50000, 50001, 50002):
            env_legacy.reset(seed=seed)
            env_default.reset(seed=seed)
            v_leg = (env_legacy.sim.victims[0]["r"], env_legacy.sim.victims[0]["c"])
            v_def = (env_default.sim.victims[0]["r"], env_default.sim.victims[0]["c"])
            assert v_leg == v_def, f"Seed {seed}: default metric did not match legacy exit metric"
    finally:
        env_legacy.close()
        env_default.close()


def test_exterior_mask_when_not_carrying():
    """Verify that moving to exterior grass is masked out when carrying is False."""
    env = RescueEnv(prevent_exterior_wandering=True)
    try:
        env.reset(seed=42)
        # Seed 42 spawns at (13, 0) adjacent to exit doorway at col 0, with exterior grass at col -1
        mask = env.action_masks()
        # Action 4 is Left (dr=0, dc=-1) -> leads to (13, -1) which is exterior grass
        assert bool(mask[4]) is False, "Moving onto exterior grass when not carrying must be masked"
    finally:
        env.close()


def test_exterior_mask_when_carrying():
    """Verify that moving to exterior grass is legal when carrying is True (for rescue)."""
    env = RescueEnv(prevent_exterior_wandering=True)
    try:
        env.reset(seed=42)
        # Move right twice to reach victim at (13, 2), pickup, then move left back toward exit
        env.step(2)  # Right -> (13, 1)
        env.step(2)  # Right -> (13, 2)
        env.step(5)  # Pickup
        assert env.sim.victims[0]["carried_by"] == 0

        # Move back to (13, 0)
        env.step(4)  # Left -> (13, 1)
        env.step(4)  # Left -> (13, 0)

        mask = env.action_masks()
        # Now carrying=True; Action 4 (Left) towards exterior grass at (13, -1) MUST be legal!
        assert bool(mask[4]) is True, "Moving onto exterior grass when carrying must be legal for rescue"

        # Take action 4 to step onto grass -> triggers rescue
        obs, reward, terminated, truncated, info = env.step(4)
        assert terminated is True
        assert reward == 10.0
        assert info["success"] is True
    finally:
        env.close()


def test_anti_oscillation_penalty_training():
    """Verify that RewardShaper applies -0.02 penalty only on repeating 2-cell cycles."""
    shaper = RewardShaper(oscillation_penalty=0.02)
    victims = [{"id": 0, "carried_by": -1, "rescued": False}]

    # Sequence of distinct cells: (0, 0) -> (0, 1) -> (0, 2) -> (0, 3)
    r1 = shaper.compute_reward(victims, current_position=(0, 0))
    r2 = shaper.compute_reward(victims, current_position=(0, 1))
    r3 = shaper.compute_reward(victims, current_position=(0, 2))
    r4 = shaper.compute_reward(victims, current_position=(0, 3))
    # Each step only has step penalty (-0.001)
    assert abs(r4 - (-0.001)) < 1e-6

    # Now simulate 2-cell oscillation: A -> B -> A -> B
    shaper.reset()
    _ = shaper.compute_reward(victims, current_position=(5, 5))  # p0 = A
    _ = shaper.compute_reward(victims, current_position=(5, 6))  # p1 = B
    _ = shaper.compute_reward(victims, current_position=(5, 5))  # p2 = A
    r_osc = shaper.compute_reward(victims, current_position=(5, 6))  # p3 = B (2-cycle!)
    # Should have step penalty (-0.001) + oscillation penalty (-0.02) = -0.021
    assert abs(r_osc - (-0.021)) < 1e-6

    # Legitimate dead-end backtracking: A -> B -> A -> C
    shaper.reset()
    _ = shaper.compute_reward(victims, current_position=(5, 5))  # A
    _ = shaper.compute_reward(victims, current_position=(5, 6))  # B
    _ = shaper.compute_reward(victims, current_position=(5, 5))  # A (backtracked)
    r_backtrack = shaper.compute_reward(victims, current_position=(6, 5))  # C (new path!)
    # Should NOT have oscillation penalty
    assert abs(r_backtrack - (-0.001)) < 1e-6


def test_unshaped_eval_invariance():
    """Verify that evaluation (reward_shaping=False) has zero shaping or oscillation penalties."""
    env = RescueEnv(reward_shaping=False, max_steps=10)
    try:
        env.reset(seed=42)
        # Oscillate between (13, 0) and (13, 1)
        r1 = env.step(2)[1]  # Right -> (13, 1)
        r2 = env.step(4)[1]  # Left -> (13, 0)
        r3 = env.step(2)[1]  # Right -> (13, 1)
        r4 = env.step(4)[1]  # Left -> (13, 0)
        assert r1 == 0.0
        assert r2 == 0.0
        assert r3 == 0.0
        assert r4 == 0.0
    finally:
        env.close()
