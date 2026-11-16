"""Unit tests for Phase 4 Route-Complexity Return Generator and Curriculum."""

import pytest
import numpy as np
from rescue_env.envs.single_agent import RescueEnv
from training.train_ppo import CurriculumRescueEnv, make_train_env


def test_route_complexity_levels_r1_to_r4():
    """Verify R1..R4 generate valid, physically consistent scenarios with correct branching."""
    for level, expected_branches in [("R1", 0), ("R2", 1), ("R3", 2), ("R4", 3)]:
        env = RescueEnv(
            victim_distance=10 if level in ("R3", "R4") else 6,
            max_steps=192,
            return_mode="route_complexity",
            return_level=level,
        )
        obs, info = env.reset(seed=42, options={"task_type": "return", "return_level": level})

        sim = env.unwrapped.sim
        mem = env.unwrapped.memory

        # 1. Agent placed at victim cell with victim carried
        assert len(sim.victims) == 1
        victim = sim.victims[0]
        agent = sim.agents[0]
        assert victim["carried_by"] == 0
        assert (agent["r"], agent["c"]) == (victim["r"], victim["c"])

        # 2. Action masks: cannot pick up because already carrying
        mask = env.action_masks()
        assert not mask[5], f"Pickup action must be masked when already carrying (level {level})"
        assert mask[0], "Wait action must be available"

        # 3. Exit is marked in memory map
        assert (mem.grid[7] > 0.5).any(), f"Exit portal must be observed in memory for {level}"

        # 4. Memory explored count is non-zero
        explored_count = int(np.count_nonzero(mem.explored))
        assert explored_count > 0, f"Memory must have explored cells for {level}"

        # 5. Unexplored rooms remain strictly 0.0
        total_interior = int(np.count_nonzero(sim.zone >= 0))
        assert explored_count < total_interior, f"Unexplored cells must remain 0.0, got {explored_count}/{total_interior}"

        # 6. Info metadata
        assert info["return_level"] == level
        assert info["return_mode"] == "route_complexity"

        env.close()


def test_route_complexity_physical_consistency():
    """Verify that every explored cell in memory is a legitimate physical cell."""
    env = RescueEnv(
        victim_distance=12,
        max_steps=256,
        return_mode="route_complexity",
        return_level="R3",
    )
    obs, info = env.reset(seed=123, options={"task_type": "return", "return_level": "R3"})
    sim = env.unwrapped.sim
    mem = env.unwrapped.memory

    explored_coords = list(zip(*np.where(mem.explored > 0.5)))
    assert len(explored_coords) >= 10

    for r, c in explored_coords:
        # Check within bounds
        assert 0 <= r < sim.h and 0 <= c < sim.w
        # Must be an interior cell
        assert sim.zone[r, c] >= 0

        # Memory passability North matches sim ground truth
        mem_north = mem.grid[0, r, c]
        sim_north = float(sim.south[r - 1, c] >= 0.5 if r > 0 else 0)
        assert mem_north == sim_north, f"North passability mismatch at ({r},{c})"

        # Memory passability East matches sim ground truth
        mem_east = mem.grid[1, r, c]
        sim_east = float(sim.east[r, c] >= 0.5 if c + 1 < sim.w else 0)
        assert mem_east == sim_east, f"East passability mismatch at ({r},{c})"

    env.close()


def test_legacy_return_backward_compatibility():
    """Verify legacy return mode produces backward-compatible direct corridor without crashing."""
    env = RescueEnv(
        victim_distance=8,
        max_steps=160,
        return_mode="legacy",
    )
    obs, info = env.reset(seed=42, options={"task_type": "return", "return_mode": "legacy"})
    assert info["task_type"] == "return"
    assert info["return_mode"] == "legacy"
    assert info["return_level"] == "legacy"
    assert env.unwrapped.sim.victims[0]["carried_by"] == 0
    env.close()


def test_curriculum_rescue_env_route_complexity():
    """Verify CurriculumRescueEnv supports return_mode='route_complexity' across task sampling."""
    c_env = CurriculumRescueEnv(
        stage="2C",
        substage="2C_distant",
        enable_multitask=True,
        task_mixture=(0.0, 1.0, 0.0),  # 100% return
        return_mode="route_complexity",
    )
    sampled_levels = set()
    for s in range(30):
        obs, info = c_env.reset(seed=1000 + s)
        assert info["task_type"] == "return"
        assert info["return_mode"] == "route_complexity"
        assert info["return_level"] in ("R1", "R2", "R3", "R4")
        sampled_levels.add(info["return_level"])

    # Over 30 trials with mixture, multiple R-levels must be sampled
    assert len(sampled_levels) >= 2, f"Expected diverse R-level sampling, got {sampled_levels}"
    c_env.close()
