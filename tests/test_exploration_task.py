import pytest
import numpy as np

from rescue_env.envs.single_agent import RescueEnv
from training.train_ppo import CurriculumRescueEnv


def test_exploration_task_initialization():
    """Verify exploration_only task has 0 victims, zero victim channel, and disabled pickup/water."""
    env = RescueEnv(task_type="exploration_only", geometry_mode="edge", reward_shaping=True)
    obs, info = env.reset(seed=123)

    assert env.task_type == "exploration_only"
    assert len(env.sim.victims) == 0
    assert float(obs["grid"][6].sum()) == 0.0
    mask = env.action_masks()
    assert bool(mask[0]) is True   # Wait legal
    assert bool(mask[5]) is False  # Pickup strictly disabled
    assert bool(mask[6]) is False  # Water hose strictly disabled in fireless
    assert info["task_type"] == "exploration_only"
    assert info["reachable_interior_cells"] > 500
    assert info["reachable_edges"] > 1000
    assert len(env._reachable_rooms) >= 4
    assert 0.0 < info["observed_coverage"] < 0.15
    env.close()


def test_exploration_coverage_and_edge_accounting():
    """Verify reachable graph calculation and legit observation pipeline tracking."""
    env = RescueEnv(task_type="exploration_only", geometry_mode="edge", reward_shaping=True)
    obs, info = env.reset(seed=456)

    initial_obs_count = len(env._observed_reachable_cells)
    initial_edge_count = len(env._discovered_edges)
    assert initial_obs_count > 0
    assert initial_edge_count > 0

    # Moving into a new cell reveals new cells and traversable edges
    mask = env.action_masks()
    move_actions = [a for a in (1, 2, 3, 4) if mask[a]]
    if move_actions:
        obs, reward, terminated, truncated, info = env.step(move_actions[0])
        assert info["steps"] == 1
        assert len(env._observed_reachable_cells) >= initial_obs_count
        assert len(env._discovered_edges) >= initial_edge_count
        assert info["visited_coverage"] > 0.0
    env.close()


def test_exploration_rewards_and_capping():
    """Verify +0.05 per cell, +0.02 per edge, -0.001 per step, 0 on revisit, and cap at +6.0."""
    env = RescueEnv(
        task_type="exploration_only",
        geometry_mode="edge",
        reward_shaping=True,
        max_exploration_reward=1.0,  # small cap for testing
        exploration_cell_bonus=0.05,
        exploration_edge_bonus=0.02,
        exploration_step_penalty=0.001,
        oscillation_penalty=0.02,
    )
    obs, info = env.reset(seed=789)

    # 1. Wait action yields -0.001 step penalty, 0 cell/edge bonus
    obs, reward, terminated, truncated, info = env.step(0)
    assert pytest.approx(reward, abs=1e-5) == -0.001
    assert env._total_exploration_reward == 0.0

    # 2. Repeated wait does not farm reward
    obs, reward, terminated, truncated, info = env.step(0)
    assert pytest.approx(reward, abs=1e-5) == -0.001
    assert env._total_exploration_reward == 0.0

    # 3. Movement rewards and cap enforcement
    total_pos_reward = 0.0
    for _ in range(50):
        mask = env.action_masks()
        move_actions = [a for a in (1, 2, 3, 4) if mask[a]]
        action = move_actions[0] if move_actions else 0
        obs, r, term, trunc, info = env.step(action)
        if r > 0:
            total_pos_reward += (r + 0.001)  # remove step penalty
        if term or trunc:
            break

    assert env._total_exploration_reward <= 1.0 + 1e-5
    assert info["exploration_reward"] <= 1.0 + 1e-5
    env.close()


def test_two_cell_oscillation_penalty():
    """Verify -0.02 penalty on 2-cell oscillation (A <-> B <-> A <-> B)."""
    env = RescueEnv(
        task_type="exploration_only",
        geometry_mode="edge",
        reward_shaping=True,
        oscillation_penalty=0.02,
    )
    obs, info = env.reset(seed=101)

    # Find two adjacent passable cells to oscillate between
    mask = env.action_masks()
    # East (2) and West (4), or North (1) and South (3)
    opp_pair = None
    if mask[1] and mask[3]:
        opp_pair = (1, 3)
    elif mask[2] and mask[4]:
        opp_pair = (2, 4)
    else:
        # Step once to enter hallway
        for a in (1, 2, 3, 4):
            if mask[a]:
                env.step(a)
                break
        mask = env.action_masks()
        if mask[1] and mask[3]:
            opp_pair = (1, 3)
        elif mask[2] and mask[4]:
            opp_pair = (2, 4)

    if opp_pair:
        a1, a2 = opp_pair
        # Step A -> B -> A -> B
        env.step(a1)
        env.step(a2)
        env.step(a1)
        obs, r4, term, trunc, info = env.step(a2)
        # 4th step completes A <-> B <-> A <-> B oscillation cycle
        # r4 should include -0.02 oscillation penalty
        assert r4 <= -0.02
        assert info["oscillation_percentage"] > 0.0
    env.close()


def test_exploration_completion_threshold():
    """Verify episode terminates with success when coverage reaches target."""
    env = RescueEnv(
        task_type="exploration_only",
        geometry_mode="edge",
        reward_shaping=True,
        coverage_target=0.03,  # reachable threshold
    )
    obs, info = env.reset(seed=42)

    terminated_flag = False
    rng = np.random.default_rng(42)
    for step in range(50):
        mask = env.action_masks()
        moves = [a for a in (1, 2, 3, 4) if mask[a]]
        action = int(rng.choice(moves)) if moves else 0
        obs, r, term, trunc, info = env.step(action)
        if term:
            terminated_flag = True
            assert info["success"] is True
            assert info["exploration_success"] is True
            assert info["observed_coverage"] >= 0.03
            break

    assert terminated_flag, "Should have reached 0.03 coverage within 50 steps"
    env.close()


def test_unshaped_evaluation_mode():
    """Verify that when reward_shaping=False, reward is strictly 0.0 in exploration_only."""
    env = RescueEnv(
        task_type="exploration_only",
        geometry_mode="edge",
        reward_shaping=False,
    )
    obs, info = env.reset(seed=303)

    for step in range(10):
        mask = env.action_masks()
        moves = [a for a in (1, 2, 3, 4) if mask[a]]
        action = moves[0] if moves else 0
        obs, reward, term, trunc, info = env.step(action)
        assert reward == 0.0, f"Expected 0.0 unshaped reward, got {reward}"
    env.close()


def test_rescue_tasks_unaffected():
    """Verify that regular full rescue task is completely unaffected by exploration changes."""
    env = RescueEnv(
        task_type="full",
        geometry_mode="edge",
        reward_shaping=False,
        max_steps=64,
    )
    obs, info = env.reset(seed=404)

    assert env.task_type == "full"
    assert len(env.sim.victims) == 1
    assert float(obs["grid"][6].sum()) > 0.0

    # Step wait
    obs, reward, term, trunc, info = env.step(0)
    assert reward == 0.0
    assert term is False
    assert info["task_type"] == "full"
    env.close()


def test_curriculum_4mixture_sampling():
    """Verify CurriculumRescueEnv correctly samples 4-tuple mixture."""
    mixture = (0.24, 0.21, 0.15, 0.40)
    env = CurriculumRescueEnv(
        stage=3,
        enable_multitask=True,
        task_mixture=mixture,
        geometry_mode="edge",
    )

    counts = {"full": 0, "return": 0, "discovery": 0, "exploration_only": 0}
    for i in range(100):
        obs, info = env.reset(seed=1000 + i)
        counts[env.task_type] += 1

    assert counts["full"] > 10
    assert counts["return"] > 10
    assert counts["discovery"] > 5
    assert counts["exploration_only"] > 25
    assert sum(counts.values()) == 100
    env.close()
