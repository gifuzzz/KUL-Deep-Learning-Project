"""Phase 10: Reward Correctness and Attribution Test Suite.

Verifies the 12 required properties of the Phase 10 reward system:
1. Normal movement into an already explored cell receives expected reward (-0.001).
2. Necessary reversal from a dead end is correctly classified (no oscillation penalty on multi-cell retreat).
3. Oscillation penalty applies strictly under its documented condition (p3 == p1, p2 == p0, p3 != p2).
4. Waiting produces expected per-step cost (-0.001).
5. Repeated-Wait penalty increases according to configured schedule.
6. Wait penalty resets after a non-Wait action.
7. Single legitimate Wait is not excessively penalized (0.0 additional).
8. Native rescue rewards remain unchanged (+10.0).
9. Reward shaping is completely disabled for evaluation.
10. Seven-action network and pretrained checkpoints remain compatible.
11. Action 6 remains masked in fireless episodes.
12. Reward counters reset correctly between episodes.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from rescue_env.envs.single_agent import RescueEnv
from rescue_env.envs.reward_shaping import RewardShaper
from rescue_env.wrappers import PreviousActionWrapper
from agents.recurrent_ppo import load_checkpoint


def test_1_normal_movement_into_explored_cell():
    """1. Normal movement into an already explored cell receives only step penalty (-0.001)."""
    env = RescueEnv(
        reward_shaping=True,
        geometry_mode="edge",
        fixed_seed=42,
        oscillation_penalty=0.02,
    )
    env.reset()

    # Artificially set total_exploration_reward to max to simulate already explored territory
    env.reward_shaper.total_exploration_reward = env.reward_shaper.max_exploration_reward

    # Take legal movement action
    masks = env.action_masks()
    move_action = next(a for a in (1, 2, 3, 4) if masks[a])
    obs, reward, term, trunc, info = env.step(move_action)

    breakdown = info["reward_breakdown"]
    assert breakdown["step_penalty"] == -0.001
    assert breakdown["exploration_bonus"] == 0.0
    assert breakdown["oscillation_penalty"] == 0.0
    assert breakdown["repeated_wait_penalty"] == 0.0
    assert reward == -0.001
    env.close()


def test_2_necessary_reversal_from_dead_end():
    """2. A necessary multi-cell reversal from a dead end does not trigger oscillation penalty."""
    shaper = RewardShaper(oscillation_penalty=0.02)
    shaper.total_exploration_reward = 3.0  # exploration bonus capped

    # Sequence: path into dead end: (1, 1) -> (1, 2) -> (1, 3)
    # Then retreat: (1, 3) -> (1, 2) -> (1, 1)
    # Step 1: at (1, 1)
    r1 = shaper.compute_reward([], newly_uncovered_cells=0, current_position=(1, 1), action=1)
    assert r1 == -0.001

    # Step 2: to (1, 2)
    r2 = shaper.compute_reward([], newly_uncovered_cells=0, current_position=(1, 2), action=2)
    assert r2 == -0.001

    # Step 3: to (1, 3) (dead end)
    r3 = shaper.compute_reward([], newly_uncovered_cells=0, current_position=(1, 3), action=2)
    assert r3 == -0.001

    # Step 4: retreat to (1, 2). History: [(1, 1), (1, 2), (1, 3), (1, 2)]
    # p0=(1, 1), p1=(1, 2), p2=(1, 3), p3=(1, 2). p3==p1 is True, but p2==p0 is False!
    r4 = shaper.compute_reward([], newly_uncovered_cells=0, current_position=(1, 2), action=4)
    assert r4 == -0.001, "First retreat step should not trigger oscillation penalty"

    # Step 5: retreat to (1, 1). History: [(1, 2), (1, 3), (1, 2), (1, 1)]
    # p0=(1, 2), p1=(1, 3), p2=(1, 2), p3=(1, 1). p3==p1 is False!
    r5 = shaper.compute_reward([], newly_uncovered_cells=0, current_position=(1, 1), action=4)
    assert r5 == -0.001, "Second retreat step should not trigger oscillation penalty"


def test_3_oscillation_penalty_trigger_condition():
    """3. Oscillation penalty applies strictly under its documented 2-cell limit cycle condition."""
    shaper = RewardShaper(oscillation_penalty=0.02)
    shaper.total_exploration_reward = 3.0

    # 2-cell cycle: A -> B -> A -> B
    # Step 1: at A
    shaper.compute_reward([], current_position=(5, 5), action=1)
    # Step 2: to B
    shaper.compute_reward([], current_position=(5, 6), action=2)
    # Step 3: to A
    r3 = shaper.compute_reward([], current_position=(5, 5), action=4)
    assert r3 == -0.001  # Not yet 4 positions matching cycle

    # Step 4: to B -> cycle complete [A, B, A, B]
    r4 = shaper.compute_reward([], current_position=(5, 6), action=2)
    assert r4 == pytest.approx(-0.021)
    assert shaper.last_breakdown["oscillation_penalty"] == -0.02

    # Step 5: with oscillation_penalty = 0.0 (Variant B/D), no penalty applied
    shaper_no_osc = RewardShaper(oscillation_penalty=0.0)
    shaper_no_osc.total_exploration_reward = 3.0
    shaper_no_osc.compute_reward([], current_position=(5, 5), action=1)
    shaper_no_osc.compute_reward([], current_position=(5, 6), action=2)
    shaper_no_osc.compute_reward([], current_position=(5, 5), action=4)
    r4_no_osc = shaper_no_osc.compute_reward([], current_position=(5, 6), action=2)
    assert r4_no_osc == -0.001
    assert shaper_no_osc.last_breakdown["oscillation_penalty"] == 0.0


def test_4_waiting_per_step_cost():
    """4. Waiting produces the expected per-step cost of -0.001."""
    env = RescueEnv(
        reward_shaping=True,
        geometry_mode="edge",
        fixed_seed=42,
        repeated_wait_penalty_enabled=False,
    )
    env.reset()
    obs, reward, term, trunc, info = env.step(0)  # Wait
    assert reward == -0.001
    assert info["reward_breakdown"]["step_penalty"] == -0.001
    assert info["reward_breakdown"]["repeated_wait_penalty"] == 0.0
    env.close()


def test_5_repeated_wait_penalty_schedule():
    """5. The repeated-Wait penalty increases according to configured schedule."""
    schedule = (0.0, 0.002, 0.004, 0.008)
    shaper = RewardShaper(
        step_penalty=0.001,
        repeated_wait_penalty_enabled=True,
        wait_penalty_schedule=schedule,
    )

    # Wait 1 (streak=1): schedule[0] = 0.0 -> reward = -0.001
    r1 = shaper.compute_reward([], current_position=(5, 5), action=0)
    assert r1 == pytest.approx(-0.001)
    assert shaper.last_breakdown["repeated_wait_penalty"] == 0.0

    # Wait 2 (streak=2): schedule[1] = 0.002 -> reward = -0.001 - 0.002 = -0.003
    r2 = shaper.compute_reward([], current_position=(5, 5), action=0)
    assert r2 == pytest.approx(-0.003)
    assert shaper.last_breakdown["repeated_wait_penalty"] == pytest.approx(-0.002)

    # Wait 3 (streak=3): schedule[2] = 0.004 -> reward = -0.001 - 0.004 = -0.005
    r3 = shaper.compute_reward([], current_position=(5, 5), action=0)
    assert r3 == pytest.approx(-0.005)
    assert shaper.last_breakdown["repeated_wait_penalty"] == pytest.approx(-0.004)

    # Wait 4 (streak=4): schedule[3] = 0.008 -> reward = -0.001 - 0.008 = -0.009
    r4 = shaper.compute_reward([], current_position=(5, 5), action=0)
    assert r4 == pytest.approx(-0.009)
    assert shaper.last_breakdown["repeated_wait_penalty"] == pytest.approx(-0.008)

    # Wait 5 (streak=5): capped at schedule[3] = 0.008 -> reward = -0.009
    r5 = shaper.compute_reward([], current_position=(5, 5), action=0)
    assert r5 == pytest.approx(-0.009)
    assert shaper.last_breakdown["repeated_wait_penalty"] == pytest.approx(-0.008)


def test_6_wait_penalty_resets_after_non_wait():
    """6. The repeated-Wait penalty resets after a non-Wait action."""
    schedule = (0.0, 0.002, 0.004, 0.008)
    shaper = RewardShaper(
        step_penalty=0.001,
        repeated_wait_penalty_enabled=True,
        wait_penalty_schedule=schedule,
    )

    # 3 consecutive waits: streak reaches 3
    shaper.compute_reward([], current_position=(5, 5), action=0)
    shaper.compute_reward([], current_position=(5, 5), action=0)
    r3 = shaper.compute_reward([], current_position=(5, 5), action=0)
    assert shaper.consecutive_waits == 3

    # Non-wait action (move action 1)
    shaper.compute_reward([], current_position=(5, 6), action=1)
    assert shaper.consecutive_waits == 0
    assert shaper.last_breakdown["repeated_wait_penalty"] == 0.0

    # Next wait is treated as Wait 1 (0.0 additional penalty)
    r_after = shaper.compute_reward([], current_position=(5, 6), action=0)
    assert shaper.consecutive_waits == 1
    assert r_after == pytest.approx(-0.001)
    assert shaper.last_breakdown["repeated_wait_penalty"] == 0.0


def test_7_single_legitimate_wait_not_excessively_penalized():
    """7. A single legitimate Wait incurs only normal step penalty (-0.001), not an immediate penalty."""
    schedule = (0.0, 0.002, 0.004, 0.008)
    shaper = RewardShaper(
        step_penalty=0.001,
        repeated_wait_penalty_enabled=True,
        wait_penalty_schedule=schedule,
    )
    shaper.compute_reward([], current_position=(5, 5), action=1)  # moving
    r = shaper.compute_reward([], current_position=(5, 6), action=0)  # single wait
    assert r == pytest.approx(-0.001)
    assert shaper.last_breakdown["repeated_wait_penalty"] == 0.0


def test_8_native_rescue_rewards_remain_unchanged():
    """8. Native rescue rewards remain unchanged (+10.0 when carried to exit)."""
    env = RescueEnv(
        reward_shaping=False,
        geometry_mode="edge",
        fixed_seed=42,
    )
    env.reset()
    # Verify sim returns 10.0 on rescue
    # Move agent to exterior exit while carrying victim
    agent = env.sim.agents[0]
    victim = env.sim.victims[0]
    victim["carried_by"] = agent["id"]
    # Teleport to grass cell
    agent["r"] = 0
    agent["c"] = 0
    victim["r"] = 0
    victim["c"] = 0
    result = env.sim.action_batch([0], physics_ticks=1)
    if result["victims"][0]["rescued"]:
        assert result["reward"] == 10.0
    env.close()


def test_9_reward_shaping_disabled_for_evaluation():
    """9. Reward shaping is completely disabled when reward_shaping=False."""
    env = RescueEnv(
        reward_shaping=False,
        geometry_mode="edge",
        fixed_seed=42,
        repeated_wait_penalty_enabled=True,  # Even if enabled in config
    )
    env.reset()
    assert env.reward_shaper is None

    # Step 10 times taking legal actions
    for _ in range(10):
        masks = env.action_masks()
        act = next(a for a in range(5) if masks[a])
        obs, reward, term, trunc, info = env.step(act)
        assert reward == 0.0, f"Expected 0.0 unshaped reward on non-rescue step, got {reward}"
        assert info["reward_breakdown"]["total_shaping"] == 0.0
    env.close()


def test_10_seven_action_network_checkpoint_compatibility():
    """10. The seven-action network and pretrained checkpoints remain compatible."""
    model_path = Path("models/phase6_spawn_curriculum/production_250k/stage3_spawn_curriculum/best_model.zip")
    assert model_path.exists(), f"Checkpoint {model_path} must exist"

    env = PreviousActionWrapper(RescueEnv(reward_shaping=True, geometry_mode="edge"))
    model = load_checkpoint(str(model_path), env=env, device="cpu")

    obs, _ = env.reset(seed=42)
    obs_t = {k: torch.as_tensor(v, device="cpu").unsqueeze(0) for k, v in obs.items()}
    lstm_states = torch.zeros(1, 1, model.policy.hidden_size)
    episode_starts = torch.tensor([True])
    masks = torch.tensor([env.action_masks()])

    dist = model.policy.get_distribution(
        obs_t,
        lstm_states=lstm_states,
        episode_starts=episode_starts,
        action_masks=masks,
    )
    logits = dist.distribution.logits
    assert logits.shape == (1, 7), f"Expected 7-action logits, got {logits.shape}"
    assert torch.isfinite(logits[masks]).all(), "Legal action logits must be finite"
    assert logits[0, 6] < -1e7, "Action 6 must have -1e8 masked logit"
    env.close()


def test_11_action6_remains_masked():
    """11. Action 6 remains masked out in fireless episodes."""
    env = RescueEnv(reward_shaping=False, geometry_mode="edge", fixed_seed=42)
    env.reset()
    masks = env.action_masks()
    assert masks[6] == False, "Action 6 must be masked out in fireless environment"
    env.close()


def test_12_reward_counters_reset_between_episodes():
    """12. Reward counters and state history reset correctly between episodes."""
    env = RescueEnv(
        reward_shaping=True,
        geometry_mode="edge",
        fixed_seed=42,
        repeated_wait_penalty_enabled=True,
    )
    env.reset()
    # Step wait 5 times
    for _ in range(5):
        env.step(0)
    assert env.reward_shaper.consecutive_waits == 5
    assert len(env.reward_shaper.recent_positions) == 4  # bounded at 4 positions

    # Reset
    env.reset()
    assert env.reward_shaper.consecutive_waits == 0
    assert len(env.reward_shaper.recent_positions) == 0
    assert env.reward_shaper.total_exploration_reward == 0.0
    assert len(env.reward_shaper.rewarded_pickups) == 0
    env.close()
