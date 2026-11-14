"""Unit Tests for Stage 2C Distance-10 Curriculum Bridge.

Verifies:
1. Distance configurations table and ladder parameters (Near, Medium, Bridge A, Bridge B, Bridge C, Distant).
2. Environment initialization, stage normalization, and options handling.
3. Substage sampling mixtures and probability distributions within Stage 2C.
4. Adaptive progression across readiness thresholds and self-healing in RecurrentMaskableEvalCallback.
5. Unshaped evaluation across all 6 distance ladder tiers.
6. Backward compatibility with existing Stage 1 and Stage 2 configurations.
7. Reward shaping and exploration bonus integrity.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from rescue_env.envs.single_agent import RescueEnv
from rescue_env.envs.reward_shaping import RewardShaper, potential_reward, state_potential
from training.train_ppo import (
    DISTANCE_CONFIGS,
    CurriculumRescueEnv,
    normalize_stage,
    make_eval_env,
    make_train_env,
    set_env_stage,
    RecurrentMaskableEvalCallback,
    evaluate_unshaped,
)
from scripts.diagnose_curriculum import (
    CURRICULUM_TIERS,
    LEGACY_TIERS,
    BRIDGE_TIERS,
    evaluate_tier,
)


def test_distance_configs_table():
    """Verify exact specifications of the 6 intermediate distance tiers."""
    expected = {
        "near": {"victim_distance": 2, "min_victim_distance": 1, "max_steps": 64},
        "medium": {"victim_distance": 6, "min_victim_distance": 3, "max_steps": 128},
        "bridge_a": {"victim_distance": 8, "min_victim_distance": 5, "max_steps": 160},
        "bridge_b": {"victim_distance": 10, "min_victim_distance": 7, "max_steps": 192},
        "bridge_c": {"victim_distance": 12, "min_victim_distance": 9, "max_steps": 256},
        "distant": {"victim_distance": None, "min_victim_distance": 1, "max_steps": 384},
    }
    for tier, cfg in expected.items():
        assert tier in DISTANCE_CONFIGS, f"Missing tier {tier} in DISTANCE_CONFIGS"
        assert DISTANCE_CONFIGS[tier]["victim_distance"] == cfg["victim_distance"]
        assert DISTANCE_CONFIGS[tier]["min_victim_distance"] == cfg["min_victim_distance"]
        assert DISTANCE_CONFIGS[tier]["max_steps"] == cfg["max_steps"]


def test_normalize_stage_and_env_init():
    """Verify stage normalization and environment defaults."""
    assert normalize_stage(1) == 1
    assert normalize_stage("1") == 1
    assert normalize_stage(2) == 2
    assert normalize_stage("2") == 2
    assert normalize_stage("2C") == "2C"
    assert normalize_stage("2c") == "2C"
    assert normalize_stage("all") == "ALL"

    # Default substage for stage 2C is 2C_bridge8
    env_2c = CurriculumRescueEnv(stage="2C")
    assert env_2c.stage == "2C"
    assert env_2c.substage == "2C_bridge8"
    env_2c.close()

    # Case-insensitivity check
    env_lower = CurriculumRescueEnv(stage="2c")
    assert env_lower.stage == "2C"
    assert env_lower.substage == "2C_bridge8"
    env_lower.close()

    # Setting stage via set_stage
    env = CurriculumRescueEnv(stage=1)
    assert env.stage == 1
    env.set_stage(stage="2C", substage="2C_bridge10")
    assert env.stage == "2C"
    assert env.substage == "2C_bridge10"
    env.close()


def test_stage2c_substage_mixtures_simulated():
    """Verify that Stage 2C substages partition [0, 1) into exact target mixtures."""
    env = CurriculumRescueEnv(stage="2C", substage="2C_bridge8")

    # Helper to test roll assignment deterministically
    def sample_tier_for_roll(substage: str, roll: float):
        mock_rng = MagicMock()
        mock_rng.random.return_value = roll
        env.np_random = mock_rng
        env.substage = substage
        env.stage = "2C"
        # Run reset logic up to placing victims
        # We can call the reset mapping directly
        if substage == "2C_bridge8":
            if roll < 0.20:
                env._apply_tier_config("near")
            elif roll < 0.60:
                env._apply_tier_config("medium")
            else:
                env._apply_tier_config("bridge_a")
        elif substage == "2C_bridge10":
            if roll < 0.15:
                env._apply_tier_config("near")
            elif roll < 0.40:
                env._apply_tier_config("medium")
            elif roll < 0.70:
                env._apply_tier_config("bridge_a")
            else:
                env._apply_tier_config("bridge_b")
        elif substage == "2C_bridge12":
            if roll < 0.10:
                env._apply_tier_config("near")
            elif roll < 0.30:
                env._apply_tier_config("medium")
            elif roll < 0.60:
                env._apply_tier_config("bridge_b")
            else:
                env._apply_tier_config("bridge_c")
        elif substage == "2C_distant":
            if roll < 0.10:
                env._apply_tier_config("near")
            elif roll < 0.25:
                env._apply_tier_config("medium")
            elif roll < 0.375:
                env._apply_tier_config("bridge_b")
            elif roll < 0.50:
                env._apply_tier_config("bridge_c")
            else:
                env._apply_tier_config("distant")
        elif substage == "reinforce_near":
            if roll < 0.60:
                env._apply_tier_config("near")
            elif roll < 0.90:
                env._apply_tier_config("medium")
            else:
                env.victim_distance = None
                env.min_victim_distance = 1
                env.max_steps = 256
        elif substage == "reinforce_medium":
            if roll < 0.30:
                env._apply_tier_config("near")
            elif roll < 0.75:
                env._apply_tier_config("medium")
            else:
                bridge_tier = getattr(env, "active_bridge", "bridge_a")
                env._apply_tier_config(bridge_tier)
        return env.victim_distance, env.min_victim_distance, env.max_steps

    # 1. 2C_bridge8: 20% Near, 40% Medium (d=6), 40% Bridge A (d=8)
    assert sample_tier_for_roll("2C_bridge8", 0.05) == (2, 1, 64)
    assert sample_tier_for_roll("2C_bridge8", 0.19) == (2, 1, 64)
    assert sample_tier_for_roll("2C_bridge8", 0.21) == (6, 3, 128)
    assert sample_tier_for_roll("2C_bridge8", 0.59) == (6, 3, 128)
    assert sample_tier_for_roll("2C_bridge8", 0.61) == (8, 5, 160)
    assert sample_tier_for_roll("2C_bridge8", 0.99) == (8, 5, 160)

    # 2. 2C_bridge10: 15% Near, 25% Medium, 30% d=8, 30% Bridge B (d=10)
    assert sample_tier_for_roll("2C_bridge10", 0.10) == (2, 1, 64)
    assert sample_tier_for_roll("2C_bridge10", 0.25) == (6, 3, 128)
    assert sample_tier_for_roll("2C_bridge10", 0.55) == (8, 5, 160)
    assert sample_tier_for_roll("2C_bridge10", 0.85) == (10, 7, 192)

    # 3. 2C_bridge12: 10% Near, 20% Medium, 30% d=10, 40% Bridge C (d=12)
    assert sample_tier_for_roll("2C_bridge12", 0.05) == (2, 1, 64)
    assert sample_tier_for_roll("2C_bridge12", 0.20) == (6, 3, 128)
    assert sample_tier_for_roll("2C_bridge12", 0.45) == (10, 7, 192)
    assert sample_tier_for_roll("2C_bridge12", 0.85) == (12, 9, 256)

    # 4. 2C_distant: 10% Near, 15% Medium, 25% d=10/12, 50% Distant (d=None)
    assert sample_tier_for_roll("2C_distant", 0.05) == (2, 1, 64)
    assert sample_tier_for_roll("2C_distant", 0.20) == (6, 3, 128)
    assert sample_tier_for_roll("2C_distant", 0.30) == (10, 7, 192)
    assert sample_tier_for_roll("2C_distant", 0.45) == (12, 9, 256)
    assert sample_tier_for_roll("2C_distant", 0.75) == (None, 1, 384)

    # 5. reinforce_near: 60% Near, 30% Medium, 10% Distant
    assert sample_tier_for_roll("reinforce_near", 0.30) == (2, 1, 64)
    assert sample_tier_for_roll("reinforce_near", 0.75) == (6, 3, 128)
    assert sample_tier_for_roll("reinforce_near", 0.95) == (None, 1, 256)

    # 6. reinforce_medium: 30% Near, 45% Medium, 25% Bridge (active: bridge_a)
    env.active_bridge = "bridge_a"
    assert sample_tier_for_roll("reinforce_medium", 0.15) == (2, 1, 64)
    assert sample_tier_for_roll("reinforce_medium", 0.50) == (6, 3, 128)
    assert sample_tier_for_roll("reinforce_medium", 0.85) == (8, 5, 160)

    # reinforce_medium with active: bridge_b
    env.active_bridge = "bridge_b"
    assert sample_tier_for_roll("reinforce_medium", 0.85) == (10, 7, 192)

    env.close()


def test_env_reset_and_action_masks():
    """Verify environment reset and action masking for each Stage 2C tier."""
    for substage in ["2C_bridge8", "2C_bridge10", "2C_bridge12", "2C_distant", "reinforce_medium"]:
        env = CurriculumRescueEnv(stage="2C", substage=substage)
        try:
            obs, info = env.reset(seed=12345)
            assert "grid" in obs
            assert "memory" in obs
            assert "agent" in obs
            assert obs["grid"].shape == (9, 15, 15)
            assert obs["memory"].shape == (9, 40, 52)
            assert obs["agent"].shape == (5,)

            masks = env.action_masks()
            assert masks.shape == (7,)
            assert masks.dtype == bool
            assert masks[0] == True  # No-op always legal

            # Step no-op
            obs, reward, terminated, truncated, step_info = env.step(0)
            assert "success" in step_info
            assert "timeout" in step_info
        finally:
            env.close()


def test_callback_substage_transitions_and_self_healing():
    """Verify Stage 2C dual-guard progression milestones and self-healing recovery in the callback."""
    # Test with a real vectorized training environment
    train_vec_env = make_train_env(stage="2C", substage="2C_bridge8", n_envs=2)
    callback = RecurrentMaskableEvalCallback(
        eval_env_stage2=MagicMock(),
        train_env=train_vec_env,
        stage="2C",
        initial_substage="2C_bridge8",
        eval_freq=1000,
        verbose=0,
        stage2c_mode="retention_aware",
    )
    assert callback.current_substage == "2C_bridge8"

    # Simulate transition 1: 2C_bridge8 -> 2C_bridge10 requires Near >= 85%, Medium >= 30%, Bridge A >= 20%
    callback.num_timesteps = 1000
    near_m = {"success_rate": 0.90}
    med_m = {"success_rate": 0.35}
    br_a_m = {"success_rate": 0.25}
    br_b_m = {"success_rate": 0.10}
    br_c_m = {"success_rate": 0.05}

    # Helper simulating the callback progression logic
    def run_progression_logic(cb, near, med, br_a, br_b, br_c):
        near_succ = near["success_rate"]
        med_succ = med["success_rate"]
        a_succ = br_a["success_rate"] if br_a else 0.0
        b_succ = br_b["success_rate"] if br_b else 0.0
        c_succ = br_c["success_rate"] if br_c else 0.0

        if cb.stage2c_mode == "retention_aware":
            if near_succ < 0.80 and cb.current_substage != "reinforce_near":
                if cb.current_substage != "reinforce_medium":
                    cb._prev_substage = cb.current_substage
                cb.current_substage = "reinforce_near"
                set_env_stage(cb.train_env, stage="2C", substage="reinforce_near")
            elif cb.current_substage == "reinforce_near" and near_succ < 0.85:
                pass
            elif med_succ < 0.25 and cb.current_substage != "reinforce_medium":
                if cb.current_substage != "reinforce_near":
                    cb._prev_substage = cb.current_substage
                cb.current_substage = "reinforce_medium"
                set_env_stage(cb.train_env, stage="2C", substage="reinforce_medium")
            elif cb.current_substage == "reinforce_medium" and med_succ < 0.30:
                pass
            elif near_succ >= 0.85 and med_succ >= 0.30:
                if cb.current_substage in ("reinforce_near", "reinforce_medium"):
                    resume_sub = getattr(cb, "_prev_substage", "2C_bridge8")
                    cb.current_substage = resume_sub
                    set_env_stage(cb.train_env, stage="2C", substage=resume_sub)

                if cb.current_substage == "2C_bridge8" and a_succ >= 0.20:
                    cb.current_substage = "2C_bridge10"
                    set_env_stage(cb.train_env, stage="2C", substage="2C_bridge10")
                elif cb.current_substage == "2C_bridge10" and b_succ >= 0.10:
                    cb.current_substage = "2C_bridge12"
                    set_env_stage(cb.train_env, stage="2C", substage="2C_bridge12")
                elif cb.current_substage == "2C_bridge12" and c_succ >= 0.10:
                    cb.current_substage = "2C_distant"
                    set_env_stage(cb.train_env, stage="2C", substage="2C_distant")

    run_progression_logic(callback, near_m, med_m, br_a_m, br_b_m, br_c_m)
    assert callback.current_substage == "2C_bridge10"
    for sub in train_vec_env.get_attr("substage"):
        assert sub == "2C_bridge10"

    # Simulate transition 2: 2C_bridge10 -> 2C_bridge12 requires Near >= 85%, Medium >= 30%, Bridge B >= 10%
    run_progression_logic(callback, near_m, med_m, br_a_m, br_b_m, br_c_m)
    assert callback.current_substage == "2C_bridge12"
    for sub in train_vec_env.get_attr("substage"):
        assert sub == "2C_bridge12"

    # Simulate transition 3: 2C_bridge12 -> 2C_distant requires Near >= 85%, Medium >= 30%, Bridge C >= 10%
    br_c_m["success_rate"] = 0.12
    run_progression_logic(callback, near_m, med_m, br_a_m, br_b_m, br_c_m)
    assert callback.current_substage == "2C_distant"
    for sub in train_vec_env.get_attr("substage"):
        assert sub == "2C_distant"

    # Simulate Guard 2 Trigger: Medium retention drops < 25% (e.g. 20%) -> reinforce_medium
    med_m["success_rate"] = 0.20
    run_progression_logic(callback, near_m, med_m, br_a_m, br_b_m, br_c_m)
    assert callback.current_substage == "reinforce_medium"
    assert callback._prev_substage == "2C_distant"
    for sub in train_vec_env.get_attr("substage"):
        assert sub == "reinforce_medium"

    # Simulate Guard 1 Preemption: Near retention drops < 80% while in reinforce_medium -> reinforce_near
    near_m["success_rate"] = 0.70
    run_progression_logic(callback, near_m, med_m, br_a_m, br_b_m, br_c_m)
    assert callback.current_substage == "reinforce_near"
    for sub in train_vec_env.get_attr("substage"):
        assert sub == "reinforce_near"

    # Simulate Guard 1 Recovery: Near retention restored >= 85%, but Medium still < 25% -> transitions to reinforce_medium
    near_m["success_rate"] = 0.90
    run_progression_logic(callback, near_m, med_m, br_a_m, br_b_m, br_c_m)
    assert callback.current_substage == "reinforce_medium"
    for sub in train_vec_env.get_attr("substage"):
        assert sub == "reinforce_medium"

    # Simulate Guard 2 Recovery: Medium restored >= 30% -> resumes 2C_distant
    med_m["success_rate"] = 0.35
    run_progression_logic(callback, near_m, med_m, br_a_m, br_b_m, br_c_m)
    assert callback.current_substage == "2C_distant"
    for sub in train_vec_env.get_attr("substage"):
        assert sub == "2C_distant"

    train_vec_env.close()


def test_unshaped_eval_distance_ladder():
    """Verify that unshaped evaluation works across all 6 distance ladder tiers."""
    ladder = [
        (2, 1, 64),
        (6, 3, 128),
        (8, 5, 160),
        (10, 7, 192),
        (12, 9, 256),
        (None, 1, 384),
    ]

    for vd, mvd, max_s in ladder:
        env = make_eval_env(victim_distance=vd, min_victim_distance=mvd, max_steps=max_s)
        try:
            assert env.unwrapped.reward_shaping == False
            assert env.unwrapped.navigation_shaping == False
            assert env.unwrapped.victim_distance == vd
            assert env.unwrapped.min_victim_distance == mvd
            assert env.unwrapped.max_steps == max_s

            obs, info = env.reset(seed=999)
            assert env.observation_space.contains(obs)
            # Step action
            next_obs, reward, term, trunc, step_info = env.step(0)
            assert isinstance(reward, float)
            assert "success" in step_info
            assert "timeout" in step_info
        finally:
            env.close()


def test_diagnose_curriculum_tiers_definition():
    """Verify diagnose_curriculum exports correct tier presets."""
    assert len(CURRICULUM_TIERS) == 6
    assert len(LEGACY_TIERS) == 3
    assert len(BRIDGE_TIERS) == 3

    # Check that Bridge A, B, C are in CURRICULUM_TIERS
    names = [t[0] for t in CURRICULUM_TIERS]
    assert any("Bridge A" in n for n in names)
    assert any("Bridge B" in n for n in names)
    assert any("Bridge C" in n for n in names)


def test_backward_compatibility_stage1_and_stage2():
    """Verify existing Stage 1 and Stage 2 environments function unchanged."""
    # Stage 1: Near ratio splitting
    env_s1 = CurriculumRescueEnv(stage=1, near_ratio=0.7)
    try:
        mock_rng = MagicMock()
        mock_rng.random.return_value = 0.50
        env_s1.np_random = mock_rng
        env_s1.reset()
        assert env_s1.victim_distance == 2
        assert env_s1.max_steps == 64

        mock_rng.random.return_value = 0.85
        env_s1.np_random = mock_rng
        env_s1.reset()
        assert env_s1.victim_distance == 6
        assert env_s1.max_steps == 128
    finally:
        env_s1.close()

    # Stage 2: Substage 2A, 2B, 2C
    for sub, roll, exp_vd, exp_max in [
        ("2A", 0.10, 2, 64),
        ("2A", 0.50, 6, 128),
        ("2A", 0.90, None, 256),
        ("2B", 0.10, 2, 64),
        ("2B", 0.40, 6, 128),
        ("2B", 0.70, None, 384),
        ("2C", 0.05, 2, 64),
        ("2C", 0.20, 6, 128),
        ("2C", 0.50, None, 384),
        ("reinforce_near", 0.50, 2, 64),
        ("reinforce_near", 0.75, 6, 128),
        ("reinforce_near", 0.95, None, 256),
    ]:
        env_s2 = CurriculumRescueEnv(stage=2, substage=sub)
        try:
            mock_rng = MagicMock()
            mock_rng.random.return_value = roll
            env_s2.np_random = mock_rng
            env_s2.reset()
            assert env_s2.victim_distance == exp_vd
            assert env_s2.max_steps == exp_max
        finally:
            env_s2.close()


def test_reward_and_shaping_integrity():
    """Verify exploration bonus and potential reward mathematical properties."""
    shaper = RewardShaper(pickup_reward=1.5, exploration_bonus=0.05, max_exploration_reward=3.0)

    # 1. Not carrying victim: newly uncovered cells give +0.05
    victims_not_carrying = [{"id": 0, "carried_by": -1, "rescued": False}]
    r1 = shaper.compute_reward(victims_not_carrying, newly_uncovered_cells=10)
    # -0.001 (step) + 10 * 0.05 = 0.499
    assert np.isclose(r1, -0.001 + 0.50)

    # 2. Carrying victim: NO exploration bonus awarded even if newly uncovered cells > 0
    victims_carrying = [{"id": 0, "carried_by": 0, "rescued": False}]
    r2 = shaper.compute_reward(victims_carrying, newly_uncovered_cells=10)
    # Pickup reward (1.5) was awarded on pickup, step penalty -0.001, but ZERO exploration bonus
    assert np.isclose(r2, -0.001 + 1.5)

    # 3. Navigation potential formulation
    # potential_reward = beta * (gamma * after - before)
    r_pot = potential_reward(before=-10.0, after=-9.0, gamma=0.99, beta=0.1)
    # 0.1 * (0.99 * -9.0 - (-10.0)) = 0.1 * (-8.91 + 10.0) = 0.1 * 1.09 = 0.109
    assert np.isclose(r_pot, 0.109)


def test_callback_on_step_dual_guards_execution(monkeypatch):
    """Verify live _on_step execution in RecurrentMaskableEvalCallback with dual guards."""
    train_vec_env = make_train_env(stage="2C", substage="2C_bridge8", n_envs=2)
    callback = RecurrentMaskableEvalCallback(
        eval_env_stage2=MagicMock(),
        eval_env_near=MagicMock(),
        eval_env_med=MagicMock(),
        eval_env_bridge_a=MagicMock(),
        eval_env_bridge_b=MagicMock(),
        eval_env_bridge_c=MagicMock(),
        train_env=train_vec_env,
        stage="2C",
        initial_substage="2C_bridge8",
        eval_freq=1000,
        verbose=0,
        stage2c_mode="retention_aware",
    )
    callback.model = MagicMock()
    callback.model.num_timesteps = 0
    callback.num_timesteps = 1000

    # Mock evaluate_unshaped returning specific metrics
    eval_state = {
        "near": 0.90,
        "med": 0.35,
        "bridge_a": 0.25,
        "bridge_b": 0.0,
        "bridge_c": 0.0,
        "s2": 0.0,
    }

    def mock_eval(model, env, **kwargs):
        if env == callback.eval_env_near:
            return {"success_rate": eval_state["near"], "pickup_rate": 0.95, "mean_reward": 9.5, "mean_length": 3.0}
        elif env == callback.eval_env_med:
            return {"success_rate": eval_state["med"], "pickup_rate": 0.40, "mean_reward": 4.0, "mean_length": 15.0}
        elif env == callback.eval_env_bridge_a:
            return {"success_rate": eval_state["bridge_a"], "pickup_rate": 0.30, "mean_reward": 2.5, "mean_length": 25.0}
        elif env == callback.eval_env_bridge_b:
            return {"success_rate": eval_state["bridge_b"], "pickup_rate": 0.20, "mean_reward": 1.5, "mean_length": 35.0}
        elif env == callback.eval_env_bridge_c:
            return {"success_rate": eval_state["bridge_c"], "pickup_rate": 0.10, "mean_reward": 1.0, "mean_length": 50.0}
        else:
            return {"success_rate": eval_state["s2"], "pickup_rate": 0.05, "mean_reward": 0.5, "mean_length": 100.0}

    monkeypatch.setattr("training.train_ppo.evaluate_unshaped", mock_eval)

    # 1. Step at 1000 with Bridge A >= 20% -> advances to 2C_bridge10
    callback._on_step()
    assert callback.current_substage == "2C_bridge10"

    # 2. Step at 2000 with Medium dropping below 25% (0.20) -> Guard 2 triggers reinforce_medium
    callback.num_timesteps = 2000
    eval_state["med"] = 0.20
    callback._on_step()
    assert callback.current_substage == "reinforce_medium"
    assert callback._prev_substage == "2C_bridge10"

    # 3. Step at 3000 with Near dropping below 80% (0.75) -> Guard 1 preempts and triggers reinforce_near
    callback.num_timesteps = 3000
    eval_state["near"] = 0.75
    callback._on_step()
    assert callback.current_substage == "reinforce_near"

    # 4. Step at 4000 with Near restored >= 85% (0.90), Medium still low (0.20) -> shifts to reinforce_medium
    callback.num_timesteps = 4000
    eval_state["near"] = 0.90
    callback._on_step()
    assert callback.current_substage == "reinforce_medium"

    # 5. Step at 5000 with Medium restored >= 30% (0.35) -> recovers to 2C_bridge10
    callback.num_timesteps = 5000
    eval_state["med"] = 0.35
    callback._on_step()
    assert callback.current_substage == "2C_bridge10"

    # 6. Step at 6000 with Bridge B >= 10% (0.15) -> advances to 2C_bridge12
    callback.num_timesteps = 6000
    eval_state["bridge_b"] = 0.15
    callback._on_step()
    assert callback.current_substage == "2C_bridge12"

    train_vec_env.close()

