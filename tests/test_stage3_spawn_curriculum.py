import pytest
import numpy as np
import torch

from training.train_ppo import (
    CurriculumRescueEnv,
    DISTANCE_CONFIGS,
    make_train_env,
    make_eval_env,
    normalize_stage,
)
from agents.recurrent_ppo import MaskableRecurrentPPO
from agents.cnn import RecurrentSpatialRescueCNN


def test_stage3_normalization():
    assert normalize_stage(3) == 3
    assert normalize_stage("3") == 3
    assert normalize_stage("3A") == "3A"
    assert normalize_stage("spawn") == "SPAWN"


def test_distance_configs_spawn_ladder():
    assert "spawn_2_6" in DISTANCE_CONFIGS
    assert DISTANCE_CONFIGS["spawn_2_6"]["min_victim_distance"] == 2
    assert DISTANCE_CONFIGS["spawn_2_6"]["victim_distance"] == 6
    assert DISTANCE_CONFIGS["spawn_2_6"]["max_steps"] == 64

    assert "spawn_7_12" in DISTANCE_CONFIGS
    assert DISTANCE_CONFIGS["spawn_7_12"]["min_victim_distance"] == 7
    assert DISTANCE_CONFIGS["spawn_7_12"]["victim_distance"] == 12
    assert DISTANCE_CONFIGS["spawn_7_12"]["max_steps"] == 128

    assert "spawn_13_20" in DISTANCE_CONFIGS
    assert DISTANCE_CONFIGS["spawn_13_20"]["min_victim_distance"] == 13
    assert DISTANCE_CONFIGS["spawn_13_20"]["victim_distance"] == 20
    assert DISTANCE_CONFIGS["spawn_13_20"]["max_steps"] == 192

    assert "spawn_21_35" in DISTANCE_CONFIGS
    assert DISTANCE_CONFIGS["spawn_21_35"]["min_victim_distance"] == 21
    assert DISTANCE_CONFIGS["spawn_21_35"]["victim_distance"] == 35
    assert DISTANCE_CONFIGS["spawn_21_35"]["max_steps"] == 288

    assert "spawn_distant" in DISTANCE_CONFIGS
    assert DISTANCE_CONFIGS["spawn_distant"]["victim_distance"] is None


def test_curriculum_env_stage3_init():
    env = CurriculumRescueEnv(stage=3, geometry_mode="edge")
    assert env.stage == 3
    assert env.substage == "3A"
    assert env.distance_metric == "spawn"
    assert env.active_bridge == "spawn_7_12"


def test_curriculum_env_stage3_reset_ladder():
    env = CurriculumRescueEnv(
        stage=3,
        substage="3A",
        geometry_mode="edge",
        enable_multitask=False,
    )
    for seed in range(10):
        obs, info = env.reset(seed=seed)
        assert obs["grid"].shape == (9, 15, 15)
        assert obs["memory"].shape == (9, 40, 52)
        assert env.distance_metric == "spawn"
        assert env.victim_distance in (6, 12, 20, 35, None)


def test_curriculum_env_stage3_multitask_integration():
    env = CurriculumRescueEnv(
        stage=3,
        substage="3B",
        geometry_mode="edge",
        enable_multitask=True,
        task_mixture=(0.34, 0.33, 0.33),
        return_mode="route_complexity",
    )
    task_types_seen = set()
    for seed in range(30):
        obs, info = env.reset(seed=seed)
        task_types_seen.add(env.task_type)
        assert env.distance_metric == "spawn"
    # Over 30 trials with 33% mixture, all 3 tasks should appear
    assert "full" in task_types_seen
    assert "return" in task_types_seen
    assert "discovery" in task_types_seen


def test_stage3_make_train_eval_env():
    train_env = make_train_env(
        stage=3,
        substage="3A",
        enable_multitask=True,
        n_envs=2,
        base_seed=100,
        geometry_mode="edge",
    )
    eval_env = make_eval_env(
        victim_distance=6,
        min_victim_distance=2,
        max_steps=64,
        distance_metric="spawn",
        geometry_mode="edge",
    )
    obs = train_env.reset()
    assert "grid" in obs
    assert "memory" in obs
    eval_obs, _ = eval_env.reset(seed=50000)
    assert eval_obs["grid"].shape == (9, 15, 15)

    train_env.close()
    eval_env.close()
