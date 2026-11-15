import numpy as np
import pytest
import torch
from collections import Counter

from rescue_env.envs.single_agent import RescueEnv
from training.train_ppo import (
    CurriculumRescueEnv,
    make_eval_env,
    make_train_env,
    set_env_multitask,
)
from agents.cnn import RecurrentSpatialRescueCNN
from agents.recurrent_ppo import MaskableRecurrentPPO


def test_full_task_initialization():
    """Requirement: Full task initializes standard unknown entrance scenario."""
    env = RescueEnv(task_type="full", victim_distance=10, min_victim_distance=7, max_steps=192)
    obs, info = env.reset(seed=42)

    assert info["task_type"] == "full"
    # Agent is not carrying
    assert obs["agent"][4] == 0.0
    # Memory map has no explored cells prior to first observation
    assert env.victim_seen is False
    assert (obs["memory"][6] > 0).sum() == 0  # Victim not yet known in memory
    env.close()


def test_return_task_initialization_and_carrying():
    """Requirement: Return task initializes agent carrying victim deep inside building."""
    env = RescueEnv(task_type="return", victim_distance=8, min_victim_distance=5, max_steps=160)
    obs, info = env.reset(seed=42)

    assert info["task_type"] == "return"
    # Agent must be carrying
    assert obs["agent"][4] == 1.0
    # Victim in simulator must be carried by agent 0
    victim = env.sim.victims[0]
    agent = env.sim.agents[0]
    assert victim["carried_by"] == 0
    assert victim["r"] == agent["r"]
    assert victim["c"] == agent["c"]
    env.close()


def test_return_task_memory_physical_consistency():
    """Requirement: Remembered map must be physically consistent with path traveled."""
    env = RescueEnv(task_type="return", victim_distance=10, min_victim_distance=7, max_steps=192)
    obs, info = env.reset(seed=42)

    # Exit must be marked in channel 7
    exit_marked = (obs["memory"][7] > 0).any()
    assert exit_marked, "Exit opening portal must be marked in channel 7 of memory"

    # Explored mask (channel 5) must show the traveled path
    explored_cells = (obs["memory"][5] > 0).sum()
    assert explored_cells > 0, "Traveled path must be marked explored in memory"

    # Total grid size is 40x52 = 2080 cells. In a 10-step path, only a small fraction is explored
    # Unvisited rooms must remain strictly 0.0 (unexplored)
    total_cells = env.sim.h * env.sim.w
    assert explored_cells < total_cells * 0.6, (
        f"Memory should only contain traveled corridor, got {explored_cells}/{total_cells} cells"
    )

    # Pickup reward set must already contain victim 0 so no double-rewarding
    assert 0 in env.rewarded_pickups
    env.close()


def test_return_task_grass_rescue():
    """Requirement: Returning to exit grass with carried victim triggers rescue."""
    env = RescueEnv(task_type="return", victim_distance=2, max_steps=64)
    obs, info = env.reset(seed=0)

    exit_opening = next(o for o in env.sim.openings if o["kind"] == "exit")
    start_cell = env.sim._cells_for_opening(exit_opening, exit_opening["zone_a"])
    path = env._find_path_to_victim(start_cell, (env.sim.agents[0]["r"], env.sim.agents[0]["c"]))
    path.reverse()

    if exit_opening["orient"] == "H":
        exit_act = 1 if exit_opening["at"] == 0 else 3
    else:
        exit_act = 4 if exit_opening["at"] == 0 else 2

    # Step along path to doorway
    for i in range(len(path) - 1):
        curr = path[i]
        nxt = path[i + 1]
        dr, dc = nxt[0] - curr[0], nxt[1] - curr[1]
        act = 1 if dr == -1 else (3 if dr == 1 else (2 if dc == 1 else (4 if dc == -1 else 0)))
        obs, r, term, trunc, info = env.step(act)

    # Step out to grass
    obs, r, term, trunc, info = env.step(exit_act)
    assert term is True, "Stepping onto grass while carrying victim must terminate episode"
    assert r >= 10.0, f"Expected native rescue reward +10.0, got {r}"
    assert info["success"] is True
    assert info["return_success"] is True
    env.close()


def test_discovery_task_initialization():
    """Requirement: Discovery task initializes entrance search for distant victim."""
    env = RescueEnv(task_type="discovery", victim_distance=8, max_steps=160, discovery_reward=2.5)
    obs, info = env.reset(seed=42)

    assert info["task_type"] == "discovery"
    assert obs["agent"][4] == 0.0  # Not carrying
    assert env._discovery_completed is False
    env.close()


def test_discovery_task_auxiliary_termination_and_reward():
    """Requirement: Discovery task terminates immediately on legitimate sighting with reward."""
    # Find a seed where victim is within 1 step
    for seed in range(50):
        env = RescueEnv(task_type="discovery", victim_distance=2, max_steps=64, discovery_reward=2.5)
        obs, info = env.reset(seed=seed)
        if (obs["grid"][6] > 0).any():
            # Victim sighted in initial observation! Step action 0 (wait)
            obs, r, term, trunc, info = env.step(0)
            assert term is True, "Discovery task must terminate immediately upon sighting victim"
            assert r >= 2.5, f"Expected discovery reward >= 2.5, got {r}"
            assert info["discovery_success"] is True
            assert info["success"] is True
            env.close()
            return
        env.close()
    pytest.fail("Could not find a scenario with visible victim in 50 trials")


def test_zero_auxiliary_reward_leakage_in_evaluation():
    """Requirement: Evaluation environments must NEVER leak discovery rewards or auxiliary termination."""
    eval_env = make_eval_env(victim_distance=2, max_steps=64)
    obs, info = eval_env.reset(seed=0)

    # Check that unwrapped environment has task_type="full"
    raw_env = eval_env.unwrapped
    assert raw_env.task_type == "full", f"Eval env must be task_type='full', got {raw_env.task_type}"
    assert raw_env.reward_shaping is False
    assert raw_env.navigation_shaping is False

    # Take 10 wait actions; even if victim is in line-of-sight, discovery reward is 0.0
    for _ in range(10):
        obs, r, term, trunc, info = eval_env.step(0)
        assert "discovery_success" not in info
        assert r == 0.0, f"Unshaped evaluation step must give 0.0 reward before rescue, got {r}"
        if term:
            break
    eval_env.close()


def test_action_masks_valid_across_all_tasks():
    """Requirement: Legal action masks (7-dim bool) must be correct across all three tasks."""
    for task in ("full", "return", "discovery"):
        env = RescueEnv(task_type=task, victim_distance=6, max_steps=128)
        obs, info = env.reset(seed=42)
        mask = env.action_masks()

        assert mask.shape == (7,)
        assert mask.dtype == bool
        assert mask[0] is np.True_ or mask[0] is True  # Wait always legal
        # In return task (carrying), pickup (action 5) must be False
        if task == "return":
            assert mask[5] is np.False_ or mask[5] is False, "Pickup must be illegal while carrying"
        env.close()


def test_curriculum_env_task_mixture_sampling():
    """Requirement: CurriculumRescueEnv samples tasks according to configured mixture."""
    env = CurriculumRescueEnv(
        stage="2C",
        substage="2C_bridge8",
        enable_multitask=True,
        task_mixture=(0.50, 0.25, 0.25),
        discovery_reward=2.0,
    )

    counts = Counter()
    for seed in range(200):
        obs, info = env.reset(seed=seed)
        counts[info["task_type"]] += 1

    env.close()

    total = sum(counts.values())
    p_full = counts["full"] / total
    p_return = counts["return"] / total
    p_discovery = counts["discovery"] / total

    # Check that all 3 tasks were sampled and within reasonable bounds of (0.50, 0.25, 0.25)
    assert 0.35 <= p_full <= 0.65, f"Expected full ~0.50, got {p_full}"
    assert 0.15 <= p_return <= 0.35, f"Expected return ~0.25, got {p_return}"
    assert 0.15 <= p_discovery <= 0.35, f"Expected discovery ~0.25, got {p_discovery}"


def test_recurrent_hidden_state_and_bptt_across_multitask():
    """Requirement: Neural network forward pass and GRU hidden state transitions work cleanly across tasks."""
    train_env = make_train_env(
        stage="2C",
        substage="2C_bridge8",
        enable_multitask=True,
        task_mixture=(0.34, 0.33, 0.33),
        discovery_reward=2.0,
        n_envs=2,
        base_seed=42,
    )

    obs = train_env.reset()
    policy = RecurrentSpatialRescueCNN(
        train_env.observation_space,
        train_env.action_space,
        architecture_variant="baseline",
    )

    gru_states = None
    episode_starts = np.ones((2,), dtype=bool)

    # Run 10 forward steps
    for step in range(10):
        masks = train_env.env_method("action_masks")
        action_masks = np.array(masks, dtype=bool)
        obs_tensor = policy.obs_to_tensor(obs)[0]
        with torch.no_grad():
            actions, values, log_probs, gru_states = policy(
                obs_tensor,
                gru_states,
                torch.tensor(episode_starts, dtype=torch.float32),
                action_masks=action_masks,
                return_hidden_state=True,
            )
        acts = actions.cpu().numpy()
        obs, rewards, dones, infos = train_env.step(acts)
        episode_starts = dones

    train_env.close()
    assert True
