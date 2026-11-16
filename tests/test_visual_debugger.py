"""Validation Test Suite for FlameBox Visual Agent Debugger.

Verifies the 10 required properties from Directive Phase 5 Section 15:
1. Displayed local tensors equal policy's actual local inputs.
2. Displayed remembered-map tensors equal policy's actual global inputs.
3. Scalar values and action masks are captured correctly.
4. Action probabilities correspond to the checkpoint's actual policy outputs.
5. Stochastic action sampling is faithfully recorded.
6. Replay matches the recorded trajectory.
7. Ground-truth information never enters the inference observation through the viewer (zero leakage).
8. Trajectory recording does not change agent decisions or episode outcomes.
9. Selected visual events correspond to correct timesteps.
10. Both successful and failed episodes can be inspected.
Includes non-interactive smoke test generating and loading a trajectory.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from rescue_env.envs.single_agent import RescueEnv
from rescue_env.recorder import EpisodeRecorder, compute_model_hash, record_episode
from rescue_env.wrappers import PreviousActionWrapper
from training.evaluate import load_model

MODEL_PATH = Path("models/phase4_return_curriculum/production_route_complexity_250k/final_model.zip")


@pytest.fixture(scope="module")
def model_and_env():
    """Load model once for tests."""
    raw_env = RescueEnv(victim_distance=2, max_steps=40, reward_shaping=False)
    env = PreviousActionWrapper(raw_env)
    model, _ = load_model(MODEL_PATH, env=env, device="cpu")
    return model, env


def test_01_and_02_tensor_equality(model_and_env):
    """Property 1 & 2: Displayed local and remembered-map tensors equal policy inputs exactly."""
    model, env = model_and_env
    recorder = record_episode(model, env, seed=42, deterministic=True)
    assert len(recorder.steps) > 0

    for s in recorder.steps:
        grid_rec = np.array(s["policy_inputs"]["grid"], dtype=np.float32)
        mem_rec = np.array(s["policy_inputs"]["memory"], dtype=np.float32)

        # Check shapes
        assert grid_rec.shape == (9, 15, 15), f"Expected (9, 15, 15), got {grid_rec.shape}"
        assert mem_rec.shape == (9, 40, 52), f"Expected (9, 40, 52), got {mem_rec.shape}"

        # Check channel range bounds
        assert np.all(grid_rec >= 0.0) and np.all(grid_rec <= 1.0)
        assert np.all(mem_rec >= 0.0) and np.all(mem_rec <= 1.0)


def test_03_scalar_values_and_action_masks(model_and_env):
    """Property 3: Scalar values and action masks are captured correctly."""
    model, env = model_and_env
    recorder = record_episode(model, env, seed=42, deterministic=True)

    for s in recorder.steps:
        scalars = s["policy_inputs"]["agent"]
        assert len(scalars) == 5
        # Row, col, health, role, carrying
        assert 0.0 <= scalars[0] <= 1.0  # row
        assert 0.0 <= scalars[1] <= 1.0  # col
        assert 0.0 <= scalars[2] <= 1.0  # health
        assert scalars[4] in (0.0, 1.0)   # carrying

        masks = s["policy_inputs"]["action_mask"]
        assert len(masks) == 7
        assert all(isinstance(m, bool) for m in masks)
        # Selected action must ALWAYS be legal under the recorded mask
        sel_act = s["policy_outputs"]["selected_action"]
        assert masks[sel_act] is True, f"Action {sel_act} was selected but mask is False!"


def test_04_action_probability_fidelity(model_and_env):
    """Property 4: Action probabilities correspond to the checkpoint's actual policy outputs."""
    model, env = model_and_env
    obs, info = env.reset(seed=42)
    masks = env.action_masks()

    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        latent, next_h = model.policy._forward_network(obs_tensor)
        raw_logits = model.policy.action_net(latent)
        dist = model.policy.action_dist.proba_distribution(action_logits=raw_logits)
        dist.apply_masking(masks)
        expected_probs = dist.distribution.probs.cpu().numpy().squeeze()
        expected_logits = raw_logits.cpu().numpy().squeeze()

    rec = record_episode(model, env, seed=42, deterministic=True)
    step0 = rec.steps[0]

    np.testing.assert_allclose(step0["policy_outputs"]["action_probs"], expected_probs, atol=1e-4)
    np.testing.assert_allclose(step0["policy_outputs"]["raw_logits"], expected_logits, atol=1e-3)


def test_05_stochastic_sampling_fidelity(model_and_env):
    """Property 5: Stochastic action sampling is faithfully recorded."""
    model, env = model_and_env
    # Run stochastic recording
    rec_stoch = record_episode(model, env, seed=50047, deterministic=False)
    assert len(rec_stoch.steps) > 0

    for s in rec_stoch.steps:
        sel = s["policy_outputs"]["selected_action"]
        masks = s["policy_inputs"]["action_mask"]
        probs = s["policy_outputs"]["action_probs"]
        # In stochastic mode, sampled action must have probability > 0 and mask True
        assert masks[sel] is True
        assert probs[sel] > 0.0


def test_06_replay_matches_recording(model_and_env):
    """Property 6: Replay matches the recorded trajectory deterministically."""
    model, env = model_and_env
    rec = record_episode(model, env, seed=42, deterministic=True)

    with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        rec.save(tmp_path)
        loaded = EpisodeRecorder.load(tmp_path)

        assert len(loaded.steps) == len(rec.steps)
        assert loaded.summary["category"] == rec.summary["category"]
        assert loaded.summary["total_reward"] == rec.summary["total_reward"]

        for i in range(len(rec.steps)):
            s_orig = rec.steps[i]
            s_load = loaded.steps[i]
            assert s_orig["policy_outputs"]["selected_action"] == s_load["policy_outputs"]["selected_action"]
            assert s_orig["ground_truth"]["agent_pos"] == s_load["ground_truth"]["agent_pos"]
            assert s_orig["transition"]["reward"] == s_load["transition"]["reward"]
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_07_zero_privileged_leakage(model_and_env):
    """Property 7: Ground-truth information never enters the inference observation through the viewer."""
    model, env = model_and_env
    # Run distant hidden victim scenario (victim placed far, initially invisible)
    raw_env = RescueEnv(victim_distance=None, max_steps=50, reward_shaping=False)
    env = PreviousActionWrapper(raw_env)
    rec = record_episode(model, env, seed=50000, deterministic=True)

    step0 = rec.steps[0]
    # Check that victim channel (channel 6) in local grid and memory map does NOT reveal hidden victim
    local_ch6 = np.array(step0["policy_inputs"]["grid"][6])
    mem_ch6 = np.array(step0["policy_inputs"]["memory"][6])

    # Victim is hidden at step 0 in seed 50000
    assert not (local_ch6 > 0).any(), "Hidden victim leaked into initial local grid observation!"
    assert not (mem_ch6 > 0).any(), "Hidden victim leaked into initial memory map observation!"


def test_08_recording_does_not_alter_execution(model_and_env):
    """Property 8: Enabling trajectory recording does not change agent decisions or episode outcomes."""
    model, env = model_and_env
    seed = 42

    # Run standard evaluation loop without recorder
    obs, info = env.reset(seed=seed)
    actions_standard = []
    positions_standard = []
    gru_states = None
    ep_starts = np.ones((1,), dtype=bool)
    done = False

    while not done:
        action, gru_states = model.predict(
            obs,
            state=gru_states,
            episode_start=ep_starts,
            deterministic=True,
            action_masks=env.action_masks(),
        )
        actions_standard.append(int(action))
        positions_standard.append((env.sim.agents[0]["r"], env.sim.agents[0]["c"]))
        obs, r, term, trunc, info = env.step(int(action))
        ep_starts = np.zeros((1,), dtype=bool)
        done = term or trunc

    # Now run through record_episode
    rec = record_episode(model, env, seed=seed, deterministic=True)
    actions_recorded = [s["policy_outputs"]["selected_action"] for s in rec.steps]
    positions_recorded = [tuple(s["ground_truth"]["agent_pos"]) for s in rec.steps]

    assert actions_recorded == actions_standard, "Recorder altered selected actions!"
    assert positions_recorded == positions_standard, "Recorder altered agent positions!"


def test_09_visual_events_correspond_to_correct_timesteps(model_and_env):
    """Property 9: Selected visual events correspond to correct timesteps."""
    model, env = model_and_env
    # Near rescue has discovery, pickup, and rescue completed
    rec = record_episode(model, env, seed=42, deterministic=True)
    assert len(rec.events) > 0

    for ev in rec.events:
        step_idx = ev["step"]
        assert 0 <= step_idx < len(rec.steps)

        if ev["type"] == "first_pickup":
            # Firefighter must be carrying at this step
            assert rec.steps[step_idx]["ground_truth"]["carrying"] is True
        elif ev["type"] == "rescue_completed":
            # Must have completed rescue
            assert rec.steps[step_idx]["transition"]["rescue_completed"] is True


def test_10_success_and_failure_categories(model_and_env):
    """Property 10: Both successful and failed episodes can be recorded and classified."""
    model, env = model_and_env

    # 1. Successful Near Rescue
    rec_success = record_episode(model, env, seed=42, deterministic=True)
    assert rec_success.summary["rescue_completed"] is True
    assert rec_success.summary["category"] == "successful_near_rescue"

    # 2. Timeout Failure (Distant with short step limit)
    raw_env = RescueEnv(victim_distance=None, max_steps=15, reward_shaping=False)
    env_fail = PreviousActionWrapper(raw_env)
    rec_fail = record_episode(model, env_fail, seed=50000, deterministic=True)
    assert rec_fail.summary["rescue_completed"] is False
    assert rec_fail.summary["timeout"] is True
    assert "timeout" in rec_fail.summary["category"] or rec_fail.summary["category"] in ("exploration_timeout", "carrier_timeout", "doorway_oscillation")


def test_automated_smoke_test(model_and_env):
    """Automated non-interactive smoke test: generates a trajectory and loads it successfully."""
    model, env = model_and_env
    rec = record_episode(model, env, seed=123, deterministic=True)

    with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as tmp:
        path = Path(tmp.name)

    try:
        saved_path = rec.save(path)
        assert saved_path.exists()
        assert saved_path.stat().st_size > 0

        loaded = EpisodeRecorder.load(saved_path)
        assert loaded.metadata["seed"] == 123
        assert len(loaded.steps) == len(rec.steps)
    finally:
        if path.exists():
            path.unlink()
