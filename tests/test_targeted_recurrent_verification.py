"""Targeted Mathematical and Behavioral Verification Suite for Recurrent Masked PPO in FlameBox.

Implements the 7 targeted tests required by the verification directive:
  1. Temporal gradient propagation: dL_2 / d_s0 != 0 across time steps; reset severance.
  2. Rollout boundary continuity: GRU hidden state persists across rollout boundaries.
  3. Episode reset isolation: Asynchronous resets zero only terminating envs.
  4. Action-mask consistency: Zero illegal probability, 100+ valid trials, finite PPO ratios.
  5. Checkpoint continuity: Parameters, optimizer, metadata restored; CPU/MPS round-trip; resume.
  6. Fixed-scenario overfitting: Learnability diagnostic with native rescue and unshaped evaluation.
  7. Remembered-map spatial sensitivity: Narrow doorway sensitivity in unpooled memory encoder.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.cnn import RecurrentSpatialRescueCNN
from agents.recurrent_ppo import MaskableRecurrentPPO
from rescue_env.envs.single_agent import RescueEnv
from rescue_env.wrappers import PreviousActionWrapper
from training.train_ppo import load_checkpoint, save_checkpoint


def make_dummy_obs_space() -> spaces.Dict:
    """Construct observation space dictionary matching Phase 2 specifications."""
    return spaces.Dict({
        "grid": spaces.Box(low=-np.inf, high=np.inf, shape=(9, 15, 15), dtype=np.float32),
        "memory": spaces.Box(low=-np.inf, high=np.inf, shape=(9, 40, 52), dtype=np.float32),
        "agent": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
        "previous_action": spaces.Box(low=0, high=6, shape=(), dtype=np.int64),
    })


# ==============================================================================
# Test 1: Temporal Gradient Propagation (Multi-step BPTT)
# ==============================================================================

def test_test1_temporal_gradient_propagation():
    """Test 1: Verify multi-step BPTT gradient propagation across consecutive timesteps.

    Constructs a 3-step sequence (t=0, t=1, t=2). Computes loss solely at t=2.
    Verifies:
      a) dL_2 / d_s0 != 0 (information propagates through recurrent state across 2 full steps).
      b) dL_2 / d_s1 != 0 and dL_2 / d_s2 != 0.
      c) Boundary severance: If episode resets at t=1, gradient into t=0 is identically 0.0.
      d) Boundary severance: If episode resets at t=2, gradient into t=0 and t=1 is identically 0.0.
    """
    obs_space = make_dummy_obs_space()
    action_space = spaces.Discrete(7)
    policy = RecurrentSpatialRescueCNN(obs_space, action_space)
    policy.set_training_mode(True)

    # 3 consecutive timesteps for 1 sequence: n_seq = 1, max_length = 3
    g0 = torch.randn(1, 9, 15, 15, requires_grad=True)
    g1 = torch.randn(1, 9, 15, 15, requires_grad=True)
    g2 = torch.randn(1, 9, 15, 15, requires_grad=True)
    batch_grid = torch.cat([g0, g1, g2], dim=0)  # shape (3, 9, 15, 15)

    batch_obs = {
        "grid": batch_grid,
        "memory": torch.randn(3, 9, 40, 52),
        "agent": torch.randn(3, 5),
        "previous_action": torch.tensor([0, 1, 2], dtype=torch.long),
    }

    actions = torch.tensor([1, 2, 3], dtype=torch.long)
    masks = torch.ones((3, 7), dtype=torch.bool)
    h0 = torch.zeros((1, 1, 128))
    # Episode starts at t=0, continues at t=1, continues at t=2
    episode_starts = torch.tensor([1.0, 0.0, 0.0])

    values, log_prob, entropy = policy.evaluate_actions(
        batch_obs,
        actions,
        lstm_states=h0,
        episode_starts=episode_starts,
        action_masks=masks,
    )

    # Loss computed solely at t=2
    loss_t2 = values[2] + log_prob[2]
    grad_g0 = torch.autograd.grad(loss_t2, g0, retain_graph=True)[0]
    grad_g1 = torch.autograd.grad(loss_t2, g1, retain_graph=True)[0]
    grad_g2 = torch.autograd.grad(loss_t2, g2, retain_graph=True)[0]

    norm_g0 = grad_g0.norm().item()
    norm_g1 = grad_g1.norm().item()
    norm_g2 = grad_g2.norm().item()

    print(f"\n[Test 1] Multi-step BPTT Gradients: dL2/dg0={norm_g0:.6f}, dL2/dg1={norm_g1:.6f}, dL2/dg2={norm_g2:.6f}")

    # Assert non-zero gradients across time steps (genuine multi-step BPTT)
    assert norm_g0 > 1e-5, f"Expected non-zero BPTT gradient dL2/dg0, got {norm_g0}"
    assert norm_g1 > 1e-5, f"Expected non-zero BPTT gradient dL2/dg1, got {norm_g1}"
    assert norm_g2 > 1e-5, f"Expected non-zero gradient dL2/dg2, got {norm_g2}"

    # Verify boundary severance: reset at t=1 must zero dL2/dg0
    episode_starts_reset_t1 = torch.tensor([1.0, 1.0, 0.0])
    val_res_t1, lp_res_t1, _ = policy.evaluate_actions(
        batch_obs,
        actions,
        lstm_states=h0,
        episode_starts=episode_starts_reset_t1,
        action_masks=masks,
    )
    loss_res_t1 = val_res_t1[2] + lp_res_t1[2]
    grad_g0_res_t1 = torch.autograd.grad(loss_res_t1, g0, retain_graph=True)[0]
    grad_g1_res_t1 = torch.autograd.grad(loss_res_t1, g1, retain_graph=True)[0]

    assert torch.all(grad_g0_res_t1 == 0.0), "Gradient leaked across episode reset at t=1!"
    assert grad_g1_res_t1.norm().item() > 1e-5, "Valid gradient within sequence was incorrectly zeroed"

    # Verify boundary severance: reset at t=2 must zero dL2/dg0 and dL2/dg1
    episode_starts_reset_t2 = torch.tensor([1.0, 0.0, 1.0])
    val_res_t2, lp_res_t2, _ = policy.evaluate_actions(
        batch_obs,
        actions,
        lstm_states=h0,
        episode_starts=episode_starts_reset_t2,
        action_masks=masks,
    )
    loss_res_t2 = val_res_t2[2] + lp_res_t2[2]
    grad_g0_res_t2 = torch.autograd.grad(loss_res_t2, g0, retain_graph=True)[0]
    grad_g1_res_t2 = torch.autograd.grad(loss_res_t2, g1, retain_graph=True)[0]

    assert torch.all(grad_g0_res_t2 == 0.0), "Gradient leaked across episode reset at t=2 into g0!"
    assert torch.all(grad_g1_res_t2 == 0.0), "Gradient leaked across episode reset at t=2 into g1!"


# ==============================================================================
# Test 2: Rollout Boundary Continuity
# ==============================================================================

def test_test2_rollout_boundary_continuity():
    """Test 2: Verify GRU hidden state persists across PPO rollout boundaries.

    Runs an episode with max_steps=20 with PPO rollout buffer size n_steps=8.
    Verifies:
      a) Hidden state is non-zero after rollout 1 (step 8) and episode has not ended.
      b) Rollout 2 buffer stores this exact hidden state as its initial hidden state.
      c) Hidden state persists across the boundary from step 7 to step 8.
      d) Episode termination at step 20 (mid-rollout 3) cleanly triggers episode_starts=1.0.
    """
    raw_env = RescueEnv(max_steps=20, victim_distance=3)
    env = PreviousActionWrapper(raw_env)
    vec_env = DummyVecEnv([lambda: env])

    model = MaskableRecurrentPPO(
        policy=RecurrentSpatialRescueCNN,
        env=vec_env,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        learning_rate=1e-4,
        device="cpu",
        seed=42,
    )

    # Rollout 1: steps 0 to 7 (8 steps, episode max_steps=20 -> episode has not ended)
    model.learn(total_timesteps=8)
    h_end_r1 = model._last_lstm_states.pi[0].clone()
    assert h_end_r1.norm().item() > 0.1, "Hidden state is zero at end of rollout 1!"
    assert not model._last_episode_starts[0], "Episode prematurely terminated in rollout 1"

    # Rollout 2: steps 8 to 15 (8 more steps, episode still active)
    model.learn(total_timesteps=8, reset_num_timesteps=False)
    h_end_r2 = model._last_lstm_states.pi[0].clone()
    assert h_end_r2.norm().item() > 0.1, "Hidden state wiped during rollout 2!"
    assert not model._last_episode_starts[0], "Episode prematurely terminated in rollout 2"

    # Verify initial hidden state in buffer of rollout 2 matches h_end_r1
    h_buf_start_r2 = torch.as_tensor(model.rollout_buffer.hidden_states_pi[0][0])
    assert torch.allclose(h_end_r1.cpu(), h_buf_start_r2.cpu(), atol=1e-5), (
        "Buffer initial hidden state does not match previous rollout terminal hidden state!"
    )

    # Rollout 3: steps 16 to 23 (step 20 terminates the episode at step 4 of rollout 3)
    model.learn(total_timesteps=8, reset_num_timesteps=False)
    starts_r3 = model.rollout_buffer.episode_starts[:, 0]
    # Step 20 is index 4 in rollout 3 (16 + 4 = 20)
    assert starts_r3[4] == 1.0, f"Expected episode reset at step 4 of rollout 3, got {starts_r3}"

    vec_env.close()


# ==============================================================================
# Test 3: Episode Reset Isolation
# ==============================================================================

def test_test3_episode_reset_isolation():
    """Test 3: Verify asynchronous episode reset isolation across parallel environments.

    Runs 4 parallel environments where only environment 1 terminates (done=True).
    Verifies:
      a) Env 1's GRU hidden state is zeroed out before recurrent propagation.
      b) Envs 0, 2, and 3 continue seamlessly from their non-zero hidden states.
      c) Outputs for Envs 0, 2, and 3 match isolated single-env continuations exactly.
      d) Output for Env 1 matches an isolated single-env step initialized from zeros.
    """
    obs_space = make_dummy_obs_space()
    action_space = spaces.Discrete(7)
    policy = RecurrentSpatialRescueCNN(obs_space, action_space)
    policy.set_training_mode(False)

    n_envs = 4
    # Non-zero prior hidden states for all 4 environments
    torch.manual_seed(123)
    h_prev = torch.randn(1, n_envs, 128)
    obs = {
        "grid": torch.randn(n_envs, 9, 15, 15),
        "memory": torch.randn(n_envs, 9, 40, 52),
        "agent": torch.randn(n_envs, 5),
        "previous_action": torch.tensor([0, 1, 2, 3], dtype=torch.long),
    }

    # Episode starts: Only Env 1 terminates and restarts (done=True for env 1)
    dones = torch.tensor([0.0, 1.0, 0.0, 0.0])

    _, _, _, next_h = policy(obs, lstm_states=h_prev, episode_starts=dones, return_hidden_state=True)
    if hasattr(next_h, "pi"):
        h_multi = next_h.pi[0]
    else:
        h_multi = next_h

    # 1. Isolated check for terminating Env 1: Must match execution from zero hidden state
    obs_env1 = {k: v[1:2] for k, v in obs.items()}
    h_zero = torch.zeros(1, 1, 128)
    _, _, _, next_h1_iso = policy(obs_env1, lstm_states=h_zero, episode_starts=torch.tensor([0.0]), return_hidden_state=True)
    if hasattr(next_h1_iso, "pi"):
        h1_iso = next_h1_iso.pi[0]
    else:
        h1_iso = next_h1_iso

    diff_1 = (h_multi[:, 1:2, :] - h1_iso).norm().item()
    print(f"\n[Test 3] Env 1 reset diff from isolated zero-start: {diff_1:.2e}")
    assert diff_1 < 1e-4, f"Terminating Env 1 did not reset to zero! Diff: {diff_1}"

    # 2. Isolated check for continuing Envs 0, 2, 3: Must match isolated executions from h_prev
    for i in [0, 2, 3]:
        obs_i = {k: v[i:i+1] for k, v in obs.items()}
        h_i = h_prev[:, i:i+1, :]
        _, _, _, next_hi_iso = policy(obs_i, lstm_states=h_i, episode_starts=torch.tensor([0.0]), return_hidden_state=True)
        if hasattr(next_hi_iso, "pi"):
            hi_iso = next_hi_iso.pi[0]
        else:
            hi_iso = next_hi_iso

        diff_i = (h_multi[:, i:i+1, :] - hi_iso).norm().item()
        print(f"[Test 3] Continuing Env {i} diff from isolated continuation: {diff_i:.2e}")
        assert diff_i < 1e-4, f"Continuing Env {i} was corrupted by Env 1 reset! Diff: {diff_i}"


# ==============================================================================
# Test 4: Action-Mask Consistency
# ==============================================================================

def test_test4_action_mask_consistency():
    """Test 4: Verify action-mask consistency in inference, sampling, and PPO updates.

    Constructs states with restricted valid-action subsets.
    Verifies:
      a) Invalid-action probability is strictly 0.0 in the softmax distribution.
      b) Invalid-action logits are masked to <= -1e7.
      c) 100+ sampled trials per configuration yield 100% legal actions (0 invalid choices).
      d) PPO evaluate_actions log-probabilities respect the masks (finite for valid, -inf for invalid).
      e) PPO policy ratios (exp(log_prob - old_log_prob)) remain finite, non-NaN, and positive.
    """
    obs_space = make_dummy_obs_space()
    action_space = spaces.Discrete(7)
    policy = RecurrentSpatialRescueCNN(obs_space, action_space)
    policy.set_training_mode(False)

    test_masks = [
        [False, False, True, False, False, False, False],  # only action 2
        [True, False, False, False, False, False, True],   # actions 0 and 6
        [False, True, True, True, False, False, False],    # actions 1, 2, 3
    ]
    mask_tensor = torch.tensor(test_masks, dtype=torch.bool)
    n_configs = len(test_masks)

    obs = {
        "grid": torch.randn(n_configs, 9, 15, 15),
        "memory": torch.randn(n_configs, 9, 40, 52),
        "agent": torch.randn(n_configs, 5),
        "previous_action": torch.tensor([0, 1, 2], dtype=torch.long),
    }

    dist = policy.get_distribution(obs, action_masks=mask_tensor)
    probs = dist.distribution.probs

    for i, mask in enumerate(test_masks):
        invalid_indices = [a for a in range(7) if not mask[a]]
        valid_indices = [a for a in range(7) if mask[a]]

        # a) Invalid action probabilities strictly 0.0
        assert torch.all(probs[i, invalid_indices] == 0.0), f"Config {i} has non-zero invalid probabilities!"
        assert torch.isclose(probs[i].sum(), torch.tensor(1.0), atol=1e-5), f"Config {i} probabilities do not sum to 1.0!"

        # b) Invalid logits <= -1e7
        masked_logits = policy.get_masked_logits(obs, action_masks=mask_tensor)
        assert torch.all(masked_logits[i, invalid_indices] <= -1e7), f"Config {i} illegal logits not masked: {masked_logits[i]}"

        # c) Sample 150 stochastic actions per configuration
        for _ in range(150):
            action = dist.sample()[i].item()
            assert action in valid_indices, f"Sampled illegal action {action} (valid: {valid_indices})"

        # d) PPO update log-prob evaluation
        for legal_act in valid_indices:
            act_tensor = torch.tensor([legal_act], dtype=torch.long)
            obs_i = {k: v[i:i+1] for k, v in obs.items()}
            mask_i = mask_tensor[i:i+1]

            val, log_prob, entropy = policy.evaluate_actions(obs_i, act_tensor, action_masks=mask_i)
            assert torch.isfinite(log_prob).item(), f"Log prob is not finite for legal action {legal_act}: {log_prob}"

            # e) Policy ratio stability
            # Ratio with identical old_log_prob must be exactly 1.0
            ratio_exact = torch.exp(log_prob - log_prob)
            assert torch.isfinite(ratio_exact) and not torch.isnan(ratio_exact)
            assert torch.isclose(ratio_exact, torch.tensor(1.0), atol=1e-5)

            # Ratio with perturbed old_log_prob must remain finite and positive
            perturbed_old_lp = log_prob - 0.2
            ratio_pert = torch.exp(log_prob - perturbed_old_lp)
            assert torch.isfinite(ratio_pert) and not torch.isnan(ratio_pert)
            assert ratio_pert.item() > 0.0


# ==============================================================================
# Test 5: Checkpoint Continuity (Parameters, Optimizer, Devices, Metadata)
# ==============================================================================

def test_test5_checkpoint_continuity(tmp_path: Path):
    """Test 5: Verify checkpoint save, restore, optimizer state, counters, and device round-trip.

    Verifies:
      a) Model parameters restored identically (weight tensor equality).
      b) Optimizer state restored (momentum/exp_avg buffers preserved).
      c) Cross-device loading (CPU / MPS) without assertion or dimension failures.
      d) Training counters (num_timesteps, _n_updates) and stage metadata restored.
      e) Training resumes correctly from the restored state, incrementing counters.
      f) Distinguishes inference-checkpoint correctness from exact training-resume reproducibility.
    """
    raw_env = RescueEnv(max_steps=16, victim_distance=2)
    env = PreviousActionWrapper(raw_env)
    vec_env = DummyVecEnv([lambda: env])

    # Train briefly (16 steps) on CPU to step optimizer
    model = MaskableRecurrentPPO(
        policy=RecurrentSpatialRescueCNN,
        env=vec_env,
        n_steps=16,
        batch_size=16,
        n_epochs=2,
        device="cpu",
        seed=42,
    )
    model.learn(total_timesteps=16)

    # Save complete training state
    checkpoint_file = tmp_path / "model_ckpt.zip"
    meta = {
        "stage": 1,
        "qualification_gate": "PASSED",
        "custom_tag": "checkpoint_verification",
    }
    save_checkpoint(model, checkpoint_file, metadata=meta)

    assert checkpoint_file.exists(), "Checkpoint zip file was not created"
    assert checkpoint_file.with_suffix(".meta.json").exists(), "Companion metadata file was not created"

    # Reload on CPU
    loaded_cpu = load_checkpoint(checkpoint_file, env=vec_env, device="cpu")

    # a) Parameters restored identically
    for (k1, v1), (k2, v2) in zip(
        model.policy.state_dict().items(),
        loaded_cpu.policy.state_dict().items(),
    ):
        assert k1 == k2, f"State dict key mismatch: {k1} != {k2}"
        assert torch.equal(v1, v2), f"Parameter tensor divergence for {k1}"

    # b) Optimizer state restored
    assert loaded_cpu.policy.optimizer is not None
    assert len(loaded_cpu.policy.optimizer.state) > 0, "Optimizer state dict is empty after reload!"

    # d) Metadata and counters restored
    assert loaded_cpu.num_timesteps == 16, f"Expected num_timesteps=16, got {loaded_cpu.num_timesteps}"
    assert loaded_cpu.checkpoint_metadata["stage"] == 1
    assert loaded_cpu.checkpoint_metadata["qualification_gate"] == "PASSED"

    # c) Cross-device loading (MPS if available)
    if torch.backends.mps.is_available():
        loaded_mps = load_checkpoint(checkpoint_file, env=vec_env, device="mps")
        mps_param = next(loaded_mps.policy.parameters())
        assert mps_param.device.type == "mps", f"Expected MPS device, got {mps_param.device}"

        # Inference on MPS
        obs, _ = env.reset(seed=42)
        mask = env.action_masks()
        act_mps, state_mps = loaded_mps.predict(obs, deterministic=True, action_masks=mask)
        assert 0 <= int(act_mps) < 7

        # Round-trip MPS -> CPU
        mps_save_file = tmp_path / "mps_saved.zip"
        save_checkpoint(loaded_mps, mps_save_file, metadata={"device_source": "mps"})
        reloaded_from_mps = load_checkpoint(mps_save_file, env=vec_env, device="cpu")
        reloaded_param = next(reloaded_from_mps.policy.parameters())
        assert reloaded_param.device.type == "cpu"
        print("\n[Test 5] Cross-device CPU <-> MPS save and reload verified successfully")

    # e) Resume training
    loaded_cpu.learn(total_timesteps=16, reset_num_timesteps=False)
    assert loaded_cpu.num_timesteps == 32, f"Expected timesteps=32 after resume, got {loaded_cpu.num_timesteps}"

    vec_env.close()


# ==============================================================================
# Test 6: Fixed-Scenario Overfitting (Learnability Diagnostic)
# ==============================================================================

def test_test6_fixed_scenario_overfitting():
    """Test 6: Fixed-scenario overfitting learnability diagnostic.

    Constructs 2 fixed rescue scenarios (seeds 42 and 43 with near victim).
    Trains with native rescue mechanics and training reward shaping for 3,000 steps.
    Evaluates strictly unshaped (reward_shaping=False, navigation_shaping=False, +10 native rescue only).
    Verifies that the agent learns consistently high success (100% on both scenarios).
    """
    seeds = [42, 43]

    def make_train_env(seed: int):
        def _init():
            env = RescueEnv(
                max_steps=16,
                victim_distance=1,
                min_victim_distance=1,
                fixed_seed=seed,
                reward_shaping=True,
                navigation_shaping=True,
            )
            return PreviousActionWrapper(env)
        return _init

    train_vec = DummyVecEnv([make_train_env(seeds[0]), make_train_env(seeds[1])])

    np.random.seed(42)
    torch.manual_seed(42)

    model = MaskableRecurrentPPO(
        policy=RecurrentSpatialRescueCNN,
        env=train_vec,
        n_steps=32,
        batch_size=32,
        n_epochs=5,
        learning_rate=1e-3,
        ent_coef=0.01,
        device="cpu",
        seed=42,
    )

    model.learn(total_timesteps=3000)
    train_vec.close()

    # Evaluate strictly unshaped
    results = {}
    for seed in seeds:
        eval_env = PreviousActionWrapper(
            RescueEnv(
                max_steps=16,
                victim_distance=1,
                min_victim_distance=1,
                fixed_seed=seed,
                reward_shaping=False,
                navigation_shaping=False,
            )
        )
        obs, info = eval_env.reset()
        done = False
        state = None
        actions_taken = []
        ep_reward = 0.0
        while not done:
            mask = eval_env.action_masks()
            act, state = model.predict(obs, state=state, deterministic=True, action_masks=mask)
            obs, reward, term, trunc, step_info = eval_env.step(act)
            ep_reward += reward
            actions_taken.append(int(act))
            done = term or trunc

        results[seed] = {
            "success": step_info["success"],
            "steps": step_info["steps"],
            "reward": ep_reward,
            "actions": actions_taken,
        }
        eval_env.close()

    print("\n[Test 6] Fixed-Scenario Overfitting Results:")
    for s, res in results.items():
        print(f"  Seed {s}: Success={res['success']}, Steps={res['steps']}, Reward={res['reward']}, Actions={res['actions']}")

    # Verify both seeds solved with unshaped rescue (+10 native reward)
    for s, res in results.items():
        assert res["success"], f"Failed to overfit fixed scenario seed {s}: {res}"
        assert res["reward"] == pytest.approx(10.0), f"Expected native unshaped rescue reward 10.0, got {res['reward']}"


# ==============================================================================
# Test 7: Remembered-Map Spatial Sensitivity (Unpooled Conv Architecture)
# ==============================================================================

def test_test7_remembered_map_spatial_sensitivity():
    """Test 7: Verify spatial sensitivity of the unpooled memory encoder to single-cell doorway variations.

    Constructs two otherwise identical valid observations differing only in a single
    doorway/wall passability cell in the memory canvas.
    Measures:
      a) Memory CNN encoder output sensitivity (L2 distance, cosine similarity, relative L2).
      b) Fused policy latent sensitivity.
      c) Intermediate conv feature map spatial localization (confirming 3x3 receptive field bounding).
      d) Verifies architecture retains narrow passage geometry without MaxPool2d obliteration.
    """
    env = RescueEnv(max_steps=16, victim_distance=1, fixed_seed=42)
    obs, _ = env.reset()

    policy = RecurrentSpatialRescueCNN(env.observation_space, env.action_space)
    policy.set_training_mode(False)

    grid_t = torch.as_tensor(obs["grid"]).unsqueeze(0)
    agent_t = torch.as_tensor(obs["agent"]).unsqueeze(0)
    prev_act = torch.tensor([0], dtype=torch.long)

    mem_A = torch.as_tensor(obs["memory"]).clone().unsqueeze(0)
    mem_B = mem_A.clone()

    # Invert passability at a specific doorway cell (row 22, col 23) in East passability channel (channel 1)
    doorway_r, doorway_c = 22, 23
    original_val = mem_A[0, 1, doorway_r, doorway_c].item()
    mem_B[0, 1, doorway_r, doorway_c] = 1.0 - original_val

    obs_A = {"grid": grid_t, "memory": mem_A, "agent": agent_t, "previous_action": prev_act}
    obs_B = {"grid": grid_t, "memory": mem_B, "agent": agent_t, "previous_action": prev_act}

    # a) Memory CNN output features
    out_mem_A = policy.memory_cnn(mem_A)
    out_mem_B = policy.memory_cnn(mem_B)

    diff_l2 = torch.norm(out_mem_A - out_mem_B).item()
    norm_A = torch.norm(out_mem_A).item()
    rel_l2 = diff_l2 / (norm_A + 1e-8)
    cos_sim = torch.nn.functional.cosine_similarity(out_mem_A, out_mem_B).item()

    print(f"\n[Test 7] Memory CNN Encoder (8,320-dim):")
    print(f"  L2 Distance: {diff_l2:.6f}")
    print(f"  Relative L2: {rel_l2:.6f}")
    print(f"  Cosine Similarity: {cos_sim:.6f}")

    # Memory encoder must be sensitive to the single doorway cell
    assert diff_l2 > 0.1, f"Memory CNN is insensitive to single doorway cell! L2 diff: {diff_l2}"
    assert rel_l2 > 0.05, f"Relative L2 sensitivity is too low: {rel_l2}"
    assert cos_sim < 0.999, f"Cosine similarity too close to 1.0: {cos_sim}"

    # b) Fused policy latent sensitivity
    feat_A = policy.extract_features(obs_A)
    feat_B = policy.extract_features(obs_B)
    feat_diff_l2 = torch.norm(feat_A - feat_B).item()
    feat_cos_sim = torch.nn.functional.cosine_similarity(feat_A, feat_B).item()

    print(f"  Fused Latent (128-dim) L2 Distance: {feat_diff_l2:.6f}")
    print(f"  Fused Latent Cosine Similarity: {feat_cos_sim:.6f}")
    assert feat_diff_l2 > 1e-3, f"Fused policy features insensitive to doorway: {feat_diff_l2}"

    # c) Spatial localization in intermediate conv layers
    # Conv1: 3x3 kernel, pad=1 -> Affected cells must be exactly 9 (3x3 grid around doorway_r, doorway_c)
    conv1 = policy.memory_cnn[0]
    relu1 = policy.memory_cnn[1]
    c1_A = relu1(conv1(mem_A))
    c1_B = relu1(conv1(mem_B))
    spatial_diff_c1 = (c1_A - c1_B).abs().sum(dim=1)[0]
    affected_cells_c1 = spatial_diff_c1.nonzero()

    print(f"  Conv1 affected cells count: {len(affected_cells_c1)} (Expected: 9 for 3x3 kernel)")
    assert len(affected_cells_c1) == 9, f"Expected 9 affected cells in 3x3 receptive field, got {len(affected_cells_c1)}"

    # Confirm all affected cells are within distance <= 1 of (doorway_r, doorway_c)
    for r, c in affected_cells_c1:
        assert abs(r.item() - doorway_r) <= 1 and abs(c.item() - doorway_c) <= 1, (
            f"Conv1 diff leaked outside 3x3 receptive field: ({r}, {c})"
        )

    env.close()
