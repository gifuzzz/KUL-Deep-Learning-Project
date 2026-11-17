#!/usr/bin/env python3
"""Rigorous Inspection of Channel 9 Trained Weights, Gradients, and Logit Sensitivity.

Validates:
1. Weight progression: memory_cnn[0].weight[:, 9] changed from 0.0 across training checkpoints.
2. Gradient verification: optimizer states (exp_avg, exp_avg_sq) confirm nonzero gradient reception.
3. Active gradient verification: backward pass on realistic exploration batch yields nonzero dL/dW[:, 9].
4. Logit response: policy logits, value estimates, and action probabilities respond systematically to realistic visit history changes.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rescue_env.envs.single_agent import RescueEnv
from rescue_env.wrappers import PreviousActionWrapper
from training.train_ppo import load_checkpoint


def inspect_weights_progression():
    """Inspect memory_cnn[0].weight[:, 9] across all checkpoints."""
    print("\n" + "=" * 90)
    print("1. WEIGHT PROGRESSION: memory_cnn[0].weight[:, 9]")
    print("=" * 90)

    configs = [
        ("Variant B (Binary Visited)", "models/phase8_binary_visited", [
            ("Step 0 (init)", "initial_transferred.zip"),
            ("Step 10k", "stage3_spawn_curriculum/checkpoint_10000.zip"),
            ("Step 25k", "stage3_spawn_curriculum/checkpoint_25000.zip"),
            ("Best Model", "stage3_spawn_curriculum/best_model.zip"),
            ("Step 50k (final)", "final_model.zip"),
        ]),
        ("Variant C (Visit Recency)", "models/phase8_recency", [
            ("Step 0 (init)", "initial_transferred.zip"),
            ("Step 10k", "stage3_spawn_curriculum/checkpoint_10000.zip"),
            ("Step 25k", "stage3_spawn_curriculum/checkpoint_25000.zip"),
            ("Best Model", "stage3_spawn_curriculum/best_model.zip"),
            ("Step 50k (final)", "final_model.zip"),
        ]),
    ]

    results = {}

    for variant_name, base_dir, ckpts in configs:
        print(f"\n--- {variant_name} ---")
        var_res = {}
        for label, rel_path in ckpts:
            full_path = ROOT / base_dir / rel_path
            if not full_path.exists():
                print(f"  {label:<18}: NOT FOUND ({rel_path})")
                continue
            model = load_checkpoint(full_path, device="cpu")
            w = model.policy.memory_cnn[0].weight.data
            w9 = w[:, 9]  # Shape: (32, 3, 3) -> 288 weights
            w0_8 = w[:, :9]  # Shape: (32, 9, 3, 3)

            norm9 = float(torch.norm(w9).item())
            mean_abs9 = float(torch.abs(w9).mean().item())
            max_abs9 = float(torch.abs(w9).max().item())
            std9 = float(torch.std(w9).item())
            nonzero_count = int((torch.abs(w9) > 1e-7).sum().item())

            # Baselines for Ch 0-8
            norms_pretrained = torch.norm(w0_8.permute(1, 0, 2, 3).reshape(9, -1), dim=1)
            mean_norm_pretrained = float(norms_pretrained.mean().item())
            mean_abs_pretrained = float(torch.abs(w0_8).mean().item())

            ratio_norm = (norm9 / mean_norm_pretrained) * 100.0 if mean_norm_pretrained > 0 else 0.0

            var_res[label] = {
                "checkpoint": str(rel_path),
                "norm": norm9,
                "mean_abs": mean_abs9,
                "max_abs": max_abs9,
                "std": std9,
                "nonzero_count": nonzero_count,
                "total_weights": 288,
                "pretrained_mean_norm": mean_norm_pretrained,
                "ratio_to_pretrained_norm_pct": ratio_norm,
            }

            print(
                f"  {label:<18}: Norm={norm9:8.6f} | MeanAbs={mean_abs9:8.6f} | "
                f"MaxAbs={max_abs9:8.6f} | Nonzero={nonzero_count}/288 ({nonzero_count/288*100:5.1f}%) | "
                f"RelToPretrained={ratio_norm:5.2f}%"
            )
        results[variant_name] = var_res

    return results


def inspect_optimizer_gradients():
    """Inspect Adam optimizer moments (exp_avg and exp_avg_sq) in checkpoint zips."""
    print("\n" + "=" * 90)
    print("2. GRADIENT VERIFICATION: Adam Optimizer Moments (exp_avg & exp_avg_sq)")
    print("=" * 90)

    configs = [
        ("Variant B (Binary Visited)", "models/phase8_binary_visited", [
            ("Step 10k", "stage3_spawn_curriculum/checkpoint_10000.zip"),
            ("Step 25k", "stage3_spawn_curriculum/checkpoint_25000.zip"),
            ("Best Model", "stage3_spawn_curriculum/best_model.zip"),
            ("Step 50k (final)", "final_model.zip"),
        ]),
        ("Variant C (Visit Recency)", "models/phase8_recency", [
            ("Step 10k", "stage3_spawn_curriculum/checkpoint_10000.zip"),
            ("Step 25k", "stage3_spawn_curriculum/checkpoint_25000.zip"),
            ("Best Model", "stage3_spawn_curriculum/best_model.zip"),
            ("Step 50k (final)", "final_model.zip"),
        ]),
    ]

    results = {}

    for variant_name, base_dir, ckpts in configs:
        print(f"\n--- {variant_name} ---")
        var_res = {}
        for label, rel_path in ckpts:
            full_path = ROOT / base_dir / rel_path
            if not full_path.exists():
                continue
            with zipfile.ZipFile(full_path) as z:
                if "policy.optimizer.pth" not in z.namelist():
                    continue
                opt_data = io.BytesIO(z.read("policy.optimizer.pth"))
                opt_state = torch.load(opt_data, map_location="cpu", weights_only=False)

            # Find memory_cnn[0].weight (shape: 32, 10, 3, 3)
            found = False
            for param_id, s in opt_state["state"].items():
                if "exp_avg" in s and s["exp_avg"].shape == torch.Size([32, 10, 3, 3]):
                    found = True
                    step = s["step"].item() if isinstance(s["step"], torch.Tensor) else s["step"]
                    m9 = s["exp_avg"][:, 9]  # First moment (momentum)
                    v9 = s["exp_avg_sq"][:, 9]  # Second moment (variance of gradient)

                    m9_norm = float(torch.norm(m9).item())
                    v9_norm = float(torch.norm(v9).item())
                    v9_mean = float(v9.mean().item())
                    v9_max = float(v9.max().item())
                    nonzero_grad_count = int((v9 > 0).sum().item())
                    rms_grad = float(torch.sqrt(v9).mean().item())

                    # Compare with pretrained channels 0-8
                    v0_8 = s["exp_avg_sq"][:, :9]
                    v0_8_mean = float(v0_8.mean().item())

                    var_res[label] = {
                        "opt_step": step,
                        "exp_avg_norm": m9_norm,
                        "exp_avg_sq_norm": v9_norm,
                        "exp_avg_sq_mean": v9_mean,
                        "exp_avg_sq_max": v9_max,
                        "rms_grad": rms_grad,
                        "nonzero_params": nonzero_grad_count,
                        "total_params": 288,
                        "pretrained_v_mean": v0_8_mean,
                    }

                    print(
                        f"  {label:<18}: OptSteps={step:6.0f} | exp_avg_norm={m9_norm:9.6f} | "
                        f"exp_avg_sq_norm={v9_norm:11.8f} | RMS Grad={rms_grad:9.6f} | "
                        f"Nonzero Grads={nonzero_grad_count}/288 ({nonzero_grad_count/288*100:5.1f}%)"
                    )
            if not found:
                print(f"  {label:<18}: memory_cnn[0].weight not in optimizer state!")

        results[variant_name] = var_res

    return results


def verify_active_gradients():
    """Run an active forward-backward pass on a realistic rollout batch and verify dL/dW[:, 9] != 0."""
    print("\n" + "=" * 90)
    print("3. ACTIVE GRADIENT VERIFICATION: Forward-Backward Pass on Realistic Batch")
    print("=" * 90)

    model_path = ROOT / "models/phase8_binary_visited/final_model.zip"
    model = load_checkpoint(model_path, device="cpu")
    model.policy.train()

    env = PreviousActionWrapper(RescueEnv(
        task_type="exploration_only",
        spatial_memory_mode="visited_binary",
        reward_shaping=False,
    ))

    # Collect 32 diverse transitions
    obs_list = []
    actions_list = []
    masks_list = []

    for seed in [60000, 60001, 60002, 60003]:
        obs, _ = env.reset(seed=seed)
        for _ in range(8):
            masks = env.action_masks()
            obs_list.append({k: torch.as_tensor(v, dtype=torch.float32) for k, v in obs.items()})
            masks_list.append(torch.as_tensor(masks, dtype=torch.bool))
            valid = [a for a in range(7) if masks[a]]
            act = valid[0] if valid else 0
            actions_list.append(torch.tensor(act, dtype=torch.long))
            obs, _, done, _, _ = env.step(act)
            if done:
                break

    env.close()

    batch_size = len(obs_list)
    batch_obs = {
        k: torch.stack([o[k] for o in obs_list])
        for k in obs_list[0].keys()
    }
    batch_actions = torch.stack(actions_list)
    batch_masks = torch.stack(masks_list)
    batch_episode_starts = torch.zeros(batch_size, dtype=torch.bool)
    batch_lstm_states = torch.zeros(1, batch_size, model.policy.hidden_size)

    # Zero all parameter gradients
    model.policy.zero_grad()

    # Forward pass
    values, log_prob, entropy = model.policy.evaluate_actions(
        batch_obs,
        batch_actions,
        lstm_states=batch_lstm_states,
        episode_starts=batch_episode_starts,
        action_masks=batch_masks,
    )

    # Surrogate exploration loss: policy loss + value loss - entropy loss
    pseudo_advantages = torch.randn_like(log_prob)
    pseudo_returns = torch.randn_like(values)
    loss = -(log_prob * pseudo_advantages).mean() + 0.5 * ((values - pseudo_returns) ** 2).mean() - 0.01 * entropy.mean()

    # Backward pass
    loss.backward()

    # Inspect gradient on memory_cnn[0].weight
    grad_all = model.policy.memory_cnn[0].weight.grad
    grad_ch9 = grad_all[:, 9]
    grad_ch0_8 = grad_all[:, :9]

    grad_norm_9 = float(torch.norm(grad_ch9).item())
    grad_mean_abs_9 = float(torch.abs(grad_ch9).mean().item())
    grad_max_abs_9 = float(torch.abs(grad_ch9).max().item())
    nonzero_grads_9 = int((torch.abs(grad_ch9) > 1e-9).sum().item())

    grad_norms_pretrained = torch.norm(grad_ch0_8.permute(1, 0, 2, 3).reshape(9, -1), dim=1)
    grad_norm_pretrained = float(grad_norms_pretrained.mean().item())

    print(f"Batch size: {batch_size} transitions")
    print(f"Surrogate Loss: {loss.item():.4f}")
    print(f"dL / dW[:, 9] Gradient Norm: {grad_norm_9:.8f} (NON-ZERO: {grad_norm_9 > 0})")
    print(f"dL / dW[:, 9] Mean Abs Grad: {grad_mean_abs_9:.8f}")
    print(f"dL / dW[:, 9] Max Abs Grad : {grad_max_abs_9:.8f}")
    print(f"Nonzero Gradient Elements  : {nonzero_grads_9}/288 ({nonzero_grads_9/288*100:.1f}%)")
    print(f"Pretrained Ch 0-8 Mean Norm: {grad_norm_pretrained:.8f}")
    print(f"Ratio (Ch 9 / Pretrained)  : {grad_norm_9 / max(1e-9, grad_norm_pretrained):.4f}")

    return {
        "grad_norm_9": grad_norm_9,
        "grad_mean_abs_9": grad_mean_abs_9,
        "grad_max_abs_9": grad_max_abs_9,
        "nonzero_grads_count": nonzero_grads_9,
        "total_grads_count": 288,
        "grad_norm_pretrained_mean": grad_norm_pretrained,
    }


def inspect_logit_sensitivity():
    """Systematically test how the network's logits respond to realistic changes in visit history."""
    print("\n" + "=" * 90)
    print("4. LOGIT SENSITIVITY: Systematic Response to Realistic Visit History Changes")
    print("=" * 90)

    model_path = ROOT / "models/phase8_binary_visited/final_model.zip"
    model = load_checkpoint(model_path, device="cpu")
    model.policy.eval()

    env = PreviousActionWrapper(RescueEnv(
        task_type="exploration_only",
        spatial_memory_mode="visited_binary",
        reward_shaping=False,
    ))

    # Test across 30 diverse states across different seeds and episode timesteps
    seeds = [60000, 60005, 60010, 60015, 60020, 60025]
    steps_to_sample = [10, 25, 50, 75, 100]

    states = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        for st in range(101):
            masks = env.action_masks()
            if st in steps_to_sample:
                states.append({
                    "seed": seed,
                    "step": st,
                    "obs": {k: np.copy(v) for k, v in obs.items()},
                    "mask": np.copy(masks),
                    "agent_pos": (env.sim.agents[0]["r"], env.sim.agents[0]["c"]),
                })
            valid = [a for a in [1, 2, 3, 4] if masks[a]]
            act = valid[0] if valid else 0
            obs, _, done, _, _ = env.step(act)
            if done:
                break

    env.close()
    print(f"Sampled {len(states)} diverse evaluation states across seeds {seeds}")

    delta_zero_logits = []
    delta_heavy_logits = []
    delta_zero_probs = []
    delta_heavy_probs = []
    delta_zero_values = []
    delta_heavy_values = []
    delta_directional_shifts = []

    lstm_states = torch.zeros(1, 1, model.policy.hidden_size)
    episode_starts = torch.tensor([False])

    for s in states:
        obs_base = {k: torch.as_tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in s["obs"].items()}
        mask_t = torch.as_tensor(np.array([s["mask"]]), dtype=torch.bool)
        valid_actions = [a for a, m in enumerate(s["mask"]) if m]
        ar, ac = s["agent_pos"]

        with torch.no_grad():
            # 1. Base prediction
            lat_b, _ = model.policy._forward_network(obs_base, lstm_states=lstm_states, episode_starts=episode_starts)
            val_b = model.policy.value_net(lat_b).item()
            d_b = model.policy.get_distribution(obs_base, lstm_states=lstm_states, episode_starts=episode_starts, action_masks=mask_t)
            log_b = d_b.distribution.logits[0].clone()
            prob_b = torch.softmax(log_b, dim=-1)

            # 2. Zero-ablation (Ch 9 set to 0.0)
            obs_zero = {k: v.clone() for k, v in obs_base.items()}
            obs_zero["memory"][:, 9, :, :] = 0.0
            lat_z, _ = model.policy._forward_network(obs_zero, lstm_states=lstm_states, episode_starts=episode_starts)
            val_z = model.policy.value_net(lat_z).item()
            d_z = model.policy.get_distribution(obs_zero, lstm_states=lstm_states, episode_starts=episode_starts, action_masks=mask_t)
            log_z = d_z.distribution.logits[0].clone()
            prob_z = torch.softmax(log_z, dim=-1)

            # 3. Heavy visit history (mark 7x7 neighborhood around agent as visited)
            obs_heavy = {k: v.clone() for k, v in obs_base.items()}
            obs_heavy["memory"][:, 9, max(0, ar-3):min(40, ar+4), max(0, ac-3):min(52, ac+4)] = 1.0
            lat_h, _ = model.policy._forward_network(obs_heavy, lstm_states=lstm_states, episode_starts=episode_starts)
            val_h = model.policy.value_net(lat_h).item()
            d_h = model.policy.get_distribution(obs_heavy, lstm_states=lstm_states, episode_starts=episode_starts, action_masks=mask_t)
            log_h = d_h.distribution.logits[0].clone()
            prob_h = torch.softmax(log_h, dim=-1)

            # Metrics
            d_zero_log = (log_b[valid_actions] - log_z[valid_actions]).abs().max().item()
            d_heavy_log = (log_b[valid_actions] - log_h[valid_actions]).abs().max().item()

            d_zero_p = 0.5 * (prob_b[valid_actions] - prob_z[valid_actions]).abs().sum().item()
            d_heavy_p = 0.5 * (prob_b[valid_actions] - prob_h[valid_actions]).abs().sum().item()

            d_zero_v = abs(val_b - val_z)
            d_heavy_v = abs(val_h - val_b)

            delta_zero_logits.append(d_zero_log)
            delta_heavy_logits.append(d_heavy_log)
            delta_zero_probs.append(d_zero_p)
            delta_heavy_probs.append(d_heavy_p)
            delta_zero_values.append(d_zero_v)
            delta_heavy_values.append(d_heavy_v)

            # Directional sensitivity test: if there are >= 2 legal movement actions
            valid_moves = [a for a in [1, 2, 3, 4] if s["mask"][a]]
            if len(valid_moves) >= 2:
                # Top move vs runner-up move
                sorted_moves = sorted(valid_moves, key=lambda a: prob_b[a].item(), reverse=True)
                top_m, sec_m = sorted_moves[0], sorted_moves[1]
                delta_vector = {1: (-2, 0), 2: (0, 2), 3: (2, 0), 4: (0, -2)}
                # Mark top move direction as visited
                obs_top = {k: v.clone() for k, v in obs_base.items()}
                dr, dc = delta_vector[top_m]
                tr, tc = max(0, min(39, ar + dr)), max(0, min(51, ac + dc))
                obs_top["memory"][:, 9, max(0, tr-1):min(40, tr+2), max(0, tc-1):min(52, tc+2)] = 1.0

                d_top = model.policy.get_distribution(obs_top, lstm_states=lstm_states, episode_starts=episode_starts, action_masks=mask_t)
                log_top = d_top.distribution.logits[0].clone()

                # Does marking top move direction as visited reduce its relative advantage?
                # Advantage = logit(top) - logit(runner-up)
                adv_base = (log_b[top_m] - log_b[sec_m]).item()
                adv_top_visited = (log_top[top_m] - log_top[sec_m]).item()
                delta_adv = adv_top_visited - adv_base
                delta_directional_shifts.append(abs(delta_adv))

    mean_d_zero_log = float(np.mean(delta_zero_logits))
    max_d_zero_log = float(np.max(delta_zero_logits))
    mean_d_heavy_log = float(np.mean(delta_heavy_logits))
    max_d_heavy_log = float(np.max(delta_heavy_logits))

    mean_d_heavy_prob = float(np.mean(delta_heavy_probs))
    max_d_heavy_prob = float(np.max(delta_heavy_probs))

    mean_d_heavy_val = float(np.mean(delta_heavy_values))
    max_d_heavy_val = float(np.max(delta_heavy_values))

    mean_dir_shift = float(np.mean(delta_directional_shifts)) if delta_directional_shifts else 0.0

    print(f"Evaluated across {len(states)} diverse realistic environmental states:")
    print(f"\n1. Effect of Zero-Ablating Channel 9 (True Visit History -> All 0s):")
    print(f"   - Max Logit Shift : Mean = {mean_d_zero_log:.6f} | Peak = {max_d_zero_log:.6f}")
    print(f"   - Value Head Shift: Mean = {np.mean(delta_zero_values):.6f} | Peak = {np.max(delta_zero_values):.6f}")

    print(f"\n2. Effect of Adding Realistic Visited Area (Local Area Visited):")
    print(f"   - Max Logit Shift : Mean = {mean_d_heavy_log:.6f} | Peak = {max_d_heavy_log:.6f}")
    print(f"   - Action Prob Shift (TVD): Mean = {mean_d_heavy_prob:.6f} | Peak = {max_d_heavy_prob:.6f}")
    print(f"   - Value Head Shift: Mean = {mean_d_heavy_val:.6f} | Peak = {max_d_heavy_val:.6f}")
    print(f"   - Relative Directional Logit Shift: Mean = {mean_dir_shift:.6f}")

    return {
        "n_states_evaluated": len(states),
        "zero_ablation": {
            "mean_max_logit_delta": mean_d_zero_log,
            "peak_max_logit_delta": max_d_zero_log,
            "mean_value_delta": float(np.mean(delta_zero_values)),
            "peak_value_delta": float(np.max(delta_zero_values)),
        },
        "realistic_visit_addition": {
            "mean_max_logit_delta": mean_d_heavy_log,
            "peak_max_logit_delta": max_d_heavy_log,
            "mean_total_variation_prob_delta": mean_d_heavy_prob,
            "peak_total_variation_prob_delta": max_d_heavy_prob,
            "mean_value_delta": mean_d_heavy_val,
            "peak_value_delta": max_d_heavy_val,
            "mean_directional_relative_shift": mean_dir_shift,
        },
    }


def main():
    print("=" * 90)
    print("PHASE 8 CHANNEL 9 WEIGHTS, GRADIENTS & LOGIT SENSITIVITY AUDIT")
    print("=" * 90)

    weights_data = inspect_weights_progression()
    optimizer_data = inspect_optimizer_gradients()
    active_grad_data = verify_active_gradients()
    sensitivity_data = inspect_logit_sensitivity()

    audit_summary = {
        "weights_progression": weights_data,
        "optimizer_moments": optimizer_data,
        "active_gradients": active_grad_data,
        "logit_sensitivity": sensitivity_data,
    }

    out_json = ROOT / "models/phase8_evaluations/channel9_weights_gradients_audit.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print(f"\n[SUCCESS] Saved comprehensive audit JSON to: {out_json}")


if __name__ == "__main__":
    main()
