#!/usr/bin/env python3
"""Phase 11: Investigate Variant C Pretrained Policy Reproduction and Zero-Shot Behavior.

Specifically addresses:
1. Weight transfer integrity from Phase 10 checkpoint into Variant C.
2. Tensor-level verification of local crop vs. raw raycast grid:
   - Unknown cells (never observed)
   - Currently visible cells (in line-of-sight)
   - Historical observations (previously observed, currently occluded)
   - Boundaries (canvas borders and building edges)
   - Channel-by-channel comparison (0..8)
3. Action logits, probability distributions, and recurrent trajectory rollout comparison:
   - Logit L1/L2 distance, KL divergence, argmax action agreement.
   - Recurrent hidden state divergence ||h_t^A - h_t^C|| over multi-step rollouts.
   - First divergence step and behavioral cause analysis.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from agents.cnn import (
    RecurrentSpatialRescueCNN,
    inspect_checkpoint,
    transfer_phase11_weights,
)
from rescue_env.envs.single_agent import RescueEnv
from rescue_env.wrappers import PreviousActionWrapper

PHASE10_CKPT = ROOT / "models/phase10_variant_b/final_model.zip"


def audit_tensor_reproduction(env: Any, model_c: RecurrentSpatialRescueCNN, n_episodes: int = 25, max_steps_per_ep: int = 64):
    """Deep tensor audit comparing obs['grid'] and _extract_memory_crop(obs['memory'])."""
    print("=" * 80)
    print("PART 2: TENSOR-LEVEL REPRODUCTION AUDIT (LOCAL CROP VS. RAYCAST GRID)")
    print("=" * 80)

    stats = {
        "total_states": 0,
        "step0_states": 0,
        "step0_bit_exact": 0,
        "channel_stats": {ch: {"diff_sum": 0.0, "diff_max": 0.0, "exact_match_cells": 0, "total_cells": 0} for ch in range(9)},
        "los_visible_cells": 0,
        "los_exact_match": 0,
        "never_seen_cells": 0,
        "never_seen_exact_match": 0,
        "historical_occluded_cells": 0,
        "historical_retained_in_crop": 0,
        "boundary_states": 0,
        "boundary_pad_exact": 0,
    }

    for ep in range(n_episodes):
        obs, info = env.reset(seed=50000 + ep)
        done = False
        step = 0

        while not done and step < max_steps_per_ep:
            stats["total_states"] += 1
            if step == 0:
                stats["step0_states"] += 1

            grid_np = obs["grid"]  # (9, 15, 15)
            mem_np = obs["memory"]  # (9, 40, 52)
            person = env.unwrapped.sim.agents[0]
            agent_r, agent_c = person["r"], person["c"]

            # PyTorch tensors
            mem_t = torch.as_tensor(mem_np, dtype=torch.float32).unsqueeze(0)
            grid_t = torch.as_tensor(grid_np, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                crop_t = model_c._extract_memory_crop(mem_t)

            crop_np = crop_t[0].cpu().numpy()

            # Check step 0 bit-exactness
            if step == 0:
                if np.allclose(grid_np, crop_np, atol=1e-6):
                    stats["step0_bit_exact"] += 1

            # Channel-by-channel comparison
            for ch in range(9):
                diff = np.abs(grid_np[ch] - crop_np[ch])
                stats["channel_stats"][ch]["diff_sum"] += float(np.sum(diff))
                stats["channel_stats"][ch]["diff_max"] = max(stats["channel_stats"][ch]["diff_max"], float(np.max(diff)))
                stats["channel_stats"][ch]["exact_match_cells"] += int(np.sum(diff < 1e-6))
                stats["channel_stats"][ch]["total_cells"] += 15 * 15

            # Cell categories:
            # Grid channel 5 > 0 is LOS visible in current step
            los_mask = (grid_np[5] > 0)
            stats["los_visible_cells"] += int(np.sum(los_mask))
            # On LOS visible cells, does crop match grid?
            los_diff = np.abs(grid_np[:, los_mask] - crop_np[:, los_mask])
            stats["los_exact_match"] += int(np.sum(np.all(los_diff < 1e-6, axis=0)))

            # Crop channel 5 == 0 and grid channel 5 == 0: Never seen cells
            never_seen_mask = (grid_np[5] == 0) & (crop_np[5] == 0)
            stats["never_seen_cells"] += int(np.sum(never_seen_mask))
            never_seen_diff = np.abs(grid_np[:, never_seen_mask] - crop_np[:, never_seen_mask])
            stats["never_seen_exact_match"] += int(np.sum(np.all(never_seen_diff < 1e-6, axis=0)))

            # Historical occluded: grid channel 5 == 0, but crop channel 5 > 0
            # (Previously explored in memory, but currently occluded by LOS raycast)
            historical_mask = (grid_np[5] == 0) & (crop_np[5] > 0)
            n_hist = int(np.sum(historical_mask))
            stats["historical_occluded_cells"] += n_hist
            if n_hist > 0:
                # In crop, explored mask is 1.0
                stats["historical_retained_in_crop"] += int(np.sum(crop_np[5, historical_mask] > 0))

            # Boundary check: Agent within 7 cells of world border
            if agent_r < 7 or agent_c < 7 or agent_r >= (env.unwrapped.memory.height - 7) or agent_c >= (env.unwrapped.memory.width - 7):
                stats["boundary_states"] += 1
                # Check center agent marker (7, 7)
                if crop_np[8, 7, 7] == 1.0 and np.sum(crop_np[8]) == 1.0:
                    stats["boundary_pad_exact"] += 1

            # Step environment with legal action
            action = int(np.random.choice(np.where(env.action_masks())[0]))
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            step += 1

    return stats


def compare_logits_and_distributions(env: Any, model_a: Any, model_c: Any, n_states: int = 1000):
    """Compares action logits and action distributions between Model A (grid) and Model C (crop)."""
    print("\n" + "=" * 80)
    print("PART 3A: ACTION LOGITS & DISTRIBUTION COMPARISON (ZERO-SHOT)")
    print("=" * 80)

    l1_dists = []
    l2_dists = []
    kl_divs = []
    cos_sims = []
    agreements = []
    state_types = {"step0": [], "historical": [], "all": []}

    obs, info = env.reset(seed=12345)
    done = False
    step_in_ep = 0

    for s_idx in range(n_states):
        masks = torch.as_tensor(env.action_masks(), dtype=torch.bool).unsqueeze(0)
        t_obs = {
            "grid": torch.as_tensor(obs["grid"], dtype=torch.float32).unsqueeze(0),
            "memory": torch.as_tensor(obs["memory"], dtype=torch.float32).unsqueeze(0),
            "agent": torch.as_tensor(obs["agent"], dtype=torch.float32).unsqueeze(0),
            "previous_action": torch.as_tensor([obs["previous_action"]], dtype=torch.int64),
        }

        with torch.no_grad():
            la = model_a.get_masked_logits(t_obs, action_masks=masks)
            lc = model_c.get_masked_logits(t_obs, action_masks=masks)

        valid_idx = masks[0].nonzero().squeeze(-1)
        valid_la = la[0, valid_idx]
        valid_lc = lc[0, valid_idx]

        # Logit differences
        l1 = (valid_la - valid_lc).abs().mean().item()
        l2 = (valid_la - valid_lc).norm(2).item()
        cos = F.cosine_similarity(valid_la.unsqueeze(0), valid_lc.unsqueeze(0)).item()

        # Probabilities
        pa = F.softmax(valid_la, dim=-1)
        pc = F.softmax(valid_lc, dim=-1)
        # KL(A || C)
        kl = (pa * (torch.log(pa + 1e-12) - torch.log(pc + 1e-12))).sum().item()

        act_a = la.argmax(dim=-1).item()
        act_c = lc.argmax(dim=-1).item()
        agree = (act_a == act_c)

        l1_dists.append(l1)
        l2_dists.append(l2)
        kl_divs.append(kl)
        cos_sims.append(cos)
        agreements.append(agree)

        crop_t = model_c._extract_memory_crop(t_obs["memory"])
        is_step0 = (step_in_ep == 0)
        is_hist = bool(torch.any((t_obs["grid"][:, 5] == 0) & (crop_t[:, 5] > 0)).item())

        state_types["all"].append(agree)
        if is_step0:
            state_types["step0"].append(agree)
        if is_hist:
            state_types["historical"].append(agree)

        # Step environment
        action = act_a if np.random.rand() < 0.5 else act_c
        if not env.action_masks()[action]:
            action = int(np.where(env.action_masks())[0][0])
        obs, r, term, trunc, info = env.step(action)
        step_in_ep += 1
        if term or trunc:
            obs, info = env.reset()
            step_in_ep = 0

    return {
        "n_states": n_states,
        "mean_l1_distance": float(np.mean(l1_dists)),
        "max_l1_distance": float(np.max(l1_dists)),
        "mean_l2_distance": float(np.mean(l2_dists)),
        "mean_cosine_similarity": float(np.mean(cos_sims)),
        "mean_kl_divergence": float(np.mean(kl_divs)),
        "argmax_agreement_overall_pct": float(100.0 * np.mean(agreements)),
        "argmax_agreement_step0_pct": float(100.0 * np.mean(state_types["step0"])) if state_types["step0"] else 100.0,
        "argmax_agreement_historical_pct": float(100.0 * np.mean(state_types["historical"])) if state_types["historical"] else 0.0,
    }


def compare_recurrent_trajectories(env: Any, model_a: Any, model_c: Any, n_episodes: int = 50, max_steps: int = 128):
    """Runs matched recurrent trajectory rollouts from identical initial seeds.

    Measures:
    - Step of first action divergence between Agent A and Agent C.
    - Total action agreement along trajectory.
    - Hidden state divergence ||h_t^A - h_t^C||_2 as a function of timestep t.
    """
    print("\n" + "=" * 80)
    print("PART 3B: RECURRENT TRAJECTORY DIVERGENCE AUDIT (50 MATCHED EPISODES)")
    print("=" * 80)

    divergence_steps = []
    agreement_rates = []
    hidden_distances_by_step = [[] for _ in range(max_steps)]

    for ep in range(n_episodes):
        ep_seed = 50000 + ep

        # Run Episode with Agent A
        obs_a, info_a = env.reset(seed=ep_seed, options={"victim_distance": 12, "min_victim_distance": 7, "max_steps": max_steps})
        h_a = torch.zeros((1, 1, model_a.hidden_size))
        actions_a = []
        hidden_states_a = []

        done = False
        step = 0
        while not done and step < max_steps:
            masks = torch.as_tensor(env.action_masks(), dtype=torch.bool).unsqueeze(0)
            t_obs = {
                "grid": torch.as_tensor(obs_a["grid"], dtype=torch.float32).unsqueeze(0),
                "memory": torch.as_tensor(obs_a["memory"], dtype=torch.float32).unsqueeze(0),
                "agent": torch.as_tensor(obs_a["agent"], dtype=torch.float32).unsqueeze(0),
                "previous_action": torch.as_tensor([obs_a["previous_action"]], dtype=torch.int64),
            }
            with torch.no_grad():
                act, val, logp, next_h = model_a.forward(
                    t_obs, lstm_states=h_a, episode_starts=torch.tensor([step == 0]), deterministic=True, action_masks=masks, return_hidden_state=True
                )
            action = act.item()
            actions_a.append(action)
            hidden_states_a.append(next_h.squeeze(0).squeeze(0).cpu().numpy())
            h_a = next_h
            obs_a, r, term, trunc, info_a = env.step(action)
            done = term or trunc
            step += 1

        # Run Episode with Agent C on EXACT SAME seed
        obs_c, info_c = env.reset(seed=ep_seed, options={"victim_distance": 12, "min_victim_distance": 7, "max_steps": max_steps})
        h_c = torch.zeros((1, 1, model_c.hidden_size))
        actions_c = []
        hidden_states_c = []

        done = False
        step = 0
        while not done and step < max_steps:
            masks = torch.as_tensor(env.action_masks(), dtype=torch.bool).unsqueeze(0)
            t_obs = {
                "grid": torch.as_tensor(obs_c["grid"], dtype=torch.float32).unsqueeze(0),
                "memory": torch.as_tensor(obs_c["memory"], dtype=torch.float32).unsqueeze(0),
                "agent": torch.as_tensor(obs_c["agent"], dtype=torch.float32).unsqueeze(0),
                "previous_action": torch.as_tensor([obs_c["previous_action"]], dtype=torch.int64),
            }
            with torch.no_grad():
                act, val, logp, next_h = model_c.forward(
                    t_obs, lstm_states=h_c, episode_starts=torch.tensor([step == 0]), deterministic=True, action_masks=masks, return_hidden_state=True
                )
            action = act.item()
            actions_c.append(action)
            hidden_states_c.append(next_h.squeeze(0).squeeze(0).cpu().numpy())
            h_c = next_h
            obs_c, r, term, trunc, info_c = env.step(action)
            done = term or trunc
            step += 1

        # Compare trajectories
        min_len = min(len(actions_a), len(actions_c))
        first_div = min_len
        matches = 0
        for t in range(min_len):
            if actions_a[t] == actions_c[t]:
                matches += 1
            elif first_div == min_len:
                first_div = t

            # Hidden distance
            dist = float(np.linalg.norm(hidden_states_a[t] - hidden_states_c[t]))
            hidden_distances_by_step[t].append(dist)

        divergence_steps.append(first_div)
        agreement_rates.append(100.0 * matches / max(1, min_len))

    # Compute step-wise hidden distance means
    mean_hidden_by_step = []
    for t in range(max_steps):
        if hidden_distances_by_step[t]:
            mean_hidden_by_step.append(float(np.mean(hidden_distances_by_step[t])))
        else:
            break

    return {
        "n_episodes": n_episodes,
        "mean_first_divergence_step": float(np.mean(divergence_steps)),
        "median_first_divergence_step": float(np.median(divergence_steps)),
        "episodes_zero_divergence_pct": float(100.0 * np.mean([d == max_steps for d in divergence_steps])),
        "mean_trajectory_action_agreement_pct": float(np.mean(agreement_rates)),
        "hidden_divergence_step_0": mean_hidden_by_step[0] if mean_hidden_by_step else 0.0,
        "hidden_divergence_step_1": mean_hidden_by_step[1] if len(mean_hidden_by_step) > 1 else 0.0,
        "hidden_divergence_step_5": mean_hidden_by_step[5] if len(mean_hidden_by_step) > 5 else 0.0,
        "hidden_divergence_step_10": mean_hidden_by_step[10] if len(mean_hidden_by_step) > 10 else 0.0,
        "hidden_divergence_step_20": mean_hidden_by_step[20] if len(mean_hidden_by_step) > 20 else 0.0,
    }


def main():
    if not PHASE10_CKPT.exists():
        print(f"Error: Phase 10 checkpoint not found: {PHASE10_CKPT}")
        sys.exit(1)

    print("=" * 80)
    print("PHASE 11: VARIANT C ZERO-SHOT INVESTIGATION & REPRODUCTION BENCHMARK")
    print(f"Base Checkpoint: {PHASE10_CKPT}")
    print("=" * 80)

    # Instantiate Environment
    env = PreviousActionWrapper(RescueEnv(reward_shaping=False, geometry_mode="edge"))

    # Part 1: Load Phase 10 Checkpoint into Variant A and Variant C
    print("\nPART 1: LOADING PRETRAINED PHASE 10 WEIGHTS VIA COMPATIBLE TRANSFER...")
    model_a = RecurrentSpatialRescueCNN(env.observation_space, env.action_space, architecture_variant="current")
    model_c = RecurrentSpatialRescueCNN(env.observation_space, env.action_space, architecture_variant="memory_crop")

    res_a = transfer_phase11_weights(PHASE10_CKPT, model_a)
    res_c = transfer_phase11_weights(PHASE10_CKPT, model_c)

    print(f"  Model A (Current) Transfer: {res_a['status']} ({res_a['transferred_params']:,} / {res_a['total_target_params']:,} params, {res_a['transfer_coverage_pct']:.2f}%)")
    print(f"  Model C (Memory Crop) Transfer: {res_c['status']} ({res_c['transferred_params']:,} / {res_c['total_target_params']:,} params, {res_c['transfer_coverage_pct']:.2f}%)")

    model_a.eval()
    model_c.eval()

    # Part 2: Tensor Reproduction Audit
    tensor_audit = audit_tensor_reproduction(env, model_c, n_episodes=25, max_steps_per_ep=64)
    print(f"  Total States Audited: {tensor_audit['total_states']:,}")
    print(f"  Step 0 States: {tensor_audit['step0_states']}, Bit-Exact: {tensor_audit['step0_bit_exact']} ({100.0 * tensor_audit['step0_bit_exact'] / max(1, tensor_audit['step0_states']):.2f}%)")
    print(f"  LOS Visible Cells Audited: {tensor_audit['los_visible_cells']:,}, Exact Match: {tensor_audit['los_exact_match']:,} ({100.0 * tensor_audit['los_exact_match'] / max(1, tensor_audit['los_visible_cells']):.4f}%)")
    print(f"  Never-Seen Cells Audited: {tensor_audit['never_seen_cells']:,}, Exact Match: {tensor_audit['never_seen_exact_match']:,} ({100.0 * tensor_audit['never_seen_exact_match'] / max(1, tensor_audit['never_seen_cells']):.4f}%)")
    print(f"  Historical Occluded Cells: {tensor_audit['historical_occluded_cells']:,}, Retained in Crop: {tensor_audit['historical_retained_in_crop']:,} ({100.0 * tensor_audit['historical_retained_in_crop'] / max(1, tensor_audit['historical_occluded_cells']):.2f}%)")
    print(f"  Boundary States Audited: {tensor_audit['boundary_states']:,}, Padding Exact: {tensor_audit['boundary_pad_exact']:,} ({100.0 * tensor_audit['boundary_pad_exact'] / max(1, tensor_audit['boundary_states']):.2f}%)")

    print("\n  Per-Channel Match Rates:")
    channel_names = ["Pass N", "Pass E", "Pass S", "Pass W", "Debris", "Explored", "Victim", "Exit", "Agent Pos"]
    for ch in range(9):
        cs = tensor_audit["channel_stats"][ch]
        pct = 100.0 * cs["exact_match_cells"] / max(1, cs["total_cells"])
        print(f"    Ch {ch} ({channel_names[ch]:10s}): Exact Match = {pct:6.2f}%, Max Diff = {cs['diff_max']:.6f}")

    # Part 3A: Logits & Probability Distributions
    logit_audit = compare_logits_and_distributions(env, model_a, model_c, n_states=1000)
    print(f"  Logits Audited: {logit_audit['n_states']:,} states")
    print(f"  Mean Valid L1 Distance: {logit_audit['mean_l1_distance']:.6f} (Max L1: {logit_audit['max_l1_distance']:.6f})")
    print(f"  Mean Valid L2 Distance: {logit_audit['mean_l2_distance']:.6f}")
    print(f"  Mean Cosine Similarity: {logit_audit['mean_cosine_similarity']:.6f}")
    print(f"  Mean KL Divergence:     {logit_audit['mean_kl_divergence']:.6f}")
    print(f"  Argmax Action Agreement (Overall):    {logit_audit['argmax_agreement_overall_pct']:.2f}%")
    print(f"  Argmax Action Agreement (Step 0):      {logit_audit['argmax_agreement_step0_pct']:.2f}%")
    print(f"  Argmax Action Agreement (Historical): {logit_audit['argmax_agreement_historical_pct']:.2f}%")

    # Part 3B: Recurrent Trajectory Rollouts
    traj_audit = compare_recurrent_trajectories(env, model_a, model_c, n_episodes=50, max_steps=128)
    print(f"  Trajectories Audited: {traj_audit['n_episodes']} matched episodes")
    print(f"  Mean First Divergence Step:   {traj_audit['mean_first_divergence_step']:.1f} steps (Median: {traj_audit['median_first_divergence_step']:.1f})")
    print(f"  Mean Trajectory Action Match: {traj_audit['mean_trajectory_action_agreement_pct']:.1f}%")
    print(f"  Recurrent Hidden State Divergence ||h_t^A - h_t^C||:")
    print(f"    Step 0:  {traj_audit['hidden_divergence_step_0']:.6f}")
    print(f"    Step 1:  {traj_audit['hidden_divergence_step_1']:.6f}")
    print(f"    Step 5:  {traj_audit['hidden_divergence_step_5']:.6f}")
    print(f"    Step 10: {traj_audit['hidden_divergence_step_10']:.6f}")
    print(f"    Step 20: {traj_audit['hidden_divergence_step_20']:.6f}")

    # Save summary json
    out_dir = ROOT / "models/phase11_evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "part1_weight_transfer": {
            "model_a": res_a,
            "model_c": res_c,
        },
        "part2_tensor_reproduction": tensor_audit,
        "part3a_logit_comparison": logit_audit,
        "part3b_trajectory_comparison": traj_audit,
    }
    out_json = out_dir / "variant_c_zero_shot_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved Zero-Shot Audit Results to: {out_json}")

    env.close()


if __name__ == "__main__":
    main()
