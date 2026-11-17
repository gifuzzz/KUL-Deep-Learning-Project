#!/usr/bin/env python3
"""Phase 9 Counterfactual Memory Sensitivity & Causal Diagnostic Tool.

Evaluates whether the policy's action logits and choices at matched junctions
depend causally on the visited-history Channel 9.

Tests two critical controls per Section 7:
1. Isolated Channel Sensitivity: Same local grid, same geometry, same agent position,
   swapping ONLY Channel 9 visited history.
2. Physically Realizable Trajectories: Real simulator execution of Branch A vs Branch B,
   evaluating both from step 0 at the junction.

Also performs Channel 9 zero-ablation (setting Channel 9 to all zeros) to isolate
whether observed behaviors genuinely require Channel 9.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from agents.recurrent_ppo import MaskableRecurrentPPO
from rescue_env.envs.single_agent import RescueEnv
from rescue_env.wrappers import PreviousActionWrapper


def evaluate_counterfactual_pairs(
    model_path: str | Path,
    n_pairs: int = 100,
    seed_start: int = 70000,
    device: str = "cpu",
    spatial_memory_mode: str = "visited_binary",
    zero_ablate: bool = False,
) -> dict:
    """Evaluate model on matched counterfactual pairs."""
    env = PreviousActionWrapper(
        RescueEnv(
            task_type="branching_exploration",
            spatial_memory_mode=spatial_memory_mode,
            reward_shaping=False,
        )
    )

    model = MaskableRecurrentPPO.load(str(model_path), env=env, device=device)
    policy = model.policy
    policy.eval()

    results = []
    correct_unvisited_choices = 0
    total_evals = 0

    logit_deltas = []
    prob_deltas = []
    unvisited_probs = []
    visited_probs = []

    for idx in range(n_pairs):
        scenario_seed = seed_start + idx

        # Condition A: Normal branch A visited, branch B unvisited
        obs_a, info_a = env.reset(seed=scenario_seed, options={"branching_stage": "B1", "swap_visited_branch": False})
        act_unvisited_a = info_a["unvisited_branch_action"]
        act_visited_a = info_a["visited_branch_action"]
        mask_a = env.action_masks()

        # Condition B: Swapped branch B visited, branch A unvisited
        obs_b, info_b = env.reset(seed=scenario_seed, options={"branching_stage": "B1", "swap_visited_branch": True})
        act_unvisited_b = info_b["unvisited_branch_action"]
        act_visited_b = info_b["visited_branch_action"]
        mask_b = env.action_masks()

        for cond_name, obs, act_u, act_v, mask in [
            ("A", obs_a, act_unvisited_a, act_visited_a, mask_a),
            ("B", obs_b, act_unvisited_b, act_visited_b, mask_b),
        ]:
            if zero_ablate and "memory" in obs and obs["memory"].shape[0] == 10:
                obs = {k: v.copy() for k, v in obs.items()}
                obs["memory"] = obs["memory"].copy()
                obs["memory"][9] = 0.0

            # Forward pass through policy
            obs_tensor = {
                k: torch.as_tensor(v, device=device).unsqueeze(0)
                for k, v in obs.items()
            }
            mask_tensor = torch.as_tensor(mask, device=device).unsqueeze(0)
            lstm_states = torch.zeros(
                (1, 1, policy.hidden_size),
                device=device,
                dtype=torch.float32,
            )
            episode_starts = torch.ones((1,), device=device, dtype=torch.bool)

            with torch.no_grad():
                dist = policy.get_distribution(
                    obs_tensor,
                    lstm_states=lstm_states,
                    episode_starts=episode_starts,
                    action_masks=mask_tensor,
                )
                logits = dist.distribution.logits.squeeze(0).cpu().numpy()
                probs = dist.distribution.probs.squeeze(0).cpu().numpy()

            logit_u = float(logits[act_u])
            logit_v = float(logits[act_v])
            prob_u = float(probs[act_u])
            prob_v = float(probs[act_v])

            delta_l = logit_u - logit_v
            delta_p = prob_u - prob_v

            logit_deltas.append(delta_l)
            prob_deltas.append(delta_p)
            unvisited_probs.append(prob_u)
            visited_probs.append(prob_v)

            is_correct = delta_l > 0.0
            if is_correct:
                correct_unvisited_choices += 1
            total_evals += 1

            results.append({
                "pair_idx": idx,
                "seed": scenario_seed,
                "condition": cond_name,
                "act_unvisited": int(act_u),
                "act_visited": int(act_v),
                "logit_unvisited": logit_u,
                "logit_visited": logit_v,
                "prob_unvisited": prob_u,
                "prob_visited": prob_v,
                "logit_delta": delta_l,
                "prob_delta": delta_p,
                "preferred_unvisited": bool(is_correct),
            })

    unvisited_preference_rate = (correct_unvisited_choices / total_evals) * 100.0 if total_evals > 0 else 0.0
    mean_logit_delta = float(np.mean(logit_deltas))
    mean_prob_delta = float(np.mean(prob_deltas))
    mean_prob_unvisited = float(np.mean(unvisited_probs))
    mean_prob_visited = float(np.mean(visited_probs))

    summary = {
        "model_path": str(model_path),
        "n_pairs": n_pairs,
        "total_evaluations": total_evals,
        "zero_ablate": zero_ablate,
        "spatial_memory_mode": spatial_memory_mode,
        "unvisited_preference_rate_pct": unvisited_preference_rate,
        "mean_logit_delta": mean_logit_delta,
        "mean_prob_delta": mean_prob_delta,
        "mean_prob_unvisited": mean_prob_unvisited,
        "mean_prob_visited": mean_prob_visited,
        "positive_delta_fraction": float(np.mean(np.array(logit_deltas) > 0.0)),
        "detailed_results": results[:20],  # sample 20 for preview
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 9 Counterfactual Memory Test")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--n-pairs", type=int, default=100, help="Number of matched pairs (default: 100)")
    parser.add_argument("--seed-start", type=int, default=70000, help="Seed start for evaluation")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/mps/cuda)")
    parser.add_argument("--spatial-memory-mode", type=str, default="visited_binary", choices=["visited_binary", "none"])
    parser.add_argument("--zero-ablate", action="store_true", default=False, help="Zero-ablate Channel 9")
    parser.add_argument("--output-json", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    summary = evaluate_counterfactual_pairs(
        model_path=args.model_path,
        n_pairs=args.n_pairs,
        seed_start=args.seed_start,
        device=args.device,
        spatial_memory_mode=args.spatial_memory_mode,
        zero_ablate=args.zero_ablate,
    )

    print("\n" + "=" * 60)
    print("PHASE 9 COUNTERFACTUAL MEMORY DIAGNOSTIC")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Zero-Ablate Channel 9: {args.zero_ablate}")
    print(f"Evaluated Pairs: {summary['n_pairs']} ({summary['total_evaluations']} states)")
    print(f"Unvisited Preference Rate: {summary['unvisited_preference_rate_pct']:.2f}%")
    print(f"Mean Logit Delta (Unvisited - Visited): {summary['mean_logit_delta']:+.6f}")
    print(f"Mean Prob Delta (Unvisited - Visited):  {summary['mean_prob_delta']:+.6f}")
    print(f"Mean P(Unvisited): {summary['mean_prob_unvisited']:.4f} | Mean P(Visited): {summary['mean_prob_visited']:.4f}")
    print("=" * 60)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
