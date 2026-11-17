#!/usr/bin/env python3
"""Phase 10 Comprehensive Evaluation Suite.

Evaluates trained Phase 10 variants (A, B, C, D) alongside references:
- Phase 6 Production Baseline
- Phase 9 Best Navigation Model

Evaluations performed:
1. Pure Navigation and Exploration Benchmark (200 episodes, seeds 60000..60199)
   - Unique cells visited
   - Unique rooms entered
   - Reachable-cell coverage
   - New information discovered after step 50
   - Wait fraction, max wait streak, wait streaks >10, >50, >100
   - 2-cell oscillation percentage, doorway reversals, productive backtracking
2. Branching Navigation Benchmark (Stages B1 & B2, 100 episodes, seeds 60000..60099)
   - Correct branch selection rate (%)
   - Entered new branch rate (%)
   - Wait action fraction (%)
3. Full Rescue Evaluation (200 matched episodes per tier, seeds 50000..50199)
   - Spawn-relative ladder (5 tiers: 2-6, 7-12, 13-20, 21-35, Distant)
   - Historical exit-relative ladder (6 tiers: Near, Medium, Bridge A, Bridge B, Bridge C, Distant)
   - Metrics: Discovery, Pickup, Rescue, Return-after-pickup, Carrier timeouts, Stalling, Oscillation
4. Failure Decomposition (mutually exclusive rescue stages + overlapping pathologies)
5. Learned Wait Preferences Analysis (action probabilities at deadlock states)

All evaluations are strictly unshaped (reward_shaping=False, native +10 rescue only, zero privileged state).
"""

from __future__ import annotations

import argparse
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

from rescue_env.envs.single_agent import RescueEnv
from rescue_env.wrappers import PreviousActionWrapper
from agents.recurrent_ppo import load_checkpoint, MaskableRecurrentPPO


SPAWN_TIERS = [
    ("spawn_2_6", 6, 2, 64),
    ("spawn_7_12", 12, 7, 128),
    ("spawn_13_20", 20, 13, 192),
    ("spawn_21_35", 35, 21, 288),
    ("spawn_distant", None, 1, 384),
]

EXIT_TIERS = [
    ("near", 2, 1, 64),
    ("medium", 6, 3, 128),
    ("bridge_a", 8, 5, 160),
    ("bridge_b", 10, 7, 192),
    ("bridge_c", 12, 9, 256),
    ("distant", None, 1, 384),
]


def evaluate_navigation(
    model: MaskableRecurrentPPO,
    n_episodes: int = 200,
    seed_start: int = 60000,
    deterministic: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluates pure exploration and navigation behavior on full procedural layouts."""
    env = PreviousActionWrapper(RescueEnv(
        task_type="exploration_only",
        geometry_mode="edge",
        reward_shaping=False,
        max_steps=384,
        distance_metric="spawn",
        spatial_memory_mode="none",
    ))

    obs_coverages = []
    vis_coverages = []
    unique_cells = []
    unique_rooms = []
    wait_fractions = []
    max_wait_streaks = []
    streaks_gt10 = 0
    streaks_gt50 = 0
    streaks_gt100 = 0
    oscillations = []
    doorway_reversals = []
    new_after_50_count = []
    lengths = []

    for ep in range(n_episodes):
        ep_seed = seed_start + ep
        obs, _ = env.reset(seed=ep_seed)
        lstm_states = torch.zeros(1, 1, model.policy.hidden_size, device=device)
        episode_starts = torch.tensor([True], device=device)

        done = False
        step_count = 0

        while not done and step_count < 384:
            action_mask = env.action_masks()
            obs_t = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}
            mask_t = torch.as_tensor(np.array([action_mask], dtype=bool), device=device)

            with torch.no_grad():
                actions, _, _, lstm_states = model.policy.forward(
                    obs_t,
                    lstm_states=lstm_states,
                    episode_starts=episode_starts,
                    deterministic=deterministic,
                    action_masks=mask_t,
                    return_hidden_state=True,
                )
                action = int(actions.item())
                episode_starts = torch.tensor([False], device=device)

            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            done = terminated or truncated

        obs_coverages.append(float(info.get("observed_coverage", 0.0)))
        vis_coverages.append(float(info.get("visited_coverage", 0.0)))
        unique_cells.append(float(info.get("visited_reachable_cells", 0)))
        unique_rooms.append(float(info.get("unique_rooms_entered", 1)))
        wait_fractions.append(float(info.get("wait_action_fraction", 0.0)))
        max_streak = int(info.get("max_consecutive_waits", 0))
        max_wait_streaks.append(max_streak)
        if max_streak >= 10:
            streaks_gt10 += 1
        if max_streak >= 50:
            streaks_gt50 += 1
        if max_streak >= 100:
            streaks_gt100 += 1
        oscillations.append(float(info.get("oscillation_percentage", 0.0)))
        doorway_reversals.append(float(info.get("doorway_reversals", 0)))
        new_after_50_count.append(float(info.get("new_cells_after_step_50", 0)))
        lengths.append(step_count)

    env.close()

    return {
        "n_episodes": n_episodes,
        "mode": "deterministic" if deterministic else "stochastic",
        "mean_unique_cells_visited": float(np.mean(unique_cells)),
        "mean_unique_rooms_entered": float(np.mean(unique_rooms)),
        "mean_observed_coverage_pct": float(np.mean(obs_coverages) * 100.0),
        "mean_visited_coverage_pct": float(np.mean(vis_coverages) * 100.0),
        "mean_wait_fraction_pct": float(np.mean(wait_fractions) * 100.0),
        "mean_max_wait_streak": float(np.mean(max_wait_streaks)),
        "episodes_wait_streak_gt10_pct": float(streaks_gt10 / n_episodes * 100.0),
        "episodes_wait_streak_gt50_pct": float(streaks_gt50 / n_episodes * 100.0),
        "episodes_wait_streak_gt100_pct": float(streaks_gt100 / n_episodes * 100.0),
        "mean_oscillation_pct": float(np.mean(oscillations)),
        "mean_doorway_reversals": float(np.mean(doorway_reversals)),
        "mean_new_cells_after_step_50": float(np.mean(new_after_50_count)),
        "episodes_with_late_discovery_pct": float(np.mean([1.0 if x > 0 else 0.0 for x in new_after_50_count]) * 100.0),
        "mean_episode_length": float(np.mean(lengths)),
    }


def evaluate_branching(
    model: MaskableRecurrentPPO,
    n_episodes: int = 100,
    seed_start: int = 60000,
    deterministic: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluates branching navigation on Stages B1 and B2."""
    env = PreviousActionWrapper(RescueEnv(
        task_type="branching_exploration",
        geometry_mode="edge",
        reward_shaping=False,
        distance_metric="spawn",
        spatial_memory_mode="none",
    ))

    results = {}
    for stage, max_s in [("B1", 64), ("B2", 128)]:
        correct = []
        entered_new = []
        waits = []

        for ep in range(n_episodes):
            ep_seed = seed_start + ep
            obs, info = env.reset(seed=ep_seed, options={"branching_stage": stage, "max_steps": max_s})
            lstm_states = torch.zeros(1, 1, model.policy.hidden_size, device=device)
            episode_starts = torch.tensor([True], device=device)

            done = False
            step_count = 0
            while not done and step_count < max_s:
                action_mask = env.action_masks()
                obs_t = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}
                mask_t = torch.as_tensor(np.array([action_mask], dtype=bool), device=device)

                with torch.no_grad():
                    actions, _, _, lstm_states = model.policy.forward(
                        obs_t,
                        lstm_states=lstm_states,
                        episode_starts=episode_starts,
                        deterministic=deterministic,
                        action_masks=mask_t,
                        return_hidden_state=True,
                    )
                    action = int(actions.item())
                    episode_starts = torch.tensor([False], device=device)

                obs, reward, terminated, truncated, info = env.step(action)
                step_count += 1
                done = terminated or truncated

            if info.get("correct_branch_selected") is not None:
                correct.append(float(info["correct_branch_selected"]))
            entered_new.append(float(info.get("entered_new_branch", False)))
            waits.append(float(info.get("wait_action_fraction", 0.0)))

        results[stage] = {
            "correct_branch_selection_rate_pct": float(np.mean(correct) * 100.0) if correct else 0.0,
            "entered_new_branch_rate_pct": float(np.mean(entered_new) * 100.0),
            "mean_wait_fraction_pct": float(np.mean(waits) * 100.0),
        }

    env.close()
    return results


def evaluate_tier_set(
    model: MaskableRecurrentPPO,
    tiers: list[tuple[str, int | None, int, int]],
    distance_metric: str = "spawn",
    n_episodes: int = 200,
    seed_start: int = 50000,
    deterministic: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluates a set of difficulty tiers for 200 episodes each."""
    env = PreviousActionWrapper(RescueEnv(
        task_type="full",
        geometry_mode="edge",
        reward_shaping=False,
        distance_metric=distance_metric,
        spatial_memory_mode="none",
    ))

    tier_results = {}
    total_failures = {
        "never_discovered": 0,
        "discovered_not_picked": 0,
        "carrier_timeout": 0,
        "rescued": 0,
        "excessive_wait_stalling": 0,
        "excessive_oscillation": 0,
        "other_timeout": 0,
    }

    for tier_name, v_dist, min_dist, max_s in tiers:
        discoveries = 0
        pickups = 0
        rescues = 0
        returns_given_pickup = []
        carrier_timeouts = 0
        osc_pcts = []
        wait_pcts = []
        ep_lens = []

        for ep in range(n_episodes):
            ep_seed = seed_start + ep
            obs, _ = env.reset(
                seed=ep_seed,
                options={
                    "victim_distance": v_dist,
                    "min_victim_distance": min_dist,
                    "max_steps": max_s,
                }
            )
            lstm_states = torch.zeros(1, 1, model.policy.hidden_size, device=device)
            episode_starts = torch.tensor([True], device=device)

            done = False
            step_count = 0
            discovered = False
            picked_up = False

            while not done and step_count < max_s:
                action_mask = env.action_masks()
                obs_t = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}
                mask_t = torch.as_tensor(np.array([action_mask], dtype=bool), device=device)

                with torch.no_grad():
                    actions, _, _, lstm_states = model.policy.forward(
                        obs_t,
                        lstm_states=lstm_states,
                        episode_starts=episode_starts,
                        deterministic=deterministic,
                        action_masks=mask_t,
                        return_hidden_state=True,
                    )
                    action = int(actions.item())
                    episode_starts = torch.tensor([False], device=device)

                obs, reward, terminated, truncated, info = env.step(action)
                step_count += 1
                done = terminated or truncated

                if info.get("discovery", False):
                    discovered = True
                if any(v["carried_by"] == 0 for v in env.sim.victims):
                    picked_up = True

            rescued = int(info.get("victims_rescued", 0)) > 0
            if discovered:
                discoveries += 1
            if picked_up:
                pickups += 1
                returns_given_pickup.append(1.0 if rescued else 0.0)
                if not rescued:
                    carrier_timeouts += 1

            if rescued:
                rescues += 1
                total_failures["rescued"] += 1
            else:
                if not discovered:
                    total_failures["never_discovered"] += 1
                elif discovered and not picked_up:
                    total_failures["discovered_not_picked"] += 1
                elif picked_up and not rescued:
                    total_failures["carrier_timeout"] += 1

                # Overlapping pathologies
                max_w = int(info.get("max_consecutive_waits", 0))
                osc_p = float(info.get("oscillation_percentage", 0.0))
                if max_w >= 20:
                    total_failures["excessive_wait_stalling"] += 1
                elif osc_p >= 15.0:
                    total_failures["excessive_oscillation"] += 1
                else:
                    total_failures["other_timeout"] += 1

            osc_pcts.append(float(info.get("oscillation_percentage", 0.0)))
            wait_pcts.append(float(info.get("wait_action_fraction", 0.0)))
            ep_lens.append(step_count)

        tier_results[tier_name] = {
            "discovery_rate_pct": float(discoveries / n_episodes * 100.0),
            "pickup_rate_pct": float(pickups / n_episodes * 100.0),
            "rescue_rate_pct": float(rescues / n_episodes * 100.0),
            "return_given_pickup_pct": float(np.mean(returns_given_pickup) * 100.0) if returns_given_pickup else 0.0,
            "carrier_timeout_pct": float(carrier_timeouts / n_episodes * 100.0),
            "mean_oscillation_pct": float(np.mean(osc_pcts)),
            "mean_wait_fraction_pct": float(np.mean(wait_pcts) * 100.0),
            "mean_episode_length": float(np.mean(ep_lens)),
        }

    env.close()
    return {"tiers": tier_results, "failure_decomposition": total_failures}


def evaluate_model_all(
    model_path: str,
    model_name: str,
    output_dir: Path,
    n_episodes: int = 200,
    device: str = "cpu",
) -> dict[str, Any]:
    """Runs complete Phase 10 evaluation battery for a single model."""
    print("=" * 80)
    print(f"EVALUATING MODEL: {model_name} ({n_episodes} eps/tier)")
    print(f"Path: {model_path}")
    print("=" * 80)

    # Use a dummy environment for loading
    dummy_env = PreviousActionWrapper(RescueEnv(reward_shaping=False, geometry_mode="edge"))
    model = load_checkpoint(model_path, env=dummy_env, device=device)

    # 1. Navigation Evaluation (Deterministic and Stochastic)
    print(f"\n--- 1. Evaluating Pure Exploration (Deterministic, {n_episodes} eps) ---")
    nav_det = evaluate_navigation(model, n_episodes=n_episodes, deterministic=True, device=device)
    print(f"  Visited Coverage: {nav_det['mean_visited_coverage_pct']:.2f}%, Cells: {nav_det['mean_unique_cells_visited']:.1f}, Wait %: {nav_det['mean_wait_fraction_pct']:.1f}%, Osc %: {nav_det['mean_oscillation_pct']:.1f}%")

    print(f"\n--- 2. Evaluating Pure Exploration (Stochastic, {n_episodes} eps) ---")
    nav_stoch = evaluate_navigation(model, n_episodes=n_episodes, deterministic=False, device=device)
    print(f"  Visited Coverage: {nav_stoch['mean_visited_coverage_pct']:.2f}%, Cells: {nav_stoch['mean_unique_cells_visited']:.1f}, Wait %: {nav_stoch['mean_wait_fraction_pct']:.1f}%, Osc %: {nav_stoch['mean_oscillation_pct']:.1f}%")

    # 2. Branching Navigation (B1 & B2)
    branch_eps = min(100, n_episodes)
    print(f"\n--- 3. Evaluating Branching Navigation (B1 & B2, {branch_eps} eps) ---")
    branch_eval = evaluate_branching(model, n_episodes=branch_eps, deterministic=True, device=device)
    print(f"  B1 Correct Selection: {branch_eval['B1']['correct_branch_selection_rate_pct']:.1f}%, B2 Entered New: {branch_eval['B2']['entered_new_branch_rate_pct']:.1f}%")

    # 3. Full Rescue: Spawn-Relative Ladder (5 tiers)
    print(f"\n--- 4. Evaluating Spawn-Relative Ladder (Deterministic, 5 tiers x {n_episodes} eps) ---")
    spawn_det = evaluate_tier_set(model, SPAWN_TIERS, distance_metric="spawn", n_episodes=n_episodes, deterministic=True, device=device)
    for t_name, t_res in spawn_det["tiers"].items():
        print(f"  {t_name:15s} Rescue: {t_res['rescue_rate_pct']:5.1f}% | Pickup: {t_res['pickup_rate_pct']:5.1f}% | Return: {t_res['return_given_pickup_pct']:5.1f}% | Osc: {t_res['mean_oscillation_pct']:4.1f}%")

    print(f"\n--- 5. Evaluating Spawn-Relative Ladder (Stochastic, 5 tiers x {n_episodes} eps) ---")
    spawn_stoch = evaluate_tier_set(model, SPAWN_TIERS, distance_metric="spawn", n_episodes=n_episodes, deterministic=False, device=device)
    for t_name, t_res in spawn_stoch["tiers"].items():
        print(f"  {t_name:15s} Rescue: {t_res['rescue_rate_pct']:5.1f}% | Pickup: {t_res['pickup_rate_pct']:5.1f}% | Return: {t_res['return_given_pickup_pct']:5.1f}% | Osc: {t_res['mean_oscillation_pct']:4.1f}%")

    # 4. Full Rescue: Exit-Relative Ladder (6 tiers)
    print(f"\n--- 6. Evaluating Exit-Relative Ladder (Deterministic, 6 tiers x {n_episodes} eps) ---")
    exit_det = evaluate_tier_set(model, EXIT_TIERS, distance_metric="exit", n_episodes=n_episodes, deterministic=True, device=device)
    for t_name, t_res in exit_det["tiers"].items():
        print(f"  {t_name:15s} Rescue: {t_res['rescue_rate_pct']:5.1f}% | Pickup: {t_res['pickup_rate_pct']:5.1f}% | Return: {t_res['return_given_pickup_pct']:5.1f}% | Osc: {t_res['mean_oscillation_pct']:4.1f}%")

    print(f"\n--- 7. Evaluating Exit-Relative Ladder (Stochastic, 6 tiers x {n_episodes} eps) ---")
    exit_stoch = evaluate_tier_set(model, EXIT_TIERS, distance_metric="exit", n_episodes=n_episodes, deterministic=False, device=device)
    for t_name, t_res in exit_stoch["tiers"].items():
        print(f"  {t_name:15s} Rescue: {t_res['rescue_rate_pct']:5.1f}% | Pickup: {t_res['pickup_rate_pct']:5.1f}% | Return: {t_res['return_given_pickup_pct']:5.1f}% | Osc: {t_res['mean_oscillation_pct']:4.1f}%")

    summary = {
        "model_name": model_name,
        "model_path": str(model_path),
        "n_episodes_per_tier": n_episodes,
        "navigation_deterministic": nav_det,
        "navigation_stochastic": nav_stoch,
        "branching": branch_eval,
        "spawn_ladder_deterministic": spawn_det["tiers"],
        "spawn_ladder_stochastic": spawn_stoch["tiers"],
        "exit_ladder_deterministic": exit_det["tiers"],
        "exit_ladder_stochastic": exit_stoch["tiers"],
        "failure_decomposition_spawn_det": spawn_det["failure_decomposition"],
        "failure_decomposition_spawn_stoch": spawn_stoch["failure_decomposition"],
    }

    out_file = output_dir / f"eval_{model_name.lower().replace(' ', '_')}.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved full evaluation battery to {out_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 10 Evaluation Battery")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--name", type=str, required=True, help="Descriptive name for model")
    parser.add_argument("--device", type=str, default="cpu", help="Device for evaluation")
    parser.add_argument("--n-episodes", type=int, default=200, help="Episodes per tier/benchmark")
    parser.add_argument("--output-dir", type=str, default="models/phase10_evaluations", help="Output directory")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluate_model_all(args.model, args.name, out_dir, n_episodes=args.n_episodes, device=args.device)


if __name__ == "__main__":
    main()
