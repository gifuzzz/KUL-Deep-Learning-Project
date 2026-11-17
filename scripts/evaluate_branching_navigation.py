#!/usr/bin/env python3
"""Phase 9 Navigation-Specific Evaluation Tool.

Evaluates trained policies on held-out procedural scenarios across the four
branching stages:
- Stage B1 (Single Junction): 2 branches, 1 visited, 1 unvisited (max_steps=64)
- Stage B2 (Multiple Junctions): 2 sequential junctions, dead-end backtracking (max_steps=128)
- Stage B3 (Multi-Room Navigation): multi-room navigation through visited corridors (max_steps=192)
- Stage B4 (Full Procedural Building): full unknown 30x46 layout (max_steps=384)

Evaluates 200 held-out scenarios per stage (seeds 60000..60199) under both
deterministic and stochastic modes.
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


def evaluate_stage(
    policy,
    env,
    stage: str,
    n_episodes: int = 200,
    seed_start: int = 60000,
    deterministic: bool = True,
    zero_ablate: bool = False,
    device: str = "cpu",
) -> dict:
    """Evaluate policy on held-out scenarios for a specific branching stage."""
    correct_selections = []
    junction_exited_list = []
    entered_new_branch_list = []
    unique_cells_list = []
    unique_rooms_list = []
    time_to_first_new_list = []
    wait_fractions = []
    max_wait_streaks = []
    oscillation_pcts = []
    doorway_reversals_list = []
    new_after_50_list = []
    episode_lengths = []

    for ep_idx in range(n_episodes):
        ep_seed = seed_start + ep_idx
        obs, info = env.reset(seed=ep_seed, options={"branching_stage": stage})

        lstm_states = torch.zeros(
            (1, 1, policy.hidden_size),
            device=device,
            dtype=torch.float32,
        )
        episode_starts = np.array([True])
        done = False
        step_count = 0

        while not done:
            if zero_ablate and "memory" in obs and obs["memory"].shape[0] == 10:
                obs = {k: v.copy() for k, v in obs.items()}
                obs["memory"] = obs["memory"].copy()
                obs["memory"][9] = 0.0

            action_mask = env.action_masks()
            action, lstm_states = policy.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=deterministic,
                action_masks=action_mask,
            )

            action_int = int(action)
            obs, reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated
            episode_starts = np.array([done])
            step_count += 1

        # Record metrics
        if info.get("correct_branch_selected") is not None:
            correct_selections.append(float(info["correct_branch_selected"]))

        junction_exited_list.append(float(info.get("junction_exited", False)))
        entered_new_branch_list.append(float(info.get("entered_new_branch", False)))
        unique_cells_list.append(float(info.get("visited_reachable_cells", 0)))
        unique_rooms_list.append(float(info.get("unique_rooms_entered", 0)))

        t_first = info.get("time_to_first_new_territory")
        if t_first is not None:
            time_to_first_new_list.append(float(t_first))

        wait_fractions.append(float(info.get("wait_action_fraction", 0.0)))
        max_wait_streaks.append(float(info.get("max_consecutive_waits", 0)))
        oscillation_pcts.append(float(info.get("oscillation_percentage", 0.0)))
        doorway_reversals_list.append(float(info.get("doorway_reversals", 0)))

        n_after_50 = info.get("new_cells_after_step_50", 0)
        new_after_50_list.append(float(n_after_50 > 0))
        episode_lengths.append(float(step_count))

    correct_rate = (np.mean(correct_selections) * 100.0) if correct_selections else None
    junction_exit_rate = float(np.mean(junction_exited_list) * 100.0)
    new_branch_rate = float(np.mean(entered_new_branch_list) * 100.0)
    mean_unique_cells = float(np.mean(unique_cells_list))
    mean_unique_rooms = float(np.mean(unique_rooms_list))
    mean_time_to_first = float(np.mean(time_to_first_new_list)) if time_to_first_new_list else None
    mean_wait_frac = float(np.mean(wait_fractions))
    mean_max_wait = float(np.mean(max_wait_streaks))
    mean_osc = float(np.mean(oscillation_pcts))
    mean_reversals = float(np.mean(doorway_reversals_list))
    after_50_rate = float(np.mean(new_after_50_list) * 100.0)
    mean_ep_len = float(np.mean(episode_lengths))

    return {
        "stage": stage,
        "episodes": n_episodes,
        "deterministic": deterministic,
        "correct_branch_selection_rate": correct_rate,
        "junction_exit_rate": junction_exit_rate,
        "new_branch_entry_rate": new_branch_rate,
        "mean_unique_cells_visited": mean_unique_cells,
        "mean_unique_rooms_entered": mean_unique_rooms,
        "mean_time_to_first_new_territory": mean_time_to_first,
        "mean_wait_fraction": mean_wait_frac,
        "mean_max_wait_streak": mean_max_wait,
        "mean_oscillation_pct": mean_osc,
        "mean_doorway_reversals": mean_reversals,
        "new_territory_after_step_50_rate": after_50_rate,
        "mean_episode_length": mean_ep_len,
    }


def evaluate_model_navigation(
    model_path: str | Path,
    stages: tuple[str, ...] = ("B1", "B2", "B3", "B4"),
    n_episodes: int = 200,
    seed_start: int = 60000,
    device: str = "cpu",
    spatial_memory_mode: str = "visited_binary",
    zero_ablate: bool = False,
) -> dict:
    """Evaluate model across all stages for both deterministic and stochastic modes."""
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

    results = {}
    for stage in stages:
        print(f"Evaluating {stage} (Deterministic)...")
        res_det = evaluate_stage(
            policy=policy,
            env=env,
            stage=stage,
            n_episodes=n_episodes,
            seed_start=seed_start,
            deterministic=True,
            zero_ablate=zero_ablate,
            device=device,
        )

        print(f"Evaluating {stage} (Stochastic)...")
        res_stoch = evaluate_stage(
            policy=policy,
            env=env,
            stage=stage,
            n_episodes=n_episodes,
            seed_start=seed_start,
            deterministic=False,
            zero_ablate=zero_ablate,
            device=device,
        )

        results[stage] = {
            "deterministic": res_det,
            "stochastic": res_stoch,
        }

    return {
        "model_path": str(model_path),
        "zero_ablate": zero_ablate,
        "spatial_memory_mode": spatial_memory_mode,
        "stages": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 9 Navigation Benchmark Runner")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--stages", nargs="+", default=["B1", "B2", "B3", "B4"], help="Stages to evaluate")
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per stage (default: 200)")
    parser.add_argument("--seed-start", type=int, default=60000, help="Held-out seed start (default: 60000)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/mps/cuda)")
    parser.add_argument("--spatial-memory-mode", type=str, default="visited_binary", choices=["visited_binary", "none"])
    parser.add_argument("--zero-ablate", action="store_true", default=False, help="Zero-ablate Channel 9")
    parser.add_argument("--output-json", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    report = evaluate_model_navigation(
        model_path=args.model_path,
        stages=tuple(args.stages),
        n_episodes=args.episodes,
        seed_start=args.seed_start,
        device=args.device,
        spatial_memory_mode=args.spatial_memory_mode,
        zero_ablate=args.zero_ablate,
    )

    print("\n" + "=" * 70)
    print("PHASE 9 NAVIGATION BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Model: {args.model_path} | Zero-Ablate: {args.zero_ablate}")
    print(f"{'Stage':<6} | {'Mode':<6} | {'Correct %':<9} | {'Exit %':<7} | {'NewBr %':<7} | {'Cells':<6} | {'Rooms':<6} | {'Wait %':<7} | {'Osc %':<6} | {'>50 %':<6}")
    print("-" * 70)

    for st, data in report["stages"].items():
        for mode in ("deterministic", "stochastic"):
            m = data[mode]
            corr = f"{m['correct_branch_selection_rate']:.1f}%" if m['correct_branch_selection_rate'] is not None else "N/A"
            print(
                f"{st:<6} | {mode[:5]:<6} | {corr:<9} | {m['junction_exit_rate']:<6.1f}% | "
                f"{m['new_branch_entry_rate']:<6.1f}% | {m['mean_unique_cells_visited']:<6.1f} | "
                f"{m['mean_unique_rooms_entered']:<6.1f} | {m['mean_wait_fraction']*100:<6.1f}% | "
                f"{m['mean_oscillation_pct']:<5.1f}% | {m['new_territory_after_step_50_rate']:<5.1f}%"
            )
    print("=" * 70)

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved navigation results to {out_p}")


if __name__ == "__main__":
    main()
