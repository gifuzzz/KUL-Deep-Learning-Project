#!/usr/bin/env python3
"""Pure Exploration Benchmark Script for Phase 7.

Evaluates trained CNN + GRU models on held-out unknown procedural buildings without victims.
Measures systematic exploration, coverage milestones, doorway oscillation, and Wait stalling.
Strictly unshaped (reward_shaping=False, 0.0 reward).

Metrics reported:
- Coverage at 50, 100, 200, 384 steps
- Final observed-cell coverage fraction
- Actual visited-cell coverage fraction
- Traversable edge discovery fraction
- Unique rooms explored
- Exploration efficiency (cells per 100 steps)
- Wait frequency (fraction of actions spent waiting)
- Long Wait streaks (>5, >10, >50 steps)
- Doorway crossings and reversals
- 2-cell oscillation percentage
- 90% coverage completion rate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from rescue_env.envs.single_agent import RescueEnv
from rescue_env.wrappers import PreviousActionWrapper
from agents.recurrent_ppo import MaskableRecurrentPPO
from agents.cnn import RecurrentSpatialRescueCNN
from training.train_ppo import load_checkpoint


def run_exploration_benchmark(
    model: MaskableRecurrentPPO,
    n_episodes: int = 200,
    seed_start: int = 60000,
    deterministic: bool = True,
    coverage_target: float = 0.90,
    max_steps: int = 384,
    device: str = "cpu",
) -> dict[str, float | list]:
    env = RescueEnv(
        task_type="exploration_only",
        max_steps=max_steps,
        reward_shaping=False,
        navigation_shaping=False,
        coverage_target=coverage_target,
        geometry_mode="edge",
    )
    env = PreviousActionWrapper(env)

    obs_coverages = []
    vis_coverages = []
    edge_coverages = []
    rooms_entered = []
    total_steps_list = []
    revisits_list = []
    doorway_crossings_list = []
    doorway_reversals_list = []
    oscillation_pct_list = []
    wait_fractions = []
    max_waits_list = []
    wait_gt5_list = []
    wait_gt10_list = []
    wait_gt50_list = []
    cells_per_100_list = []
    cov_50_list = []
    cov_100_list = []
    cov_200_list = []
    cov_384_list = []
    completed_target = []

    for ep in range(n_episodes):
        seed = seed_start + ep
        obs, info = env.reset(seed=seed)
        if not deterministic:
            torch.manual_seed(seed)
            np.random.seed(seed)

        done = False
        gru_states = None
        episode_starts = np.ones((1,), dtype=bool)

        while not done:
            action, gru_states = model.predict(
                obs,
                state=gru_states,
                episode_start=episode_starts,
                deterministic=deterministic,
                action_masks=env.action_masks(),
            )
            episode_starts = np.zeros((1,), dtype=bool)
            act_int = int(action)

            obs, reward, terminated, truncated, info = env.step(act_int)
            done = terminated or truncated

        # Collect metrics from final info
        final_obs_cov = float(info["observed_coverage"])
        final_vis_cov = float(info["visited_coverage"])
        final_edge_cov = float(info["edge_coverage"])
        steps_taken = int(info["steps"])

        obs_coverages.append(final_obs_cov)
        vis_coverages.append(final_vis_cov)
        edge_coverages.append(final_edge_cov)
        rooms_entered.append(int(info["unique_rooms_entered"]))
        total_steps_list.append(steps_taken)
        revisits_list.append(int(info["revisits"]))
        doorway_crossings_list.append(int(info["doorway_crossings"]))
        doorway_reversals_list.append(int(info["doorway_reversals"]))
        oscillation_pct_list.append(float(info["oscillation_percentage"]))
        wait_fractions.append(float(info["wait_action_fraction"]))
        max_waits_list.append(int(info["max_consecutive_waits"]))
        wait_gt5_list.append(int(info["wait_streaks_gt5"]))
        wait_gt10_list.append(int(info["wait_streaks_gt10"]))
        wait_gt50_list.append(int(info["wait_streaks_gt50"]))
        cells_per_100_list.append(float(info["cells_per_100_steps"]))

        # Coverage milestones
        cov_50 = info.get("coverage_at_50")
        cov_100 = info.get("coverage_at_100")
        cov_200 = info.get("coverage_at_200")
        cov_384 = info.get("coverage_at_384")

        cov_50_list.append(float(cov_50) if cov_50 is not None else final_obs_cov)
        cov_100_list.append(float(cov_100) if cov_100 is not None else final_obs_cov)
        cov_200_list.append(float(cov_200) if cov_200 is not None else final_obs_cov)
        cov_384_list.append(float(cov_384) if cov_384 is not None else final_obs_cov)

        completed_target.append(1.0 if final_obs_cov >= coverage_target else 0.0)

    env.close()

    def mean_pct(lst):
        return float(np.mean(lst) * 100.0)

    def mean_val(lst):
        return float(np.mean(lst))

    def std_val(lst):
        return float(np.std(lst))

    results = {
        "n_episodes": n_episodes,
        "seed_start": seed_start,
        "deterministic": deterministic,
        "coverage_target": coverage_target,
        "observed_coverage_mean": mean_pct(obs_coverages),
        "observed_coverage_std": float(np.std(obs_coverages) * 100.0),
        "visited_coverage_mean": mean_pct(vis_coverages),
        "visited_coverage_std": float(np.std(vis_coverages) * 100.0),
        "edge_coverage_mean": mean_pct(edge_coverages),
        "edge_coverage_std": float(np.std(edge_coverages) * 100.0),
        "rooms_entered_mean": mean_val(rooms_entered),
        "completion_rate_90pct": mean_pct(completed_target),
        "coverage_at_50_mean": mean_pct(cov_50_list),
        "coverage_at_100_mean": mean_pct(cov_100_list),
        "coverage_at_200_mean": mean_pct(cov_200_list),
        "coverage_at_384_mean": mean_pct(cov_384_list),
        "steps_mean": mean_val(total_steps_list),
        "cells_per_100_mean": mean_val(cells_per_100_list),
        "wait_action_pct": mean_pct(wait_fractions),
        "max_wait_streak_mean": mean_val(max_waits_list),
        "wait_streaks_gt5_mean": mean_val(wait_gt5_list),
        "wait_streaks_gt10_mean": mean_val(wait_gt10_list),
        "wait_streaks_gt50_mean": mean_val(wait_gt50_list),
        "doorway_crossings_mean": mean_val(doorway_crossings_list),
        "doorway_reversals_mean": mean_val(doorway_reversals_list),
        "oscillation_pct_mean": mean_val(oscillation_pct_list),
    }

    return results


def print_exploration_table(results_det: dict, results_stoch: dict | None = None):
    print("\n" + "=" * 90)
    print(" PURE EXPLORATION BENCHMARK RESULTS (200 Held-Out Scenarios, Seeds 60000..60199)")
    print("=" * 90)
    header = f"{'Metric':<38} | {'Deterministic':<22} | {'Stochastic':<22}"
    print(header)
    print("-" * 90)

    rows = [
        ("Observed Reachable Cells (%)", f"{results_det['observed_coverage_mean']:.1f}% ± {results_det['observed_coverage_std']:.1f}%",
         f"{results_stoch['observed_coverage_mean']:.1f}% ± {results_stoch['observed_coverage_std']:.1f}%" if results_stoch else "N/A"),
        ("Visited Reachable Cells (%)", f"{results_det['visited_coverage_mean']:.1f}% ± {results_det['visited_coverage_std']:.1f}%",
         f"{results_stoch['visited_coverage_mean']:.1f}% ± {results_stoch['visited_coverage_std']:.1f}%" if results_stoch else "N/A"),
        ("Discovered Traversable Edges (%)", f"{results_det['edge_coverage_mean']:.1f}% ± {results_det['edge_coverage_std']:.1f}%",
         f"{results_stoch['edge_coverage_mean']:.1f}% ± {results_stoch['edge_coverage_std']:.1f}%" if results_stoch else "N/A"),
        ("Unique Rooms Explored", f"{results_det['rooms_entered_mean']:.2f}",
         f"{results_stoch['rooms_entered_mean']:.2f}" if results_stoch else "N/A"),
        ("90% Coverage Completion Rate (%)", f"{results_det['completion_rate_90pct']:.1f}%",
         f"{results_stoch['completion_rate_90pct']:.1f}%" if results_stoch else "N/A"),
        ("Coverage @ 50 Steps (%)", f"{results_det['coverage_at_50_mean']:.1f}%",
         f"{results_stoch['coverage_at_50_mean']:.1f}%" if results_stoch else "N/A"),
        ("Coverage @ 100 Steps (%)", f"{results_det['coverage_at_100_mean']:.1f}%",
         f"{results_stoch['coverage_at_100_mean']:.1f}%" if results_stoch else "N/A"),
        ("Coverage @ 200 Steps (%)", f"{results_det['coverage_at_200_mean']:.1f}%",
         f"{results_stoch['coverage_at_200_mean']:.1f}%" if results_stoch else "N/A"),
        ("Coverage @ 384 Steps (%)", f"{results_det['coverage_at_384_mean']:.1f}%",
         f"{results_stoch['coverage_at_384_mean']:.1f}%" if results_stoch else "N/A"),
        ("Efficiency (Cells/100 Steps)", f"{results_det['cells_per_100_mean']:.1f}",
         f"{results_stoch['cells_per_100_mean']:.1f}" if results_stoch else "N/A"),
        ("Actions Spent Waiting (%)", f"{results_det['wait_action_pct']:.1f}%",
         f"{results_stoch['wait_action_pct']:.1f}%" if results_stoch else "N/A"),
        ("Max Wait Streak (steps)", f"{results_det['max_wait_streak_mean']:.1f}",
         f"{results_stoch['max_wait_streak_mean']:.1f}" if results_stoch else "N/A"),
        ("Wait Streaks > 5 (count/ep)", f"{results_det['wait_streaks_gt5_mean']:.2f}",
         f"{results_stoch['wait_streaks_gt5_mean']:.2f}" if results_stoch else "N/A"),
        ("Wait Streaks > 10 (count/ep)", f"{results_det['wait_streaks_gt10_mean']:.2f}",
         f"{results_stoch['wait_streaks_gt10_mean']:.2f}" if results_stoch else "N/A"),
        ("Wait Streaks > 50 (count/ep)", f"{results_det['wait_streaks_gt50_mean']:.2f}",
         f"{results_stoch['wait_streaks_gt50_mean']:.2f}" if results_stoch else "N/A"),
        ("Doorway Crossings (count/ep)", f"{results_det['doorway_crossings_mean']:.2f}",
         f"{results_stoch['doorway_crossings_mean']:.2f}" if results_stoch else "N/A"),
        ("Doorway Reversals (count/ep)", f"{results_det['doorway_reversals_mean']:.2f}",
         f"{results_stoch['doorway_reversals_mean']:.2f}" if results_stoch else "N/A"),
        ("2-Cell Oscillation (% steps)", f"{results_det['oscillation_pct_mean']:.1f}%",
         f"{results_stoch['oscillation_pct_mean']:.1f}%" if results_stoch else "N/A"),
        ("Mean Episode Length (steps)", f"{results_det['steps_mean']:.1f}",
         f"{results_stoch['steps_mean']:.1f}" if results_stoch else "N/A"),
    ]

    for label, det_val, stoch_val in rows:
        print(f"{label:<38} | {det_val:<22} | {stoch_val:<22}")
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Pure Exploration Benchmark Runner")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint zip")
    parser.add_argument("--episodes", type=int, default=200, help="Number of evaluation episodes")
    parser.add_argument("--seed-start", type=int, default=60000, help="Starting scenario seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or mps)")
    parser.add_argument("--architecture", type=str, default="baseline", choices=["baseline", "deep128", "deep256"])
    parser.add_argument("--coverage-target", type=float, default=0.90, help="Coverage target threshold")
    parser.add_argument("--max-steps", type=int, default=384, help="Maximum steps per episode")
    parser.add_argument("--mode", type=str, default="both", choices=["deterministic", "stochastic", "both"])
    parser.add_argument("--output-json", type=str, default=None, help="Save machine-readable results to JSON")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: checkpoint {checkpoint_path} not found.")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_checkpoint(
        checkpoint_path,
        env=None,
        device=args.device,
        expected_architecture=args.architecture,
    )

    results_det = None
    results_stoch = None

    if args.mode in ("deterministic", "both"):
        print(f"Running Deterministic Exploration Benchmark ({args.episodes} episodes)...")
        results_det = run_exploration_benchmark(
            model=model,
            n_episodes=args.episodes,
            seed_start=args.seed_start,
            deterministic=True,
            coverage_target=args.coverage_target,
            max_steps=args.max_steps,
            device=args.device,
        )

    if args.mode in ("stochastic", "both"):
        print(f"Running Stochastic Exploration Benchmark ({args.episodes} episodes)...")
        results_stoch = run_exploration_benchmark(
            model=model,
            n_episodes=args.episodes,
            seed_start=args.seed_start,
            deterministic=False,
            coverage_target=args.coverage_target,
            max_steps=args.max_steps,
            device=args.device,
        )

    if results_det is None:
        results_det = results_stoch

    print_exploration_table(results_det, results_stoch)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(checkpoint_path),
            "episodes": args.episodes,
            "seed_start": args.seed_start,
            "coverage_target": args.coverage_target,
            "deterministic": results_det,
            "stochastic": results_stoch,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
