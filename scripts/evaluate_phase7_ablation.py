#!/usr/bin/env python3
"""Unified Phase 7 Evaluation and Failure Decomposition Runner.

Evaluates checkpoints across:
1. Pure Exploration Benchmark: 200 held-out unknown procedural buildings without victims (seeds 60000..60199).
2. Spawn Ladder Rescue Benchmark: 200 episodes/tier across 5 spawn distance tiers (seeds 50000..50199).
3. Exit Ladder Rescue Benchmark: 200 episodes/tier across 6 exit distance tiers (seeds 50000..50199).
4. Failure Decomposition: Overlapping and mutually-exclusive breakdown of failed episodes.

Outputs structured JSON results.
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
from scripts.benchmark_exploration import run_exploration_benchmark

SPAWN_LADDER_TIERS = [
    ("Spawn 2-6", 6, 2, 64),
    ("Spawn 7-12", 12, 7, 128),
    ("Spawn 13-20", 20, 13, 192),
    ("Spawn 21-35", 35, 21, 288),
    ("Distant Multi-Room", None, 1, 384),
]

EXIT_LADDER_TIERS = [
    ("Near (d=2)", 2, 1, 64),
    ("Medium (d=6)", 6, 3, 128),
    ("Bridge A (d=8)", 8, 5, 160),
    ("Bridge B (d=10)", 10, 7, 192),
    ("Bridge C (d=12)", 12, 9, 256),
    ("Distant (d=None)", None, 1, 384),
]


def evaluate_rescue_tier(
    model: MaskableRecurrentPPO,
    tier_name: str,
    victim_distance: int | None,
    min_victim_distance: int,
    max_steps: int,
    distance_metric: str,
    n_episodes: int = 200,
    seed_start: int = 50000,
    deterministic: bool = True,
    geometry_mode: str = "edge",
) -> dict:
    env = RescueEnv(
        victim_distance=victim_distance,
        min_victim_distance=min_victim_distance,
        max_steps=max_steps,
        reward_shaping=False,
        navigation_shaping=False,
        distance_metric=distance_metric,
        geometry_mode=geometry_mode,
    )
    env = PreviousActionWrapper(env)

    discoveries = []
    pickups = []
    rescues = []
    rescues_after_pickup = []
    timeouts = []
    timeouts_while_carrying = []
    lengths = []
    observed_cov_list = []
    wait_fractions = []
    doorway_reversals_list = []
    oscillation_pct_list = []

    # Failure counts
    failure_records = []

    for ep in range(n_episodes):
        seed = seed_start + ep
        obs, info = env.reset(seed=seed)
        if not deterministic:
            torch.manual_seed(seed)
            np.random.seed(seed)

        done = False
        gru_states = None
        episode_starts = np.ones((1,), dtype=bool)

        ep_ever_discovered = bool((obs["grid"][6] > 0).any())
        ep_ever_picked_up = False
        ep_currently_carrying = False

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

            if not ep_ever_discovered and (obs["grid"][6] > 0).any():
                ep_ever_discovered = True

            ep_currently_carrying = any(
                v["carried_by"] == 0 and not v["dead"] and not v["rescued"]
                for v in env.unwrapped.sim.victims
            )
            if ep_currently_carrying:
                ep_ever_picked_up = True

        is_success = bool(info.get("success", False))
        is_timeout = bool(info.get("timeout", False))
        step_count = int(info.get("steps", max_steps))
        obs_cov = float(info.get("observed_coverage", 0.0))
        wait_frac = float(info.get("wait_action_fraction", 0.0))
        door_rev = int(info.get("doorway_reversals", 0))
        osc_pct = float(info.get("oscillation_percentage", 0.0))

        if is_success:
            ep_ever_discovered = True
            ep_ever_picked_up = True

        discoveries.append(ep_ever_discovered)
        pickups.append(ep_ever_picked_up)
        if ep_ever_picked_up:
            rescues_after_pickup.append(is_success)
        rescues.append(is_success)
        timeouts.append(is_timeout)
        timeouts_while_carrying.append(is_timeout and ep_currently_carrying)
        if is_success:
            lengths.append(step_count)
        observed_cov_list.append(obs_cov)
        wait_fractions.append(wait_frac)
        doorway_reversals_list.append(door_rev)
        oscillation_pct_list.append(osc_pct)

        if not is_success:
            failure_records.append({
                "seed": seed,
                "discovered": ep_ever_discovered,
                "picked_up": ep_ever_picked_up,
                "timeout": is_timeout,
                "carrying_at_end": ep_currently_carrying,
                "wait_fraction": wait_frac,
                "doorway_reversals": door_rev,
                "oscillation_pct": osc_pct,
                "steps": step_count,
            })

    env.close()

    total_failures = len(failure_records)

    # Classify failures
    # Overlapping categories:
    cat_never_discovered = sum(1 for f in failure_records if not f["discovered"])
    cat_disc_not_picked = sum(1 for f in failure_records if f["discovered"] and not f["picked_up"])
    cat_picked_not_returned = sum(1 for f in failure_records if f["picked_up"])
    cat_excessive_wait = sum(1 for f in failure_records if f["wait_fraction"] >= 0.35)
    cat_door_oscillation = sum(1 for f in failure_records if f["doorway_reversals"] >= 6 or f["oscillation_pct"] >= 20.0)
    cat_other_timeout = sum(1 for f in failure_records if f["timeout"] and not (f["wait_fraction"] >= 0.35 or f["doorway_reversals"] >= 6 or f["oscillation_pct"] >= 20.0))

    # Mutually exclusive precedence:
    # 1. Picked up but not returned
    # 2. Discovered but not picked up
    # 3. Excessive stationary waiting
    # 4. Repeated doorway/cell oscillation
    # 5. Never discovered
    # 6. Other timeout
    mut_picked_not_returned = 0
    mut_disc_not_picked = 0
    mut_excessive_wait = 0
    mut_door_oscillation = 0
    mut_never_discovered = 0
    mut_other = 0

    for f in failure_records:
        if f["picked_up"]:
            mut_picked_not_returned += 1
        elif f["discovered"]:
            mut_disc_not_picked += 1
        elif f["wait_fraction"] >= 0.35:
            mut_excessive_wait += 1
        elif f["doorway_reversals"] >= 6 or f["oscillation_pct"] >= 20.0:
            mut_door_oscillation += 1
        elif not f["discovered"]:
            mut_never_discovered += 1
        else:
            mut_other += 1

    return {
        "tier_name": tier_name,
        "n_episodes": n_episodes,
        "deterministic": deterministic,
        "discovery_rate": float(np.mean(discoveries) * 100.0),
        "pickup_rate": float(np.mean(pickups) * 100.0),
        "rescue_success": float(np.mean(rescues) * 100.0),
        "return_success_conditional": float(np.mean(rescues_after_pickup) * 100.0) if rescues_after_pickup else 0.0,
        "timeout_rate": float(np.mean(timeouts) * 100.0),
        "timeout_while_carrying": float(np.mean(timeouts_while_carrying) * 100.0),
        "mean_successful_length": float(np.mean(lengths)) if lengths else 0.0,
        "mean_observed_coverage": float(np.mean(observed_cov_list) * 100.0),
        "mean_wait_fraction": float(np.mean(wait_fractions) * 100.0),
        "mean_doorway_reversals": float(np.mean(doorway_reversals_list)),
        "mean_oscillation_pct": float(np.mean(oscillation_pct_list)),
        "total_failures": total_failures,
        "failure_decomposition_overlapping": {
            "never_discovered_pct": float(cat_never_discovered / max(1, total_failures) * 100.0),
            "discovered_not_picked_pct": float(cat_disc_not_picked / max(1, total_failures) * 100.0),
            "picked_up_not_returned_pct": float(cat_picked_not_returned / max(1, total_failures) * 100.0),
            "excessive_wait_pct": float(cat_excessive_wait / max(1, total_failures) * 100.0),
            "doorway_oscillation_pct": float(cat_door_oscillation / max(1, total_failures) * 100.0),
            "other_timeout_pct": float(cat_other_timeout / max(1, total_failures) * 100.0),
        },
        "failure_decomposition_mutually_exclusive": {
            "picked_up_not_returned_count": mut_picked_not_returned,
            "discovered_not_picked_count": mut_disc_not_picked,
            "excessive_wait_count": mut_excessive_wait,
            "doorway_oscillation_count": mut_door_oscillation,
            "never_discovered_count": mut_never_discovered,
            "other_count": mut_other,
        },
    }


def evaluate_full_model(
    checkpoint_path: str | Path,
    episodes_per_tier: int = 200,
    seed_start: int = 50000,
    exploration_seed_start: int = 60000,
    device: str = "cpu",
    architecture: str = "baseline",
    modes: tuple[str, ...] = ("deterministic", "stochastic"),
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    print(f"\n=======================================================")
    print(f" FULL EVALUATION FOR: {checkpoint_path}")
    print(f"=======================================================")

    model = load_checkpoint(
        checkpoint_path,
        env=None,
        device=device,
        expected_architecture=architecture,
    )

    results = {
        "checkpoint": str(checkpoint_path),
        "episodes_per_tier": episodes_per_tier,
        "exploration_benchmark": {},
        "spawn_ladder": {},
        "exit_ladder": {},
    }

    # 1. Pure Exploration Benchmark
    print("\n--- Running Pure Exploration Benchmark (200 episodes) ---")
    for mode in modes:
        det = (mode == "deterministic")
        print(f"  Mode: {mode.upper()}...")
        results["exploration_benchmark"][mode] = run_exploration_benchmark(
            model=model,
            n_episodes=episodes_per_tier,
            seed_start=exploration_seed_start,
            deterministic=det,
            coverage_target=0.90,
            max_steps=384,
            device=device,
        )

    # 2. Spawn Ladder Rescue Benchmark
    print("\n--- Running Spawn-Relative Ladder Rescue Benchmark (5 tiers x 200 eps) ---")
    for mode in modes:
        det = (mode == "deterministic")
        results["spawn_ladder"][mode] = {}
        for tname, vdist, mdist, msteps in SPAWN_LADDER_TIERS:
            print(f"  Tier [{tname}] - {mode.upper()}...")
            tier_res = evaluate_rescue_tier(
                model=model,
                tier_name=tname,
                victim_distance=vdist,
                min_victim_distance=mdist,
                max_steps=msteps,
                distance_metric="spawn",
                n_episodes=episodes_per_tier,
                seed_start=seed_start,
                deterministic=det,
                geometry_mode="edge",
            )
            results["spawn_ladder"][mode][tname] = tier_res
            print(f"    Rescue: {tier_res['rescue_success']:.1f}% | Disc: {tier_res['discovery_rate']:.1f}% | Pick: {tier_res['pickup_rate']:.1f}% | Ret|Pick: {tier_res['return_success_conditional']:.1f}% | Cov: {tier_res['mean_observed_coverage']:.1f}%")

    # 3. Exit Ladder Rescue Benchmark (historical comparison)
    print("\n--- Running Exit-Relative Ladder Rescue Benchmark (6 tiers x 200 eps) ---")
    for mode in modes:
        det = (mode == "deterministic")
        results["exit_ladder"][mode] = {}
        for tname, vdist, mdist, msteps in EXIT_LADDER_TIERS:
            print(f"  Tier [{tname}] - {mode.upper()}...")
            tier_res = evaluate_rescue_tier(
                model=model,
                tier_name=tname,
                victim_distance=vdist,
                min_victim_distance=mdist,
                max_steps=msteps,
                distance_metric="exit",
                n_episodes=episodes_per_tier,
                seed_start=seed_start,
                deterministic=det,
                geometry_mode="edge",
            )
            results["exit_ladder"][mode][tname] = tier_res
            print(f"    Rescue: {tier_res['rescue_success']:.1f}% | Disc: {tier_res['discovery_rate']:.1f}% | Pick: {tier_res['pickup_rate']:.1f}% | Ret|Pick: {tier_res['return_success_conditional']:.1f}% | Cov: {tier_res['mean_observed_coverage']:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 7 Ablation Evaluator")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint zip")
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per tier (default: 200)")
    parser.add_argument("--seed-start", type=int, default=50000, help="Rescue evaluation start seed")
    parser.add_argument("--exploration-seed-start", type=int, default=60000, help="Exploration benchmark start seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or mps)")
    parser.add_argument("--architecture", type=str, default="baseline")
    parser.add_argument("--modes", type=str, default="deterministic,stochastic")
    parser.add_argument("--output-json", type=str, required=True, help="Path to save output JSON")
    args = parser.parse_args()

    mode_list = tuple(m.strip() for m in args.modes.split(","))

    res = evaluate_full_model(
        checkpoint_path=args.checkpoint,
        episodes_per_tier=args.episodes,
        seed_start=args.seed_start,
        exploration_seed_start=args.exploration_seed_start,
        device=args.device,
        architecture=args.architecture,
        modes=mode_list,
    )

    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nAll results saved to: {out_file}")


if __name__ == "__main__":
    main()
