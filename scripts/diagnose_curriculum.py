#!/usr/bin/env python3
"""Diagnostic curriculum evaluation script for fire-rescue-project-2.0.

Evaluates checkpoints across 3 difficulty tiers:
1. Near (d=2, max_steps=64)
2. Medium (d=6, max_steps=128)
3. Distant (d=None, max_steps=384)

Reports both Deterministic and Stochastic metrics on identical held-out seed sets:
- Discovery rate (%)
- Pickup rate (%)
- Rescue success (%)
- Return success conditional on pickup (%)
- Timeout while carrying (%)
- Mean successful episode length (steps)
- Oscillation frequency (%)
"""

from __future__ import annotations

import argparse
import concurrent.futures
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
from training.train_ppo import load_checkpoint, make_eval_env


OPPOSITE_ACTIONS = {
    1: 3,  # Up <-> Down
    3: 1,
    2: 4,  # Right <-> Left
    4: 2,
}


CURRICULUM_TIERS = [
    ("Near (d=2, steps=64)", 2, 1, 64),
    ("Medium (d=6, steps=128)", 6, 3, 128),
    ("Bridge A (d=8, steps=160)", 8, 5, 160),
    ("Bridge B (d=10, steps=192)", 10, 7, 192),
    ("Bridge C (d=12, steps=256)", 12, 9, 256),
    ("Distant (d=None, steps=384)", None, 1, 384),
]

LEGACY_TIERS = [
    ("Near (d=2, steps=64)", 2, 1, 64),
    ("Medium (d=6, steps=128)", 6, 3, 128),
    ("Distant (d=None, steps=384)", None, 1, 384),
]

BRIDGE_TIERS = [
    ("Bridge A (d=8, steps=160)", 8, 5, 160),
    ("Bridge B (d=10, steps=192)", 10, 7, 192),
    ("Bridge C (d=12, steps=256)", 12, 9, 256),
]


def evaluate_tier(
    model,
    victim_distance: int | None,
    max_steps: int,
    n_episodes: int = 20,
    seed_start: int = 50000,
    deterministic: bool = True,
    device: str = "cpu",
    min_victim_distance: int = 1,
) -> dict[str, float]:
    env = RescueEnv(
        victim_distance=victim_distance,
        min_victim_distance=min_victim_distance,
        max_steps=max_steps,
        reward_shaping=False,
        navigation_shaping=False,
    )
    env = PreviousActionWrapper(env)

    discoveries = []
    pickups = []
    rescues = []
    rescues_after_pickup = []
    timeouts_while_carrying = []
    successful_lengths = []
    total_steps = 0
    total_oscillations = 0

    for ep in range(n_episodes):
        seed = seed_start + ep
        obs, _ = env.reset(seed=seed)
        if not deterministic:
            torch.manual_seed(seed)

        done = False
        ep_discovered = bool((obs["grid"][6] > 0).any())
        ep_ever_picked_up = False
        ep_carrying = False
        step_count = 0
        pos_history = []
        action_history = []

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
            action_history.append(act_int)

            obs, reward, terminated, truncated, info = env.step(act_int)
            step_count += 1
            total_steps += 1
            done = terminated or truncated

            agent_pos = (env.unwrapped.sim.agents[0]["r"], env.unwrapped.sim.agents[0]["c"])
            pos_history.append(agent_pos)

            # Check 2-step oscillation (A -> B -> A) or immediate reversal action
            if len(pos_history) >= 3 and pos_history[-1] == pos_history[-3]:
                total_oscillations += 1
            elif len(action_history) >= 2 and OPPOSITE_ACTIONS.get(action_history[-1]) == action_history[-2]:
                total_oscillations += 1

            if not ep_discovered and (obs["grid"][6] > 0).any():
                ep_discovered = True

            ep_carrying = any(
                v["carried_by"] == 0 and not v["dead"] and not v["rescued"]
                for v in env.unwrapped.sim.victims
            )
            if ep_carrying:
                ep_ever_picked_up = True

        is_success = bool(info.get("success", False))
        is_timeout = bool(info.get("timeout", False))

        discoveries.append(ep_discovered or is_success)
        pickups.append(ep_ever_picked_up or is_success)
        if ep_ever_picked_up or is_success:
            rescues_after_pickup.append(is_success)
        rescues.append(is_success)
        timeouts_while_carrying.append(is_timeout and ep_carrying)
        if is_success:
            successful_lengths.append(step_count)

    env.close()

    osc_freq = float(total_oscillations / max(1, total_steps))
    return {
        "discovery_rate": float(np.mean(discoveries)) * 100.0,
        "pickup_rate": float(np.mean(pickups)) * 100.0,
        "rescue_success": float(np.mean(rescues)) * 100.0,
        "return_success_after_pickup": float(np.mean(rescues_after_pickup)) * 100.0 if rescues_after_pickup else 0.0,
        "timeout_while_carrying": float(np.mean(timeouts_while_carrying)) * 100.0,
        "mean_successful_length": float(np.mean(successful_lengths)) if successful_lengths else 0.0,
        "oscillation_frequency": osc_freq * 100.0,
    }


def _eval_worker(task):
    model_path, tier_name, dist, min_dist, max_s, n_episodes, seed_start, deterministic, device = task
    dummy_env = make_eval_env(victim_distance=2, max_steps=64)
    model = load_checkpoint(model_path, env=dummy_env, device=device)
    dummy_env.close()
    res = evaluate_tier(
        model,
        victim_distance=dist,
        min_victim_distance=min_dist,
        max_steps=max_s,
        n_episodes=n_episodes,
        seed_start=seed_start,
        deterministic=deterministic,
        device=device,
    )
    mode_str = "deterministic" if deterministic else "stochastic"
    print(
        f"[{tier_name}] {mode_str:<13}: Rescue={res['rescue_success']:>5.1f}%, "
        f"Pick={res['pickup_rate']:>5.1f}%, Disc={res['discovery_rate']:>5.1f}%, "
        f"Osc={res['oscillation_frequency']:>4.1f}%",
        flush=True,
    )
    return tier_name, mode_str, res


def run_full_diagnosis(
    model_path: str,
    device: str = "cpu",
    n_episodes: int = 20,
    seed_start: int = 50000,
    preset: str = "all",
    tiers: list[tuple[str, int | None, int, int]] | None = None,
    workers: int = 1,
) -> dict:
    if tiers is None:
        if preset == "legacy":
            tiers = LEGACY_TIERS
        elif preset == "bridge":
            tiers = BRIDGE_TIERS
        else:
            tiers = CURRICULUM_TIERS

    if workers > 1:
        tasks = []
        for tier_name, dist, min_dist, max_s in tiers:
            tasks.append((model_path, tier_name, dist, min_dist, max_s, n_episodes, seed_start, True, device))
            tasks.append((model_path, tier_name, dist, min_dist, max_s, n_episodes, seed_start, False, device))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            worker_results = list(executor.map(_eval_worker, tasks))
        results = {tier[0]: {} for tier in tiers}
        for tier_name, mode_str, res in worker_results:
            results[tier_name][mode_str] = res
        return results

    dummy_env = make_eval_env(victim_distance=2, max_steps=64)
    model = load_checkpoint(model_path, env=dummy_env, device=device)
    dummy_env.close()

    results = {}
    for tier_name, dist, min_dist, max_s in tiers:
        det = evaluate_tier(
            model,
            victim_distance=dist,
            min_victim_distance=min_dist,
            max_steps=max_s,
            n_episodes=n_episodes,
            seed_start=seed_start,
            deterministic=True,
            device=device,
        )
        print(f"[{tier_name}] deterministic: Rescue={det['rescue_success']:>5.1f}%, Pick={det['pickup_rate']:>5.1f}%, Disc={det['discovery_rate']:>5.1f}%, Osc={det['oscillation_frequency']:>4.1f}%", flush=True)
        stoch = evaluate_tier(
            model,
            victim_distance=dist,
            min_victim_distance=min_dist,
            max_steps=max_s,
            n_episodes=n_episodes,
            seed_start=seed_start,
            deterministic=False,
            device=device,
        )
        print(f"[{tier_name}] stochastic   : Rescue={stoch['rescue_success']:>5.1f}%, Pick={stoch['pickup_rate']:>5.1f}%, Disc={stoch['discovery_rate']:>5.1f}%, Osc={stoch['oscillation_frequency']:>4.1f}%", flush=True)
        results[tier_name] = {
            "deterministic": det,
            "stochastic": stoch,
        }
    return results


def print_table(results: dict, title: str):
    print("\n" + "=" * 95)
    print(f"DIAGNOSTIC BENCHMARK: {title}")
    print("=" * 95)
    header = f"{'Tier':<28} | {'Mode':<5} | {'Disc%':<6} | {'Pick%':<6} | {'Resc%':<6} | {'Ret|Pick%':<9} | {'T/O Carr%':<9} | {'AvgLen':<6} | {'Osc%':<5}"
    print(header)
    print("-" * 95)
    for tier_name, modes in results.items():
        for mode, m in modes.items():
            m_label = "Det" if mode == "deterministic" else "Stoch"
            print(
                f"{tier_name:<28} | {m_label:<5} | "
                f"{m['discovery_rate']:>5.1f}% | "
                f"{m['pickup_rate']:>5.1f}% | "
                f"{m['rescue_success']:>5.1f}% | "
                f"{m['return_success_after_pickup']:>8.1f}% | "
                f"{m['timeout_while_carrying']:>8.1f}% | "
                f"{m['mean_successful_length']:>6.1f} | "
                f"{m['oscillation_frequency']:>4.1f}%"
            )
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=50000)
    parser.add_argument(
        "--tier",
        choices=["all", "legacy", "bridge"],
        default="all",
        help="Evaluation tiers: all (2, 6, 8, 10, 12, Distant), legacy (2, 6, Distant), or bridge (8, 10, 12)",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        choices=["baseline", "deep128", "deep256"],
        help="Optional architecture variant override",
    )
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes for tiers (default: 1)")
    args = parser.parse_args()

    res = run_full_diagnosis(
        args.model,
        device=args.device,
        n_episodes=args.episodes,
        seed_start=args.seed_start,
        preset=args.tier,
        workers=args.workers,
    )
    print_table(res, args.model)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Results saved to {args.output_json}")
