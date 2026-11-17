"""Comprehensive Evaluation Suite for Phase 8 Ablation.

Evaluates trained models across:
1. Pure Exploration Benchmark (200 episodes, seeds 60000-60199)
2. Spawn-Relative Rescue Ladder (5 tiers x 200 episodes, seeds 50000-50199)
3. Exit-Relative Rescue Ladder (6 tiers x 200 episodes, seeds 50000-50199)
4. Zero-Ablation Diagnostic (Channel 9 set to 0.0 at inference)

Strictly unshaped (reward_shaping=False, native +10 rescue only).
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


def evaluate_exploration(
    model: MaskableRecurrentPPO,
    spatial_memory_mode: str = "none",
    n_episodes: int = 200,
    seed_start: int = 60000,
    deterministic: bool = True,
    zero_ablate_ch9: bool = False,
    device: str = "cpu",
) -> dict[str, float]:
    env = PreviousActionWrapper(RescueEnv(
        task_type="exploration_only",
        geometry_mode="edge",
        reward_shaping=False,
        max_steps=384,
        distance_metric="spawn",
        spatial_memory_mode=spatial_memory_mode,
    ))

    obs_coverages = []
    vis_coverages = []
    edge_coverages = []
    wait_fractions = []
    oscillations = []
    unique_rooms = []
    cov_50 = []
    cov_100 = []
    cov_200 = []
    cov_384 = []
    lengths = []

    for ep in range(n_episodes):
        ep_seed = seed_start + ep
        obs, _ = env.reset(seed=ep_seed)
        lstm_states = torch.zeros(1, 1, model.policy.hidden_size, device=device)
        episode_starts = torch.tensor([True], device=device)

        done = False
        step_count = 0

        while not done and step_count < 384:
            if zero_ablate_ch9 and obs["memory"].shape[0] >= 10:
                obs["memory"][9, :, :] = 0.0

            obs_t = {
                k: torch.as_tensor(v, device=device).unsqueeze(0)
                for k, v in obs.items()
            }
            legal_mask = env.action_masks()
            legal_mask_t = torch.as_tensor(np.array([legal_mask]), device=device)

            with torch.no_grad():
                actions, _, _, next_h = model.policy(
                    obs_t,
                    lstm_states=lstm_states,
                    episode_starts=episode_starts,
                    deterministic=deterministic,
                    action_masks=legal_mask_t,
                )
                action = int(actions[0].item())
                lstm_states = next_h
                episode_starts = torch.tensor([False], device=device)

            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            done = terminated or truncated

        obs_coverages.append(info.get("observed_coverage", 0.0))
        vis_coverages.append(info.get("visited_coverage", 0.0))
        edge_coverages.append(info.get("edge_coverage", 0.0))
        wait_fractions.append(info.get("wait_action_fraction", 0.0))
        oscillations.append(info.get("oscillation_percentage", 0.0))
        unique_rooms.append(info.get("unique_rooms_entered", 0))
        if info.get("coverage_at_50") is not None:
            cov_50.append(info["coverage_at_50"])
        if info.get("coverage_at_100") is not None:
            cov_100.append(info["coverage_at_100"])
        if info.get("coverage_at_200") is not None:
            cov_200.append(info["coverage_at_200"])
        if info.get("coverage_at_384") is not None:
            cov_384.append(info["coverage_at_384"])
        lengths.append(step_count)

    env.close()

    return {
        "observed_coverage_mean": float(np.mean(obs_coverages) * 100.0),
        "visited_coverage_mean": float(np.mean(vis_coverages) * 100.0),
        "edge_coverage_mean": float(np.mean(edge_coverages) * 100.0),
        "wait_action_fraction_mean": float(np.mean(wait_fractions) * 100.0),
        "oscillation_percentage_mean": float(np.mean(oscillations)),
        "unique_rooms_entered_mean": float(np.mean(unique_rooms)),
        "coverage_at_50_mean": float(np.mean(cov_50) * 100.0) if cov_50 else 0.0,
        "coverage_at_100_mean": float(np.mean(cov_100) * 100.0) if cov_100 else 0.0,
        "coverage_at_200_mean": float(np.mean(cov_200) * 100.0) if cov_200 else 0.0,
        "coverage_at_384_mean": float(np.mean(cov_384) * 100.0) if cov_384 else 0.0,
        "mean_episode_length": float(np.mean(lengths)),
    }


def evaluate_rescue_tier(
    model: MaskableRecurrentPPO,
    victim_distance: int | None,
    min_victim_distance: int,
    max_steps: int,
    distance_metric: str = "spawn",
    spatial_memory_mode: str = "none",
    n_episodes: int = 200,
    seed_start: int = 50000,
    deterministic: bool = True,
    zero_ablate_ch9: bool = False,
    device: str = "cpu",
) -> dict[str, float]:
    env = PreviousActionWrapper(RescueEnv(
        task_type="full",
        victim_distance=victim_distance,
        min_victim_distance=min_victim_distance,
        max_steps=max_steps,
        distance_metric=distance_metric,
        geometry_mode="edge",
        reward_shaping=False,
        navigation_shaping=False,
        spatial_memory_mode=spatial_memory_mode,
    ))

    rescues = []
    pickups = []
    discoveries = []
    returns_cond = []
    carrier_timeouts = []
    successful_lengths = []
    oscillations = []
    wait_fractions = []

    for ep in range(n_episodes):
        ep_seed = seed_start + ep
        obs, _ = env.reset(seed=ep_seed)
        lstm_states = torch.zeros(1, 1, model.policy.hidden_size, device=device)
        episode_starts = torch.tensor([True], device=device)

        done = False
        step_count = 0
        picked_up_ever = False

        while not done and step_count < max_steps:
            if zero_ablate_ch9 and obs["memory"].shape[0] >= 10:
                obs["memory"][9, :, :] = 0.0

            obs_t = {
                k: torch.as_tensor(v, device=device).unsqueeze(0)
                for k, v in obs.items()
            }
            legal_mask = env.action_masks()
            legal_mask_t = torch.as_tensor(np.array([legal_mask]), device=device)

            with torch.no_grad():
                actions, _, _, next_h = model.policy(
                    obs_t,
                    lstm_states=lstm_states,
                    episode_starts=episode_starts,
                    deterministic=deterministic,
                    action_masks=legal_mask_t,
                )
                action = int(actions[0].item())
                lstm_states = next_h
                episode_starts = torch.tensor([False], device=device)

            carrying_before = any(
                v["carried_by"] == 0 for v in env.sim.victims
            ) if env.sim is not None else False
            if carrying_before:
                picked_up_ever = True

            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            done = terminated or truncated

        rescued = info.get("victims_rescued", 0) > 0
        discovered = env.victim_seen
        carrying_at_end = any(v["carried_by"] == 0 for v in env.sim.victims) if env.sim is not None else False
        if carrying_at_end:
            picked_up_ever = True

        rescues.append(float(rescued))
        pickups.append(float(picked_up_ever))
        discoveries.append(float(discovered))
        oscillations.append(info.get("oscillation_percentage", 0.0))
        wait_fractions.append(info.get("wait_action_fraction", 0.0))

        if picked_up_ever:
            returns_cond.append(float(rescued))
        if truncated and picked_up_ever and not rescued:
            carrier_timeouts.append(1.0)
        else:
            carrier_timeouts.append(0.0)

        if rescued:
            successful_lengths.append(step_count)

    env.close()

    return {
        "rescue_rate": float(np.mean(rescues) * 100.0),
        "pickup_rate": float(np.mean(pickups) * 100.0),
        "discovery_rate": float(np.mean(discoveries) * 100.0),
        "return_conditional": float(np.mean(returns_cond) * 100.0) if returns_cond else 0.0,
        "carrier_timeout_rate": float(np.mean(carrier_timeouts) * 100.0),
        "mean_successful_length": float(np.mean(successful_lengths)) if successful_lengths else 0.0,
        "oscillation_percentage_mean": float(np.mean(oscillations)),
        "wait_action_fraction_mean": float(np.mean(wait_fractions) * 100.0),
    }


def evaluate_model_full(
    model_path: str | Path,
    model_name: str,
    spatial_memory_mode: str = "none",
    n_episodes: int = 200,
    zero_ablate_ch9: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Evaluating: {model_name} (Mode: {spatial_memory_mode}, Zero-Ablate Ch9: {zero_ablate_ch9})")
    print(f"=======================================================")

    dummy_env = PreviousActionWrapper(RescueEnv(
        task_type="exploration_only",
        spatial_memory_mode=spatial_memory_mode,
        reward_shaping=False,
    ))
    model = load_checkpoint(model_path, env=dummy_env, device=device)
    dummy_env.close()

    # 1. Pure Exploration Benchmark
    print(">> Running Pure Exploration Benchmark (200 episodes)...")
    exp_res = evaluate_exploration(
        model,
        spatial_memory_mode=spatial_memory_mode,
        n_episodes=n_episodes,
        seed_start=60000,
        deterministic=True,
        zero_ablate_ch9=zero_ablate_ch9,
        device=device,
    )
    print(f"   Obs Coverage: {exp_res['observed_coverage_mean']:.2f}% | Visited Coverage: {exp_res['visited_coverage_mean']:.2f}% | Wait: {exp_res['wait_action_fraction_mean']:.1f}% | Osc: {exp_res['oscillation_percentage_mean']:.2f}%")

    # 2. Spawn-Relative Rescue Ladder
    print(">> Running Spawn-Relative Rescue Ladder (5 tiers x 200 episodes)...")
    spawn_ladder = {}
    for tier_name, v_dist, min_dist, max_steps in SPAWN_TIERS:
        res = evaluate_rescue_tier(
            model,
            victim_distance=v_dist,
            min_victim_distance=min_dist,
            max_steps=max_steps,
            distance_metric="spawn",
            spatial_memory_mode=spatial_memory_mode,
            n_episodes=n_episodes,
            seed_start=50000,
            deterministic=True,
            zero_ablate_ch9=zero_ablate_ch9,
            device=device,
        )
        spawn_ladder[tier_name] = res
        print(f"   {tier_name:<15}: Rescue={res['rescue_rate']:5.1f}% | Pickup={res['pickup_rate']:5.1f}% | Disc={res['discovery_rate']:5.1f}% | RetCond={res['return_conditional']:5.1f}% | Osc={res['oscillation_percentage_mean']:4.1f}%")

    # 3. Exit-Relative Rescue Ladder
    print(">> Running Exit-Relative Rescue Ladder (6 tiers x 200 episodes)...")
    exit_ladder = {}
    for tier_name, v_dist, min_dist, max_steps in EXIT_TIERS:
        res = evaluate_rescue_tier(
            model,
            victim_distance=v_dist,
            min_victim_distance=min_dist,
            max_steps=max_steps,
            distance_metric="exit",
            spatial_memory_mode=spatial_memory_mode,
            n_episodes=n_episodes,
            seed_start=50000,
            deterministic=True,
            zero_ablate_ch9=zero_ablate_ch9,
            device=device,
        )
        exit_ladder[tier_name] = res
        print(f"   {tier_name:<15}: Rescue={res['rescue_rate']:5.1f}% | Pickup={res['pickup_rate']:5.1f}% | Disc={res['discovery_rate']:5.1f}% | RetCond={res['return_conditional']:5.1f}% | Osc={res['oscillation_percentage_mean']:4.1f}%")

    return {
        "model_name": model_name,
        "model_path": str(model_path),
        "spatial_memory_mode": spatial_memory_mode,
        "zero_ablate_ch9": zero_ablate_ch9,
        "exploration_benchmark": exp_res,
        "spawn_ladder": spawn_ladder,
        "exit_ladder": exit_ladder,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 8 Comprehensive Evaluation Suite")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint zip")
    parser.add_argument("--model-name", type=str, default="Model", help="Human-readable model name")
    parser.add_argument("--spatial-memory-mode", choices=["none", "visited_binary", "visit_recency"], default="none")
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per evaluation tier (default: 200)")
    parser.add_argument("--zero-ablate-ch9", action="store_true", default=False, help="Zero out channel 9 at inference")
    parser.add_argument("--output-json", type=str, required=True, help="Output path for JSON evaluation results")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    results = evaluate_model_full(
        model_path=args.model_path,
        model_name=args.model_name,
        spatial_memory_mode=args.spatial_memory_mode,
        n_episodes=args.episodes,
        zero_ablate_ch9=args.zero_ablate_ch9,
        device=args.device,
    )
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] Saved evaluation results to {out_path}")


if __name__ == "__main__":
    main()
