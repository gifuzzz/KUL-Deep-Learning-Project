"""FlameBox Agent Evaluation Script.

Evaluates trained agents strictly using unshaped native rescue rewards (+10 only)
with zero privileged simulator state, no waypoints, and no teacher calls (reward_shaping=False).
Supports recurrent hidden state propagation (GRU), action masking, deterministic/stochastic modes,
and saving results to JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from agents.cnn import RecurrentSpatialRescueCNN
from rescue_env.envs.single_agent import RescueEnv


def load_model(
    model_path: str | Path,
    env: RescueEnv | None = None,
    device: str | torch.device = "auto",
) -> tuple[MaskablePPO, torch.device]:
    """Load model checkpoint safely mapping tensors to the target device."""
    if device == "auto":
        target_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    custom_objects = {"policy_class": RecurrentSpatialRescueCNN}
    try:
        model = MaskablePPO.load(
            str(model_path),
            env=env,
            device=target_device,
            custom_objects=custom_objects,
        )
    except Exception:
        model = MaskablePPO.load(
            str(model_path),
            env=env,
            device=target_device,
        )

    if hasattr(model, "policy") and model.policy is not None:
        model.policy.to(target_device)
        if hasattr(model.policy, "optimizer") and model.policy.optimizer is not None:
            for state in model.policy.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(target_device)

    return model, target_device


def parse_victim_distance(val: str | None) -> int | None:
    if val is None or str(val).strip().lower() in ("none", "null", ""):
        return None
    return int(val)


def main():
    parser = argparse.ArgumentParser(description="FlameBox Evaluation")
    parser.add_argument(
        "--model",
        nargs="?",
        default="models/baseline/best_model.zip",
        help="Path to trained model checkpoint zip file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions stochastically instead of choosing the most likely action",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=None,
        help="Force deterministic action selection (default: True unless --stochastic is set)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Evaluate uniformly random valid actions",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for action sampling",
    )
    parser.add_argument(
        "--victim-distance",
        type=str,
        default="2",
        help="Maximum victim placement distance from exit (integer or 'none' for multi-room)",
    )
    parser.add_argument(
        "--min-victim-distance",
        type=int,
        default=1,
        help="Minimum victim placement distance from exit",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=64,
        help="Maximum allowed environment steps per episode",
    )
    parser.add_argument(
        "--scenario-start",
        type=int,
        default=20_000,
        help="First environment reset seed; use disjoint ranges for validation/test",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "auto"),
        default="auto",
        help="Inference device (auto, cpu, mps)",
    )
    parser.add_argument(
        "--output-json",
        "--output",
        dest="output_json",
        type=Path,
        default=None,
        help="Save evaluation summary and per-episode results to JSON file",
    )
    parser.add_argument(
        "--trace-failures",
        action="store_true",
        help="Print trajectory for the first episode that times out while carrying a victim",
    )
    parser.add_argument(
        "--memory-agent-marker",
        action="store_true",
        help="Include transient agent marker in memory observation channel",
    )

    args = parser.parse_args()

    if args.episodes < 1 or args.scenario_start < 0 or args.max_steps < 1:
        parser.error("episodes must be positive, scenario-start nonnegative, max-steps positive")

    deterministic = True
    if args.stochastic:
        deterministic = False
    elif args.deterministic is not None:
        deterministic = args.deterministic

    victim_dist = parse_victim_distance(args.victim_distance)

    # Strictly unshaped evaluation environment:
    # reward_shaping=False, navigation_shaping=False (+10 native rescue only)
    env = RescueEnv(
        max_steps=args.max_steps,
        victim_distance=victim_dist,
        min_victim_distance=args.min_victim_distance,
        reward_shaping=False,
        navigation_shaping=False,
        memory_agent_marker=args.memory_agent_marker,
    )

    model = None
    target_device = torch.device("cpu")
    if not args.random:
        model, target_device = load_model(args.model, env=env, device=args.device)

    print(f"Device: {target_device}")
    print(f"Mode  : {'Random' if args.random else ('Deterministic' if deterministic else 'Sampled Stochastic')}")
    print(f"Victim Distance: {victim_dist} | Max Steps: {args.max_steps} | Episodes: {args.episodes}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    records = []
    rewards = []
    successes = []
    successful_lengths = []
    timeouts = []

    pickups = []
    rescues_after_pickup = []
    timeouts_without_pickup = []
    timeouts_while_carrying = []
    drop_counts = []
    failure_printed = False

    try:
        for seed in range(args.scenario_start, args.scenario_start + args.episodes):
            obs, reset_info = env.reset(seed=seed)
            # Channel 6 is Victim in Phase 2 contract
            initially_visible = bool((obs["grid"][6] > 0).any())

            # Seed action sampling per episode
            torch.manual_seed(args.seed + seed)

            done = False
            total_reward = 0.0
            ever_picked_up = False
            carrying_at_end = False
            drops_this_episode = 0
            previously_carrying = False
            episode_trace = []
            visited_positions = set()
            outside_steps = 0

            # Recurrent hidden state handling
            gru_states = None
            episode_starts = np.ones((1,), dtype=bool)

            while not done:
                if args.random:
                    mask = env.action_masks()
                    valid_actions = np.flatnonzero(mask)
                    action = int(np.random.choice(valid_actions))
                else:
                    action, gru_states = model.predict(
                        obs,
                        state=gru_states,
                        episode_start=episode_starts,
                        deterministic=deterministic,
                        action_masks=env.action_masks(),
                    )
                    episode_starts = np.zeros((1,), dtype=bool)

                obs, reward, terminated, truncated, info = env.step(int(action))

                total_reward += reward
                done = terminated or truncated

                victims = env.sim.victims
                person = env.sim.agents[0]
                position = (person["r"], person["c"])
                visited_positions.add(position)
                outside_steps += int(not (0 <= position[0] < env.sim.h and 0 <= position[1] < env.sim.w))

                carrying = any(
                    v["carried_by"] == 0 and not v["dead"] and not v["rescued"]
                    for v in victims
                )
                ever_picked_up = ever_picked_up or carrying
                carrying_at_end = carrying

                if previously_carrying and not carrying:
                    dropped = any(
                        v["carried_by"] < 0 and not v["rescued"] and not v["dead"]
                        for v in env.sim.victims
                    )
                    if dropped:
                        drops_this_episode += 1

                previously_carrying = carrying

                if args.trace_failures and not failure_printed:
                    episode_trace.append({
                        "step": int(info["steps"]),
                        "action": int(action),
                        "position": (person["r"], person["c"]),
                        "carrying": carrying,
                        "reward": float(reward),
                    })

            if (
                args.trace_failures
                and not failure_printed
                and info["timeout"]
                and carrying_at_end
            ):
                failure_printed = True
                print(f"\nFailed return episode: environment seed {seed}")
                print("Step | Action  | Position | Carrying | Reward")
                print("-" * 55)
                action_names = {0: "Wait", 1: "Up", 2: "Right", 3: "Down", 4: "Left", 5: "Pickup", 6: "Water"}
                for entry in episode_trace:
                    print(
                        f"{entry['step']:>4} | "
                        f"{action_names.get(entry['action'], str(entry['action'])):<7} | "
                        f"{str(entry['position']):<8} | "
                        f"{str(entry['carrying']):<8} | "
                        f"{entry['reward']:.3f}"
                    )

            ever_picked_up = ever_picked_up or info["success"]
            pickups.append(ever_picked_up)
            if ever_picked_up:
                rescues_after_pickup.append(info["success"])

            timeouts_without_pickup.append(info["timeout"] and not ever_picked_up)
            timeouts_while_carrying.append(info["timeout"] and carrying_at_end)
            drop_counts.append(drops_this_episode)

            rewards.append(total_reward)
            successes.append(info["success"])
            timeouts.append(info["timeout"])

            records.append(dict(
                seed=seed,
                success=bool(info["success"]),
                initially_visible=initially_visible,
                victim_exit_distance=reset_info.get("victim_exit_distance"),
                pickup=bool(ever_picked_up),
                steps=int(info["steps"]),
                timeout=bool(info["timeout"]),
                carrying_timeout=bool(info["timeout"] and carrying_at_end),
                unique_positions=len(visited_positions),
                outside_steps=outside_steps,
                drops=int(drops_this_episode),
            ))

            if info["success"]:
                successful_lengths.append(info["steps"])

        mean_reward = float(np.mean(rewards))
        success_rate = float(np.mean(successes)) * 100
        timeout_rate = float(np.mean(timeouts)) * 100
        pickup_rate = float(np.mean(pickups)) * 100
        rescue_rate_after_pickup = float(np.mean(rescues_after_pickup) * 100) if rescues_after_pickup else 0.0

        print("\nEvaluation results")
        print("------------------")
        print(f"Episodes: {args.episodes}")
        print(f"Mean reward: {mean_reward:.3f}")
        print(f"Success rate: {success_rate:.1f} %")
        print(f"Timeout rate: {timeout_rate:.1f} %")
        print(
            "Mean successful episode length:",
            f"{np.mean(successful_lengths):.1f}" if successful_lengths else "No successful rescues",
        )

        print("\nRescue diagnostics")
        print("------------------")
        print(f"Pickup rate: {pickup_rate:.1f} %")
        print(f"Rescue rate after pickup: {rescue_rate_after_pickup:.1f} %")
        print(f"Timeout without pickup: {np.mean(timeouts_without_pickup) * 100:.1f} %")
        print(f"Timeout while carrying: {np.mean(timeouts_while_carrying) * 100:.1f} %")
        print(f"Episodes with drops: {np.mean(np.array(drop_counts) > 0) * 100:.1f} %")
        print(f"Average drops per episode: {np.mean(drop_counts):.2f}")

        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            summary_output = {
                "model": str(args.model) if not args.random else None,
                "device": str(target_device),
                "deterministic": deterministic,
                "stochastic": not deterministic,
                "random": args.random,
                "seed": args.seed,
                "victim_distance": victim_dist,
                "min_victim_distance": args.min_victim_distance,
                "max_steps": args.max_steps,
                "episodes_count": args.episodes,
                "mean_reward": mean_reward,
                "success_rate": success_rate,
                "timeout_rate": timeout_rate,
                "pickup_rate": pickup_rate,
                "rescue_rate_after_pickup": rescue_rate_after_pickup,
                "actions": list(env.ACTIONS),
                "episodes": records,
            }
            with args.output_json.open("w") as f:
                json.dump(summary_output, f, indent=2)
            print(f"\nResults successfully saved to {args.output_json}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
