import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from rescue_env.envs.single_agent import RescueEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        nargs="?",
        default="models/baseline/best_model.zip",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of choosing the most likely action",
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
        help="Random seed for stochastic evaluation",
    )
    parser.add_argument(
        "--victim-distance",
        type=int,
        default=2,
        help="Maximum victim placement distance from an exit",
    )
    parser.add_argument(
        "--trace-failures",
        action="store_true",
        help="Print the first episode that times out while carrying a victim",
    )
    parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="auto")
    parser.add_argument("--scenario-start", type=int, default=20_000,
                        help="First environment reset seed; use disjoint ranges for validation/test")
    parser.add_argument("--output", type=Path, help="Save per-episode results to a new JSON file")
    parser.add_argument("--memory-agent-marker", action="store_true")
    parser.add_argument("--min-victim-distance", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=64)
    args = parser.parse_args()
    if args.episodes < 1 or args.scenario_start < 0 or args.max_steps < 1:
        parser.error("episodes must be positive and scenario-start nonnegative")
    if args.output and args.output.exists():
        parser.error("output already exists; choose a new path")
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    print("Device:", device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.set_num_threads(1)

    # Evaluation environment: same task as training,
    # but no reward shaping.
    env = RescueEnv(
        max_steps=args.max_steps,
        victim_distance=args.victim_distance,
        min_victim_distance=args.min_victim_distance,
        reward_shaping=False,
        navigation_shaping=False,
        memory_agent_marker=args.memory_agent_marker,
    )

    model = None if args.random else MaskablePPO.load(args.model, env=env, device=device)
    # Space checks cannot identify old checkpoints with different action semantics.
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
            initially_visible = bool((obs["grid"][3] > 0).any())

             
            # Set the action-sampling seed after resetting the environment.
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

            while not done:
                if args.random:
                    mask = env.action_masks()
                    valid_actions = np.flatnonzero(mask)
                    action = int(np.random.choice(valid_actions))
                else:
                    action, _ = model.predict(
                        obs,
                        deterministic=not args.stochastic,
                        action_masks=env.action_masks(),
                    )

                obs, reward, terminated, truncated, info = env.step(
                    int(action)
                )

                total_reward += reward
                done = terminated or truncated
                victims = env.sim.victims
                person = env.sim.agents[0]
                position = (person["r"], person["c"])
                visited_positions.add(position)
                outside_steps += int(not (0 <= position[0] < env.sim.h
                                          and 0 <= position[1] < env.sim.w))

                carrying = any(
                    v["carried_by"] == 0
                    and not v["dead"]
                    and not v["rescued"]
                    for v in victims
                )

                ever_picked_up = ever_picked_up or carrying

                carrying_at_end = carrying

                if previously_carrying and not carrying:
                    dropped = any(
                        v["carried_by"] < 0
                        and not v["rescued"]
                        and not v["dead"]
                        for v in env.sim.victims
                    )

                    if dropped:
                        drops_this_episode += 1

                previously_carrying = carrying

                if args.trace_failures and not failure_printed:
                    agent = env.sim.agents[0]

                    episode_trace.append({
                        "step": info["steps"],
                        "action": int(action),
                        "position": (agent["r"], agent["c"]),
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
                print("Step | Action | Position | Carrying | Reward")
                print("-" * 55)

                action_names = {
                    0: "Wait",
                    1: "Up",
                    2: "Right",
                    3: "Down",
                    4: "Left",
                    5: "Pickup",
                    6: "Water",
                }

                for entry in episode_trace:
                    action_id = entry["action"]

                    print(
                        f"{entry['step']:>4} | "
                        f"{action_names[action_id]:<7} | "
                        f"{str(entry['position']):<10} | "
                        f"{str(entry['carrying']):<8} | "
                        f"{entry['reward']:.3f}"
                    )
            ever_picked_up = ever_picked_up or info["success"]

            pickups.append(ever_picked_up)

            if ever_picked_up:
                rescues_after_pickup.append(info["success"])

            timeouts_without_pickup.append(
                info["timeout"] and not ever_picked_up
            )

            timeouts_while_carrying.append(
                info["timeout"] and carrying_at_end
            )

            drop_counts.append(drops_this_episode)

            rewards.append(total_reward)
            successes.append(info["success"])
            timeouts.append(info["timeout"])

            records.append(dict(seed=seed, success=bool(info["success"]),
                                initially_visible=initially_visible,
                                victim_exit_distance=reset_info["victim_exit_distance"],
                                pickup=bool(ever_picked_up), steps=int(info["steps"]),
                                timeout=bool(info["timeout"]), carrying_timeout=bool(info["timeout"] and carrying_at_end),
                                unique_positions=len(visited_positions), outside_steps=outside_steps,
                                drops=int(drops_this_episode)))
            if info["success"]:
                successful_lengths.append(info["steps"])

        print("\nEvaluation results")
        print("------------------")
        print("Episodes:", args.episodes)
        print("Mean reward:", np.mean(rewards))
        print("Success rate:", np.mean(successes) * 100, "%")
        print("Timeout rate:", np.mean(timeouts) * 100, "%")

        print(
            "Mean successful episode length:",
            np.mean(successful_lengths)
            if successful_lengths
            else "No successful rescues",
        )

        print("\nRescue diagnostics")
        print("------------------")

        print("Pickup rate:", np.mean(pickups) * 100, "%")

        print(
            "Rescue rate after pickup:",
            np.mean(rescues_after_pickup) * 100
            if rescues_after_pickup else 0,
            "%",
        )

        print(
            "Timeout without pickup:",
            np.mean(timeouts_without_pickup) * 100,
            "%",
        )

        print(
            "Timeout while carrying:",
            np.mean(timeouts_while_carrying) * 100,
            "%",
        )
        print("Episodes with drops:", np.mean(np.array(drop_counts) > 0) * 100, "%")
        print("Average drops per episode:", np.mean(drop_counts))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as output:
                json.dump(dict(model=None if args.random else args.model, device=device,
                               stochastic=args.stochastic, random=args.random, action_seed=args.seed,
                               victim_distance=args.victim_distance,
                               min_victim_distance=args.min_victim_distance,
                               max_steps=args.max_steps,
                               memory_agent_marker=args.memory_agent_marker,
                               actions=list(env.ACTIONS), episodes=records), output, indent=2)

    finally:
        env.close()


if __name__ == "__main__":
    main()
