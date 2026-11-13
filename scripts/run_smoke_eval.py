"""Run 50-episode unshaped evaluation on smoke_model.zip (deterministic vs stochastic).

Saves structured results to models/autonomous_rl_v1/smoke_eval.json.
"""
import json
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from agents.cnn import RecurrentSpatialRescueCNN
from rescue_env.envs.single_agent import RescueEnv
from training.evaluate import load_model


def evaluate_policy(
    model,
    env: RescueEnv,
    episodes: int = 50,
    deterministic: bool = True,
    scenario_start: int = 10000,
    seed: int = 42,
) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    records = []
    rewards = []
    successes = []
    lengths = []
    successful_lengths = []
    timeouts = []
    pickups = []
    rescues_after_pickup = []

    for ep_idx, sc_seed in enumerate(range(scenario_start, scenario_start + episodes)):
        obs, reset_info = env.reset(seed=sc_seed)
        torch.manual_seed(seed + sc_seed)

        done = False
        total_reward = 0.0
        ever_picked_up = False
        carrying_at_end = False
        step_count = 0

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

            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            step_count += 1
            done = terminated or truncated

            carrying = any(
                v["carried_by"] == 0 and not v["dead"] and not v["rescued"]
                for v in env.sim.victims
            )
            ever_picked_up = ever_picked_up or carrying
            carrying_at_end = carrying

        ever_picked_up = ever_picked_up or info["success"]
        pickups.append(ever_picked_up)
        if ever_picked_up:
            rescues_after_pickup.append(info["success"])

        rewards.append(total_reward)
        successes.append(info["success"])
        timeouts.append(info["timeout"])
        lengths.append(step_count)
        if info["success"]:
            successful_lengths.append(step_count)

        records.append({
            "episode": ep_idx + 1,
            "seed": sc_seed,
            "success": bool(info["success"]),
            "reward": float(total_reward),
            "steps": int(step_count),
            "timeout": bool(info["timeout"]),
            "pickup": bool(ever_picked_up),
            "carrying_at_end": bool(carrying_at_end),
        })

    mean_reward = float(np.mean(rewards))
    success_rate = float(np.mean(successes)) * 100.0
    timeout_rate = float(np.mean(timeouts)) * 100.0
    pickup_rate = float(np.mean(pickups)) * 100.0
    mean_length = float(np.mean(lengths))
    mean_successful_length = float(np.mean(successful_lengths)) if successful_lengths else None

    return {
        "episodes": episodes,
        "deterministic": deterministic,
        "success_rate_percent": success_rate,
        "mean_reward": mean_reward,
        "mean_episode_length": mean_length,
        "mean_successful_episode_length": mean_successful_length,
        "timeout_rate_percent": timeout_rate,
        "pickup_rate_percent": pickup_rate,
        "episodes_detail": records,
    }


def main():
    model_path = Path("models/autonomous_rl_v1/smoke_model.zip")
    output_path = Path("models/autonomous_rl_v1/smoke_eval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading smoke model from {model_path}...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # Unshaped evaluation environment
    env = RescueEnv(
        max_steps=64,
        victim_distance=2,
        min_victim_distance=1,
        reward_shaping=False,
        navigation_shaping=False,
    )

    try:
        model, target_device = load_model(model_path, env=env, device=device)
        print(f"Loaded model on device: {target_device}")

        print("\n--- Running 50 Episodes Deterministic Unshaped Evaluation ---")
        det_results = evaluate_policy(
            model=model,
            env=env,
            episodes=50,
            deterministic=True,
            scenario_start=20000,
        )
        print(f"Deterministic: Success Rate = {det_results['success_rate_percent']:.1f}%, Mean Reward = {det_results['mean_reward']:.3f}, Mean Length = {det_results['mean_episode_length']:.1f}")

        print("\n--- Running 50 Episodes Stochastic Unshaped Evaluation ---")
        stoch_results = evaluate_policy(
            model=model,
            env=env,
            episodes=50,
            deterministic=False,
            scenario_start=30000,
        )
        print(f"Stochastic: Success Rate = {stoch_results['success_rate_percent']:.1f}%, Mean Reward = {stoch_results['mean_reward']:.3f}, Mean Length = {stoch_results['mean_episode_length']:.1f}")

        eval_summary = {
            "model_path": str(model_path),
            "device": str(target_device),
            "evaluation_environment": {
                "max_steps": 64,
                "victim_distance": 2,
                "min_victim_distance": 1,
                "reward_shaping": False,
                "navigation_shaping": False,
                "native_reward_only": "+10 native rescue only",
            },
            "metrics": {
                "deterministic": {
                    "episodes": det_results["episodes"],
                    "success_rate_percent": det_results["success_rate_percent"],
                    "mean_reward": det_results["mean_reward"],
                    "mean_episode_length": det_results["mean_episode_length"],
                    "mean_successful_episode_length": det_results["mean_successful_episode_length"],
                    "pickup_rate_percent": det_results["pickup_rate_percent"],
                    "timeout_rate_percent": det_results["timeout_rate_percent"],
                },
                "stochastic": {
                    "episodes": stoch_results["episodes"],
                    "success_rate_percent": stoch_results["success_rate_percent"],
                    "mean_reward": stoch_results["mean_reward"],
                    "mean_episode_length": stoch_results["mean_episode_length"],
                    "mean_successful_episode_length": stoch_results["mean_successful_episode_length"],
                    "pickup_rate_percent": stoch_results["pickup_rate_percent"],
                    "timeout_rate_percent": stoch_results["timeout_rate_percent"],
                },
            },
            "deterministic_detail": det_results["episodes_detail"],
            "stochastic_detail": stoch_results["episodes_detail"],
        }

        with open(output_path, "w") as f:
            json.dump(eval_summary, f, indent=2)

        print(f"\nSaved structured evaluation results to {output_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
