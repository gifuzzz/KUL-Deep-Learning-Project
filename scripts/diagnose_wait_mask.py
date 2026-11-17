"""Zero-shot Wait-mask diagnostic.

Evaluates Phase 6 Base and Phase 7 Best models with and without masking the Wait action (action 0)
when valid movement actions (actions 1..4) are available.
Measures whether forced movement breaks stationary traps or induces navigation loops.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import torch

from rescue_env.envs.single_agent import RescueEnv
from rescue_env.wrappers import PreviousActionWrapper
from agents.recurrent_ppo import load_checkpoint
from training.train_ppo import DISTANCE_CONFIGS


def run_wait_mask_eval(model_path: str, model_name: str, mask_wait: bool, n_episodes: int = 50, seed_start: int = 60000):
    print(f"\n--- Evaluating {model_name} | Mask Wait: {mask_wait} ({n_episodes} eps) ---")

    # Pure exploration benchmark
    env = PreviousActionWrapper(RescueEnv(
        task_type="exploration_only",
        geometry_mode="edge",
        reward_shaping=False,
        max_steps=384,
        distance_metric="spawn",
    ))
    model = load_checkpoint(model_path, env=env, device="cpu")

    obs_coverages = []
    vis_coverages = []
    wait_fractions = []
    oscillations = []
    lengths = []

    for ep in range(n_episodes):
        ep_seed = seed_start + ep
        obs, _ = env.reset(seed=ep_seed)
        lstm_states = torch.zeros(1, 1, model.policy.hidden_size)
        episode_starts = torch.tensor([True])

        done = False
        step_count = 0
        wait_count = 0

        while not done and step_count < 384:
            legal_mask = env.action_masks()
            if mask_wait:
                has_move = any(legal_mask[a] for a in (1, 2, 3, 4))
                if has_move:
                    legal_mask[0] = False

            obs_t = {
                k: torch.as_tensor(v, device="cpu").unsqueeze(0)
                for k, v in obs.items()
            }

            with torch.no_grad():
                dist = model.policy.get_distribution(
                    obs_t,
                    lstm_states=lstm_states,
                    episode_starts=episode_starts,
                    action_masks=torch.tensor([legal_mask]),
                )
                action = int(dist.distribution.logits.argmax(dim=-1).item())
                _, lstm_states = model.policy._forward_network(
                    obs_t,
                    lstm_states=lstm_states,
                    episode_starts=episode_starts,
                )
                episode_starts = torch.tensor([False])

            if action == 0:
                wait_count += 1

            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            done = terminated or truncated

        obs_coverages.append(info.get("observed_coverage", 0.0))
        vis_coverages.append(info.get("visited_coverage", 0.0))
        wait_fractions.append(info.get("wait_action_fraction", 0.0))
        oscillations.append(info.get("oscillation_percentage", 0.0))
        lengths.append(step_count)

    env.close()

    res = {
        "model": model_name,
        "mask_wait": mask_wait,
        "n_episodes": n_episodes,
        "mean_observed_coverage": float(np.mean(obs_coverages)),
        "mean_visited_coverage": float(np.mean(vis_coverages)),
        "mean_wait_fraction": float(np.mean(wait_fractions)),
        "mean_oscillation_pct": float(np.mean(oscillations)),
        "mean_length": float(np.mean(lengths)),
    }
    print(f"Result: Obs Cov: {res['mean_observed_coverage']*100:.2f}%, Vis Cov: {res['mean_visited_coverage']*100:.2f}%, Wait: {res['mean_wait_fraction']*100:.1f}%, Osc: {res['mean_oscillation_pct']:.2f}%")
    return res


def main():
    p6_path = "models/phase6_spawn_curriculum/production_250k/stage3_spawn_curriculum/best_model.zip"
    p7_path = "models/phase7_explore20/stage3_spawn_curriculum/best_model.zip"

    results = []
    # Test Phase 6: Standard vs Masked Wait
    results.append(run_wait_mask_eval(p6_path, "Phase 6 Base", mask_wait=False, n_episodes=50))
    results.append(run_wait_mask_eval(p6_path, "Phase 6 Base", mask_wait=True, n_episodes=50))

    # Test Phase 7: Standard vs Masked Wait
    results.append(run_wait_mask_eval(p7_path, "Phase 7 Best", mask_wait=False, n_episodes=50))
    results.append(run_wait_mask_eval(p7_path, "Phase 7 Best", mask_wait=True, n_episodes=50))

    out_file = Path("models/phase8_evaluations/wait_mask_diagnostic.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved wait mask diagnostic results to {out_file}")


if __name__ == "__main__":
    main()
