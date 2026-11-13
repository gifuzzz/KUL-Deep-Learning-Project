"""Phase 4: Controlled Training Validation Script for fire-rescue-project-2.0.

Executes a controlled training validation experiment comparing performance before
and after training on a fixed evaluation scenario set:
1. Define 50 fixed evaluation scenarios (seeds 50000-50049: 25 near/visible + 25 medium/hidden).
2. Evaluate initialized agent (seed 42) BEFORE training in both Deterministic and Stochastic modes.
3. Train pilot run (16,384 steps) with MaskableRecurrentPPO under Stage 1 Mixed Visibility Curriculum.
   Records per-update training metrics: policy loss, value loss, entropy, approx KL,
   GRU gradient norms, and explained variance.
4. Evaluate checkpoint AFTER training on the EXACT SAME 50 scenarios in Deterministic and Stochastic modes.
5. Generate side-by-side comparison tables and save machine-readable outputs:
   - models/controlled_pilot/pilot_eval_before.json
   - models/controlled_pilot/pilot_eval_after.json
   - models/controlled_pilot/pilot_summary.json
   - models/controlled_pilot/pilot_model.zip
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.cnn import RecurrentSpatialRescueCNN
from agents.recurrent_ppo import MaskableRecurrentPPO
from rescue_env.envs.single_agent import RescueEnv
from rescue_env.wrappers import PreviousActionWrapper
from training.train_ppo import CurriculumRescueEnv, get_git_commit, load_checkpoint, make_train_env, save_checkpoint


def generate_fixed_scenarios(n_episodes: int = 50, seed_start: int = 50000) -> list[dict[str, Any]]:
    """Define fixed 50 evaluation scenarios: 25 near/visible and 25 medium/hidden."""
    scenarios = []
    n_near = n_episodes // 2
    n_medium = n_episodes - n_near

    for i in range(n_near):
        scenarios.append({
            "scenario_index": i,
            "seed": seed_start + i,
            "scenario_type": "near_visible",
            "victim_distance": 2,
            "min_victim_distance": 1,
            "max_steps": 64,
        })

    for i in range(n_medium):
        scenarios.append({
            "scenario_index": n_near + i,
            "seed": seed_start + n_near + i,
            "scenario_type": "medium_hidden",
            "victim_distance": 6,
            "min_victim_distance": 3,
            "max_steps": 128,
        })

    return scenarios


def evaluate_scenario_set(
    model: MaskableRecurrentPPO,
    scenarios: list[dict[str, Any]],
    deterministic: bool = True,
    device: str | torch.device = "cpu",
    base_seed: int = 42,
) -> dict[str, Any]:
    """Evaluate model strictly using unshaped native rescue rewards (+10 only)."""
    records = []
    rewards = []
    successes = []
    pickups = []
    rescues_after_pickup = []
    timeouts_before_pickup = []
    timeouts_while_carrying = []
    successful_lengths = []
    drop_counts = []

    hidden_episodes_count = 0
    hidden_discovered_count = 0
    hidden_rescued_count = 0

    visible_episodes_count = 0
    visible_rescued_count = 0

    mode_str = "Deterministic" if deterministic else "Sampled Stochastic"
    print(f"\n[EVAL] Running {mode_str} Evaluation on {len(scenarios)} Fixed Scenarios...")

    for sc in scenarios:
        sc_seed = sc["seed"]
        max_steps = sc["max_steps"]
        victim_dist = sc["victim_distance"]
        min_victim_dist = sc["min_victim_distance"]
        sc_type = sc["scenario_type"]

        # Strictly unshaped evaluation environment:
        raw_env = RescueEnv(
            max_steps=max_steps,
            victim_distance=victim_dist,
            min_victim_distance=min_victim_dist,
            reward_shaping=False,
            navigation_shaping=False,
        )
        eval_env = PreviousActionWrapper(raw_env)

        try:
            obs, reset_info = eval_env.reset(seed=sc_seed)
            # Channel 6 is victim presence
            initially_visible = bool((obs["grid"][6] > 0).any())
            victim_exit_dist = reset_info.get("victim_exit_distance")

            # Seed action sampling per episode
            torch.manual_seed(base_seed + sc_seed)
            np.random.seed(base_seed + sc_seed)

            done = False
            total_reward = 0.0
            ever_picked_up = False
            carrying_at_end = False
            drops_this_episode = 0
            previously_carrying = False
            step_count = 0
            victim_seen = initially_visible

            gru_states = None
            episode_starts = np.ones((1,), dtype=bool)

            while not done:
                action, gru_states = model.predict(
                    obs,
                    state=gru_states,
                    episode_start=episode_starts,
                    deterministic=deterministic,
                    action_masks=eval_env.action_masks(),
                )
                episode_starts = np.zeros((1,), dtype=bool)

                obs, reward, terminated, truncated, info = eval_env.step(int(action))
                total_reward += reward
                step_count += 1
                done = terminated or truncated

                # Check victim visibility during episode
                if (obs["grid"][6] > 0).any():
                    victim_seen = True

                victims = eval_env.sim.victims
                carrying = any(
                    v["carried_by"] == 0 and not v["dead"] and not v["rescued"]
                    for v in victims
                )
                ever_picked_up = ever_picked_up or carrying
                carrying_at_end = carrying

                if previously_carrying and not carrying:
                    dropped = any(
                        v["carried_by"] < 0 and not v["rescued"] and not v["dead"]
                        for v in eval_env.sim.victims
                    )
                    if dropped:
                        drops_this_episode += 1

                previously_carrying = carrying

            success = bool(info["success"])
            timeout = bool(info["timeout"])
            ever_picked_up = ever_picked_up or success

            # Track hidden-victim discovery and rescue
            if not initially_visible:
                hidden_episodes_count += 1
                discovered = bool(victim_seen or ever_picked_up or success)
                if discovered:
                    hidden_discovered_count += 1
                if success:
                    hidden_rescued_count += 1
            else:
                visible_episodes_count += 1
                if success:
                    visible_rescued_count += 1

            rewards.append(total_reward)
            successes.append(success)
            pickups.append(ever_picked_up)
            drop_counts.append(drops_this_episode)

            if ever_picked_up:
                rescues_after_pickup.append(success)

            timeouts_before_pickup.append(timeout and not ever_picked_up)
            timeouts_while_carrying.append(timeout and carrying_at_end)

            if success:
                successful_lengths.append(step_count)

            records.append({
                "scenario_index": sc["scenario_index"],
                "seed": sc_seed,
                "scenario_type": sc_type,
                "victim_distance_param": victim_dist,
                "victim_exit_distance": victim_exit_dist,
                "max_steps": max_steps,
                "initially_visible": initially_visible,
                "victim_discovered": bool(victim_seen or ever_picked_up or success),
                "pickup": ever_picked_up,
                "success": success,
                "timeout": timeout,
                "timeout_before_pickup": bool(timeout and not ever_picked_up),
                "timeout_while_carrying": bool(timeout and carrying_at_end),
                "carrying_at_end": carrying_at_end,
                "steps": step_count,
                "reward": float(total_reward),
                "drops": drops_this_episode,
            })

        finally:
            eval_env.close()

    total_episodes = len(scenarios)
    success_rate = (sum(successes) / total_episodes) * 100.0
    pickup_rate = (sum(pickups) / total_episodes) * 100.0
    return_success_rate = (
        (sum(rescues_after_pickup) / len(rescues_after_pickup) * 100.0)
        if rescues_after_pickup
        else 0.0
    )
    timeout_before_pickup_rate = (sum(timeouts_before_pickup) / total_episodes) * 100.0
    timeout_while_carrying_rate = (sum(timeouts_while_carrying) / total_episodes) * 100.0

    hidden_discovery_rate = (
        (hidden_discovered_count / hidden_episodes_count * 100.0)
        if hidden_episodes_count > 0
        else 0.0
    )
    hidden_rescue_rate = (
        (hidden_rescued_count / hidden_episodes_count * 100.0)
        if hidden_episodes_count > 0
        else 0.0
    )
    visible_rescue_rate = (
        (visible_rescued_count / visible_episodes_count * 100.0)
        if visible_episodes_count > 0
        else 0.0
    )

    mean_successful_length = (
        float(np.mean(successful_lengths)) if successful_lengths else None
    )
    mean_reward = float(np.mean(rewards))

    summary = {
        "mode": mode_str,
        "deterministic": deterministic,
        "total_episodes": total_episodes,
        "mean_reward": mean_reward,
        "success_rate_pct": success_rate,
        "pickup_rate_pct": pickup_rate,
        "return_success_conditional_pct": return_success_rate,
        "hidden_victim_discovery_rate_pct": hidden_discovery_rate,
        "hidden_victim_rescue_rate_pct": hidden_rescue_rate,
        "visible_victim_rescue_rate_pct": visible_rescue_rate,
        "timeout_before_pickup_pct": timeout_before_pickup_rate,
        "timeout_while_carrying_pct": timeout_while_carrying_rate,
        "mean_successful_episode_length": mean_successful_length,
        "successful_episodes_count": len(successful_lengths),
        "total_hidden_episodes": hidden_episodes_count,
        "total_visible_episodes": visible_episodes_count,
        "episodes_with_drops_pct": float(np.mean(np.array(drop_counts) > 0) * 100.0),
        "mean_drops_per_episode": float(np.mean(drop_counts)),
    }

    print(f"  -> Success Rate: {success_rate:.1f}% | Pickup Rate: {pickup_rate:.1f}% | Return Cond: {return_success_rate:.1f}%")
    print(f"  -> Hidden Discovery: {hidden_discovery_rate:.1f}% | Hidden Rescue: {hidden_rescue_rate:.1f}%")
    print(f"  -> Timeout (no pickup): {timeout_before_pickup_rate:.1f}% | Timeout (carrying): {timeout_while_carrying_rate:.1f}%")
    if mean_successful_length is not None:
        print(f"  -> Mean Successful Length: {mean_successful_length:.1f} steps")
    else:
        print("  -> Mean Successful Length: N/A (0 rescues)")

    return {
        "summary": summary,
        "episodes": records,
    }


def main():
    parser = argparse.ArgumentParser(description="FlameBox Controlled Pilot Experiment")
    parser.add_argument("--timesteps", type=int, default=16384, help="Pilot training timesteps")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel environments for training")
    parser.add_argument("--n-steps", type=int, default=32, help="Rollout steps per env")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO minibatch size")
    parser.add_argument("--n-epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Pilot random seed")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--save-dir", type=str, default="models/controlled_pilot", help="Output directory")
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print("=" * 70)
    print("PHASE 4: CONTROLLED TRAINING VALIDATION EXPERIMENT")
    print("=" * 70)
    print(f"Device        : {device}")
    print(f"Model Seed    : {args.seed}")
    print(f"Pilot Steps   : {args.timesteps}")
    print(f"Parallel Envs : {args.n_envs}")
    print(f"Rollout n_steps: {args.n_steps} (Buffer: {args.n_envs * args.n_steps})")
    print(f"Batch Size    : {args.batch_size} | Epochs: {args.n_epochs} | LR: {args.lr}")
    print(f"Output Dir    : {args.save_dir}")
    print("=" * 70)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate 50 Fixed Evaluation Scenarios
    scenarios = generate_fixed_scenarios(n_episodes=50, seed_start=50000)
    scenarios_meta_file = save_dir / "fixed_scenarios_definition.json"
    with open(scenarios_meta_file, "w") as f:
        json.dump(scenarios, f, indent=2)
    print(f"\n[SETUP] 50 Fixed Scenarios defined and saved to {scenarios_meta_file}")

    # Set master seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)

    # 2. Initialize Model (seed 42) for BEFORE Evaluation
    def make_train_subenv():
        return make_train_env(stage=1, reward_shaping=True, navigation_shaping=False)

    train_vec_env = DummyVecEnv([make_train_subenv for _ in range(args.n_envs)])

    print(f"\n[INIT] Initializing MaskableRecurrentPPO (seed={args.seed}) on {device}...")
    model = MaskableRecurrentPPO(
        policy=RecurrentSpatialRescueCNN,
        env=train_vec_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.99,
        ent_coef=0.01,
        max_grad_norm=0.5,
        verbose=0,
        device=device,
        seed=args.seed,
    )

    # 3. BEFORE Evaluation (Deterministic & Stochastic)
    print("\n" + "=" * 70)
    print("STEP 1: MEASURING BEFORE METRICS (Untrained Initialized Model)")
    print("=" * 70)
    t0_before = time.time()
    before_det = evaluate_scenario_set(model, scenarios, deterministic=True, device=device, base_seed=args.seed)
    before_stoch = evaluate_scenario_set(model, scenarios, deterministic=False, device=device, base_seed=args.seed)
    t_before_elapsed = time.time() - t0_before
    print(f"[EVAL] Before evaluation completed in {t_before_elapsed:.2f}s")

    eval_before_output = {
        "experiment": "Phase 4 Controlled Training Validation",
        "evaluation_state": "BEFORE",
        "date": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "device": device,
        "seed": args.seed,
        "deterministic": before_det,
        "stochastic": before_stoch,
    }
    before_json_path = save_dir / "pilot_eval_before.json"
    with open(before_json_path, "w") as f:
        json.dump(eval_before_output, f, indent=2)
    print(f"[SAVE] Saved {before_json_path}")

    # 4. Controlled Pilot Training Run (e.g. 16,384 steps)
    print("\n" + "=" * 70)
    print(f"STEP 2: TRAINING PILOT RUN ({args.timesteps} timesteps)")
    print("=" * 70)
    t0_train = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=False,
    )
    t_train_elapsed = time.time() - t0_train
    sps = args.timesteps / t_train_elapsed
    print(f"\n[TRAIN] Pilot training complete: {args.timesteps} steps in {t_train_elapsed:.2f}s ({sps:.1f} SPS)")

    # Save trained checkpoint
    pilot_model_path = save_dir / "pilot_model.zip"
    save_checkpoint(
        model,
        pilot_model_path,
        metadata={
            "experiment": "controlled_pilot",
            "timesteps": args.timesteps,
            "device": device,
            "seed": args.seed,
            "hyperparameters": {
                "n_steps": args.n_steps,
                "n_envs": args.n_envs,
                "batch_size": args.batch_size,
                "n_epochs": args.n_epochs,
                "learning_rate": args.lr,
                "ent_coef": 0.01,
            },
        },
    )
    print(f"[SAVE] Checkpoint saved to {pilot_model_path}")

    # 5. AFTER Evaluation on the EXACT SAME 50 Fixed Scenarios
    print("\n" + "=" * 70)
    print("STEP 3: MEASURING AFTER METRICS (Trained Pilot Model)")
    print("=" * 70)
    print(f"[LOAD] Verifying and loading saved checkpoint from {pilot_model_path} onto {device}...")
    eval_model = load_checkpoint(pilot_model_path, device=device)
    t0_after = time.time()
    after_det = evaluate_scenario_set(eval_model, scenarios, deterministic=True, device=device, base_seed=args.seed)
    after_stoch = evaluate_scenario_set(eval_model, scenarios, deterministic=False, device=device, base_seed=args.seed)
    t_after_elapsed = time.time() - t0_after
    print(f"[EVAL] After evaluation completed in {t_after_elapsed:.2f}s")

    eval_after_output = {
        "experiment": "Phase 4 Controlled Training Validation",
        "evaluation_state": "AFTER",
        "date": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "device": device,
        "seed": args.seed,
        "pilot_timesteps": args.timesteps,
        "deterministic": after_det,
        "stochastic": after_stoch,
    }
    after_json_path = save_dir / "pilot_eval_after.json"
    with open(after_json_path, "w") as f:
        json.dump(eval_after_output, f, indent=2)
    print(f"[SAVE] Saved {after_json_path}")

    # 6. Extract Training Metrics Dynamics
    training_history = getattr(model, "training_metrics_history", [])
    print(f"\n[METRICS] Captured {len(training_history)} training update records.")

    policy_losses = [m["policy_loss"] for m in training_history]
    value_losses = [m["value_loss"] for m in training_history]
    entropies = [m["entropy"] for m in training_history]
    approx_kls = [m["approx_kl"] for m in training_history]
    gru_norms = [m["gru_grad_norm"] for m in training_history]
    exp_vars = [m["explained_variance"] for m in training_history]

    training_summary = {
        "total_updates": len(training_history),
        "total_timesteps": args.timesteps,
        "training_time_seconds": t_train_elapsed,
        "throughput_sps": sps,
        "initial_update": training_history[0] if training_history else None,
        "mid_update": training_history[len(training_history) // 2] if training_history else None,
        "final_update": training_history[-1] if training_history else None,
        "aggregate_metrics": {
            "policy_loss": {
                "mean": float(np.mean(policy_losses)),
                "min": float(np.min(policy_losses)),
                "max": float(np.max(policy_losses)),
                "initial": float(policy_losses[0]),
                "final": float(policy_losses[-1]),
            },
            "value_loss": {
                "mean": float(np.mean(value_losses)),
                "min": float(np.min(value_losses)),
                "max": float(np.max(value_losses)),
                "initial": float(value_losses[0]),
                "final": float(value_losses[-1]),
            },
            "entropy": {
                "mean": float(np.mean(entropies)),
                "min": float(np.min(entropies)),
                "max": float(np.max(entropies)),
                "initial": float(entropies[0]),
                "final": float(entropies[-1]),
            },
            "approx_kl": {
                "mean": float(np.mean(approx_kls)),
                "min": float(np.min(approx_kls)),
                "max": float(np.max(approx_kls)),
                "initial": float(approx_kls[0]),
                "final": float(approx_kls[-1]),
            },
            "gru_grad_norm": {
                "mean": float(np.mean(gru_norms)),
                "min": float(np.min(gru_norms)),
                "max": float(np.max(gru_norms)),
                "initial": float(gru_norms[0]),
                "final": float(gru_norms[-1]),
            },
            "explained_variance": {
                "mean": float(np.mean(exp_vars)),
                "min": float(np.min(exp_vars)),
                "max": float(np.max(exp_vars)),
                "initial": float(exp_vars[0]),
                "final": float(exp_vars[-1]),
            },
        },
        "history": training_history,
    }

    # 7. Construct Side-by-Side Comparison Table
    def make_comparison_block(before_sum: dict, after_sum: dict) -> dict[str, Any]:
        metrics = [
            "success_rate_pct",
            "pickup_rate_pct",
            "return_success_conditional_pct",
            "hidden_victim_discovery_rate_pct",
            "hidden_victim_rescue_rate_pct",
            "visible_victim_rescue_rate_pct",
            "timeout_before_pickup_pct",
            "timeout_while_carrying_pct",
            "mean_reward",
            "episodes_with_drops_pct",
            "mean_drops_per_episode",
        ]
        comp = {}
        for k in metrics:
            b_val = before_sum.get(k, 0.0)
            a_val = after_sum.get(k, 0.0)
            comp[k] = {
                "before": b_val,
                "after": a_val,
                "delta": a_val - b_val,
            }
        comp["mean_successful_episode_length"] = {
            "before": before_sum.get("mean_successful_episode_length"),
            "after": after_sum.get("mean_successful_episode_length"),
        }
        return comp

    comparison_deterministic = make_comparison_block(before_det["summary"], after_det["summary"])
    comparison_stochastic = make_comparison_block(before_stoch["summary"], after_stoch["summary"])

    pilot_summary = {
        "experiment": "Phase 4 Controlled Training Validation",
        "date": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "platform": {
            "device": device,
            "seed": args.seed,
            "timesteps": args.timesteps,
            "sps": sps,
            "training_time_sec": t_train_elapsed,
        },
        "scenarios_breakdown": {
            "total_episodes": len(scenarios),
            "near_visible_episodes": 25,
            "medium_hidden_episodes": 25,
            "initial_visible_count": before_det["summary"]["total_visible_episodes"],
            "initial_hidden_count": before_det["summary"]["total_hidden_episodes"],
        },
        "comparison": {
            "deterministic": comparison_deterministic,
            "stochastic": comparison_stochastic,
        },
        "training_dynamics": training_summary,
    }

    summary_json_path = save_dir / "pilot_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(pilot_summary, f, indent=2)
    print(f"[SAVE] Pilot summary saved to {summary_json_path}")

    # 8. Print Terminal Summary Comparison Table
    print("\n" + "=" * 82)
    print("SIDE-BY-SIDE CONTROLLED PILOT RESULTS: BEFORE VS AFTER TRAINING")
    print("=" * 82)
    header = f"{'Metric':<36} | {'Before (Det)':<12} | {'After (Det)':<12} | {'Before (Stoch)':<14} | {'After (Stoch)':<14}"
    print(header)
    print("-" * 82)

    rows = [
        ("Rescue Success Rate (%)", "success_rate_pct", "{:.1f}%"),
        ("Victim Pickup Rate (%)", "pickup_rate_pct", "{:.1f}%"),
        ("Return Success (cond. pickup) (%)", "return_success_conditional_pct", "{:.1f}%"),
        ("Hidden-Victim Discovery Rate (%)", "hidden_victim_discovery_rate_pct", "{:.1f}%"),
        ("Hidden-Victim Rescue Rate (%)", "hidden_victim_rescue_rate_pct", "{:.1f}%"),
        ("Timeout Before Pickup (%)", "timeout_before_pickup_pct", "{:.1f}%"),
        ("Timeout While Carrying (%)", "timeout_while_carrying_pct", "{:.1f}%"),
        ("Mean Reward", "mean_reward", "{:.3f}"),
    ]

    for label, key, fmt in rows:
        b_det = fmt.format(before_det["summary"][key])
        a_det = fmt.format(after_det["summary"][key])
        b_stoch = fmt.format(before_stoch["summary"][key])
        a_stoch = fmt.format(after_stoch["summary"][key])
        print(f"{label:<36} | {b_det:<12} | {a_det:<12} | {b_stoch:<14} | {a_stoch:<14}")

    def format_len(val):
        return f"{val:.1f} steps" if val is not None else "None (0 res)"

    b_len_det = format_len(before_det["summary"]["mean_successful_episode_length"])
    a_len_det = format_len(after_det["summary"]["mean_successful_episode_length"])
    b_len_stoch = format_len(before_stoch["summary"]["mean_successful_episode_length"])
    a_len_stoch = format_len(after_stoch["summary"]["mean_successful_episode_length"])
    print(f"{'Mean Successful Length (steps)':<36} | {b_len_det:<12} | {a_len_det:<12} | {b_len_stoch:<14} | {a_len_stoch:<14}")

    print("-" * 82)
    print("TRAINING DYNAMICS UPDATE SUMMARY (16,384 STEPS)")
    print("-" * 82)
    agg = training_summary["aggregate_metrics"]
    print(f"Policy Loss        : Initial = {agg['policy_loss']['initial']:.6f} | Final = {agg['policy_loss']['final']:.6f} | Mean = {agg['policy_loss']['mean']:.6f}")
    print(f"Value Loss         : Initial = {agg['value_loss']['initial']:.6f} | Final = {agg['value_loss']['final']:.6f} | Mean = {agg['value_loss']['mean']:.6f}")
    print(f"Entropy            : Initial = {agg['entropy']['initial']:.4f} | Final = {agg['entropy']['final']:.4f} | Mean = {agg['entropy']['mean']:.4f}")
    print(f"Approx KL          : Initial = {agg['approx_kl']['initial']:.6f} | Final = {agg['approx_kl']['final']:.6f} | Mean = {agg['approx_kl']['mean']:.6f}")
    print(f"GRU Gradient Norm  : Initial = {agg['gru_grad_norm']['initial']:.4f} | Final = {agg['gru_grad_norm']['final']:.4f} | Mean = {agg['gru_grad_norm']['mean']:.4f}")
    print(f"Explained Variance : Initial = {agg['explained_variance']['initial']:.4f} | Final = {agg['explained_variance']['final']:.4f} | Mean = {agg['explained_variance']['mean']:.4f}")
    print("=" * 82)

    train_vec_env.close()
    print("\nPhase 4 Experiment Finished Successfully!")


if __name__ == "__main__":
    main()
