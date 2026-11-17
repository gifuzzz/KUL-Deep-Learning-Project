#!/usr/bin/env python3
"""Phase 9 Controlled Branching Navigation Training Pipeline.

Trains recurrent CNN policies under the Phase 9 protocol:
- Controlled 50,000 environment interaction steps per variant.
- Multitask mixture:
    * 40% branching_exploration (progressing B1 -> B2 -> B3)
    * 20% return_navigation (route-complexity R1..R4)
    * 20% discovery (distant hidden victim search)
    * 20% full_rescue (spawn-relative ladder: 2-6, 7-12, 13-20, 21-35, distant)
- Preserves recurrent state handling, partial observability, edge geometry, Action 6 disabled.
- Evaluates and saves:
    * initial_checkpoint.zip
    * checkpoint_10000.zip, checkpoint_25000.zip
    * best_navigation_model.zip
    * best_rescue_model.zip
    * final_model.zip
    * checkpoint_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.recurrent_ppo import MaskableRecurrentPPO, get_git_commit
from rescue_env.envs.single_agent import RescueEnv
from rescue_env.wrappers import PreviousActionWrapper


class Phase9CurriculumEnv(RescueEnv):
    """Multitask training environment for Phase 9."""

    def __init__(
        self,
        task_mixture: tuple[float, float, float, float] = (0.40, 0.20, 0.20, 0.20),
        spatial_memory_mode: str = "visited_binary",
        **kwargs,
    ):
        super().__init__(
            task_type="branching_exploration",
            spatial_memory_mode=spatial_memory_mode,
            reward_shaping=True,
            distance_metric="spawn",
            **kwargs,
        )
        self.task_mixture = task_mixture
        self.total_env_steps = 0
        self.task_counts = {
            "branching_B1": 0,
            "branching_B2": 0,
            "branching_B3": 0,
            "return": 0,
            "discovery": 0,
            "full_rescue": 0,
        }

    def reset(self, *, seed=None, options=None):
        if options is None:
            options = {}
        else:
            options = dict(options)

        rng = getattr(self, "np_random", None)
        if rng is None:
            rng = np.random.default_rng(seed)

        roll = float(rng.random())
        p_branch = self.task_mixture[0]
        p_return = self.task_mixture[1]
        p_disc = self.task_mixture[2]
        # remaining is full_rescue

        if roll < p_branch:
            options["task_type"] = "branching_exploration"
            # Progress through B1 -> B2 -> B3 based on total_env_steps
            if self.total_env_steps < 15_000:
                stage = "B1"
            elif self.total_env_steps < 30_000:
                stage = "B2" if float(rng.random()) < 0.60 else "B1"
            else:
                r_stage = float(rng.random())
                if r_stage < 0.20:
                    stage = "B1"
                elif r_stage < 0.60:
                    stage = "B2"
                else:
                    stage = "B3"
            options["branching_stage"] = stage
            self.task_counts[f"branching_{stage}"] = self.task_counts.get(f"branching_{stage}", 0) + 1

        elif roll < p_branch + p_return:
            options["task_type"] = "return"
            options["return_mode"] = "route_complexity"
            r_level = float(rng.random())
            if r_level < 0.25:
                options["return_level"] = "R1"
                options["max_steps"] = 64
            elif r_level < 0.55:
                options["return_level"] = "R2"
                options["max_steps"] = 128
            elif r_level < 0.85:
                options["return_level"] = "R3"
                options["max_steps"] = 192
            else:
                options["return_level"] = "R4"
                options["max_steps"] = 288
            self.task_counts["return"] += 1

        elif roll < p_branch + p_return + p_disc:
            options["task_type"] = "discovery"
            r_dist = float(rng.random())
            if r_dist < 0.30:
                self.victim_distance = 12
                self.min_victim_distance = 7
                options["max_steps"] = 128
            elif r_dist < 0.70:
                self.victim_distance = 20
                self.min_victim_distance = 13
                options["max_steps"] = 192
            else:
                self.victim_distance = None
                self.min_victim_distance = 21
                options["max_steps"] = 288
            self.task_counts["discovery"] += 1

        else:
            options["task_type"] = "full"
            r_tier = float(rng.random())
            if r_tier < 0.25:
                self.victim_distance = 6
                self.min_victim_distance = 2
                options["max_steps"] = 64
            elif r_tier < 0.55:
                self.victim_distance = 12
                self.min_victim_distance = 7
                options["max_steps"] = 128
            elif r_tier < 0.80:
                self.victim_distance = 20
                self.min_victim_distance = 13
                options["max_steps"] = 192
            else:
                self.victim_distance = 35
                self.min_victim_distance = 21
                options["max_steps"] = 288
            self.task_counts["full_rescue"] += 1

        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.total_env_steps += 1
        return obs, reward, terminated, truncated, info


class Phase9EvaluationCallback(BaseCallback):
    """Periodic evaluation of branching navigation and rescue transfer."""

    def __init__(
        self,
        eval_freq: int = 5_000,
        n_eval_episodes: int = 20,
        save_dir: Path | None = None,
        spatial_memory_mode: str = "visited_binary",
        device: str = "cpu",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_dir = Path(save_dir) if save_dir else None
        self.spatial_memory_mode = spatial_memory_mode
        self.device = device

        self.best_nav_score = -1.0
        self.best_rescue_rate = -1.0
        self.history = []
        self.saved_milestones = set()
        self.initial_timesteps = None
        self.last_eval_step = 0

    def _on_step(self) -> bool:
        if self.initial_timesteps is None:
            self.initial_timesteps = self.model.num_timesteps
        relative_step = self.num_timesteps - self.initial_timesteps

        # Check and save periodic milestone checkpoints
        if self.save_dir:
            for milestone in (10_000, 25_000, 50_000):
                if relative_step >= milestone and milestone not in self.saved_milestones:
                    self.saved_milestones.add(milestone)
                    ckpt_path = self.save_dir / f"checkpoint_{milestone}.zip"
                    self.model.save(str(ckpt_path))
                    if self.verbose:
                        print(f"\n[Step {relative_step:,}] Saved milestone checkpoint to {ckpt_path}", flush=True)

        if self.eval_freq > 0 and (relative_step - self.last_eval_step >= self.eval_freq):
            self.last_eval_step = relative_step
        else:
            return True

        if self.verbose:
            print(f"\n[Step {relative_step:,}] Running Phase 9 Evaluation...", flush=True)

        # 1. Navigation Evaluation (B1)
        nav_env = PreviousActionWrapper(
            RescueEnv(
                task_type="branching_exploration",
                spatial_memory_mode=self.spatial_memory_mode,
                reward_shaping=False,
            )
        )
        correct_b1 = 0
        total_b1 = 0
        for ep in range(self.n_eval_episodes):
            obs, info = nav_env.reset(seed=55000 + ep, options={"branching_stage": "B1"})
            lstm_states = torch.zeros((1, 1, self.model.policy.hidden_size), device=self.device)
            ep_starts = np.array([True])
            done = False
            while not done:
                action, lstm_states = self.model.policy.predict(
                    obs,
                    state=lstm_states,
                    episode_start=ep_starts,
                    deterministic=True,
                    action_masks=nav_env.action_masks(),
                )
                obs, r, term, trunc, info = nav_env.step(int(action))
                done = term or trunc
                ep_starts = np.array([done])
            if info.get("correct_branch_selected") is True:
                correct_b1 += 1
            total_b1 += 1
        nav_score = (correct_b1 / total_b1) * 100.0 if total_b1 > 0 else 0.0

        # 2. Rescue Transfer Evaluation (Spawn 2-6 and 7-12)
        rescue_env = PreviousActionWrapper(
            RescueEnv(
                task_type="full",
                spatial_memory_mode=self.spatial_memory_mode,
                reward_shaping=False,
                distance_metric="spawn",
            )
        )
        rescued = 0
        total_rescue = 0
        for ep in range(self.n_eval_episodes):
            tier_dist = 6 if ep % 2 == 0 else 12
            rescue_env.victim_distance = tier_dist
            rescue_env.min_victim_distance = 2 if tier_dist == 6 else 7
            rescue_env.max_steps = 64 if tier_dist == 6 else 128
            obs, info = rescue_env.reset(seed=56000 + ep)
            lstm_states = torch.zeros((1, 1, self.model.policy.hidden_size), device=self.device)
            ep_starts = np.array([True])
            done = False
            while not done:
                action, lstm_states = self.model.policy.predict(
                    obs,
                    state=lstm_states,
                    episode_start=ep_starts,
                    deterministic=True,
                    action_masks=rescue_env.action_masks(),
                )
                obs, r, term, trunc, info = rescue_env.step(int(action))
                done = term or trunc
                ep_starts = np.array([done])
            if info.get("victims_rescued", 0) > 0:
                rescued += 1
            total_rescue += 1
        rescue_rate = (rescued / total_rescue) * 100.0 if total_rescue > 0 else 0.0

        if self.verbose:
            print(f"  B1 Unvisited Branch Selection: {nav_score:.1f}%")
            print(f"  Spawn 2-12 Rescue Rate:        {rescue_rate:.1f}%")

        self.history.append({
            "step": relative_step,
            "b1_nav_score_pct": nav_score,
            "rescue_rate_pct": rescue_rate,
        })

        if self.save_dir:
            # Save best navigation
            if nav_score > self.best_nav_score:
                self.best_nav_score = nav_score
                best_nav_path = self.save_dir / "best_navigation_model.zip"
                self.model.save(str(best_nav_path))
                if self.verbose:
                    print(f"  New best navigation model ({nav_score:.1f}%): saved to {best_nav_path}")

            # Save best rescue
            if rescue_rate >= self.best_rescue_rate:
                self.best_rescue_rate = rescue_rate
                best_rescue_path = self.save_dir / "best_rescue_model.zip"
                self.model.save(str(best_rescue_path))
                if self.verbose:
                    print(f"  New best rescue model ({rescue_rate:.1f}%): saved to {best_rescue_path}")

        return True


def train_variant(
    variant: str = "binary_visited",
    initial_model_path: str | Path | None = None,
    total_timesteps: int = 50_000,
    seed: int = 42,
    device: str = "auto",
    save_dir: str | Path | None = None,
) -> Path:
    """Train a single model variant for 50,000 steps."""
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    spatial_mode = "visited_binary" if variant in ("binary_visited", "warm_start") else "none"

    if initial_model_path is None:
        if variant == "control":
            initial_model_path = ROOT / "models/phase6_spawn_curriculum/production_250k/stage3_spawn_curriculum/best_model.zip"
        elif variant == "binary_visited":
            initial_model_path = ROOT / "models/phase8_binary_visited/initial_transferred.zip"
        else:  # warm_start
            initial_model_path = ROOT / "models/phase8_binary_visited/stage3_spawn_curriculum/best_model.zip"

    initial_model_path = Path(initial_model_path)
    if not initial_model_path.exists():
        raise FileNotFoundError(f"Initial model not found at {initial_model_path}")

    if save_dir is None:
        save_dir = ROOT / f"models/phase9_{variant}"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"LAUNCHING PHASE 9 TRAINING: VARIANT '{variant}'")
    print(f"Spatial Memory Mode: {spatial_mode}")
    print(f"Initial Checkpoint:  {initial_model_path}")
    print(f"Total Budget:        {total_timesteps:,} steps")
    print(f"Target Device:       {device}")
    print(f"Output Directory:    {save_dir}")
    print("=" * 70)

    # Create vectorized environment
    n_envs = 4
    def make_env(rank: int):
        def _thunk():
            base_env = Phase9CurriculumEnv(
                task_mixture=(0.40, 0.20, 0.20, 0.20),
                spatial_memory_mode=spatial_mode,
            )
            return PreviousActionWrapper(base_env)
        return _thunk

    vec_env = DummyVecEnv([make_env(i) for i in range(n_envs)])

    # Load model
    print("Loading pretrained checkpoint...")
    model = MaskableRecurrentPPO.load(
        str(initial_model_path),
        env=vec_env,
        device=device,
        learning_rate=3e-4,
        ent_coef=0.001,
        gamma=0.99,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
    )

    # Save copy of initial checkpoint
    initial_ckpt_path = save_dir / "initial_checkpoint.zip"
    model.save(str(initial_ckpt_path))
    print(f"Saved initial checkpoint copy to {initial_ckpt_path}")

    # Create evaluation callback
    eval_callback = Phase9EvaluationCallback(
        eval_freq=5_000,
        n_eval_episodes=20,
        save_dir=save_dir,
        spatial_memory_mode=spatial_mode,
        device=device,
        verbose=1,
    )

    start_time = time.time()
    print("\nStarting PPO training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        reset_num_timesteps=False,
    )
    elapsed = time.time() - start_time

    # Save final model
    final_path = save_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed/60:.1f}m). Saved final model to {final_path}")

    # Ensure checkpoint_50000 and best checkpoints exist
    ckpt_50k = save_dir / "checkpoint_50000.zip"
    if not ckpt_50k.exists():
        model.save(str(ckpt_50k))
    best_nav = save_dir / "best_navigation_model.zip"
    if not best_nav.exists():
        model.save(str(best_nav))
    best_res = save_dir / "best_rescue_model.zip"
    if not best_res.exists():
        model.save(str(best_res))

    # Aggregate task exposure from parallel environments
    realized_tasks = {}
    for env_i in vec_env.envs:
        base_e = env_i.env  # unwrapped PreviousActionWrapper
        for k, v in base_e.task_counts.items():
            realized_tasks[k] = realized_tasks.get(k, 0) + v

    total_episodes = sum(realized_tasks.values())
    task_proportions = {
        k: (v / total_episodes) * 100.0 if total_episodes > 0 else 0.0
        for k, v in realized_tasks.items()
    }

    manifest = {
        "variant": variant,
        "spatial_memory_mode": spatial_mode,
        "initial_model_path": str(initial_model_path),
        "total_timesteps": total_timesteps,
        "training_duration_seconds": elapsed,
        "device": device,
        "git_commit": get_git_commit(),
        "realized_task_counts": realized_tasks,
        "task_proportions_pct": task_proportions,
        "total_training_episodes": total_episodes,
        "evaluation_history": eval_callback.history,
        "best_nav_score": eval_callback.best_nav_score,
        "best_rescue_rate": eval_callback.best_rescue_rate,
        "checkpoints": {
            "initial": str(initial_ckpt_path),
            "checkpoint_10000": str(save_dir / "checkpoint_10000.zip"),
            "checkpoint_25000": str(save_dir / "checkpoint_25000.zip"),
            "checkpoint_50000": str(save_dir / "checkpoint_50000.zip"),
            "best_navigation": str(save_dir / "best_navigation_model.zip"),
            "best_rescue": str(save_dir / "best_rescue_model.zip"),
            "final": str(final_path),
        },
    }

    manifest_path = save_dir / "checkpoint_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved checkpoint manifest to {manifest_path}")

    vec_env.close()
    return final_path


def main():
    parser = argparse.ArgumentParser(description="Phase 9 Controlled Training Runner")
    parser.add_argument(
        "--variant",
        choices=["control", "binary_visited", "warm_start"],
        required=True,
        help="Model variant to train: control (9-ch), binary_visited (10-ch zero-init), or warm_start (10-ch phase 8)",
    )
    parser.add_argument("--initial-model", type=str, default=None, help="Path to initial model checkpoint")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Timesteps to train (default: 50,000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/mps/cuda/auto)")
    parser.add_argument("--save-dir", type=str, default=None, help="Save directory")
    args = parser.parse_args()

    train_variant(
        variant=args.variant,
        initial_model_path=args.initial_model,
        total_timesteps=args.timesteps,
        seed=args.seed,
        device=args.device,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
