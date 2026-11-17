#!/usr/bin/env python3
"""Phase 10: Controlled Reward Incentive Ablation Training Pipeline.

Trains recurrent CNN policies under the Phase 10 protocol:
- Starting model: strongest validated Phase 6 production checkpoint.
- Architecture: 9-channel baseline CNN + 128-unit GRU (architecture_variant="baseline").
- Fixed training task mixture: Phase 6 mixture:
    * 40% full rescue (spawn-relative distance curriculum)
    * 35% return navigation (route-complexity R1..R4)
    * 25% discovery training (distant victim search)
- Controlled 50,000 PPO environment interaction steps per variant.
- The 4 Ablation Variants:
    * Variant A (Control): Currently implemented reward function unchanged (oscillation=-0.02, wait_penalty=None).
    * Variant B (No Oscillation Penalty): oscillation_penalty=0.0, wait_penalty=None.
    * Variant C (Repeated-Wait Penalty): oscillation_penalty=0.02, repeated_wait_penalty_schedule=(0.0, 0.002, 0.004, 0.008).
    * Variant D (Combined Correction): oscillation_penalty=0.0, repeated_wait_penalty_schedule=(0.0, 0.002, 0.004, 0.008).
- Saves:
    * initial_checkpoint.zip
    * checkpoint_10000.zip, checkpoint_25000.zip, checkpoint_50000.zip
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


VARIANT_CONFIGS = {
    "A": {
        "name": "control",
        "description": "Baseline Phase 6 rewards unchanged (oscillation=-0.02, repeated_wait=False)",
        "oscillation_penalty": 0.02,
        "repeated_wait_penalty_enabled": False,
        "wait_penalty_schedule": (0.0, 0.002, 0.004, 0.008),
    },
    "B": {
        "name": "no_oscillation_penalty",
        "description": "Remove oscillation penalty (oscillation=0.0, repeated_wait=False)",
        "oscillation_penalty": 0.0,
        "repeated_wait_penalty_enabled": False,
        "wait_penalty_schedule": (0.0, 0.002, 0.004, 0.008),
    },
    "C": {
        "name": "repeated_wait_penalty",
        "description": "Retain oscillation penalty + add repeated-Wait penalty schedule (oscillation=-0.02, repeated_wait=True)",
        "oscillation_penalty": 0.02,
        "repeated_wait_penalty_enabled": True,
        "wait_penalty_schedule": (0.0, 0.002, 0.004, 0.008),
    },
    "D": {
        "name": "combined_correction",
        "description": "Remove oscillation penalty + add repeated-Wait penalty schedule (oscillation=0.0, repeated_wait=True)",
        "oscillation_penalty": 0.0,
        "repeated_wait_penalty_enabled": True,
        "wait_penalty_schedule": (0.0, 0.002, 0.004, 0.008),
    },
}


class Phase10CurriculumEnv(RescueEnv):
    """Multitask training environment for Phase 10 reward ablation."""

    def __init__(
        self,
        variant_key: str = "A",
        task_mixture: tuple[float, float, float] = (0.40, 0.35, 0.25),
        **kwargs,
    ):
        v_cfg = VARIANT_CONFIGS[variant_key.upper()]
        super().__init__(
            task_type="full",
            spatial_memory_mode="none",
            reward_shaping=True,
            distance_metric="spawn",
            geometry_mode="edge",
            oscillation_penalty=v_cfg["oscillation_penalty"],
            repeated_wait_penalty_enabled=v_cfg["repeated_wait_penalty_enabled"],
            wait_penalty_schedule=v_cfg["wait_penalty_schedule"],
            **kwargs,
        )
        self.variant_key = variant_key.upper()
        self.task_mixture = task_mixture
        self.total_env_steps = 0
        self.task_counts = {
            "full_rescue": 0,
            "return": 0,
            "discovery": 0,
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
        p_full = self.task_mixture[0]
        p_return = self.task_mixture[1]
        # remaining is discovery

        if roll < p_full:
            options["task_type"] = "full"
            r_dist = float(rng.random())
            if r_dist < 0.20:
                options["victim_distance"] = 6
                options["min_victim_distance"] = 2
                options["max_steps"] = 64
            elif r_dist < 0.55:
                options["victim_distance"] = 12
                options["min_victim_distance"] = 7
                options["max_steps"] = 128
            elif r_dist < 0.80:
                options["victim_distance"] = 20
                options["min_victim_distance"] = 13
                options["max_steps"] = 192
            elif r_dist < 0.90:
                options["victim_distance"] = 35
                options["min_victim_distance"] = 21
                options["max_steps"] = 256
            else:
                options["victim_distance"] = None
                options["min_victim_distance"] = 1
                options["max_steps"] = 384
            self.task_counts["full_rescue"] += 1

        elif roll < p_full + p_return:
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
                options["max_steps"] = 256
            self.task_counts["return"] += 1

        else:
            options["task_type"] = "discovery"
            options["discovery_reward"] = 2.0
            r_tier = float(rng.random())
            if r_tier < 0.35:
                options["victim_distance"] = 12
                options["min_victim_distance"] = 7
                options["max_steps"] = 128
            elif r_tier < 0.70:
                options["victim_distance"] = 20
                options["min_victim_distance"] = 13
                options["max_steps"] = 192
            else:
                options["victim_distance"] = 35
                options["min_victim_distance"] = 21
                options["max_steps"] = 256
            self.task_counts["discovery"] += 1

        return super().reset(seed=seed, options=options)

    def step(self, action):
        self.total_env_steps += 1
        return super().step(action)


class Phase10EvaluationCallback(BaseCallback):
    """Evaluates unshaped held-out rescue performance and saves milestone checkpoints."""

    def __init__(
        self,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 20,
        save_dir: Path | None = None,
        device: str = "cpu",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_dir = save_dir
        self.device = device
        self.best_rescue_rate = -1.0
        self.last_eval_step = 0
        self.history: list[dict[str, Any]] = []
        self.milestones_saved: set[int] = set()
        self.initial_timesteps = 0

    def _on_training_start(self) -> None:
        self.initial_timesteps = self.model.num_timesteps
        if self.verbose:
            print(f"[Callback] Training started. Initial model timesteps: {self.initial_timesteps}")

    def _on_step(self) -> bool:
        relative_step = self.num_timesteps - self.initial_timesteps

        # Checkpoint milestones: 10k, 25k, 50k
        if self.save_dir:
            for milestone in (10_000, 25_000, 50_000):
                if relative_step >= milestone and milestone not in self.milestones_saved:
                    self.milestones_saved.add(milestone)
                    ckpt_path = self.save_dir / f"checkpoint_{milestone}.zip"
                    self.model.save(str(ckpt_path))
                    if self.verbose:
                        print(f"\n[Milestone {milestone} steps] Saved to {ckpt_path}")

        # Regular periodic evaluation
        if relative_step - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = relative_step
            self._evaluate(relative_step)

        return True

    def _evaluate(self, relative_step: int) -> bool:
        if self.verbose:
            print(f"\n--- Phase 10 Validation at Step {relative_step} ---")

        rescue_env = PreviousActionWrapper(
            RescueEnv(
                task_type="full",
                spatial_memory_mode="none",
                reward_shaping=False,
                distance_metric="spawn",
                geometry_mode="edge",
            )
        )
        rescued = 0
        total_rescue = 0
        for ep in range(self.n_eval_episodes):
            tier_dist = 6 if ep % 2 == 0 else 12
            obs, info = rescue_env.reset(
                seed=56000 + ep,
                options={
                    "victim_distance": tier_dist,
                    "min_victim_distance": 2 if tier_dist == 6 else 7,
                    "max_steps": 64 if tier_dist == 6 else 128,
                }
            )
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
        rescue_env.close()

        rescue_rate = (rescued / total_rescue) * 100.0 if total_rescue > 0 else 0.0

        if self.verbose:
            print(f"  Spawn 2-12 Validation Rescue Rate: {rescue_rate:.1f}% ({rescued}/{total_rescue})")

        self.history.append({
            "step": relative_step,
            "rescue_rate_pct": rescue_rate,
        })

        if self.save_dir:
            # Save best rescue
            if rescue_rate >= self.best_rescue_rate:
                self.best_rescue_rate = rescue_rate
                best_rescue_path = self.save_dir / "best_rescue_model.zip"
                self.model.save(str(best_rescue_path))
                if self.verbose:
                    print(f"  New best rescue model ({rescue_rate:.1f}%): saved to {best_rescue_path}")

        return True


def train_variant(
    variant_key: str = "A",
    total_timesteps: int = 50_000,
    seed: int = 42,
    device: str = "cpu",
    n_envs: int = 8,
    lr: float = 1e-4,
    initial_checkpoint: str | None = None,
) -> Path:
    """Trains a single Phase 10 reward ablation variant for total_timesteps steps."""
    v_key = variant_key.upper()
    if v_key not in VARIANT_CONFIGS:
        raise ValueError(f"Unknown variant key: '{variant_key}'. Valid: A, B, C, D")

    v_cfg = VARIANT_CONFIGS[v_key]
    save_dir = ROOT / f"models/phase10_variant_{v_key.lower()}"
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = ROOT / f"runs/phase10_variant_{v_key.lower()}"
    tb_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Starting Phase 10 Training: Variant {v_key} ({v_cfg['name']})")
    print(f"Description:        {v_cfg['description']}")
    print(f"Oscillation Penalty: {v_cfg['oscillation_penalty']}")
    print(f"Repeated Wait Pen:  {v_cfg['repeated_wait_penalty_enabled']}")
    print(f"Schedule:           {v_cfg['wait_penalty_schedule']}")
    print(f"Device:             {device}")
    print(f"Timesteps:          {total_timesteps:,}")
    print(f"Envs:               {n_envs}")
    print(f"Save Directory:     {save_dir}")
    print("=" * 80)

    # Factory for parallel training environments
    def make_env(rank: int):
        def _init():
            env = Phase10CurriculumEnv(
                variant_key=v_key,
            )
            return PreviousActionWrapper(env)
        return _init

    vec_env = DummyVecEnv([make_env(i) for i in range(n_envs)])

    # Determine initial checkpoint
    if initial_checkpoint is None:
        initial_model_path = ROOT / "models/phase6_spawn_curriculum/production_250k/stage3_spawn_curriculum/best_model.zip"
    else:
        initial_model_path = Path(initial_checkpoint)

    if not initial_model_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {initial_model_path}")

    print(f"Loading initial model from: {initial_model_path}")
    model = MaskableRecurrentPPO.load(
        str(initial_model_path),
        env=vec_env,
        device=device,
        learning_rate=lr,
        tensorboard_log=str(tb_dir),
    )

    # Save initial checkpoint
    init_ckpt_path = save_dir / "initial_checkpoint.zip"
    model.save(str(init_ckpt_path))
    print(f"Saved initial checkpoint to {init_ckpt_path}")

    callback = Phase10EvaluationCallback(
        eval_freq=10_000,
        n_eval_episodes=20,
        save_dir=save_dir,
        device=device,
        verbose=1,
    )

    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        reset_num_timesteps=False,
    )
    elapsed = time.time() - start_time
    fps = total_timesteps / elapsed

    final_model_path = save_dir / "final_model.zip"
    model.save(str(final_model_path))
    print(f"\nTraining completed in {elapsed:.1f}s ({fps:.1f} fps). Final model saved to {final_model_path}")

    # Build and write manifest
    manifest = {
        "variant": f"Variant {v_key}",
        "name": v_cfg["name"],
        "description": v_cfg["description"],
        "oscillation_penalty": v_cfg["oscillation_penalty"],
        "repeated_wait_penalty_enabled": v_cfg["repeated_wait_penalty_enabled"],
        "wait_penalty_schedule": list(v_cfg["wait_penalty_schedule"]),
        "base_model": str(initial_model_path),
        "total_timesteps_trained": total_timesteps,
        "elapsed_seconds": round(elapsed, 1),
        "fps": round(fps, 1),
        "device": device,
        "seed": seed,
        "n_envs": n_envs,
        "git_commit": get_git_commit(),
        "evaluation_history": callback.history,
        "best_rescue_rate_pct": callback.best_rescue_rate,
        "checkpoints": {
            "initial": str(init_ckpt_path.name),
            "milestone_10k": "checkpoint_10000.zip" if 10000 in callback.milestones_saved else None,
            "milestone_25k": "checkpoint_25000.zip" if 25000 in callback.milestones_saved else None,
            "milestone_50k": "checkpoint_50000.zip" if 50000 in callback.milestones_saved else None,
            "best_rescue": "best_rescue_model.zip",
            "final": str(final_model_path.name),
        },
    }

    manifest_path = save_dir / "checkpoint_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved checkpoint manifest to {manifest_path}")

    vec_env.close()
    return save_dir


def main():
    parser = argparse.ArgumentParser(description="Phase 10 Controlled Reward Ablation Training")
    parser.add_argument("--variant", choices=["A", "B", "C", "D", "all"], default="A", help="Variant to train")
    parser.add_argument("--total-timesteps", type=int, default=50_000, help="Timesteps per variant")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    variants = ["A", "B", "C", "D"] if args.variant == "all" else [args.variant]
    for v in variants:
        train_variant(
            variant_key=v,
            total_timesteps=args.total_timesteps,
            seed=args.seed,
            device=args.device,
            n_envs=args.n_envs,
            lr=args.lr,
        )


if __name__ == "__main__":
    main()
