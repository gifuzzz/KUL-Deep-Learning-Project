#!/usr/bin/env python3
"""Phase 11: Local View vs. Remembered Map Architecture Ablation Training Pipeline.

Trains recurrent CNN policies under the Phase 11 protocol:
- Base checkpoint: Phase 10 Variant B checkpoint (models/phase10_variant_b/final_model.zip).
- Phase 10 reward structure: oscillation_penalty=0.0, repeated_wait_penalty=False, Action 6 disabled for fireless.
- Fixed training task mixture: Phase 6/10 mixture:
    * 40% full rescue (spawn-relative distance ladder: 2-6, 7-12, 13-20, 21-35, distant)
    * 35% return navigation (route-complexity R1..R4)
    * 25% discovery training (distant victim search)
- Controlled 50,000 PPO environment interaction steps per variant.
- The 3 Architecture Variants:
    * Variant A (Current): Local CNN (9, 15, 15) + Memory CNN (9, 40, 52) -> Fusion (10132->128) -> GRU128 (1.43M params).
    * Variant B (Memory Only): Memory CNN (9, 40, 52) -> Fusion (8332->128) -> GRU128 (1.18M params).
    * Variant C (Memory Crop): Global Memory CNN (9, 40, 52) + Local Crop (9, 15, 15) derived from memory -> Fusion (10132->128) -> GRU128 (1.43M params).
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

from agents.cnn import (
    RecurrentSpatialRescueCNN,
    transfer_phase11_weights,
)
from agents.recurrent_ppo import MaskableRecurrentPPO, get_git_commit
from rescue_env.envs.single_agent import RescueEnv
from rescue_env.wrappers import PreviousActionWrapper


VARIANT_CONFIGS = {
    "A": {
        "name": "current_architecture",
        "arch_name": "baseline",
        "description": "Variant A (Current): Separate Local CNN (9, 15, 15) + Memory CNN (9, 40, 52) -> Fusion -> GRU128 (1.43M params)",
        "params": 1_431_168,
        "fusion_dim": 10132,
    },
    "B": {
        "name": "memory_only",
        "arch_name": "memory_only",
        "description": "Variant B (Memory Only): Memory CNN (9, 40, 52) -> Fusion -> GRU128 (1.18M params, local CNN omitted)",
        "params": 1_179_128,
        "fusion_dim": 8332,
    },
    "C": {
        "name": "memory_crop",
        "arch_name": "memory_crop",
        "description": "Variant C (Memory Crop): Global Memory CNN (9, 40, 52) + Memory Crop (9, 15, 15) -> Fusion -> GRU128 (1.43M params)",
        "params": 1_431_168,
        "fusion_dim": 10132,
    },
}


class Phase11CurriculumEnv(RescueEnv):
    """Multitask training environment for Phase 11 architecture ablation."""

    def __init__(
        self,
        task_mixture: tuple[float, float, float] = (0.40, 0.35, 0.25),
        **kwargs,
    ):
        super().__init__(
            task_type="full",
            spatial_memory_mode="none",
            reward_shaping=True,
            distance_metric="spawn",
            geometry_mode="edge",
            oscillation_penalty=0.0,
            repeated_wait_penalty_enabled=False,
            **kwargs,
        )
        self.task_mixture = task_mixture
        self._mixture_cum = np.cumsum(task_mixture)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        options = dict(options or {})

        if "task_type" not in options:
            rng = np.random.RandomState(seed) if seed is not None else np.random
            r = rng.rand()
            if r < self._mixture_cum[0]:
                chosen_task = "full"
            elif r < self._mixture_cum[1]:
                chosen_task = "return"
            else:
                chosen_task = "discovery"
            options["task_type"] = chosen_task

        task_type = options["task_type"]
        self.task_type = task_type

        if task_type == "full":
            if "victim_distance" not in options:
                rng = np.random.RandomState(seed) if seed is not None else np.random
                roll = rng.rand()
                if roll < 0.20:
                    options["victim_distance"] = 6
                    options["min_victim_distance"] = 2
                    options["max_steps"] = 64
                elif roll < 0.45:
                    options["victim_distance"] = 12
                    options["min_victim_distance"] = 7
                    options["max_steps"] = 128
                elif roll < 0.70:
                    options["victim_distance"] = 20
                    options["min_victim_distance"] = 13
                    options["max_steps"] = 192
                elif roll < 0.85:
                    options["victim_distance"] = 35
                    options["min_victim_distance"] = 21
                    options["max_steps"] = 288
                else:
                    options["victim_distance"] = None
                    options["min_victim_distance"] = 1
                    options["max_steps"] = 384
        elif task_type == "return":
            if "max_steps" not in options:
                options["max_steps"] = 256
        elif task_type == "discovery":
            if "max_steps" not in options:
                options["max_steps"] = 256

        return super().reset(seed=seed, options=options)


class Phase11EvaluationCallback(BaseCallback):
    """Periodic evaluation and milestone checkpointing callback."""

    def __init__(
        self,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 20,
        save_dir: Path | None = None,
        device: str = "cpu",
        milestones: tuple[int, ...] = (10_000, 25_000, 50_000),
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_dir = save_dir
        self.device = device
        self.milestones = sorted(milestones)
        self.saved_milestones: set[int] = set()
        self.history: list[dict[str, Any]] = []
        self.best_rescue_rate = -1.0
        self.initial_timesteps = 0

    def _on_training_start(self) -> None:
        self.initial_timesteps = self.model.num_timesteps

    def _on_step(self) -> bool:
        current_step = self.model.num_timesteps
        relative_step = current_step - self.initial_timesteps

        # Check for milestone checkpoints
        if self.save_dir:
            for ms in self.milestones:
                if ms not in self.saved_milestones and relative_step >= ms:
                    self.saved_milestones.add(ms)
                    ckpt_path = self.save_dir / f"checkpoint_{ms}.zip"
                    self.model.save(str(ckpt_path))
                    if self.verbose:
                        print(f"  Milestone checkpoint reached ({ms:,} steps): saved to {ckpt_path}")

        # Periodic evaluation
        if relative_step > 0 and relative_step % self.eval_freq == 0:
            self._run_evaluation(relative_step)

        return True

    def _run_evaluation(self, relative_step: int) -> None:
        if self.verbose:
            print(f"\n--- Phase 11 Periodic Validation (Step {relative_step:,}) ---")

        rescue_env = PreviousActionWrapper(
            RescueEnv(
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
            if rescue_rate >= self.best_rescue_rate:
                self.best_rescue_rate = rescue_rate
                best_rescue_path = self.save_dir / "best_rescue_model.zip"
                self.model.save(str(best_rescue_path))
                if self.verbose:
                    print(f"  New best rescue model ({rescue_rate:.1f}%): saved to {best_rescue_path}")


def train_variant(
    variant_key: str = "A",
    total_timesteps: int = 50_000,
    seed: int = 42,
    device: str = "auto",
    n_envs: int = 8,
    lr: float = 1e-4,
    initial_checkpoint: str | None = None,
) -> Path:
    """Trains a single Phase 11 architecture ablation variant."""
    v_key = variant_key.upper()
    if v_key not in VARIANT_CONFIGS:
        raise ValueError(f"Unknown variant key: '{variant_key}'. Valid: A, B, C")

    v_cfg = VARIANT_CONFIGS[v_key]
    arch_name = v_cfg["arch_name"]

    save_dir = ROOT / f"models/phase11_variant_{v_key.lower()}"
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = ROOT / f"runs/phase11_variant_{v_key.lower()}"
    tb_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("=" * 80)
    print(f"Starting Phase 11 Training: Variant {v_key} ({v_cfg['name']})")
    print(f"Architecture:       {arch_name}")
    print(f"Description:        {v_cfg['description']}")
    print(f"Trainable Params:   {v_cfg['params']:,}")
    print(f"Fusion In-Dim:      {v_cfg['fusion_dim']}")
    print(f"Device:             {device}")
    print(f"Timesteps:          {total_timesteps:,}")
    print(f"Envs:               {n_envs}")
    print(f"Save Directory:     {save_dir}")
    print("=" * 80)

    # Factory for parallel training environments
    def make_env(rank: int):
        def _init():
            env = Phase11CurriculumEnv()
            return PreviousActionWrapper(env)
        return _init

    vec_env = DummyVecEnv([make_env(i) for i in range(n_envs)])

    # Determine initial checkpoint
    if initial_checkpoint is None:
        initial_model_path = ROOT / "models/phase10_variant_b/final_model.zip"
    else:
        initial_model_path = Path(initial_checkpoint)

    if not initial_model_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {initial_model_path}")

    print(f"Source checkpoint: {initial_model_path}")

    # Instantiate model with the target architecture
    model = MaskableRecurrentPPO(
        policy=RecurrentSpatialRescueCNN,
        env=vec_env,
        architecture_variant=arch_name,
        learning_rate=lr,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        ent_coef=0.001,
        verbose=1,
        device=device,
        tensorboard_log=str(tb_dir),
        seed=seed,
    )

    # Transfer compatible pretrained weights
    transfer_res = transfer_phase11_weights(initial_model_path, model)
    print(f"Pretrained weight transfer result:")
    print(f"  Status:               {transfer_res['status']}")
    print(f"  Transferred keys:     {len(transfer_res['transferred_keys'])}")
    print(f"  Transferred params:   {transfer_res['transferred_params']:,} / {transfer_res['total_target_params']:,} ({transfer_res['transfer_coverage_pct']:.2f}%)")

    # Save initial checkpoint
    init_ckpt_path = save_dir / "initial_checkpoint.zip"
    model.save(str(init_ckpt_path))
    print(f"Saved initial checkpoint to {init_ckpt_path}")

    callback = Phase11EvaluationCallback(
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
        "arch_name": arch_name,
        "description": v_cfg["description"],
        "params": v_cfg["params"],
        "fusion_dim": v_cfg["fusion_dim"],
        "base_model": str(initial_model_path),
        "total_timesteps_trained": total_timesteps,
        "elapsed_seconds": round(elapsed, 1),
        "fps": round(fps, 1),
        "device": device,
        "seed": seed,
        "n_envs": n_envs,
        "learning_rate": lr,
        "git_commit": get_git_commit(),
        "weight_transfer": {
            "status": transfer_res["status"],
            "transferred_keys_count": len(transfer_res["transferred_keys"]),
            "transferred_params": transfer_res["transferred_params"],
            "total_target_params": transfer_res["total_target_params"],
            "coverage_pct": transfer_res["transfer_coverage_pct"],
        },
        "evaluation_history": callback.history,
        "best_rescue_rate_pct": callback.best_rescue_rate,
        "checkpoints": {
            "initial": "initial_checkpoint.zip",
            "milestone_10k": "checkpoint_10000.zip",
            "milestone_25k": "checkpoint_25000.zip",
            "milestone_50k": "checkpoint_50000.zip",
            "best_rescue": "best_rescue_model.zip",
            "final": "final_model.zip",
        },
    }

    manifest_path = save_dir / "checkpoint_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to {manifest_path}")

    vec_env.close()
    return final_model_path


def main():
    parser = argparse.ArgumentParser(description="Phase 11: Architecture Ablation Training")
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["A", "B", "C", "a", "b", "c", "all"],
        help="Variant to train: A (current), B (memory_only), C (memory_crop), or all (default: all)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Total environment interaction timesteps per variant (default: 50,000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device to train on (default: auto)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments (default: 8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning (default: 1e-4)",
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=str,
        default=None,
        help="Source checkpoint to transfer weights from (default: models/phase10_variant_b/final_model.zip)",
    )
    args = parser.parse_args()

    v_str = args.variant.upper()
    variants_to_run = ["A", "B", "C"] if v_str == "ALL" else [v_str]

    for v in variants_to_run:
        train_variant(
            variant_key=v,
            total_timesteps=args.timesteps,
            seed=args.seed,
            device=args.device,
            n_envs=args.n_envs,
            lr=args.lr,
            initial_checkpoint=args.initial_checkpoint,
        )


if __name__ == "__main__":
    main()
