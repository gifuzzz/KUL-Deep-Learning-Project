"""FlameBox Reinforcement Learning Training Pipeline.

Implements PPO training with MaskablePPO and RecurrentSpatialRescueCNN
under the Mixed Visibility Curriculum:
  - Stage 1 (Qualification Gate - 250k steps):
    70% near/visible victims (64 steps) + 30% medium/hidden victims (128 steps).
    Requires >= 90% deterministic and sampled unshaped success on near-rescue
    before advancing to Stage 2.
  - Stage 2 (Distant Search - 500k steps):
    30% visible victims + 70% hidden multi-room victims (256-512 steps).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.cnn import RecurrentSpatialRescueCNN
from agents.recurrent_ppo import MaskableRecurrentPPO
from rescue_env.envs.single_agent import RescueEnv
from rescue_env.wrappers import PreviousActionWrapper


def get_git_commit() -> str:
    """Retrieve git commit hash or return unknown."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return commit
    except Exception:
        return "unknown"


def save_checkpoint(
    model: MaskableRecurrentPPO | MaskablePPO,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save model checkpoint safely casting policy and optimizer state dicts to CPU.

    Supports CPU/MPS/CUDA cross-device transfer and saves metadata (stage, timesteps,
    git commit, hyperparameters) into the checkpoint zip and companion JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    orig_device = model.device

    meta_dict = {
        "git_commit": get_git_commit(),
        "total_timesteps": getattr(model, "num_timesteps", 0),
        "policy_class": model.policy.__class__.__name__,
        "algorithm": model.__class__.__name__,
    }
    if metadata:
        meta_dict.update(metadata)

    try:
        model.policy.to("cpu")
        if hasattr(model.policy, "optimizer") and model.policy.optimizer is not None:
            for state in model.policy.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to("cpu")
        model.save(str(path))

        # Append metadata.json inside the zip
        with zipfile.ZipFile(str(path), mode="a") as zf:
            zf.writestr("metadata.json", json.dumps(meta_dict, indent=2))

        # Also write companion .meta.json file alongside
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)

    finally:
        model.policy.to(orig_device)
        if hasattr(model.policy, "optimizer") and model.policy.optimizer is not None:
            for state in model.policy.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(orig_device)


def load_checkpoint(
    path: str | Path,
    env=None,
    device: str | torch.device = "auto",
) -> MaskableRecurrentPPO | MaskablePPO:
    """Load model checkpoint safely onto specified device (handling CUDA/MPS/CPU compatibility)."""
    if device == "auto":
        if torch.cuda.is_available():
            target_device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            target_device = torch.device("mps")
        else:
            target_device = torch.device("cpu")
    else:
        target_device = torch.device(device)

    custom_objects = {"policy_class": RecurrentSpatialRescueCNN}
    model = None
    for cls in (MaskableRecurrentPPO, MaskablePPO):
        try:
            model = cls.load(
                str(path),
                env=env,
                device=target_device,
                custom_objects=custom_objects,
            )
            break
        except Exception:
            continue

    if model is None:
        try:
            model = MaskableRecurrentPPO.load(str(path), env=env, device=target_device)
        except Exception:
            model = MaskablePPO.load(str(path), env=env, device=target_device)

    if hasattr(model, "policy") and model.policy is not None:
        model.policy.to(target_device)
        if hasattr(model.policy, "optimizer") and model.policy.optimizer is not None:
            for state in model.policy.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(target_device)

    try:
        with zipfile.ZipFile(str(path), mode="r") as zf:
            if "metadata.json" in zf.namelist():
                model.checkpoint_metadata = json.loads(zf.read("metadata.json"))
    except Exception:
        pass

    return model


class CurriculumRescueEnv(RescueEnv):
    """Mixed Visibility Curriculum Environment.

    Stage 1 (Qualification Gate):
      - 70% near/visible victims (max_steps=64, victim_distance=2, min_victim_distance=1)
      - 30% medium/hidden victims (max_steps=128, victim_distance=6, min_victim_distance=3)

    Stage 2 (Distant Search):
      - 30% visible victims (max_steps=128, victim_distance=2, min_victim_distance=1)
      - 70% hidden multi-room victims (max_steps=384, victim_distance=None, min_victim_distance=1)
    """

    def __init__(self, stage: int = 1, near_ratio: float = 0.70, **kwargs):
        super().__init__(**kwargs)
        self.stage = int(stage)
        self.near_ratio = float(near_ratio)

    def set_stage(self, stage: int, near_ratio: float | None = None) -> None:
        self.stage = int(stage)
        if near_ratio is not None:
            self.near_ratio = float(near_ratio)

    def reset(self, *, seed=None, options=None):
        if options and "stage" in options:
            self.stage = int(options["stage"])
        if options and "near_ratio" in options:
            self.near_ratio = float(options["near_ratio"])

        rng = getattr(self, "np_random", None)
        roll = float(rng.random()) if rng is not None else float(np.random.random())

        if self.stage == 1:
            if roll < self.near_ratio:
                self.victim_distance = 2
                self.min_victim_distance = 1
                self.max_steps = 64
            else:
                self.victim_distance = 6
                self.min_victim_distance = 3
                self.max_steps = 128
        else:
            if roll < 0.30:
                self.victim_distance = 2
                self.min_victim_distance = 1
                self.max_steps = 128
            else:
                self.victim_distance = None
                self.min_victim_distance = 1
                self.max_steps = 384

        return super().reset(seed=seed, options=options)


def set_env_stage(env: Any, stage: int, near_ratio: float | None = None) -> None:
    """Set curriculum stage and optional near victim ratio across single or vectorized environments."""
    if hasattr(env, "env_method"):
        env.env_method("set_stage", stage, near_ratio)
    elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "set_stage"):
        env.unwrapped.set_stage(stage, near_ratio)


def make_single_train_env(
    stage: int = 1,
    reward_shaping: bool = True,
    navigation_shaping: bool = False,
    near_ratio: float = 0.70,
    seed: int | None = None,
):
    """Factory creating an individual monitored environment."""
    def _init():
        env = CurriculumRescueEnv(
            stage=stage,
            reward_shaping=reward_shaping,
            navigation_shaping=navigation_shaping,
            near_ratio=near_ratio,
        )
        env = PreviousActionWrapper(env)
        if seed is not None:
            env.reset(seed=seed)
        return Monitor(env, info_keywords=("success", "timeout"))

    return _init


def make_train_env(
    stage: int = 1,
    reward_shaping: bool = True,
    navigation_shaping: bool = False,
    near_ratio: float = 0.70,
    n_envs: int = 8,
    base_seed: int = 42,
) -> DummyVecEnv | Monitor:
    """Create vectorized or single monitored training environment."""
    if n_envs <= 1:
        env = CurriculumRescueEnv(
            stage=stage,
            reward_shaping=reward_shaping,
            navigation_shaping=navigation_shaping,
            near_ratio=near_ratio,
        )
        env = PreviousActionWrapper(env)
        return Monitor(env, info_keywords=("success", "timeout"))

    env_fns = [
        make_single_train_env(
            stage=stage,
            reward_shaping=reward_shaping,
            navigation_shaping=navigation_shaping,
            near_ratio=near_ratio,
            seed=base_seed + i * 1000,
        )
        for i in range(n_envs)
    ]
    return DummyVecEnv(env_fns)


def make_eval_env(
    victim_distance: int | None = 2,
    max_steps: int = 64,
    min_victim_distance: int = 1,
) -> PreviousActionWrapper:
    """Unshaped evaluation environment (+10 native rescue only)."""
    env = RescueEnv(
        max_steps=max_steps,
        victim_distance=victim_distance,
        min_victim_distance=min_victim_distance,
        reward_shaping=False,
        navigation_shaping=False,
    )
    return PreviousActionWrapper(env)


def evaluate_unshaped(
    model: MaskableRecurrentPPO | MaskablePPO,
    eval_env: PreviousActionWrapper | RescueEnv,
    n_episodes: int = 20,
    deterministic: bool = True,
    seed_start: int = 20000,
) -> dict[str, float]:
    """Evaluate model with unshaped rewards using proper recurrent state handling."""
    rewards = []
    successes = []
    lengths = []

    for ep in range(n_episodes):
        ep_seed = seed_start + ep
        obs, _ = eval_env.reset(seed=ep_seed)
        if not deterministic:
            torch.manual_seed(ep_seed)

        done = False
        ep_reward = 0.0
        steps = 0
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
            ep_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(ep_reward)
        lengths.append(steps)
        successes.append(bool(info.get("success", False)))

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
        "mean_length": float(np.mean(lengths)),
    }


class RecurrentMaskableEvalCallback(BaseCallback):
    """Periodic evaluation callback supporting recurrent hidden state and action masks."""

    def __init__(
        self,
        eval_env: RescueEnv,
        best_model_save_path: str | Path | None = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 20,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.best_model_save_path = Path(best_model_save_path) if best_model_save_path else None
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.best_mean_reward = -float("inf")
        self.best_success_rate = -1.0
        self.last_eval_step = 0

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and (self.num_timesteps - self.last_eval_step >= self.eval_freq):
            self.last_eval_step = self.num_timesteps
            metrics = evaluate_unshaped(
                self.model,
                self.eval_env,
                n_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                seed_start=30000 + self.num_timesteps,
            )
            if self.verbose > 0:
                print(
                    f"\n[Eval @ {self.num_timesteps} steps] "
                    f"Mean Reward: {metrics['mean_reward']:.3f} | "
                    f"Success Rate: {metrics['success_rate'] * 100:.1f}% | "
                    f"Avg Steps: {metrics['mean_length']:.1f}"
                )
            if self.best_model_save_path and metrics["success_rate"] >= self.best_success_rate:
                self.best_mean_reward = metrics["mean_reward"]
                self.best_success_rate = metrics["success_rate"]
                save_checkpoint(self.model, self.best_model_save_path / "best_model.zip")
                if self.verbose > 0:
                    print(f"New best model saved to {self.best_model_save_path / 'best_model.zip'}")
        return True


def check_stage1_qualification(
    model: MaskablePPO,
    n_episodes: int = 50,
    seed_start: int = 40000,
) -> bool:
    """Verify Stage 1 Qualification Gate: >= 90% deterministic AND sampled unshaped success on near-rescue."""
    eval_env = make_eval_env(victim_distance=2, max_steps=64)
    try:
        det_metrics = evaluate_unshaped(
            model, eval_env, n_episodes=n_episodes, deterministic=True, seed_start=seed_start
        )
        sampled_metrics = evaluate_unshaped(
            model, eval_env, n_episodes=n_episodes, deterministic=False, seed_start=seed_start + 1000
        )

        det_rate = det_metrics["success_rate"]
        sampled_rate = sampled_metrics["success_rate"]
        passed = (det_rate >= 0.90) and (sampled_rate >= 0.90)

        print("\n" + "=" * 60)
        print("STAGE 1 QUALIFICATION GATE VERIFICATION (Near-Rescue, Unshaped)")
        print("=" * 60)
        print(f"Evaluation episodes        : {n_episodes}")
        print(f"Deterministic Success Rate : {det_rate * 100:5.1f}% (Required: >= 90.0%)")
        print(f"Sampled Stochastic Success : {sampled_rate * 100:5.1f}% (Required: >= 90.0%)")
        print(f"Qualification Gate Result  : {'PASSED' if passed else 'FAILED'}")
        print("=" * 60 + "\n")
        return passed
    finally:
        eval_env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="FlameBox PPO Curriculum Training")
    parser.add_argument(
        "--stage",
        choices=["1", "2", "all"],
        default="all",
        help="Curriculum stage to run: 1, 2, or all (default: all)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Total timesteps to train (overrides stage defaults if provided)",
    )
    parser.add_argument(
        "--stage1-timesteps",
        type=int,
        default=250_000,
        help="Timesteps for Stage 1 Qualification (default: 250,000)",
    )
    parser.add_argument(
        "--stage2-timesteps",
        type=int,
        default=500_000,
        help="Timesteps for Stage 2 Distant Search (default: 500,000)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Device to train on (default: auto)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--n-steps", type=int, default=128, help="Rollout steps per update")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=4, help="Epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--ent-coef", type=float, default=0.001, help="Entropy coefficient (default: 0.001)")
    parser.add_argument("--eval-freq", type=int, default=5_000, help="Evaluation frequency in steps (default: 5,000)")
    parser.add_argument("--n-eval-episodes", type=int, default=20, help="Episodes per evaluation (default: 20)")
    parser.add_argument("--save-dir", type=str, default="./models/curriculum", help="Directory to save models")
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="Direct path to save the final model zip (e.g. ./models/autonomous_rl_v1/smoke_model.zip)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel vectorized environments (default: 8)",
    )
    parser.add_argument(
        "--near-ratio",
        type=float,
        default=1.0,
        help="Stage 1 near victim ratio (default: 1.0)",
    )
    parser.add_argument(
        "--navigation-shaping",
        action="store_true",
        default=True,
        help="Enable navigation potential shaping during training (default: True)",
    )
    parser.add_argument(
        "--no-navigation-shaping",
        action="store_false",
        dest="navigation_shaping",
        help="Disable navigation shaping during training",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Initial checkpoint path to load",
    )
    parser.add_argument(
        "--skip-qualification-gate",
        action="store_true",
        help="Skip strict >=90%% qualification gate failure check (useful for test runs)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

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
    print(f"Device: {device}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    hyperparameters = {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "ent_coef": args.ent_coef,
        "n_envs": args.n_envs,
        "near_ratio": args.near_ratio,
        "seed": args.seed,
        "device": device,
    }

    # Determine timesteps for stages
    if args.stage == "1":
        stage1_steps = args.total_timesteps if args.total_timesteps is not None else args.stage1_timesteps
        stage2_steps = 0
    elif args.stage == "2":
        stage1_steps = 0
        stage2_steps = args.total_timesteps if args.total_timesteps is not None else args.stage2_timesteps
    else:  # "all"
        if args.total_timesteps is not None:
            total = args.total_timesteps
            s1_default = args.stage1_timesteps
            s2_default = args.stage2_timesteps
            stage1_steps = int(total * (s1_default / (s1_default + s2_default)))
            stage2_steps = total - stage1_steps
        else:
            stage1_steps = args.stage1_timesteps
            stage2_steps = args.stage2_timesteps

    initial_stage = 2 if args.stage == "2" else 1
    train_env = make_train_env(
        stage=initial_stage,
        reward_shaping=True,
        navigation_shaping=args.navigation_shaping,
        near_ratio=args.near_ratio,
        n_envs=args.n_envs,
        base_seed=args.seed,
    )
    eval_env = make_eval_env(victim_distance=2, max_steps=64)

    if args.checkpoint:
        print(f"Loading checkpoint from: {args.checkpoint}")
        model = load_checkpoint(args.checkpoint, env=train_env, device=device)
    else:
        model = MaskableRecurrentPPO(
            policy=RecurrentSpatialRescueCNN,
            env=train_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            verbose=1,
            device=device,
            tensorboard_log="./logs/",
            seed=args.seed,
        )

    try:
        # ---------------- STAGE 1 ----------------
        if stage1_steps > 0:
            print(f"\n>>> Starting Stage 1: Qualification Gate ({stage1_steps} steps, {args.n_envs} envs)")
            stage1_dir = save_dir / "stage1"
            stage1_dir.mkdir(parents=True, exist_ok=True)

            callback1 = RecurrentMaskableEvalCallback(
                eval_env=eval_env,
                best_model_save_path=stage1_dir,
                eval_freq=args.eval_freq,
                n_eval_episodes=args.n_eval_episodes,
                deterministic=True,
            )

            print(f"--- Stage 1: Near-Rescue Mastery ({stage1_steps} steps, {int(args.near_ratio*100)}% near victims) ---")
            set_env_stage(train_env, stage=1, near_ratio=args.near_ratio)
            model.learn(
                total_timesteps=stage1_steps,
                callback=callback1,
                tb_log_name="PPO_Curriculum_Stage1",
                reset_num_timesteps=False,
            )

            stage1_final = stage1_dir / "stage1_final.zip"
            save_checkpoint(
                model,
                stage1_final,
                metadata={"stage": 1, "hyperparameters": hyperparameters},
            )
            print(f"Stage 1 completed. Model saved to {stage1_final}")

            # Check Qualification Gate
            if not args.skip_qualification_gate:
                passed = check_stage1_qualification(model, n_episodes=args.n_eval_episodes)
                best_model_path = stage1_dir / "best_model.zip"
                if not passed and best_model_path.exists():
                    print(f"Checking if best checkpoint ({best_model_path}) passes qualification...")
                    best_model = load_checkpoint(best_model_path, env=eval_env, device=device)
                    best_passed = check_stage1_qualification(best_model, n_episodes=args.n_eval_episodes)
                    if best_passed:
                        print(f"Best checkpoint qualified! Using {best_model_path} as Stage 1 final model.")
                        model = best_model
                        passed = True

                if not passed and args.stage == "all":
                    raise RuntimeError(
                        "Stage 1 Qualification Gate FAILED: Required >= 90% deterministic and "
                        "sampled unshaped success on near-rescue before advancing to Stage 2."
                    )
            else:
                print("Skipping Stage 1 Qualification Gate check (--skip-qualification-gate).")

        # ---------------- STAGE 2 ----------------
        if stage2_steps > 0:
            print(f"\n>>> Starting Stage 2: Distant Search ({stage2_steps} steps, {args.n_envs} envs)")
            set_env_stage(train_env, stage=2)
            stage2_dir = save_dir / "stage2"
            stage2_dir.mkdir(parents=True, exist_ok=True)

            eval_env_stage2 = make_eval_env(victim_distance=None, max_steps=384)
            callback2 = RecurrentMaskableEvalCallback(
                eval_env=eval_env_stage2,
                best_model_save_path=stage2_dir,
                eval_freq=args.eval_freq,
                n_eval_episodes=args.n_eval_episodes,
                deterministic=True,
            )

            model.learn(
                total_timesteps=stage2_steps,
                callback=callback2,
                tb_log_name="PPO_Curriculum_Stage2",
                reset_num_timesteps=False,
            )
            eval_env_stage2.close()

            stage2_final = stage2_dir / "stage2_final.zip"
            save_checkpoint(
                model,
                stage2_final,
                metadata={"stage": 2, "hyperparameters": hyperparameters},
            )
            print(f"Stage 2 completed. Model saved to {stage2_final}")

        final_model_path = save_dir / "final_model.zip"
        save_checkpoint(
            model,
            final_model_path,
            metadata={
                "stage": 2 if stage2_steps > 0 else 1,
                "hyperparameters": hyperparameters,
            },
        )
        print(f"Training complete. Final checkpoint saved to {final_model_path}")

        if args.save_model:
            smoke_path = Path(args.save_model)
            smoke_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                model,
                smoke_path,
                metadata={
                    "stage": 2 if stage2_steps > 0 else 1,
                    "hyperparameters": hyperparameters,
                },
            )
            print(f"Model saved to {smoke_path}")

    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
