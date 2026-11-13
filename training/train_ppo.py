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
import sys
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from agents.cnn import RecurrentSpatialRescueCNN
from rescue_env.envs.single_agent import RescueEnv


def save_checkpoint(model: MaskablePPO, path: str | Path) -> None:
    """Save model checkpoint safely casting policy and optimizer state dicts to CPU.

    This prevents device serialization mismatch (e.g. AssertionError: mps:0 != cpu)
    when transferring between MPS and CPU runtimes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    orig_device = model.device
    try:
        model.policy.to("cpu")
        if hasattr(model.policy, "optimizer") and model.policy.optimizer is not None:
            for state in model.policy.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to("cpu")
        model.save(str(path))
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
) -> MaskablePPO:
    """Load model checkpoint safely onto specified device (handling MPS/CPU compatibility)."""
    if device == "auto":
        target_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    custom_objects = {"policy_class": RecurrentSpatialRescueCNN}
    try:
        model = MaskablePPO.load(
            str(path),
            env=env,
            device=target_device,
            custom_objects=custom_objects,
        )
    except Exception:
        model = MaskablePPO.load(
            str(path),
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

    def __init__(self, stage: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.stage = int(stage)

    def set_stage(self, stage: int) -> None:
        self.stage = int(stage)

    def reset(self, *, seed=None, options=None):
        if options and "stage" in options:
            self.stage = int(options["stage"])

        rng = getattr(self, "np_random", None)
        roll = float(rng.random()) if rng is not None else float(np.random.random())

        if self.stage == 1:
            if roll < 0.70:
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


def make_train_env(
    stage: int = 1,
    reward_shaping: bool = True,
    navigation_shaping: bool = False,
) -> Monitor:
    return Monitor(
        CurriculumRescueEnv(
            stage=stage,
            reward_shaping=reward_shaping,
            navigation_shaping=navigation_shaping,
        ),
        info_keywords=("success", "timeout"),
    )


def make_eval_env(
    victim_distance: int | None = 2,
    max_steps: int = 64,
    min_victim_distance: int = 1,
) -> RescueEnv:
    """Unshaped evaluation environment (+10 native rescue only)."""
    return RescueEnv(
        max_steps=max_steps,
        victim_distance=victim_distance,
        min_victim_distance=min_victim_distance,
        reward_shaping=False,
        navigation_shaping=False,
    )


def evaluate_unshaped(
    model: MaskablePPO,
    eval_env: RescueEnv,
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

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            metrics = evaluate_unshaped(
                self.model,
                self.eval_env,
                n_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                seed_start=30000 + self.n_calls,
            )
            if self.verbose > 0:
                print(
                    f"\n[Eval @ {self.num_timesteps} steps] "
                    f"Mean Reward: {metrics['mean_reward']:.3f} | "
                    f"Success Rate: {metrics['success_rate'] * 100:.1f}% | "
                    f"Avg Steps: {metrics['mean_length']:.1f}"
                )
            if self.best_model_save_path and metrics["mean_reward"] > self.best_mean_reward:
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
        choices=["auto", "cpu", "mps"],
        default="auto",
        help="Device to train on (default: auto)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=128, help="Rollout steps per update")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=4, help="Epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Evaluation frequency in steps")
    parser.add_argument("--n-eval-episodes", type=int, default=20, help="Episodes per evaluation")
    parser.add_argument("--save-dir", type=str, default="./models/curriculum", help="Directory to save models")
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="Direct path to save the final model zip (e.g. ./models/autonomous_rl_v1/smoke_model.zip)",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Initial checkpoint path to load")
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
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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
    train_env = make_train_env(stage=initial_stage, reward_shaping=True)
    eval_env = make_eval_env(victim_distance=2, max_steps=64)

    if args.checkpoint:
        print(f"Loading checkpoint from: {args.checkpoint}")
        model = load_checkpoint(args.checkpoint, env=train_env, device=device)
    else:
        model = MaskablePPO(
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
            print(f"\n>>> Starting Stage 1: Qualification Gate ({stage1_steps} steps)")
            train_env.unwrapped.set_stage(1)
            stage1_dir = save_dir / "stage1"
            stage1_dir.mkdir(parents=True, exist_ok=True)

            callback1 = RecurrentMaskableEvalCallback(
                eval_env=eval_env,
                best_model_save_path=stage1_dir,
                eval_freq=args.eval_freq,
                n_eval_episodes=args.n_eval_episodes,
                deterministic=True,
            )

            model.learn(
                total_timesteps=stage1_steps,
                callback=callback1,
                tb_log_name="PPO_Curriculum_Stage1",
                reset_num_timesteps=False,
            )

            stage1_final = stage1_dir / "stage1_final.zip"
            save_checkpoint(model, stage1_final)
            print(f"Stage 1 completed. Model saved to {stage1_final}")

            # Check Qualification Gate
            if not args.skip_qualification_gate:
                passed = check_stage1_qualification(model, n_episodes=args.n_eval_episodes)
                if not passed and args.stage == "all":
                    raise RuntimeError(
                        "Stage 1 Qualification Gate FAILED: Required >= 90% deterministic and "
                        "sampled unshaped success on near-rescue before advancing to Stage 2."
                    )
            else:
                print("Skipping Stage 1 Qualification Gate check (--skip-qualification-gate).")

        # ---------------- STAGE 2 ----------------
        if stage2_steps > 0:
            print(f"\n>>> Starting Stage 2: Distant Search ({stage2_steps} steps)")
            train_env.unwrapped.set_stage(2)
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
            save_checkpoint(model, stage2_final)
            print(f"Stage 2 completed. Model saved to {stage2_final}")

        final_model_path = save_dir / "final_model.zip"
        save_checkpoint(model, final_model_path)
        print(f"Training complete. Final checkpoint saved to {final_model_path}")

        if args.save_model:
            smoke_path = Path(args.save_model)
            smoke_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, smoke_path)
            print(f"Model saved to {smoke_path}")

    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
