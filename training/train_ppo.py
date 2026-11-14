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
    if "stage1_qualified.zip" in str(path) and path.exists():
        raise RuntimeError("stage1_qualified.zip is immutable and cannot be overwritten!")
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
    """Progressive Mixed-Visibility Curriculum Environment.

    Stage 1 (Qualification Gate):
      - near_ratio% near/visible victims (max_steps=64, victim_distance=2, min_victim_distance=1)
      - (1 - near_ratio)% medium/hidden victims (max_steps=128, victim_distance=6, min_victim_distance=3)

    Stage 2 (Staged Progression with Near-Rescue Retention):
      - Substage '2A' (Medium Bridge):
          30% Near (d=2, max_steps=64)
          50% Medium (d=6, min_dist=3, max_steps=128)
          20% Distant (d=None, max_steps=256)
      - Substage '2B' (Distant Search):
          20% Near (d=2, max_steps=64)
          30% Medium (d=6, min_dist=3, max_steps=128)
          50% Distant (d=None, max_steps=384)
      - Substage '2C' (Distant Mastery):
          10% Near (d=2, max_steps=64)
          20% Medium (d=6, min_dist=3, max_steps=128)
          70% Distant (d=None, max_steps=384)
      - Substage 'reinforce_near' (Healing mode when prerequisite near retention drops):
          60% Near (d=2, max_steps=64)
          30% Medium (d=6, min_dist=3, max_steps=128)
          10% Distant (d=None, max_steps=256)
    """

    def __init__(self, stage: int = 1, near_ratio: float = 0.70, substage: str = "2A", **kwargs):
        super().__init__(**kwargs)
        self.stage = int(stage)
        self.near_ratio = float(near_ratio)
        self.substage = str(substage)

    def set_stage(self, stage: int, near_ratio: float | None = None, substage: str | None = None) -> None:
        self.stage = int(stage)
        if near_ratio is not None:
            self.near_ratio = float(near_ratio)
        if substage is not None:
            self.substage = str(substage)

    def reset(self, *, seed=None, options=None):
        if options and "stage" in options:
            self.stage = int(options["stage"])
        if options and "near_ratio" in options:
            self.near_ratio = float(options["near_ratio"])
        if options and "substage" in options:
            self.substage = str(options["substage"])

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
            # Stage 2 Staged Progression
            if self.substage == "reinforce_near":
                if roll < 0.60:
                    self.victim_distance = 2
                    self.min_victim_distance = 1
                    self.max_steps = 64
                elif roll < 0.90:
                    self.victim_distance = 6
                    self.min_victim_distance = 3
                    self.max_steps = 128
                else:
                    self.victim_distance = None
                    self.min_victim_distance = 1
                    self.max_steps = 256
            elif self.substage == "2A":
                if roll < 0.30:
                    self.victim_distance = 2
                    self.min_victim_distance = 1
                    self.max_steps = 64
                elif roll < 0.80:
                    self.victim_distance = 6
                    self.min_victim_distance = 3
                    self.max_steps = 128
                else:
                    self.victim_distance = None
                    self.min_victim_distance = 1
                    self.max_steps = 256
            elif self.substage == "2B":
                if roll < 0.20:
                    self.victim_distance = 2
                    self.min_victim_distance = 1
                    self.max_steps = 64
                elif roll < 0.50:
                    self.victim_distance = 6
                    self.min_victim_distance = 3
                    self.max_steps = 128
                else:
                    self.victim_distance = None
                    self.min_victim_distance = 1
                    self.max_steps = 384
            else:  # "2C" Distant Mastery
                if roll < 0.10:
                    self.victim_distance = 2
                    self.min_victim_distance = 1
                    self.max_steps = 64
                elif roll < 0.30:
                    self.victim_distance = 6
                    self.min_victim_distance = 3
                    self.max_steps = 128
                else:
                    self.victim_distance = None
                    self.min_victim_distance = 1
                    self.max_steps = 384

        return super().reset(seed=seed, options=options)


def set_env_stage(env: Any, stage: int, near_ratio: float | None = None, substage: str | None = None) -> None:
    """Set curriculum stage, near ratio, and progressive substage across vectorized environments."""
    if hasattr(env, "env_method"):
        env.env_method("set_stage", stage, near_ratio, substage)
    elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "set_stage"):
        env.unwrapped.set_stage(stage, near_ratio, substage)


def make_single_train_env(
    stage: int = 1,
    reward_shaping: bool = True,
    navigation_shaping: bool = False,
    near_ratio: float = 0.70,
    substage: str = "2A",
    seed: int | None = None,
):
    """Factory creating an individual monitored environment."""
    def _init():
        env = CurriculumRescueEnv(
            stage=stage,
            reward_shaping=reward_shaping,
            navigation_shaping=navigation_shaping,
            near_ratio=near_ratio,
            substage=substage,
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
    substage: str = "2A",
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
            substage=substage,
        )
        env = PreviousActionWrapper(env)
        return Monitor(env, info_keywords=("success", "timeout"))

    env_fns = [
        make_single_train_env(
            stage=stage,
            reward_shaping=reward_shaping,
            navigation_shaping=navigation_shaping,
            near_ratio=near_ratio,
            substage=substage,
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
    pickups = []
    lengths = []

    for ep in range(n_episodes):
        ep_seed = seed_start + ep
        obs, _ = eval_env.reset(seed=ep_seed)
        if not deterministic:
            torch.manual_seed(ep_seed)

        done = False
        ep_reward = 0.0
        steps = 0
        ever_picked_up = False
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

            carrying = any(
                v["carried_by"] == 0 and not v["dead"] and not v["rescued"]
                for v in eval_env.unwrapped.sim.victims
            )
            if carrying:
                ever_picked_up = True

        is_succ = bool(info.get("success", False))
        rewards.append(ep_reward)
        lengths.append(steps)
        successes.append(is_succ)
        pickups.append(ever_picked_up or is_succ)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
        "pickup_rate": float(np.mean(pickups)),
        "mean_length": float(np.mean(lengths)),
    }


class RecurrentMaskableEvalCallback(BaseCallback):
    """Periodic evaluation callback supporting recurrent hidden state, action masks,

    strict improvement checking, immutable qualified stage preservation, separate
    latest_model.zip saving, and adaptive curriculum management.
    """

    def __init__(
        self,
        eval_env_stage2: RescueEnv | PreviousActionWrapper,
        eval_env_near: RescueEnv | PreviousActionWrapper | None = None,
        eval_env_med: RescueEnv | PreviousActionWrapper | None = None,
        train_env: Any = None,
        best_model_save_path: str | Path | None = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 20,
        deterministic: bool = True,
        hyperparameters: dict[str, Any] | None = None,
        seed_start: int = 50000,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.eval_env_stage2 = eval_env_stage2
        self.eval_env_near = eval_env_near
        self.eval_env_med = eval_env_med
        self.train_env = train_env
        self.best_model_save_path = Path(best_model_save_path) if best_model_save_path else None
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.hyperparameters = hyperparameters or {}
        self.seed_start = seed_start

        # Best metric tracking for strict improvement
        self.best_s2_success = 0.0
        self.best_near_retention = 0.0
        self.best_s2_pickup = 0.0
        self.best_mean_reward = -float("inf")
        self.best_mean_length = float("inf")
        self.has_saved_best = False
        self.last_eval_step = 0
        self.current_substage = "2A"

    def _is_strictly_better(
        self,
        s2_succ: float,
        near_succ: float,
        s2_pick: float,
        mean_rew: float,
        mean_len: float,
    ) -> bool:
        """Strict improvement evaluation policy with explicit tie-breaking."""
        # 1. Primary metric: Stage 2 hidden-victim rescue success (S_2 > 0)
        if s2_succ > 0.0:
            if s2_succ > self.best_s2_success:
                return True
            if s2_succ == self.best_s2_success:
                # 2. Secondary metric: Near-rescue retention
                if near_succ > self.best_near_retention:
                    return True
                if near_succ == self.best_near_retention:
                    # 3. Tertiary metric: Distant pickup rate
                    if s2_pick > self.best_s2_pickup:
                        return True
                    if s2_pick == self.best_s2_pickup:
                        # 4. Quaternary metric: Combined mean reward
                        if mean_rew > self.best_mean_reward + 0.05:
                            return True
                        if np.isclose(mean_rew, self.best_mean_reward, atol=0.05):
                            # 5. Quinary metric: Shorter average steps
                            return mean_len < self.best_mean_length - 2.0
            return False

        # 2. Early Phase (S_2 == 0.0): Track exploration progress on pickups
        # Only allow saving exploratory best IF near retention is solidly preserved (>= 80%)
        if self.best_s2_success == 0.0 and near_succ >= 0.80:
            if s2_pick > self.best_s2_pickup and s2_pick > 0.0:
                return True
            if s2_pick == self.best_s2_pickup and s2_pick > 0.0:
                if near_succ > self.best_near_retention:
                    return True
                if near_succ == self.best_near_retention and mean_rew > self.best_mean_reward + 0.10:
                    return True

        return False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and (self.num_timesteps - self.last_eval_step >= self.eval_freq):
            self.last_eval_step = self.num_timesteps

            # 1. Evaluate Distant (Stage 2 primary target)
            s2_metrics = evaluate_unshaped(
                self.model,
                self.eval_env_stage2,
                n_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                seed_start=self.seed_start,
            )

            # 2. Evaluate Near (retention check)
            if self.eval_env_near is not None:
                near_metrics = evaluate_unshaped(
                    self.model,
                    self.eval_env_near,
                    n_episodes=self.n_eval_episodes,
                    deterministic=self.deterministic,
                    seed_start=self.seed_start,
                )
            else:
                near_metrics = {"success_rate": 1.0, "pickup_rate": 1.0, "mean_reward": 10.0, "mean_length": 3.0}

            # 3. Evaluate Medium (bridge check)
            if self.eval_env_med is not None:
                med_metrics = evaluate_unshaped(
                    self.model,
                    self.eval_env_med,
                    n_episodes=self.n_eval_episodes,
                    deterministic=self.deterministic,
                    seed_start=self.seed_start,
                )
            else:
                med_metrics = {"success_rate": 0.0, "pickup_rate": 0.0, "mean_reward": 0.0, "mean_length": 64.0}

            combined_reward = (s2_metrics["mean_reward"] + near_metrics["mean_reward"]) / 2.0
            combined_length = s2_metrics["mean_length"]

            if self.verbose > 0:
                print(
                    f"\n[Eval @ {self.num_timesteps} steps] Substage: {self.current_substage} | "
                    f"Distant Succ: {s2_metrics['success_rate']*100:.1f}% (Pick: {s2_metrics['pickup_rate']*100:.1f}%) | "
                    f"Near Ret: {near_metrics['success_rate']*100:.1f}% | "
                    f"Med Succ: {med_metrics['success_rate']*100:.1f}% | "
                    f"Mean Rew: {combined_reward:.3f}"
                )

            # Unconditionally save latest_model.zip separately
            if self.best_model_save_path:
                latest_path = self.best_model_save_path / "latest_model.zip"
                save_checkpoint(
                    self.model,
                    latest_path,
                    metadata={
                        "evaluation_step": self.num_timesteps,
                        "substage": self.current_substage,
                        "distant_success_rate": s2_metrics["success_rate"],
                        "distant_pickup_rate": s2_metrics["pickup_rate"],
                        "near_retention_rate": near_metrics["success_rate"],
                        "medium_success_rate": med_metrics["success_rate"],
                        "mean_reward": combined_reward,
                        "seed_set": f"{self.seed_start}..{self.seed_start + self.n_eval_episodes - 1}",
                        "training_hyperparameters": self.hyperparameters,
                        "checkpoint_type": "latest_model",
                    },
                )

            # Check strict improvement for best_model.zip
            if self.best_model_save_path and self._is_strictly_better(
                s2_metrics["success_rate"],
                near_metrics["success_rate"],
                s2_metrics["pickup_rate"],
                combined_reward,
                combined_length,
            ):
                self.best_s2_success = s2_metrics["success_rate"]
                self.best_near_retention = near_metrics["success_rate"]
                self.best_s2_pickup = s2_metrics["pickup_rate"]
                self.best_mean_reward = combined_reward
                self.best_mean_length = combined_length
                self.has_saved_best = True

                best_path = self.best_model_save_path / "best_model.zip"
                save_checkpoint(
                    self.model,
                    best_path,
                    metadata={
                        "evaluation_step": self.num_timesteps,
                        "substage": self.current_substage,
                        "distant_success_rate": s2_metrics["success_rate"],
                        "distant_pickup_rate": s2_metrics["pickup_rate"],
                        "near_retention_rate": near_metrics["success_rate"],
                        "medium_success_rate": med_metrics["success_rate"],
                        "mean_reward": combined_reward,
                        "seed_set": f"{self.seed_start}..{self.seed_start + self.n_eval_episodes - 1}",
                        "training_hyperparameters": self.hyperparameters,
                        "checkpoint_type": "best_model",
                    },
                )
                if self.verbose > 0:
                    print(
                        f"*** STRICT IMPROVEMENT DETECTED! New best model saved to {best_path} "
                        f"(Distant Succ: {self.best_s2_success*100:.1f}%, Near Ret: {self.best_near_retention*100:.1f}%)"
                    )

            # Adaptive Curriculum Self-Healing & Progression
            if self.train_env is not None:
                if near_metrics["success_rate"] < 0.80 and self.current_substage != "reinforce_near":
                    self.current_substage = "reinforce_near"
                    set_env_stage(self.train_env, stage=2, substage="reinforce_near")
                    if self.verbose > 0:
                        print(">>> Curriculum Alert: Near retention dropped below 80%. Shifting to 'reinforce_near' mode.")
                elif near_metrics["success_rate"] >= 0.85:
                    if self.current_substage in ("2A", "reinforce_near") and med_metrics["success_rate"] >= 0.40:
                        self.current_substage = "2B"
                        set_env_stage(self.train_env, stage=2, substage="2B")
                        if self.verbose > 0:
                            print(">>> Curriculum Milestone: Near retention >= 85% and Medium success >= 40%. Advancing to Substage 2B.")
                    elif self.current_substage == "2B" and med_metrics["success_rate"] >= 0.60 and s2_metrics["success_rate"] >= 0.15:
                        self.current_substage = "2C"
                        set_env_stage(self.train_env, stage=2, substage="2C")
                        if self.verbose > 0:
                            print(">>> Curriculum Milestone: Advancing to Substage 2C (Full Distant Mastery).")

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

            eval_env_near = make_eval_env(victim_distance=2, max_steps=64)
            eval_env_med = make_eval_env(victim_distance=6, max_steps=128)
            callback1 = RecurrentMaskableEvalCallback(
                eval_env_stage2=eval_env_near,
                eval_env_near=eval_env_near,
                eval_env_med=eval_env_med,
                train_env=train_env,
                best_model_save_path=stage1_dir,
                eval_freq=args.eval_freq,
                n_eval_episodes=args.n_eval_episodes,
                deterministic=True,
                hyperparameters=hyperparameters,
                seed_start=40000,
            )

            print(f"--- Stage 1: Near-Rescue Mastery ({stage1_steps} steps, {int(args.near_ratio*100)}% near victims) ---")
            set_env_stage(train_env, stage=1, near_ratio=args.near_ratio)
            model.learn(
                total_timesteps=stage1_steps,
                callback=callback1,
                tb_log_name="PPO_Curriculum_Stage1",
                reset_num_timesteps=False,
            )
            eval_env_near.close()
            eval_env_med.close()

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
            set_env_stage(train_env, stage=2, substage="2A")
            stage2_dir = save_dir / "stage2"
            stage2_dir.mkdir(parents=True, exist_ok=True)

            eval_env_stage2 = make_eval_env(victim_distance=None, max_steps=384)
            eval_env_near = make_eval_env(victim_distance=2, max_steps=64)
            eval_env_med = make_eval_env(victim_distance=6, max_steps=128)

            callback2 = RecurrentMaskableEvalCallback(
                eval_env_stage2=eval_env_stage2,
                eval_env_near=eval_env_near,
                eval_env_med=eval_env_med,
                train_env=train_env,
                best_model_save_path=stage2_dir,
                eval_freq=args.eval_freq,
                n_eval_episodes=args.n_eval_episodes,
                deterministic=True,
                hyperparameters=hyperparameters,
                seed_start=50000,
            )

            model.learn(
                total_timesteps=stage2_steps,
                callback=callback2,
                tb_log_name="PPO_Curriculum_Stage2",
                reset_num_timesteps=False,
            )
            eval_env_stage2.close()
            eval_env_near.close()
            eval_env_med.close()

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
