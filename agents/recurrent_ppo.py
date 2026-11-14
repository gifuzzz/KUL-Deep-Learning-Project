"""Maskable Recurrent PPO implementation for FlameBox.

Extends sb3_contrib.RecurrentPPO with legal action masking:
- MaskableRecurrentDictRolloutBuffer: stores action_masks and sequences with hidden states.
- MaskableRecurrentPPO: orchestrates recurrent rollout collection, action masking,
  sequence unrolling, and PPO updates with BPTT.
"""

from __future__ import annotations

import json
import subprocess
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Generator, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TensorDict
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import (
    RecurrentDictRolloutBuffer,
    RNNStates,
    create_sequencers,
)
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks

from agents.cnn import (
    RecurrentSpatialRescueCNN,
    inspect_checkpoint,
    transfer_compatible_weights,
)


class MaskableRecurrentDictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    lstm_states: RNNStates
    episode_starts: th.Tensor
    mask: th.Tensor
    action_masks: th.Tensor


class MaskableRecurrentDictRolloutBuffer(RecurrentDictRolloutBuffer):
    """Dict Rollout Buffer for recurrent policies with legal action masking."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_state_shape: tuple[int, int, int, int],
        device: th.device | str = "auto",
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            hidden_state_shape,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )
        self.mask_dim = action_space.n if hasattr(action_space, "n") else int(np.prod(action_space.shape))
        self.action_masks: np.ndarray | None = None
        self.reset()

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "mask_dim"):
            self.action_masks = np.ones((self.buffer_size, self.n_envs, self.mask_dim), dtype=bool)

    def add(
        self,
        *args,
        lstm_states: RNNStates,
        action_masks: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        if action_masks is not None:
            self.action_masks[self.pos] = np.array(action_masks, dtype=bool).copy()
        else:
            self.action_masks[self.pos] = True

        super().add(*args, lstm_states=lstm_states, **kwargs)

    def get(self, batch_size: int | None = None) -> Generator[MaskableRecurrentDictRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"
        if not self.generator_ready:
            self.action_masks = self.swap_and_flatten(self.action_masks)
        yield from super().get(batch_size)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: VecNormalize | None = None,
    ) -> MaskableRecurrentDictRolloutBufferSamples:
        self.seq_start_indices, self.pad, self.pad_and_flatten = create_sequencers(
            self.episode_starts[batch_inds], env_change[batch_inds], self.device
        )

        n_seq = len(self.seq_start_indices)
        max_length = self.pad(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length

        lstm_states_pi = (
            self.hidden_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_vf = (
            self.hidden_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_pi = (
            self.to_torch(lstm_states_pi[0]).contiguous(),
            self.to_torch(lstm_states_pi[1]).contiguous(),
        )
        lstm_states_vf = (
            self.to_torch(lstm_states_vf[0]).contiguous(),
            self.to_torch(lstm_states_vf[1]).contiguous(),
        )

        observations = {key: self.pad(obs[batch_inds]) for (key, obs) in self.observations.items()}
        observations = {
            key: obs.reshape((padded_batch_size,) + self.obs_shape[key])
            for (key, obs) in observations.items()
        }

        padded_action_masks = self.pad(self.action_masks[batch_inds], padding_value=1.0)
        action_masks = padded_action_masks.reshape((padded_batch_size, self.mask_dim)).bool()

        return MaskableRecurrentDictRolloutBufferSamples(
            observations=observations,
            actions=self.pad(self.actions[batch_inds]).reshape((padded_batch_size, *self.actions.shape[1:])),
            old_values=self.pad_and_flatten(self.values[batch_inds]),
            old_log_prob=self.pad_and_flatten(self.log_probs[batch_inds]),
            advantages=self.pad_and_flatten(self.advantages[batch_inds]),
            returns=self.pad_and_flatten(self.returns[batch_inds]),
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=self.pad_and_flatten(self.episode_starts[batch_inds]),
            mask=self.pad_and_flatten(np.ones_like(self.returns[batch_inds])),
            action_masks=action_masks,
        )


class MaskableRecurrentPPO(RecurrentPPO):
    """Maskable Recurrent PPO with legal action masking and multi-step BPTT."""

    def __init__(
        self,
        policy: type[RecurrentActorCriticPolicy] | str,
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 128,
        batch_size: int = 64,
        n_epochs: int = 4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: None | float | Schedule = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: float | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
        architecture_variant: str | None = None,
        architecture: str | None = None,
    ):
        policy_kwargs = dict(policy_kwargs or {})
        arch = architecture_variant or architecture
        if arch is not None:
            policy_kwargs["architecture_variant"] = arch

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        self.architecture_variant = policy_kwargs.get("architecture_variant", "baseline")

    @classmethod
    def load(
        cls,
        path: str | Path,
        env: GymEnv | None = None,
        device: th.device | str = "auto",
        custom_objects: dict[str, Any] | None = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        expected_architecture: str | None = None,
        architecture: str | None = None,
        **kwargs,
    ) -> "MaskableRecurrentPPO":
        """Load model with explicit Checkpoint Guard against architecture and tensor shape mismatches."""
        expected_arch = expected_architecture or architecture

        ckpt_info = inspect_checkpoint(path)
        ckpt_arch = ckpt_info["architecture_variant"]

        # Checkpoint Guard: Fail explicitly on architecture mismatch
        if expected_arch is not None and expected_arch != ckpt_arch:
            raise ValueError(
                f"Checkpoint architecture mismatch for '{path}':\n"
                f"  Checkpoint was saved with architecture : '{ckpt_arch}'\n"
                f"  Expected architecture                  : '{expected_arch}'\n"
                f"Loading mismatched weights is prohibited to prevent silent partial loading or corrupted inference.\n"
                f"To transfer compatible feature extractor weights instead, call transfer_compatible_weights()."
            )

        target_arch = expected_arch or ckpt_arch

        # Ensure custom_objects specifies correct policy class and architecture variant
        custom_objects = dict(custom_objects or {})
        custom_objects.setdefault("policy_class", RecurrentSpatialRescueCNN)
        custom_policy_kwargs = dict(custom_objects.get("policy_kwargs") or {})
        custom_policy_kwargs.setdefault("architecture_variant", target_arch)
        custom_objects["policy_kwargs"] = custom_policy_kwargs

        model = super().load(
            str(path),
            env=env,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
            force_reset=force_reset,
            **kwargs,
        )

        # Tensor Shape Checkpoint Guard
        ckpt_state = ckpt_info["state_dict"]
        model_state = model.policy.state_dict()
        mismatches = []
        for key, param in model_state.items():
            if key not in ckpt_state:
                mismatches.append(f"Missing key in checkpoint: {key}")
            elif param.shape != ckpt_state[key].shape:
                mismatches.append(
                    f"Shape mismatch for '{key}': model shape {param.shape} vs checkpoint shape {ckpt_state[key].shape}"
                )
        for key in ckpt_state.keys():
            if key not in model_state:
                mismatches.append(f"Unexpected key in checkpoint: {key}")

        if mismatches:
            raise RuntimeError(
                f"Checkpoint weight shape guard failed for '{path}':\n"
                + "\n".join(f"  - {m}" for m in mismatches)
                + "\nSilent partial loading is prohibited!"
            )

        model.checkpoint_metadata = ckpt_info.get("metadata", {})
        model.architecture_variant = target_arch
        return model

    def _setup_model(self) -> None:
        super()._setup_model()

        lstm = getattr(self.policy, "gru", getattr(self.policy, "lstm_actor", None))
        if lstm is None:
            raise AttributeError("Policy must define self.gru or self.lstm_actor")

        single_hidden_state_shape = (lstm.num_layers, self.n_envs, lstm.hidden_size)
        self._last_lstm_states = RNNStates(
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
        )

        hidden_state_buffer_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)

        self.rollout_buffer = MaskableRecurrentDictRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: MaskableRecurrentDictRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert isinstance(
            rollout_buffer, MaskableRecurrentDictRolloutBuffer
        ), f"{rollout_buffer} doesn't support maskable recurrent policy"

        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        callback.on_rollout_start()

        lstm_states = deepcopy(self._last_lstm_states)

        while n_steps < n_rollout_steps:
            action_masks = get_action_masks(env)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = th.tensor(self._last_episode_starts, dtype=th.float32, device=self.device)
                actions, values, log_probs, lstm_states = self.policy(
                    obs_tensor, lstm_states, episode_starts, action_masks=action_masks
                )

            actions = actions.cpu().numpy()
            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            # Handle timeout bootstrapping with value function
            for idx, done_ in enumerate(dones):
                if (
                    done_
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, idx : idx + 1, :].contiguous(),
                            lstm_states.vf[1][:, idx : idx + 1, :].contiguous(),
                        )
                        episode_starts_val = th.tensor([False], dtype=th.float32, device=self.device)
                        terminal_value = self.policy.predict_values(
                            terminal_obs, terminal_lstm_state, episode_starts_val
                        )[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
                action_masks=action_masks,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

        with th.no_grad():
            episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), lstm_states.vf, episode_starts)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()

        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        gru_grad_norms = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                mask = rollout_data.mask > 1e-8

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                    action_masks=rollout_data.action_masks,
                )

                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()[mask]).item()
                clip_fractions.append(clip_fraction)

                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob[mask])
                else:
                    entropy_loss = -th.mean(entropy[mask])
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()

                # Measure GRU gradient norm before clipping
                gru_module = getattr(self.policy, "gru", None) or getattr(self.policy, "lstm_actor", None)
                if gru_module is not None:
                    gru_grads = [p.grad for p in gru_module.parameters() if p.grad is not None]
                    if gru_grads:
                        gru_norm = th.norm(th.stack([th.norm(g.detach(), 2) for g in gru_grads]), 2).item()
                    else:
                        gru_norm = 0.0
                else:
                    gru_norm = 0.0
                gru_grad_norms.append(gru_norm)

                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        mean_gru_grad_norm = float(np.mean(gru_grad_norms)) if gru_grad_norms else 0.0

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/gru_grad_norm", mean_gru_grad_norm)

        update_record = {
            "update": int(self._n_updates),
            "timesteps": int(self.num_timesteps),
            "policy_loss": float(np.mean(pg_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(-np.mean(entropy_losses)),
            "entropy_loss": float(np.mean(entropy_losses)),
            "approx_kl": float(np.mean(approx_kl_divs)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "loss": float(loss.item()),
            "explained_variance": float(explained_var),
            "gru_grad_norm": mean_gru_grad_norm,
        }
        if not hasattr(self, "training_metrics_history"):
            self.training_metrics_history = []
        self.training_metrics_history.append(update_record)

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | np.ndarray | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | np.ndarray | None]:
        return self.policy.predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
            action_masks=action_masks,
        )


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
    model: MaskableRecurrentPPO | Any,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save model checkpoint safely casting policy and optimizer state dicts to CPU.

    Supports CPU/MPS/CUDA cross-device transfer and saves metadata (stage, timesteps,
    git commit, hyperparameters, architecture variant, hidden_size, total_params)
    into the checkpoint zip and companion JSON file.
    """
    path = Path(path)
    if "stage1_qualified.zip" in str(path) and path.exists():
        raise RuntimeError("stage1_qualified.zip is immutable and cannot be overwritten!")
    if "stage2_final.zip" in str(path) and path.exists():
        raise RuntimeError("stage2_final.zip is immutable and cannot be overwritten!")
    path.parent.mkdir(parents=True, exist_ok=True)
    orig_device = model.device

    arch_variant = getattr(
        model.policy,
        "architecture_variant",
        getattr(model, "architecture_variant", "baseline"),
    )
    hidden_size = getattr(model.policy, "hidden_size", 128)
    total_params = sum(p.numel() for p in model.policy.parameters())

    meta_dict = {
        "git_commit": get_git_commit(),
        "total_timesteps": getattr(model, "num_timesteps", 0),
        "policy_class": model.policy.__class__.__name__,
        "algorithm": model.__class__.__name__,
        "architecture_variant": str(arch_variant),
        "hidden_size": int(hidden_size),
        "total_params": int(total_params),
        "warm_started": getattr(model.policy, "is_warm_started", getattr(model, "is_warm_started", False)),
        "warm_start_source": getattr(model.policy, "warm_start_source", getattr(model, "warm_start_source", None)),
        "warm_start_transferred_layers": getattr(
            model.policy,
            "warm_start_transferred_layers",
            getattr(model, "warm_start_transferred_layers", []),
        ),
    }
    if metadata:
        meta_dict.update(metadata)

    try:
        model.policy.to("cpu")
        if hasattr(model.policy, "optimizer") and model.policy.optimizer is not None:
            for state in model.policy.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, th.Tensor):
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
                    if isinstance(v, th.Tensor):
                        state[k] = v.to(orig_device)


def load_checkpoint(
    path: str | Path,
    env: GymEnv | None = None,
    device: th.device | str = "auto",
    expected_architecture: str | None = None,
    architecture: str | None = None,
) -> MaskableRecurrentPPO | Any:
    """Load model checkpoint safely with explicit Checkpoint Guard against architecture and tensor shape mismatches."""
    expected_arch = expected_architecture or architecture

    ckpt_info = inspect_checkpoint(path)
    ckpt_arch = ckpt_info["architecture_variant"]

    # Checkpoint Guard: Fail explicitly on architecture mismatch
    if expected_arch is not None and expected_arch != ckpt_arch:
        raise ValueError(
            f"Checkpoint architecture mismatch for '{path}':\n"
            f"  Checkpoint was saved with architecture : '{ckpt_arch}'\n"
            f"  Expected architecture                  : '{expected_arch}'\n"
            f"Loading mismatched weights is prohibited to prevent silent partial loading or corrupted inference.\n"
            f"To transfer compatible feature extractor weights instead, call transfer_compatible_weights()."
        )

    target_arch = expected_arch or ckpt_arch

    # Determine device
    if device == "auto":
        if th.cuda.is_available():
            target_device = th.device("cuda")
        elif th.backends.mps.is_available():
            target_device = th.device("mps")
        else:
            target_device = th.device("cpu")
    else:
        target_device = th.device(device)

    custom_objects = {
        "policy_class": RecurrentSpatialRescueCNN,
        "policy_kwargs": {"architecture_variant": target_arch},
    }

    # Try MaskableRecurrentPPO first, then MaskablePPO
    model = None
    try:
        model = MaskableRecurrentPPO.load(
            str(path),
            env=env,
            device=target_device,
            custom_objects=custom_objects,
            expected_architecture=target_arch,
        )
    except Exception as e:
        # If it was an explicit architecture mismatch error, re-raise!
        if isinstance(e, (ValueError, RuntimeError)) and ("mismatch" in str(e) or "guard" in str(e)):
            raise e
        try:
            from sb3_contrib import MaskablePPO
            model = MaskablePPO.load(
                str(path),
                env=env,
                device=target_device,
                custom_objects=custom_objects,
            )
        except Exception:
            raise e

    # Tensor shape guard for MaskablePPO fallback if loaded
    ckpt_state = ckpt_info["state_dict"]
    model_state = model.policy.state_dict()
    mismatches = []
    for key, param in model_state.items():
        if key not in ckpt_state:
            mismatches.append(f"Missing key in checkpoint: {key}")
        elif param.shape != ckpt_state[key].shape:
            mismatches.append(
                f"Shape mismatch for '{key}': model shape {param.shape} vs checkpoint shape {ckpt_state[key].shape}"
            )
    for key in ckpt_state.keys():
        if key not in model_state:
            mismatches.append(f"Unexpected key in checkpoint: {key}")

    if mismatches:
        raise RuntimeError(
            f"Checkpoint weight shape guard failed for '{path}':\n"
            + "\n".join(f"  - {m}" for m in mismatches)
            + "\nSilent partial loading is prohibited!"
        )

    model.checkpoint_metadata = ckpt_info.get("metadata", {})
    model.architecture_variant = target_arch
    return model


