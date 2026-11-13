"""Maskable Recurrent PPO implementation for FlameBox.

Extends sb3_contrib.RecurrentPPO with legal action masking:
- MaskableRecurrentDictRolloutBuffer: stores action_masks and sequences with hidden states.
- MaskableRecurrentPPO: orchestrates recurrent rollout collection, action masking,
  sequence unrolling, and PPO updates with BPTT.
"""

from __future__ import annotations

from copy import deepcopy
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
