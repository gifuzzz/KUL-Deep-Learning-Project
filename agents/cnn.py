from functools import partial
from typing import Any

import numpy as np
import torch
from torch import nn
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from sb3_contrib.ppo_mask.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.distributions import (
    MaskableDistribution,
    make_masked_proba_distribution,
)


class RescueCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=128)

        local_channels = observation_space["grid"].shape[0]
        memory_channels = observation_space["memory"].shape[0]
        agent_dim = observation_space["agent"].shape[0]

        # CNN for nearby information
        self.local_cnn = nn.Sequential(
            nn.Conv2d(local_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
        )

        # CNN for the remembered building
        self.memory_cnn = nn.Sequential(
            nn.Conv2d(memory_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        local_features = 64 * 3 * 3
        memory_features = 64 * 4 * 4

        # Combine both maps and the firefighter's own state
        self.fusion = nn.Sequential(
            nn.Linear(
                local_features + memory_features + agent_dim,
                128,
            ),
            nn.ReLU(),
        )

    def forward(self, observations):
        local_features = self.local_cnn(observations["grid"])
        memory_features = self.memory_cnn(observations["memory"])
        agent_features = observations["agent"]

        combined = torch.cat(
            [local_features, memory_features, agent_features],
            dim=1,
        )

        return self.fusion(combined)


class SpatialRescueCNN(RescueCNN):
    """Compress local channels without averaging away their cell positions."""

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space)
        self.local_cnn = nn.Sequential(
            self.local_cnn[:-2],
            nn.Conv2d(64, 8, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        height, width = observation_space["grid"].shape[1:]
        agent_dim = observation_space["agent"].shape[0]
        self.fusion[0] = nn.Linear(8 * height * width + 64 * 4 * 4 + agent_dim, 128)


class RelativeSpatialRescueCNN(SpatialRescueCNN):
    """Give remembered cells coordinates relative to the known firefighter position."""

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space)
        channels, height, width = observation_space["memory"].shape
        self.memory_cnn[0] = nn.Conv2d(channels + 2, 32, kernel_size=3, padding=1)
        self.register_buffer("map_rows", torch.arange(height).float()[None, :, None], persistent=False)
        self.register_buffer("map_cols", torch.arange(width).float()[None, None, :], persistent=False)

    def memory_input(self, observations):
        memory = observations["memory"]
        footprint = memory[:, 12]
        height = footprint.amax(dim=2).sum(dim=1)
        width = footprint.amax(dim=1).sum(dim=1)
        row = observations["agent"][:, 0] * height
        col = observations["agent"][:, 1] * width
        relative_row = (self.map_rows - row[:, None, None]) / memory.shape[2]
        relative_col = (self.map_cols - col[:, None, None]) / memory.shape[3]
        coordinates = torch.stack((relative_row * footprint, relative_col * footprint), dim=1)
        return torch.cat((memory, coordinates), dim=1)

    def forward(self, observations):
        return super().forward({**observations, "memory": self.memory_input(observations)})


class DetailedMemoryRescueCNN(RelativeSpatialRescueCNN):
    """Retain fine remembered geometry alongside the original coarse map features."""

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space)
        height, width = observation_space["memory"].shape[1:]
        self.memory_projection = nn.Sequential(nn.Conv2d(64, 8, 1), nn.ReLU(), nn.Flatten())
        self.memory_correction = nn.Linear(8 * (height // 4) * (width // 4), 128)

    def forward(self, observations):
        spatial_memory = self.memory_cnn[:-2](self.memory_input(observations))
        memory_features = self.memory_cnn[-2:](spatial_memory)
        local_features = self.local_cnn(observations["grid"])
        features = self.fusion(torch.cat((local_features, memory_features,
                                         observations["agent"]), dim=1))
        correction = self.memory_correction(self.memory_projection(spatial_memory))
        return torch.relu(features + correction)


class RememberedLocalRescueCNN(DetailedMemoryRescueCNN):
    """Read remembered nearby cells at the same spatial scale as current vision."""

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space)
        _, height, width = observation_space["grid"].shape
        self.register_buffer("crop_rows", (torch.arange(height).float() - height // 2)[None, :, None], persistent=False)
        self.register_buffer("crop_cols", (torch.arange(width).float() - width // 2)[None, None, :], persistent=False)
        self.remembered_correction = nn.Linear(8 * height * width, 128)

    def forward(self, observations):
        features = super().forward(observations)
        memory = observations["memory"]
        footprint = memory[:, 12]
        row = observations["agent"][:, 0] * footprint.amax(dim=2).sum(dim=1)
        col = observations["agent"][:, 1] * footprint.amax(dim=1).sum(dim=1)
        x = 2 * (col[:, None, None] + self.crop_cols) / (memory.shape[3] - 1) - 1
        y = 2 * (row[:, None, None] + self.crop_rows) / (memory.shape[2] - 1) - 1
        grid = torch.stack(torch.broadcast_tensors(x, y), dim=-1)
        crop = torch.nn.functional.grid_sample(memory[:, :11], grid, mode="nearest",
                                               padding_mode="zeros", align_corners=True)
        correction = self.remembered_correction(self.local_cnn(crop))
        return torch.relu(features + correction)


class _DummyFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim=features_dim)

    def forward(self, observations):
        return observations


class RecurrentSpatialRescueCNN(MaskableActorCriticPolicy):
    """Recurrent Spatial Rescue CNN Policy compatible with SB3-contrib MaskablePPO.

    Architecture:
      - Local Branch: Conv2d(9, 32, 3, pad=1) -> ReLU -> Conv2d(32, 64, 3, pad=1) -> ReLU -> Conv2d(64, 8, 1) -> Flatten (1,800)
      - Memory Branch (Zero MaxPool2d): Conv2d(9, 32, 3, pad=1) -> ReLU -> Conv2d(32, 32, 3, stride=2, pad=1) -> ReLU -> Conv2d(32, 16, 1) -> Flatten (8,320)
      - Agent Scalars: 5 features
      - Previous Action: 7-dim one-hot vector
      - Fusion: Linear(10,132, 128) -> GRU(128, 128)
      - Output Heads: Separate 7-action masked policy head and scalar value head.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space | None = None,
        lr_schedule: Schedule | None = None,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        features_extractor_class: type[BaseFeaturesExtractor] | None = None,
        features_extractor_kwargs: dict[str, Any] | None = None,
        share_features_extractor: bool = True,
        normalize_images: bool = False,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        hidden_size: int = 128,
    ):
        if action_space is None:
            action_space = spaces.Discrete(7)
        if lr_schedule is None:
            lr_schedule = lambda _: 3e-4

        if isinstance(observation_space, spaces.Dict):
            space_dict = dict(observation_space.spaces)
            if "previous_action" not in space_dict:
                space_dict["previous_action"] = spaces.Box(
                    low=0, high=6, shape=(), dtype=np.float32
                )
            observation_space = spaces.Dict(space_dict)

        self.hidden_size = hidden_size

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            features_extractor_class=features_extractor_class or _DummyFeaturesExtractor,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self.features_extractor = nn.Identity()
        self.pi_features_extractor = self.features_extractor
        self.vf_features_extractor = self.features_extractor

        local_channels = 9
        local_h, local_w = 15, 15
        mem_channels = 9
        mem_h, mem_w = 40, 52
        agent_dim = 5
        action_dim = 7

        if isinstance(self.observation_space, spaces.Dict):
            if "grid" in self.observation_space.spaces:
                local_channels, local_h, local_w = self.observation_space["grid"].shape
            if "memory" in self.observation_space.spaces:
                mem_channels, mem_h, mem_w = self.observation_space["memory"].shape
            if "agent" in self.observation_space.spaces:
                agent_dim = self.observation_space["agent"].shape[0]

        if hasattr(self.action_space, "n"):
            action_dim = self.action_space.n

        # Local Branch: Unpooled SpatialRescueCNN head
        self.local_cnn = nn.Sequential(
            nn.Conv2d(local_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 8, kernel_size=1),
            nn.Flatten(),
        )
        self.local_features_dim = 8 * local_h * local_w

        # Memory Branch: ZERO MaxPool2d
        self.memory_cnn = nn.Sequential(
            nn.Conv2d(mem_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.Flatten(),
        )
        stride_h = (mem_h + 2 * 1 - 3) // 2 + 1
        stride_w = (mem_w + 2 * 1 - 3) // 2 + 1
        self.memory_features_dim = 16 * stride_h * stride_w

        self.agent_dim = agent_dim
        self.action_dim = action_dim
        total_fusion_dim = (
            self.local_features_dim
            + self.memory_features_dim
            + self.agent_dim
            + self.action_dim
        )

        # Fusion: Linear(..., 128)
        self.fusion = nn.Linear(total_fusion_dim, self.hidden_size)

        # Recurrence: GRU(128, 128)
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            batch_first=True,
        )

        # Output heads
        self.action_net = nn.Linear(self.hidden_size, self.action_dim)
        self.policy_head = self.action_net
        self.value_net = nn.Linear(self.hidden_size, 1)
        self.value_head = self.value_net

        self.hidden_state = None

        if self.ortho_init:
            module_gains = {
                self.local_cnn: np.sqrt(2),
                self.memory_cnn: np.sqrt(2),
                self.fusion: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1.0,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _encode_previous_action(
        self,
        prev_action: torch.Tensor | np.ndarray | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if prev_action is None:
            return torch.zeros((batch_size, self.action_dim), dtype=torch.float32, device=device)
        if not isinstance(prev_action, torch.Tensor):
            prev_action = torch.as_tensor(prev_action, device=device)
        else:
            prev_action = prev_action.to(device)

        if prev_action.dim() >= 2 and prev_action.shape[-1] == self.action_dim:
            return prev_action.view(batch_size, self.action_dim).float()

        flat = prev_action.long().view(-1)
        if flat.numel() == 0:
            return torch.zeros((batch_size, self.action_dim), dtype=torch.float32, device=device)
        if flat.shape[0] != batch_size:
            if flat.numel() == 1:
                flat = flat.expand(batch_size)
            else:
                flat = flat[:batch_size]

        valid = (flat >= 0) & (flat < self.action_dim)
        safe_idx = torch.where(valid, flat, torch.zeros_like(flat))
        one_hot = torch.nn.functional.one_hot(safe_idx, num_classes=self.action_dim).float()
        one_hot = one_hot * valid.unsqueeze(-1).float()
        return one_hot

    def _forward_network(
        self,
        obs: dict[str, torch.Tensor] | torch.Tensor,
        lstm_states: torch.Tensor | None = None,
        episode_starts: torch.Tensor | np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(obs, dict):
            raise TypeError(f"RecurrentSpatialRescueCNN expects observation dict, got {type(obs)}")

        grid = obs["grid"]
        memory = obs["memory"]
        agent = obs["agent"]
        prev_action = obs.get("previous_action", None)

        batch_size = grid.shape[0]
        device = grid.device

        local_features = self.local_cnn(grid)
        memory_features = self.memory_cnn(memory)
        agent_features = agent.view(batch_size, -1)
        prev_action_features = self._encode_previous_action(prev_action, batch_size, device)

        combined = torch.cat(
            [local_features, memory_features, agent_features, prev_action_features],
            dim=-1,
        )
        fused = self.fusion(combined)

        h = lstm_states if lstm_states is not None else self.hidden_state
        if h is None or h.shape[1] != batch_size or h.device != device:
            h = torch.zeros((1, batch_size, self.hidden_size), dtype=torch.float32, device=device)

        if episode_starts is not None:
            starts = torch.as_tensor(episode_starts, dtype=torch.float32, device=device).view(1, batch_size, 1)
            h = h * (1.0 - starts)

        gru_out, next_hidden = self.gru(fused.unsqueeze(1), h)
        gru_out = gru_out.squeeze(1)

        self.hidden_state = next_hidden.detach()
        return gru_out, next_hidden

    def reset_hidden_state(
        self,
        batch_size: int = 1,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        self.hidden_state = torch.zeros((1, batch_size, self.hidden_size), dtype=torch.float32, device=device)
        return self.hidden_state

    def get_masked_logits(
        self,
        obs: dict[str, torch.Tensor] | torch.Tensor,
        action_masks: torch.Tensor | np.ndarray | None = None,
        lstm_states: torch.Tensor | None = None,
        episode_starts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent, _ = self._forward_network(obs, lstm_states=lstm_states, episode_starts=episode_starts)
        logits = self.action_net(latent)
        if action_masks is not None:
            dist = self.action_dist.proba_distribution(action_logits=logits)
            dist.apply_masking(action_masks)
            return dist.distribution.logits
        return logits

    def get_action_logits_and_values(
        self,
        obs: dict[str, torch.Tensor] | torch.Tensor,
        action_masks: torch.Tensor | np.ndarray | None = None,
        lstm_states: torch.Tensor | None = None,
        episode_starts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent, _ = self._forward_network(obs, lstm_states=lstm_states, episode_starts=episode_starts)
        logits = self.action_net(latent)
        values = self.value_net(latent)
        if action_masks is not None:
            dist = self.action_dist.proba_distribution(action_logits=logits)
            dist.apply_masking(action_masks)
            logits = dist.distribution.logits
        return logits, values

    def extract_features(
        self,
        obs: dict[str, torch.Tensor] | torch.Tensor,
        features_extractor: nn.Module | None = None,
        lstm_states: torch.Tensor | None = None,
        episode_starts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent, _ = self._forward_network(obs, lstm_states=lstm_states, episode_starts=episode_starts)
        return latent

    def forward(
        self,
        obs: PyTorchObs,
        *args,
        deterministic: bool = False,
        action_masks: np.ndarray | torch.Tensor | None = None,
        lstm_states: torch.Tensor | None = None,
        episode_starts: torch.Tensor | None = None,
        return_hidden_state: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(args) > 0:
            if isinstance(args[0], bool):
                deterministic = args[0]
                if len(args) > 1:
                    action_masks = args[1]
            elif isinstance(args[0], (torch.Tensor, tuple)):
                lstm_states = args[0]
                if len(args) > 1:
                    episode_starts = args[1]
                if len(args) > 2:
                    deterministic = args[2]
                if len(args) > 3:
                    action_masks = args[3]
                if return_hidden_state is None:
                    return_hidden_state = True

        if "action_masks" in kwargs:
            action_masks = kwargs["action_masks"]
        if "episode_starts" in kwargs:
            episode_starts = kwargs["episode_starts"]
        if "dones" in kwargs:
            episode_starts = kwargs["dones"]
        if "lstm_states" in kwargs:
            lstm_states = kwargs["lstm_states"]
        if "hidden_state" in kwargs:
            lstm_states = kwargs["hidden_state"]

        latent, next_hidden = self._forward_network(
            obs, lstm_states=lstm_states, episode_starts=episode_starts
        )
        values = self.value_net(latent)
        logits = self.action_net(latent)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))

        if return_hidden_state or (return_hidden_state is None and lstm_states is not None):
            return actions, values, log_prob, next_hidden
        return actions, values, log_prob

    def get_distribution(
        self,
        obs: PyTorchObs,
        action_masks: np.ndarray | torch.Tensor | None = None,
        lstm_states: torch.Tensor | None = None,
        episode_starts: torch.Tensor | None = None,
    ) -> MaskableDistribution:
        latent, _ = self._forward_network(obs, lstm_states=lstm_states, episode_starts=episode_starts)
        logits = self.action_net(latent)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def evaluate_actions(
        self,
        obs: PyTorchObs,
        actions: torch.Tensor,
        action_masks: torch.Tensor | None = None,
        lstm_states: torch.Tensor | None = None,
        episode_starts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, _ = self._forward_network(obs, lstm_states=lstm_states, episode_starts=episode_starts)
        values = self.value_net(latent)
        logits = self.action_net(latent)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        return values, log_prob, distribution.entropy()

    def predict_values(
        self,
        obs: PyTorchObs,
        lstm_states: torch.Tensor | None = None,
        episode_starts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent, _ = self._forward_network(obs, lstm_states=lstm_states, episode_starts=episode_starts)
        return self.value_net(latent)

    def _predict(
        self,
        observation: PyTorchObs,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> torch.Tensor:
        return self.get_distribution(observation, action_masks=action_masks).get_actions(
            deterministic=deterministic
        )

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | torch.Tensor | np.ndarray | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        self.set_training_mode(False)
        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        state_tensor = None
        if state is not None:
            if isinstance(state, np.ndarray):
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            elif isinstance(state, torch.Tensor):
                state_tensor = state.to(self.device)

        with torch.no_grad():
            actions, _, _, next_hidden = self.forward(
                obs_tensor,
                lstm_states=state_tensor,
                episode_starts=episode_start,
                action_masks=action_masks,
                deterministic=deterministic,
                return_hidden_state=True,
            )

        actions_np = actions.cpu().numpy().reshape((-1, *self.action_space.shape))
        if not vectorized_env:
            actions_np = actions_np.squeeze(axis=0)
            if actions_np.ndim == 0:
                actions_np = int(actions_np)

        next_state_np = next_hidden.cpu().numpy()
        return actions_np, next_state_np


class SpatialRescueFeaturesExtractor(BaseFeaturesExtractor):
    """Unpooled Spatial Feature Extractor for FlameBox:
    - Local: Conv(9, 32, 3, pad=1) -> ReLU -> Conv(32, 64, 3, pad=1) -> ReLU -> Conv(64, 8, 1) -> Flatten (1,800)
    - Memory (Zero MaxPool2d): Conv(9, 32, 3, pad=1) -> ReLU -> Conv(32, 32, 3, s=2, pad=1) -> ReLU -> Conv(32, 16, 1) -> Flatten (8,320)
    - Agent: 5 scalars
    - Previous Action: 7-dim one-hot
    - Fusion: Linear(10,132, 128) -> ReLU
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim=features_dim)

        local_channels = observation_space["grid"].shape[0] if "grid" in observation_space.spaces else 9
        local_h, local_w = observation_space["grid"].shape[1:] if "grid" in observation_space.spaces else (15, 15)
        mem_channels = observation_space["memory"].shape[0] if "memory" in observation_space.spaces else 9
        mem_h, mem_w = observation_space["memory"].shape[1:] if "memory" in observation_space.spaces else (40, 52)
        agent_dim = observation_space["agent"].shape[0] if "agent" in observation_space.spaces else 5
        action_dim = 7

        self.local_cnn = nn.Sequential(
            nn.Conv2d(local_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 8, kernel_size=1),
            nn.Flatten(),
        )
        local_features = 8 * local_h * local_w

        self.memory_cnn = nn.Sequential(
            nn.Conv2d(mem_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.Flatten(),
        )
        stride_h = (mem_h + 2 * 1 - 3) // 2 + 1
        stride_w = (mem_w + 2 * 1 - 3) // 2 + 1
        memory_features = 16 * stride_h * stride_w

        total_dim = local_features + memory_features + agent_dim + action_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, features_dim),
            nn.ReLU(),
        )

    def _encode_previous_action(self, prev_action, batch_size, device):
        if prev_action is None:
            return torch.zeros((batch_size, 7), dtype=torch.float32, device=device)
        if not isinstance(prev_action, torch.Tensor):
            prev_action = torch.as_tensor(prev_action, device=device)
        else:
            prev_action = prev_action.to(device)
        if prev_action.dim() >= 2 and prev_action.shape[-1] == 7:
            return prev_action.view(batch_size, 7).float()
        flat = prev_action.long().view(-1)
        if flat.numel() == 0:
            return torch.zeros((batch_size, 7), dtype=torch.float32, device=device)
        if flat.shape[0] != batch_size:
            flat = flat.expand(batch_size) if flat.numel() == 1 else flat[:batch_size]
        valid = (flat >= 0) & (flat < 7)
        safe = torch.where(valid, flat, torch.zeros_like(flat))
        one_hot = torch.nn.functional.one_hot(safe, num_classes=7).float() * valid.unsqueeze(-1).float()
        return one_hot

    def forward(self, observations):
        b = observations["grid"].shape[0]
        dev = observations["grid"].device
        loc = self.local_cnn(observations["grid"])
        mem = self.memory_cnn(observations["memory"])
        ag = observations["agent"].view(b, -1)
        prev = self._encode_previous_action(observations.get("previous_action", None), b, dev)
        combined = torch.cat([loc, mem, ag, prev], dim=-1)
        return self.fusion(combined)
