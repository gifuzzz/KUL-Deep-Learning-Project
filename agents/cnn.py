import json
import zipfile
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib.ppo_mask.policies import MaskableActorCriticPolicy
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.buffers import RNNStates
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


class RecurrentSpatialRescueCNN(RecurrentActorCriticPolicy):
    """Recurrent Spatial Rescue CNN Policy with explicit recurrent state propagation and BPTT.

    Architecture Variants:
      1. Variant A ('baseline'):
         - Local Branch: Conv2d(9, 32, 3, pad=1) -> ReLU -> Conv2d(32, 64, 3, pad=1) -> ReLU -> Conv2d(64, 8, 1) -> Flatten (1,800)
         - Memory Branch (Zero MaxPool2d): Conv2d(9, 32, 3, pad=1) -> ReLU -> Conv2d(32, 32, 3, stride=2, pad=1) -> ReLU -> Conv2d(32, 16, 1) -> Flatten (8,320)
         - Agent Scalars: 5 features; Previous Action: 7-dim one-hot vector
         - Fusion: Linear(10132, 128)
         - Recurrence: GRU(128, 128)
         - Output Heads: Linear(128, 7) policy head, Linear(128, 1) value head.
         - Total params: 1,431,168
      2. Variant B ('deep128'):
         - Local & Memory Branches: Identical to baseline
         - Fusion: Linear(10132, 256) -> LayerNorm(256) -> ReLU -> Linear(256, 128) -> ReLU
         - Recurrence: GRU(128, 128)
         - Output Heads: Linear(128, 7) policy head, Linear(128, 1) value head.
         - Total params: 2,761,600
      3. Variant C ('deep256'):
         - Local & Memory Branches: Identical to baseline
         - Fusion: Linear(10132, 512) -> LayerNorm(512) -> ReLU -> Linear(512, 256) -> ReLU
         - Recurrence: GRU(256, 256) (hidden_size=256)
         - Output Heads: Linear(256, 64) -> ReLU -> Linear(64, 7) policy head; Linear(256, 64) -> ReLU -> Linear(64, 1) value head.
         - Total params: 5,782,656
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space | None = None,
        lr_schedule: Schedule | None = None,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] | None = None,
        features_extractor_kwargs: dict[str, Any] | None = None,
        share_features_extractor: bool = True,
        normalize_images: bool = False,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        hidden_size: int = 128,
        lstm_hidden_size: int | None = None,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = False,
        lstm_kwargs: dict[str, Any] | None = None,
        architecture_variant: str = "baseline",
    ):
        if action_space is None:
            action_space = spaces.Discrete(7)
        if lr_schedule is None:
            lr_schedule = lambda _: 3e-4

        if architecture_variant not in ("baseline", "deep128", "deep256"):
            raise ValueError(
                f"Unknown architecture_variant: '{architecture_variant}'. "
                f"Must be one of ['baseline', 'deep128', 'deep256']"
            )
        self.architecture_variant = str(architecture_variant)

        if isinstance(observation_space, spaces.Dict):
            space_dict = dict(observation_space.spaces)
            if "previous_action" not in space_dict:
                space_dict["previous_action"] = spaces.Box(
                    low=0, high=6, shape=(), dtype=np.float32
                )
            observation_space = spaces.Dict(space_dict)

        if self.architecture_variant == "deep256":
            self.hidden_size = lstm_hidden_size or (256 if hidden_size == 128 else hidden_size)
        else:
            self.hidden_size = lstm_hidden_size or hidden_size
        self.lstm_output_dim = self.hidden_size
        self.lstm_hidden_state_shape = (1, 1, self.hidden_size)

        ActorCriticPolicy.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class or _DummyFeaturesExtractor,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    @property
    def total_params(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())

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

        # Fusion Branch based on architecture_variant:
        if self.architecture_variant == "baseline":
            # Variant A: Direct linear compression without non-linear depth
            self.fusion = nn.Linear(total_fusion_dim, self.hidden_size)
        elif self.architecture_variant == "deep128":
            # Variant B: 2-stage projection with LayerNorm and ReLU
            self.fusion = nn.Sequential(
                nn.Linear(total_fusion_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
            )
        elif self.architecture_variant == "deep256":
            # Variant C: 2-stage 512->256 projection with LayerNorm and ReLU
            self.fusion = nn.Sequential(
                nn.Linear(total_fusion_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
            )

        # Recurrence: GRU(hidden_size, hidden_size)
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            batch_first=True,
        )
        # Expose attributes for RecurrentPPO compatibility:
        self.lstm_actor = self.gru
        self.lstm_critic = None
        self.shared_lstm = True
        self.enable_critic_lstm = False
        self.lstm_hidden_state_shape = (1, 1, self.hidden_size)

        # Masked action distribution
        self.action_dist = make_masked_proba_distribution(self.action_space)

        # Output heads based on architecture_variant:
        if self.architecture_variant in ("baseline", "deep128"):
            self.action_net = nn.Linear(self.hidden_size, self.action_dim)
            self.value_net = nn.Linear(self.hidden_size, 1)
        elif self.architecture_variant == "deep256":
            self.action_net = nn.Sequential(
                nn.Linear(self.hidden_size, 64),
                nn.ReLU(),
                nn.Linear(64, self.action_dim),
            )
            self.value_net = nn.Sequential(
                nn.Linear(self.hidden_size, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        self.policy_head = self.action_net
        self.value_head = self.value_net

        if self.ortho_init:
            module_gains = {
                self.local_cnn: np.sqrt(2),
                self.memory_cnn: np.sqrt(2),
                self.fusion: np.sqrt(2),
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

            if isinstance(self.action_net, nn.Sequential):
                self.action_net[0].apply(partial(self.init_weights, gain=np.sqrt(2)))
                self.action_net[-1].apply(partial(self.init_weights, gain=0.01))
            else:
                self.action_net.apply(partial(self.init_weights, gain=0.01))

            if isinstance(self.value_net, nn.Sequential):
                self.value_net[0].apply(partial(self.init_weights, gain=np.sqrt(2)))
                self.value_net[-1].apply(partial(self.init_weights, gain=1.0))
            else:
                self.value_net.apply(partial(self.init_weights, gain=1.0))

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

    def _unpack_hidden_state(
        self,
        lstm_states: RNNStates | tuple[torch.Tensor, ...] | torch.Tensor | None,
        n_seq: int,
        device: torch.device,
    ) -> torch.Tensor:
        if lstm_states is None:
            return torch.zeros((1, n_seq, self.hidden_size), dtype=torch.float32, device=device)
        if isinstance(lstm_states, RNNStates):
            h = lstm_states.pi[0]
        elif isinstance(lstm_states, (tuple, list)):
            h = lstm_states[0]
        elif isinstance(lstm_states, torch.Tensor):
            h = lstm_states
        else:
            raise TypeError(f"Unexpected lstm_states type: {type(lstm_states)}")

        if h.dim() == 2:
            h = h.unsqueeze(0)
        if h.shape[1] != n_seq or h.shape[2] != self.hidden_size or h.device != device:
            # Shape adjustment if necessary
            if h.shape[1] == n_seq and h.shape[2] == self.hidden_size:
                h = h.to(device)
            else:
                h = torch.zeros((1, n_seq, self.hidden_size), dtype=torch.float32, device=device)
        return h

    def _pack_hidden_state(self, h: torch.Tensor) -> RNNStates:
        zeros = torch.zeros_like(h)
        return RNNStates(pi=(h, zeros), vf=(h, zeros))

    def _forward_network(
        self,
        obs: dict[str, torch.Tensor] | torch.Tensor,
        lstm_states: RNNStates | tuple[torch.Tensor, ...] | torch.Tensor | None = None,
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

        # Determine sequence shape:
        if lstm_states is not None:
            raw_h = (
                lstm_states.pi[0]
                if isinstance(lstm_states, RNNStates)
                else (lstm_states[0] if isinstance(lstm_states, (tuple, list)) else lstm_states)
            )
            if raw_h.dim() >= 2 and batch_size % raw_h.shape[1] == 0:
                n_seq = raw_h.shape[1]
            else:
                n_seq = batch_size
        else:
            n_seq = batch_size

        max_length = batch_size // n_seq
        h = self._unpack_hidden_state(lstm_states, n_seq, device)
        fused_sequence = fused.view(n_seq, max_length, self.hidden_size)

        if episode_starts is not None:
            starts = torch.as_tensor(episode_starts, dtype=torch.float32, device=device).view(n_seq, max_length)
        else:
            starts = torch.zeros((n_seq, max_length), dtype=torch.float32, device=device)

        if max_length == 1:
            # Single-step transition (e.g. rollout collection or step-by-step inference)
            h = (1.0 - starts[:, 0].view(1, n_seq, 1)) * h
            gru_out, next_h = self.gru(fused_sequence, h)
            gru_out = gru_out.view(batch_size, self.hidden_size)
        else:
            # Multi-step sequence unrolling with full BPTT gradient propagation
            if torch.all(starts[:, 1:] == 0.0):
                # Clean trajectory chunk: resets only at sequence start
                h = (1.0 - starts[:, 0].view(1, n_seq, 1)) * h
                gru_out, next_h = self.gru(fused_sequence, h)
                gru_out = gru_out.contiguous().view(batch_size, self.hidden_size)
            else:
                # Trajectory contains mid-sequence resets: unroll step-by-step
                gru_outputs = []
                for t in range(max_length):
                    step_reset = starts[:, t].view(1, n_seq, 1)
                    h = (1.0 - step_reset) * h
                    out_t, h = self.gru(fused_sequence[:, t : t + 1, :], h)
                    gru_outputs.append(out_t)
                gru_out = torch.cat(gru_outputs, dim=1).contiguous().view(batch_size, self.hidden_size)
                next_h = h

        return gru_out, next_h

    def reset_hidden_state(
        self,
        batch_size: int = 1,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros((1, batch_size, self.hidden_size), dtype=torch.float32, device=device)

    def get_masked_logits(
        self,
        obs: dict[str, torch.Tensor] | torch.Tensor,
        action_masks: torch.Tensor | np.ndarray | None = None,
        lstm_states: RNNStates | tuple[torch.Tensor, ...] | torch.Tensor | None = None,
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
        lstm_states: RNNStates | tuple[torch.Tensor, ...] | torch.Tensor | None = None,
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
        lstm_states: RNNStates | tuple[torch.Tensor, ...] | torch.Tensor | None = None,
        episode_starts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent, _ = self._forward_network(obs, lstm_states=lstm_states, episode_starts=episode_starts)
        return latent

    def forward(
        self,
        obs: PyTorchObs,
        *args,
        lstm_states: RNNStates | tuple[torch.Tensor, ...] | torch.Tensor | None = None,
        episode_starts: torch.Tensor | np.ndarray | None = None,
        deterministic: bool = False,
        action_masks: np.ndarray | torch.Tensor | None = None,
        return_hidden_state: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        if len(args) > 0:
            if isinstance(args[0], bool):
                deterministic = args[0]
                if len(args) > 1:
                    action_masks = args[1]
            elif isinstance(args[0], (torch.Tensor, tuple, RNNStates)):
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

        latent, next_h = self._forward_network(obs, lstm_states=lstm_states, episode_starts=episode_starts)
        values = self.value_net(latent)
        logits = self.action_net(latent)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))

        if return_hidden_state or (return_hidden_state is None and lstm_states is not None):
            if isinstance(lstm_states, torch.Tensor) and not isinstance(lstm_states, RNNStates):
                return actions, values, log_prob, next_h
            return actions, values, log_prob, self._pack_hidden_state(next_h)
        return actions, values, log_prob

    def get_distribution(
        self,
        obs: PyTorchObs,
        lstm_states: RNNStates | tuple[torch.Tensor, ...] | torch.Tensor | None = None,
        episode_starts: torch.Tensor | None = None,
        action_masks: np.ndarray | torch.Tensor | None = None,
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
        lstm_states: RNNStates | tuple[torch.Tensor, ...] | torch.Tensor | None = None,
        episode_starts: torch.Tensor | None = None,
        action_masks: torch.Tensor | None = None,
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
        lstm_states: RNNStates | tuple[torch.Tensor, ...] | torch.Tensor | None = None,
        episode_starts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent, _ = self._forward_network(obs, lstm_states=lstm_states, episode_starts=episode_starts)
        return self.value_net(latent)

    def _predict(
        self,
        observation: PyTorchObs,
        lstm_states: RNNStates | tuple[torch.Tensor, ...] | torch.Tensor | None = None,
        episode_starts: torch.Tensor | None = None,
        deterministic: bool = False,
        action_masks: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RNNStates]:
        distribution = self.get_distribution(
            observation,
            lstm_states=lstm_states,
            episode_starts=episode_starts,
            action_masks=action_masks,
        )
        actions = distribution.get_actions(deterministic=deterministic)
        _, next_h = self._forward_network(observation, lstm_states=lstm_states, episode_starts=episode_starts)
        return actions, self._pack_hidden_state(next_h)

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | np.ndarray | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        self.set_training_mode(False)
        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        state_tensor = None
        if state is not None:
            if isinstance(state, tuple):
                state_tensor = torch.as_tensor(state[0], dtype=torch.float32, device=self.device)
            elif isinstance(state, np.ndarray):
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            elif isinstance(state, torch.Tensor):
                state_tensor = state.to(self.device)

        episode_start_tensor = None
        if episode_start is not None:
            episode_start_tensor = torch.as_tensor(episode_start, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            actions, _, _, next_lstm_states = self.forward(
                obs_tensor,
                lstm_states=state_tensor,
                episode_starts=episode_start_tensor,
                deterministic=deterministic,
                action_masks=action_masks,
                return_hidden_state=True,
            )

        actions_np = actions.cpu().numpy().reshape((-1, *self.action_space.shape))
        if not vectorized_env:
            actions_np = actions_np.squeeze(axis=0)
            if actions_np.ndim == 0:
                actions_np = int(actions_np)

        if isinstance(next_lstm_states, RNNStates):
            next_state_np = next_lstm_states.pi[0].cpu().numpy()
        elif isinstance(next_lstm_states, (tuple, list)):
            next_state_np = next_lstm_states[0].cpu().numpy()
        else:
            next_state_np = next_lstm_states.cpu().numpy()

        return actions_np, next_state_np


# Register with MaskableActorCriticPolicy for sb3_contrib isinstance checks
MaskableActorCriticPolicy.register(RecurrentSpatialRescueCNN)


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


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    """Inspect checkpoint file and determine its architecture variant and weight shapes.

    Supports reading metadata.json, SB3 data payload, and policy.pth state dict.
    Returns:
        dict containing:
          - path: str
          - architecture_variant: str ('baseline', 'deep128', or 'deep256')
          - hidden_size: int (128 or 256)
          - metadata: dict
          - policy_kwargs: dict
          - total_params: int
          - state_dict: dict[str, torch.Tensor]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    metadata: dict[str, Any] = {}
    policy_kwargs: dict[str, Any] = {}
    state_dict: dict[str, torch.Tensor] = {}

    with zipfile.ZipFile(str(path), mode="r") as zf:
        namelist = zf.namelist()
        if "metadata.json" in namelist:
            try:
                metadata = json.loads(zf.read("metadata.json").decode("utf-8"))
            except Exception:
                pass
        if "data" in namelist:
            try:
                data = json.loads(zf.read("data").decode("utf-8"))
                policy_kwargs = data.get("policy_kwargs", {}) or {}
            except Exception:
                pass
        if "policy.pth" in namelist:
            with zf.open("policy.pth") as f:
                state_dict = torch.load(f, map_location="cpu", weights_only=False)

    # Deducing architecture variant:
    # 1. From metadata.json
    arch = metadata.get("architecture_variant", None)
    # 2. From policy_kwargs
    if arch is None:
        arch = policy_kwargs.get("architecture_variant", None)
    # 3. Deduce from tensor keys and shapes in policy.pth
    if arch is None and state_dict:
        if "fusion.0.weight" in state_dict:
            f0_shape = state_dict["fusion.0.weight"].shape
            if f0_shape[0] == 512:
                arch = "deep256"
            elif f0_shape[0] == 256:
                arch = "deep128"
        elif "fusion.weight" in state_dict:
            arch = "baseline"

    if arch is None:
        arch = "baseline"

    hidden_size = 256 if arch == "deep256" else 128
    alias_prefixes = ("lstm_actor.", "policy_head.", "value_head.")
    total_params = metadata.get("total_params")
    if total_params is None and state_dict:
        total_params = sum(
            p.numel()
            for k, p in state_dict.items()
            if not any(k.startswith(pfx) for pfx in alias_prefixes)
        )

    return {
        "path": str(path),
        "architecture_variant": str(arch),
        "hidden_size": hidden_size,
        "metadata": metadata,
        "policy_kwargs": policy_kwargs,
        "total_params": total_params,
        "state_dict": state_dict,
    }


def transfer_compatible_weights(
    source_model: Any,
    target_model: Any,
    compatible_layers: tuple[str, ...] = ("local_cnn", "memory_cnn"),
) -> dict[str, Any]:
    """Explicit utility function to transfer only identical layers between architecture variants.

    Transfers identical layers (such as 'local_cnn' and 'memory_cnn') from source to target
    while leaving differing layers (fusion, GRU, policy/value heads) untouched.
    Records warm-start tracking attributes on target_model.

    Args:
        source_model: Source model, policy, checkpoint path, or state dict.
        target_model: Target model or policy to receive the compatible weights.
        compatible_layers: Tuple of layer names to transfer. Default: ("local_cnn", "memory_cnn").

    Returns:
        dict with transfer summary, transferred parameters count, and status.
    """
    # Extract source state dict
    if isinstance(source_model, (str, Path)):
        ckpt_info = inspect_checkpoint(source_model)
        source_state = ckpt_info["state_dict"]
        source_arch = ckpt_info["architecture_variant"]
        source_desc = str(source_model)
    elif hasattr(source_model, "policy"):
        source_state = source_model.policy.state_dict()
        source_arch = getattr(source_model.policy, "architecture_variant", "unknown")
        source_desc = f"Model({source_arch})"
    elif isinstance(source_model, nn.Module):
        source_state = source_model.state_dict()
        source_arch = getattr(source_model, "architecture_variant", "unknown")
        source_desc = f"Policy({source_arch})"
    elif isinstance(source_model, dict):
        source_state = source_model
        source_arch = "state_dict"
        source_desc = "state_dict"
    else:
        raise TypeError(f"Unsupported source_model type: {type(source_model)}")

    # Extract target policy
    if hasattr(target_model, "policy"):
        target_policy = target_model.policy
        target_algo = target_model
    elif isinstance(target_model, nn.Module):
        target_policy = target_model
        target_algo = None
    else:
        raise TypeError(f"Unsupported target_model type: {type(target_model)}")

    target_arch = getattr(target_policy, "architecture_variant", "unknown")
    target_state = target_policy.state_dict()

    transferred_layers: list[str] = []
    transferred_params: int = 0
    skipped_layers: list[tuple[str, str]] = []

    for layer_name in compatible_layers:
        if not hasattr(target_policy, layer_name):
            skipped_layers.append((layer_name, "missing in target"))
            continue

        prefix = f"{layer_name}."
        layer_src_keys = [k for k in source_state.keys() if k.startswith(prefix)]
        if not layer_src_keys:
            skipped_layers.append((layer_name, "missing in source"))
            continue

        layer_params = 0
        all_matched = True
        for key in layer_src_keys:
            if key not in target_state:
                all_matched = False
                skipped_layers.append((key, "key missing in target"))
                break
            src_tensor = source_state[key]
            tgt_tensor = target_state[key]
            if src_tensor.shape != tgt_tensor.shape:
                all_matched = False
                skipped_layers.append((key, f"shape mismatch: {src_tensor.shape} vs {tgt_tensor.shape}"))
                break

        if all_matched:
            for key in layer_src_keys:
                src_tensor = source_state[key]
                target_state[key].copy_(
                    src_tensor.to(device=target_state[key].device, dtype=target_state[key].dtype)
                )
                layer_params += src_tensor.numel()
            transferred_layers.append(layer_name)
            transferred_params += layer_params

    is_warm = len(transferred_layers) > 0
    setattr(target_policy, "is_warm_started", is_warm)
    setattr(target_policy, "warm_start_source", source_desc)
    setattr(target_policy, "warm_start_transferred_layers", transferred_layers)
    if target_algo is not None:
        setattr(target_algo, "is_warm_started", is_warm)
        setattr(target_algo, "warm_start_source", source_desc)
        setattr(target_algo, "warm_start_transferred_layers", transferred_layers)

    return {
        "status": "success" if is_warm else "no_layers_transferred",
        "source_variant": source_arch,
        "target_variant": target_arch,
        "transferred_layers": transferred_layers,
        "transferred_params": transferred_params,
        "skipped_layers": skipped_layers,
        "is_warm_started": is_warm,
        "warm_start_source": source_desc,
    }

