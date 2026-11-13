
import torch
from torch import nn

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


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
