import numpy as np


class MemoryMap:
    def __init__(
        self,
        height,
        width,
        channels=11,
        max_height=64,
        max_width=64,
    ):
        if height > max_height or width > max_width:
            raise ValueError(
                f"Building {height}x{width} exceeds "
                f"memory capacity {max_height}x{max_width}"
            )

        self.height = height
        self.width = width
        self.channels = channels
        self.max_height = max_height
        self.max_width = max_width

        self.reset()

    def reset(self):
        # Last observed values for each cell
        self.grid = np.zeros(
            (self.channels, self.height, self.width),
            dtype=np.float32
        )

        # 1 = previously observed, 0 = unknown
        self.explored = np.zeros(
            (self.height, self.width),
            dtype=np.float32
        )

    def update(self, local_grid, agent_position):
        agent_r, agent_c = agent_position

        # Firefighter is in the center of the local grid
        radius_r = local_grid.shape[1] // 2
        radius_c = local_grid.shape[2] // 2

        for r in range(local_grid.shape[1]):
            for c in range(local_grid.shape[2]):

                # Local coordinates -> global coordinates
                global_r = agent_r + r - radius_r
                global_c = agent_c + c - radius_c

                # Ignore cells outside the building
                if not (
                    0 <= global_r < self.height
                    and 0 <= global_c < self.width
                ):
                    continue

                # Channel 2 is visibility in our current grid
                if local_grid[2, r, c] <= 0:
                    continue

                # Store the latest observation
                self.grid[:, global_r, global_c] = local_grid[:, r, c]

                # Mark this cell as explored
                self.explored[global_r, global_c] = 1.0

    def get_observation(self):
        # Store remembered information in a fixed-size array
        padded = np.zeros(
            (self.channels + 2, self.max_height, self.max_width),
            dtype=np.float32,
        )

        # Copy memory channels into the top-left region
        padded[:self.channels, :self.height, :self.width] = self.grid

        # Explored mask: 1 = observed
        padded[self.channels, :self.height, :self.width] = self.explored

        # Building mask: 1 = actual building cell, 0 = padding
        padded[self.channels + 1, :self.height, :self.width] = 1.0

        return padded
