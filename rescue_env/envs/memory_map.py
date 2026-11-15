import numpy as np


class MemoryMap:
    def __init__(
        self,
        height: int,
        width: int,
        channels: int = 9,
        max_height: int = 40,
        max_width: int = 52,
    ):
        if height > max_height or width > max_width:
            raise ValueError(
                f"Building {height}x{width} exceeds "
                f"memory capacity {max_height}x{max_width}"
            )

        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)
        self.max_height = int(max_height)
        self.max_width = int(max_width)

        self.pad_top = (self.max_height - self.height) // 2
        self.pad_left = (self.max_width - self.width) // 2

        self.reset()

    def reset(self):
        # Channels 0..8 stored for the building footprint (height, width)
        self.grid = np.zeros(
            (self.channels, self.height, self.width),
            dtype=np.float32,
        )

        # 1.0 = previously observed in line-of-sight, 0.0 = unknown
        self.explored = np.zeros(
            (self.height, self.width),
            dtype=np.float32,
        )

        self.last_agent_position = None

    def prefill_geometry(
        self,
        full_geometry: np.ndarray,
        revealed_mask: np.ndarray | None = None,
        victim_pos: tuple[int, int] | None = None,
    ):
        """Prefill known building geometry and/or victim position into memory.

        Channels 0, 1, 2, 3, 4, 7 are copied for revealed cells.
        If victim_pos is provided, channel 6 is populated at that location.
        CRITICAL: self.explored and self.grid[5] remain strictly 0.0, ensuring
        that prefilled geometry does not trigger false exploration rewards and
        the policy can distinguish between known blueprint and physically visited cells.
        """
        if revealed_mask is None:
            revealed_mask = np.ones((self.height, self.width), dtype=bool)

        for ch in (0, 1, 2, 3, 4, 7):
            if ch < self.channels and ch < full_geometry.shape[0]:
                self.grid[ch, revealed_mask] = full_geometry[ch, revealed_mask]

        if victim_pos is not None and self.channels > 6:
            vr, vc = victim_pos
            if 0 <= vr < self.height and 0 <= vc < self.width:
                self.grid[6, vr, vc] = 1.0

    def update(self, local_grid, agent_position):
        self.last_agent_position = agent_position
        agent_r, agent_c = agent_position

        radius_r = local_grid.shape[1] // 2
        radius_c = local_grid.shape[2] // 2

        r_min_local = max(0, radius_r - agent_r)
        r_max_local = min(local_grid.shape[1], radius_r + self.height - agent_r)
        c_min_local = max(0, radius_c - agent_c)
        c_max_local = min(local_grid.shape[2], radius_c + self.width - agent_c)

        if r_min_local < r_max_local and c_min_local < c_max_local:
            r_min_global = agent_r + r_min_local - radius_r
            r_max_global = agent_r + r_max_local - radius_r
            c_min_global = agent_c + c_min_local - radius_c
            c_max_global = agent_c + c_max_local - radius_c

            sub_local = local_grid[:, r_min_local:r_max_local, c_min_local:c_max_local]
            vis_ch = 5 if local_grid.shape[0] == 9 else 2
            vis_mask = sub_local[vis_ch] > 0

            if np.any(vis_mask):
                if local_grid.shape[0] == 9:
                    for ch in (0, 1, 2, 3, 4, 6, 7):
                        dest = self.grid[ch, r_min_global:r_max_global, c_min_global:c_max_global]
                        dest[vis_mask] = sub_local[ch][vis_mask]

                    self.explored[r_min_global:r_max_global, c_min_global:c_max_global][vis_mask] = 1.0
                    self.grid[5, r_min_global:r_max_global, c_min_global:c_max_global][vis_mask] = 1.0
                else:
                    for ch in range(min(self.channels, local_grid.shape[0])):
                        dest = self.grid[ch, r_min_global:r_max_global, c_min_global:c_max_global]
                        dest[vis_mask] = sub_local[ch][vis_mask]
                    self.explored[r_min_global:r_max_global, c_min_global:c_max_global][vis_mask] = 1.0

    def get_observation(self, agent_position=None):
        padded = np.zeros(
            (self.channels, self.max_height, self.max_width),
            dtype=np.float32,
        )

        pad_top = self.pad_top
        pad_left = self.pad_left

        num_copy = min(8, self.channels)
        padded[:num_copy, pad_top : pad_top + self.height, pad_left : pad_left + self.width] = self.grid[:num_copy]

        if self.channels > 5:
            padded[5, pad_top : pad_top + self.height, pad_left : pad_left + self.width] = self.explored

        pos = agent_position if agent_position is not None else self.last_agent_position
        if pos is not None and self.channels > 8:
            ar, ac = pos
            if 0 <= ar < self.height and 0 <= ac < self.width:
                padded[8, pad_top + ar, pad_left + ac] = 1.0

        return padded
