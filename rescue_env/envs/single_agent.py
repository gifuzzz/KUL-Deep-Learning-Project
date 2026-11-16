import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque

from rescue_env.flamebox.flamebox import Environment
from .memory_map import MemoryMap
from .reward_shaping import state_potential, potential_reward, RewardShaper


def extract_full_building_geometry(sim) -> np.ndarray:
    """Extract ground-truth 9-channel building geometry from sim.

    Channels:
      Ch 0: Passability North (1.0 if face to North is passable / flux >= 0.5)
      Ch 1: Passability East (1.0 if face to East is passable / flux >= 0.5)
      Ch 2: Passability South (1.0 if face to South is passable / flux >= 0.5)
      Ch 3: Passability West (1.0 if face to West is passable / flux >= 0.5)
      Ch 4: Debris (1.0 if debris present)
      Ch 5: 0.0 (Explored mask remains unvisited)
      Ch 6: 0.0 (Victim populated separately if reveal_victim=True)
      Ch 7: Exit (1.0 at exit portal cells)
      Ch 8: 0.0 (Agent position populated during observation generation)
    """
    geom = np.zeros((9, sim.h, sim.w), dtype=np.float32)

    # Passability North (r > 0)
    if sim.h > 1:
        geom[0, 1:, :] = (sim.south >= 0.5).astype(np.float32)

    # Passability East (c+1 < w)
    if sim.w > 1:
        geom[1, :, :-1] = (sim.east >= 0.5).astype(np.float32)

    # Passability South (r+1 < h)
    if sim.h > 1:
        geom[2, :-1, :] = (sim.south >= 0.5).astype(np.float32)

    # Passability West (c > 0)
    if sim.w > 1:
        geom[3, :, 1:] = (sim.east >= 0.5).astype(np.float32)

    # Debris
    geom[4] = (sim.debris > 0).astype(np.float32)

    # Exit portals and exit doorway passability
    for opening in sim.openings:
        if opening["kind"] != "exit":
            continue
        for row, col in sim._portal_cells(opening, opening["zone_a"]):
            if 0 <= row < sim.h and 0 <= col < sim.w:
                geom[7, row, col] = 1.0
                face = (
                    (0 if opening["at"] == 0 else 2)
                    if opening["orient"] == "H"
                    else (3 if opening["at"] == 0 else 1)
                )
                is_passable = bool(
                    opening["open"]
                    and not opening.get("blocked")
                    and not (sim.debris[row, col] > 0)
                )
                geom[face, row, col] = 1.0 if is_passable else 0.0

    return geom


def get_connected_revealed_mask(sim, start_pos: tuple[int, int], fraction: float) -> np.ndarray:
    """Generate a contiguous connected mask of revealed cells via grid-BFS."""
    if fraction >= 1.0:
        return (sim.zone >= 0)
    if fraction <= 0.0:
        return np.zeros((sim.h, sim.w), dtype=bool)

    total_interior = int(np.count_nonzero(sim.zone >= 0))
    target_count = int(round(fraction * total_interior))
    if target_count <= 0:
        return np.zeros((sim.h, sim.w), dtype=bool)

    revealed = np.zeros((sim.h, sim.w), dtype=bool)
    sr, sc = start_pos
    if not (0 <= sr < sim.h and 0 <= sc < sim.w):
        return revealed

    queue = deque([(sr, sc)])
    revealed[sr, sc] = True
    count = 1

    while queue and count < target_count:
        r, c = queue.popleft()
        neighbors = (
            (r - 1, c, r > 0 and sim.south[r - 1, c] >= 0.5),
            (r, c + 1, c + 1 < sim.w and sim.east[r, c] >= 0.5),
            (r + 1, c, r + 1 < sim.h and sim.south[r, c] >= 0.5),
            (r, c - 1, c > 0 and sim.east[r, c - 1] >= 0.5),
        )
        for nr, nc, passable in neighbors:
            if passable and not revealed[nr, nc] and not sim.debris[nr, nc]:
                revealed[nr, nc] = True
                count += 1
                queue.append((nr, nc))
                if count >= target_count:
                    break

    return revealed


class RescueEnv(gym.Env):
    """Gymnasium interface for one FlameBox firefighter."""

    ACTIONS = (0, 1, 2, 3, 4, 10, 5)
    NUM_CHANNELS = 9
    MEMORY_HEIGHT = 40
    MEMORY_WIDTH = 52

    def __init__(self, max_steps=500, visibility_radius=7, victim_distance=3,
                 reward_shaping=False, fixed_seed=None, scenario_seeds=None,
                 min_victim_distance=1, navigation_shaping=False,
                 memory_agent_marker=False, exploration_bonus=0.0,
                 geometry_reveal_fraction=0.0, reveal_victim=False,
                 observability_stage="A", task_type="full", discovery_reward=2.0,
                 return_mode="legacy", return_level=None, distance_metric="exit",
                 prevent_exterior_wandering=True, geometry_mode="edge",
                 fire_suppression_enabled=False):
        super().__init__()

        self.max_steps = max_steps
        self.visibility_radius = visibility_radius
        self.victim_distance = victim_distance
        self.reward_shaping = reward_shaping
        self.navigation_shaping = navigation_shaping
        self.memory_agent_marker = memory_agent_marker
        self.return_mode = str(return_mode).lower()
        self.return_level = str(return_level).upper() if return_level is not None else None
        self.current_return_level = None
        self.current_return_mode = self.return_mode
        self.reward_shaper = RewardShaper() if reward_shaping else None
        if not np.isfinite(exploration_bonus) or exploration_bonus < 0:
            raise ValueError("exploration_bonus must be finite and nonnegative")
        self.exploration_bonus = exploration_bonus

        if str(distance_metric).lower() not in ("exit", "spawn"):
            raise ValueError(f"Unknown distance_metric: '{distance_metric}'. Valid: 'exit', 'spawn'")
        self.distance_metric = str(distance_metric).lower()
        self.prevent_exterior_wandering = bool(prevent_exterior_wandering)
        self.fire_suppression_enabled = bool(fire_suppression_enabled)
        if str(geometry_mode).lower() not in ("edge", "legacy_cell"):
            raise ValueError(f"Unknown geometry_mode: '{geometry_mode}'. Valid: 'edge', 'legacy_cell'")
        self.geometry_mode = str(geometry_mode).lower()

        self.geometry_reveal_fraction = float(geometry_reveal_fraction)
        self.reveal_victim = bool(reveal_victim)
        self.observability_stage = str(observability_stage)

        if str(task_type).lower() not in ("full", "return", "discovery"):
            raise ValueError(f"Unknown task_type: '{task_type}'. Valid: 'full', 'return', 'discovery'")
        self.task_type = str(task_type).lower()
        self.discovery_reward = float(discovery_reward)
        self._discovery_completed = False

        self.fixed_seed = fixed_seed
        self.min_victim_distance = min_victim_distance
        self.memory = None

        if (not isinstance(min_victim_distance, int) or min_victim_distance < 1
                or (victim_distance is not None and min_victim_distance > victim_distance)
                or (victim_distance is None and min_victim_distance > 1)):
            raise ValueError("Require 1 <= min_victim_distance <= victim_distance")
        self.scenario_seeds = tuple(scenario_seeds) if scenario_seeds is not None else None
        if self.scenario_seeds is not None:
            if fixed_seed is not None:
                raise ValueError("Choose fixed_seed or scenario_seeds, not both")
            if not self.scenario_seeds or any(
                not isinstance(seed, int) or seed < 0 for seed in self.scenario_seeds
            ):
                raise ValueError("scenario_seeds must contain nonnegative integer seeds")
        self._scenario_index = 0

        self.sim = None
        self.steps = 0

        # Seven possible actions
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        # Local observation: 9 channels, square grid
        side = 2 * visibility_radius + 1
        self.max_height = self.MEMORY_HEIGHT
        self.max_width = self.MEMORY_WIDTH

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.NUM_CHANNELS, side, side),
                dtype=np.float32,
            ),
            "memory": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.NUM_CHANNELS, self.max_height, self.max_width),
                dtype=np.float32,
            ),
            "agent": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(5,),
                dtype=np.float32,
            ),
        })

    def _place_victims(self, rng):
        """Place victims in the environment."""

        if self.victim_distance is None:
            return

        e = self.sim

        exit_ = next(
            (o for o in e.openings
             if o["kind"] == "exit"
             and o["open"]
             and not o.get("blocked")),
            None,
        )

        if exit_ is None:
            raise ValueError("No open exit found in the environment.")

        if self.distance_metric == "spawn":
            # True distance measured strictly from the firefighter's actual starting position
            agent_spawn = (
                (self.sim.agents[0]["r"], self.sim.agents[0]["c"])
                if len(self.sim.agents) > 0
                else e._cells_for_opening(exit_, exit_["zone_a"])
            )
            starts = [agent_spawn]
        else:
            # Legacy behavior: starts from all exit portal cells when min_victim_distance > 1
            start = e._cells_for_opening(exit_, exit_["zone_a"])
            starts = [start]
            if self.min_victim_distance > 1:
                starts = [cell for o in e.openings
                          if o["kind"] == "exit" and o["open"] and not o.get("blocked")
                          for cell in e._portal_cells(o, o["zone_a"])]
        seen = set(starts)
        queue = deque((cell, 0) for cell in starts)
        candidates = []
        while queue:
            (r, c), distance = queue.popleft()

            if (
                distance >= self.min_victim_distance
                and not e.debris[r, c]
                and e.fire[r, c] < 0.05
            ):
                candidates.append((r, c, distance))

            if distance >= self.victim_distance:
                continue

            neighbors = (
                (r-1, c, e.south[r-1, c] if r > 0 else 0),
                (r, c+1, e.east[r, c] if c+1 < e.w else 0),
                (r+1, c, e.south[r, c] if r+1 < e.h else 0),
                (r, c-1, e.east[r, c-1] if c > 0 else 0),
            )

            for nr, nc, conductance in neighbors:
                if (
                    0 <= nr < e.h
                    and 0 <= nc < e.w
                    and conductance >= 0.5
                    and (nr, nc) not in seen
                    and not e.debris[nr, nc]
                ):
                    seen.add((nr, nc))
                    queue.append(((nr, nc), distance + 1))

        if len(candidates) < len(e.victims):
            raise ValueError(
                "Not enough reachable cells for victims"
            )

        indices = rng.choice(
            len(candidates),
            size=len(e.victims),
            replace=False,
        )

        for victim, index in zip(e.victims, indices):
            victim["r"], victim["c"], self.victim_exit_distance = candidates[int(index)]

        # Synchronize with native C++ simulation
        e._core.set_people(e.victims, e.agents)
        e.tick(0)

    def _find_path_to_victim(self, start, target):
        """Find shortest traversable grid path from start to target via BFS."""
        e = self.sim
        if start == target:
            return [start]
        queue = deque([start])
        parent = {start: None}
        visited = {start}
        while queue:
            curr = queue.popleft()
            if curr == target:
                break
            r, c = curr
            neighbors = (
                (r - 1, c, e.south[r - 1, c] if r > 0 else 0),
                (r, c + 1, e.east[r, c] if c + 1 < e.w else 0),
                (r + 1, c, e.south[r, c] if r + 1 < e.h else 0),
                (r, c - 1, e.east[r, c - 1] if c > 0 else 0),
            )
            for nr, nc, cond in neighbors:
                if (
                    0 <= nr < e.h
                    and 0 <= nc < e.w
                    and cond >= 0.5
                    and not e.debris[nr, nc]
                    and (nr, nc) not in visited
                ):
                    visited.add((nr, nc))
                    parent[(nr, nc)] = curr
                    queue.append((nr, nc))

        if target not in parent:
            return [start, target]

        path = []
        curr = target
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        path.reverse()
        return path

    def _explore_side_branches(
        self,
        main_path: list[tuple[int, int]],
        num_branches: int = 2,
        branch_depth: int = 5,
        rng=None,
    ) -> set[tuple[int, int]]:
        """Sample realistic exploration side branches off the main path.

        Simulates legitimate prior exploration into side rooms/corridors that were
        checked and abandoned before reaching the victim. Every branch cell is physically
        passable and connected to the main path.
        """
        if num_branches <= 0 or len(main_path) <= 2:
            return set()

        sim = self.sim
        path_set = set(main_path)
        h, w = sim.h, sim.w
        branch_cells = set()

        candidate_forks = []
        for r, c in main_path[1:-1]:
            neighbors = (
                (r - 1, c, r > 0 and sim.south[r - 1, c] >= 0.5),
                (r, c + 1, c + 1 < w and sim.east[r, c] >= 0.5),
                (r + 1, c, r + 1 < h and sim.south[r, c] >= 0.5),
                (r, c - 1, c > 0 and sim.east[r, c - 1] >= 0.5),
            )
            for nr, nc, passable in neighbors:
                if passable and (nr, nc) not in path_set and not sim.debris[nr, nc]:
                    candidate_forks.append((nr, nc))

        if not candidate_forks:
            return branch_cells

        if rng is None:
            rng = np.random.default_rng()

        perm = rng.permutation(len(candidate_forks))[:num_branches]
        for idx in perm:
            fork_start = candidate_forks[idx]
            queue = deque([(fork_start[0], fork_start[1], 1)])
            b_visited = {fork_start}
            while queue:
                br, bc, depth = queue.popleft()
                branch_cells.add((br, bc))
                if depth >= branch_depth:
                    continue
                b_nbrs = (
                    (br - 1, bc, br > 0 and sim.south[br - 1, bc] >= 0.5),
                    (br, bc + 1, bc + 1 < w and sim.east[br, bc] >= 0.5),
                    (br + 1, bc, br + 1 < h and sim.south[br, bc] >= 0.5),
                    (br, bc - 1, bc > 0 and sim.east[br, bc - 1] >= 0.5),
                )
                for nbr_r, nbr_c, passable in b_nbrs:
                    if (
                        passable
                        and (nbr_r, nbr_c) not in path_set
                        and (nbr_r, nbr_c) not in b_visited
                        and not sim.debris[nbr_r, nbr_c]
                    ):
                        b_visited.add((nbr_r, nbr_c))
                        queue.append((nbr_r, nbr_c, depth + 1))

        return branch_cells

    def _populate_path_memory(self, path, exit_opening, victim_pos, additional_cells=None):
        """Populate memory with ground-truth observations along traversed corridors and explored branches.

        Only the path from the exit to the victim, any legitimately explored side branches,
        their adjacent visible open cells, the exit portal, and the victim are marked explored.
        Unvisited rooms remain strictly unknown (0.0).
        """
        e = self.sim
        # Mark exit opening portal cells
        for er, ec in e._portal_cells(exit_opening, exit_opening["zone_a"]):
            if 0 <= er < self.memory.height and 0 <= ec < self.memory.width:
                self.memory.grid[7, er, ec] = 1.0
                self.memory.grid[5, er, ec] = 1.0
                self.memory.explored[er, ec] = 1.0
                face = (
                    (0 if exit_opening["at"] == 0 else 2)
                    if exit_opening["orient"] == "H"
                    else (3 if exit_opening["at"] == 0 else 1)
                )
                self.memory.grid[face, er, ec] = 1.0

        all_explored_cells = list(path)
        if additional_cells:
            all_explored_cells.extend(additional_cells)

        for r, c in all_explored_cells:
            if 0 <= r < self.memory.height and 0 <= c < self.memory.width:
                self.memory.grid[0, r, c] = float(e.south[r - 1, c] >= 0.5 if r > 0 else 0)
                self.memory.grid[1, r, c] = float(e.east[r, c] >= 0.5 if c + 1 < e.w else 0)
                self.memory.grid[2, r, c] = float(e.south[r, c] >= 0.5 if r + 1 < e.h else 0)
                self.memory.grid[3, r, c] = float(e.east[r, c - 1] >= 0.5 if c > 0 else 0)
                self.memory.grid[4, r, c] = float(e.debris[r, c] > 0)
                self.memory.grid[5, r, c] = 1.0
                self.memory.explored[r, c] = 1.0

                # Also observe immediate adjacent open cells visible from path
                neighbors = (
                    (r - 1, c, e.south[r - 1, c] if r > 0 else 0),
                    (r, c + 1, e.east[r, c] if c + 1 < e.w else 0),
                    (r + 1, c, e.south[r, c] if r + 1 < e.h else 0),
                    (r, c - 1, e.east[r, c - 1] if c > 0 else 0),
                )
                for nr, nc, cond in neighbors:
                    if 0 <= nr < self.memory.height and 0 <= nc < self.memory.width and cond >= 0.5:
                        self.memory.grid[0, nr, nc] = float(e.south[nr - 1, nc] >= 0.5 if nr > 0 else 0)
                        self.memory.grid[1, nr, nc] = float(e.east[nr, nc] >= 0.5 if nc + 1 < e.w else 0)
                        self.memory.grid[2, nr, nc] = float(e.south[nr, nc] >= 0.5 if nr + 1 < e.h else 0)
                        self.memory.grid[3, nr, nc] = float(e.east[nr, nc - 1] >= 0.5 if nc > 0 else 0)
                        self.memory.grid[4, nr, nc] = float(e.debris[nr, nc] > 0)
                        self.memory.grid[5, nr, nc] = 1.0
                        self.memory.explored[nr, nc] = 1.0

        vr, vc = victim_pos
        if 0 <= vr < self.memory.height and 0 <= vc < self.memory.width:
            self.memory.grid[6, vr, vc] = 1.0

    def _setup_return_task(
        self,
        return_level: str | None = None,
        return_mode: str = "legacy",
        rng=None,
    ):
        """Configure scenario where agent is already carrying victim deep inside the building,
        with remembered map physically consistent with path traveled from an exit portal.

        Supports Route-Complexity levels:
        - R1: Short, simple return (d=2..4, 0 branches, direct corridor)
        - R2: Longer corridor return (d=6..8, 1 short branch, 2-3 turns, doorways)
        - R3: Branching multi-room return (d=10..12, 2 branches, doorways & intersections)
        - R4: Realistic distant pickup states (d>=12/distant, 2-3 deep branches, multi-room)
        - legacy: Phase 3 direct single-corridor return
        """
        if len(self.sim.victims) == 0:
            return

        victim = self.sim.victims[0]
        vr, vc = int(victim["r"]), int(victim["c"])

        # Find exit opening
        exit_opening = next(
            (o for o in self.sim.openings if o["kind"] == "exit" and o["open"] and not o.get("blocked")),
            None,
        )
        if exit_opening is None:
            return

        start_cell = self.sim._cells_for_opening(exit_opening, exit_opening["zone_a"])

        # Find shortest traversable path from exit portal to victim cell
        path = self._find_path_to_victim(start_cell, (vr, vc))

        # Position agent at victim position
        self.sim.agents[0]["r"] = vr
        self.sim.agents[0]["c"] = vc

        # Attach victim as carried by agent 0
        victim["r"] = vr
        victim["c"] = vc
        victim["carried_by"] = 0

        # Sync people with C++ core
        self.sim._core.set_people(self.sim.victims, self.sim.agents[:1])
        self.sim.tick(0)

        # Mark pickup as already rewarded so agent does not get unearned pickup shaping
        self.rewarded_pickups.add(0)
        if self.reward_shaper is not None:
            self.reward_shaper.rewarded_pickups.add(0)

        # Route-complexity configuration
        num_branches = 0
        branch_depth = 0
        effective_level = return_level

        if return_mode == "route_complexity":
            if effective_level is None:
                plen = len(path)
                if plen <= 4:
                    effective_level = "R1"
                elif plen <= 8:
                    effective_level = "R2"
                elif plen <= 12:
                    effective_level = "R3"
                else:
                    effective_level = "R4"

            if effective_level == "R1":
                num_branches = 0
                branch_depth = 0
            elif effective_level == "R2":
                num_branches = 1
                branch_depth = 4
            elif effective_level == "R3":
                num_branches = 2
                branch_depth = 6
            elif effective_level == "R4":
                num_branches = 3
                branch_depth = 8
        else:
            effective_level = "legacy"

        self.current_return_level = effective_level
        self.current_return_mode = return_mode

        branch_cells = set()
        if num_branches > 0:
            branch_cells = self._explore_side_branches(
                path,
                num_branches=num_branches,
                branch_depth=branch_depth,
                rng=rng,
            )

        # Populate memory map with physical observations along traveled path + branches
        self._populate_path_memory(path, exit_opening, (vr, vc), additional_cells=branch_cells)

    def action_masks(self):
        """Valid actions in Gymnasium action order; True means available."""
        if self.sim is None:
            raise RuntimeError("Call reset() before action_masks().")

        person = self.sim.agents[0]
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[0] = True
        if person["dead"]:
            return mask

        raw = self.sim.observe(agent_id=0, visibility_radius=self.visibility_radius)
        mask[1:5] = raw["actions_mask"][1:5]
        nearby = [
            v for v in self.sim.victims
            if not v["dead"] and not v["rescued"]
            and abs(v["r"] - person["r"]) + abs(v["c"] - person["c"]) <= 1
        ]
        carrying = any(
            v["carried_by"] == person["id"]
            and not v["dead"]
            and not v["rescued"]
            for v in self.sim.victims
        )

        mask[5] = (
            not carrying
            and any(v["carried_by"] < 0 for v in nearby)
        )
        fire_active = (
            getattr(self, "fire_suppression_enabled", False)
            or getattr(self.sim, "ignitions", 0) > 0
            or (hasattr(self.sim, "fire") and bool((self.sim.fire > 0).any()))
        )
        mask[6] = bool(fire_active)

        if self.prevent_exterior_wandering and not carrying:
            pr, pc = person["r"], person["c"]
            deltas = ((-1, 0), (0, 1), (1, 0), (0, -1))
            for action_idx, (dr, dc) in enumerate(deltas, start=1):
                if mask[action_idx]:
                    nr, nc = pr + dr, pc + dc
                    is_exterior = (
                        nr < 0
                        or nr >= self.sim.h
                        or nc < 0
                        or nc >= self.sim.w
                        or (self.geometry_mode == "legacy_cell" and self.sim.zone[nr, nc] < 0)
                    )
                    if is_exterior:
                        mask[action_idx] = False

        return mask

    def _get_observation(self):
        """Retrieve firefighter 0's clean 9-channel local and memory observation."""

        raw = self.sim.observe(
            agent_id=0,
            visibility_radius=self.visibility_radius,
        )

        side = 2 * self.visibility_radius + 1
        person = self.sim.agents[0]

        # 9 channels:
        # Ch 0: Passability North (raw[8] >= 0.5)
        # Ch 1: Passability East (raw[9] >= 0.5)
        # Ch 2: Passability South (raw[10] >= 0.5)
        # Ch 3: Passability West (raw[11] >= 0.5)
        # Ch 4: Debris / Obstacle (raw[4] > 0)
        # Ch 5: Explored Mask / Line-of-sight (raw[5] > 0)
        # Ch 6: Victim (raw[6])
        # Ch 7: Exit (1.0 if exit portal cell visible)
        # Ch 8: Agent Position (1.0 at agent local center)
        local_grid = np.zeros((self.NUM_CHANNELS, side, side), dtype=np.float32)
        local_grid[0] = (raw["grid"][8] >= 0.5).astype(np.float32)
        local_grid[1] = (raw["grid"][9] >= 0.5).astype(np.float32)
        local_grid[2] = (raw["grid"][10] >= 0.5).astype(np.float32)
        local_grid[3] = (raw["grid"][11] >= 0.5).astype(np.float32)
        local_grid[4] = (raw["grid"][4] > 0.0).astype(np.float32)
        local_grid[5] = (raw["grid"][5] > 0.0).astype(np.float32)
        local_grid[6] = raw["grid"][6].astype(np.float32)
        local_grid[8, self.visibility_radius, self.visibility_radius] = 1.0

        for opening in self.sim.openings:
            if opening["kind"] != "exit":
                continue
            for row, col in self.sim._portal_cells(opening, opening["zone_a"]):
                r = row - person["r"] + self.visibility_radius
                c = col - person["c"] + self.visibility_radius
                if (0 <= r < side and 0 <= c < side and raw["grid"][5, r, c] > 0):
                    local_grid[7, r, c] = 1.0
                    face = ((0 if opening["at"] == 0 else 2) if opening["orient"] == "H"
                            else (3 if opening["at"] == 0 else 1))
                    is_passable = bool(
                        opening["open"] and not opening.get("blocked") and not (local_grid[4, r, c] > 0)
                    )
                    local_grid[face, r, c] = 1.0 if is_passable else 0.0

        carrying = any(
            v["carried_by"] == 0 and not v["dead"] and not v["rescued"]
            for v in self.sim.victims
        )

        self.victim_seen = self.victim_seen or bool((local_grid[6] > 0).any())
        agent = np.append(raw["agent"], float(carrying)).astype(np.float32)

        self.memory.update(
            local_grid=local_grid,
            agent_position=(person["r"], person["c"]),
        )

        memory = self.memory.get_observation(
            agent_position=(person["r"], person["c"]),
        )

        return {
            "grid": local_grid,
            "memory": memory,
            "agent": agent,
        }

    def reset(self, *, seed=None, options=None):
        """Start a new episode."""

        super().reset(seed=seed)
        scenario_seed = self.fixed_seed
        if self.scenario_seeds is not None:
            if seed is not None:
                self._scenario_index = 0
            scenario_seed = self.scenario_seeds[self._scenario_index]
            self._scenario_index = (self._scenario_index + 1) % len(self.scenario_seeds)
        # A separate RNG repeats both placement and map without replacing Gym's seeded RNG.
        episode_rng = (self.np_random if scenario_seed is None
                       else np.random.default_rng(scenario_seed))

        # Close the previous simulator before creating another
        self.close()

        if options:
            if "distance_metric" in options:
                dm = str(options["distance_metric"]).lower()
                if dm not in ("exit", "spawn"):
                    raise ValueError(f"Unknown distance_metric: '{dm}'. Valid: 'exit', 'spawn'")
                self.distance_metric = dm
            if "prevent_exterior_wandering" in options:
                self.prevent_exterior_wandering = bool(options["prevent_exterior_wandering"])
            if "geometry_mode" in options:
                gm = str(options["geometry_mode"]).lower()
                if gm not in ("edge", "legacy_cell"):
                    raise ValueError(f"Unknown geometry_mode: '{gm}'. Valid: 'edge', 'legacy_cell'")
                self.geometry_mode = gm
            if "fire_suppression_enabled" in options:
                self.fire_suppression_enabled = bool(options["fire_suppression_enabled"])

        # Generate a reproducible episode seed
        episode_seed = int(
            episode_rng.integers(0, 2**31 - 1)
        )

        self.sim = Environment(
            size=4,
            resolution=2,
            seed=episode_seed,
            victims=1,
            ignitions=0,
            params={
                "initial_blocked_percent": 0.0,
                "opening_open_percent": 100.0,
            },
            geometry_mode=self.geometry_mode,
        )
        self.sim._core.set_people(self.sim.victims, self.sim.agents[:1])
        self.sim.tick(0)
        self.victim_exit_distance = None
        self._place_victims(episode_rng)

        self.steps = 0
        self.rewarded_pickups = set()
        self.victim_seen = False
        if self.reward_shaper is not None:
            self.reward_shaper.reset()

        if options:
            if "geometry_reveal_fraction" in options:
                self.geometry_reveal_fraction = float(options["geometry_reveal_fraction"])
            if "reveal_victim" in options:
                self.reveal_victim = bool(options["reveal_victim"])
            if "observability_stage" in options:
                self.observability_stage = str(options["observability_stage"])
            if "task_type" in options:
                self.task_type = str(options["task_type"]).lower()
            if "discovery_reward" in options:
                self.discovery_reward = float(options["discovery_reward"])
            if "return_mode" in options:
                self.return_mode = str(options["return_mode"]).lower()
            if "return_level" in options:
                self.return_level = str(options["return_level"]).upper()

        self._discovery_completed = False

        self.memory = MemoryMap(
            height=self.sim.h,
            width=self.sim.w,
            channels=self.NUM_CHANNELS,
            max_height=self.max_height,
            max_width=self.max_width,
        )

        if self.task_type == "return":
            r_level = options.get("return_level") if (options and "return_level" in options) else getattr(self, "return_level", None)
            r_mode = options.get("return_mode") if (options and "return_mode" in options) else getattr(self, "return_mode", "legacy")
            self._setup_return_task(return_level=r_level, return_mode=r_mode, rng=episode_rng)
        elif self.geometry_reveal_fraction > 0.0 or self.reveal_victim:
            # Prefill privileged geometry or victim location if enabled by curriculum
            full_geom = extract_full_building_geometry(self.sim)
            agent_pos = (self.sim.agents[0]["r"], self.sim.agents[0]["c"])
            revealed_mask = get_connected_revealed_mask(
                self.sim,
                start_pos=agent_pos,
                fraction=self.geometry_reveal_fraction,
            )
            victim_pos = (
                (self.sim.victims[0]["r"], self.sim.victims[0]["c"])
                if (self.reveal_victim and len(self.sim.victims) > 0)
                else None
            )
            self.memory.prefill_geometry(
                full_geometry=full_geom,
                revealed_mask=revealed_mask,
                victim_pos=victim_pos,
            )

        observation = self._get_observation()
        info = {
            "seed": episode_seed,
            "scenario_seed": scenario_seed,
            "victim_exit_distance": self.victim_exit_distance,
            "geometry_reveal_fraction": self.geometry_reveal_fraction,
            "reveal_victim": self.reveal_victim,
            "observability_stage": self.observability_stage,
            "task_type": self.task_type,
            "return_mode": self.current_return_mode if self.task_type == "return" else None,
            "return_level": self.current_return_level if self.task_type == "return" else None,
        }

        return observation, info

    
    def step(self, action):
        """Execute one action in the simulation."""

        if self.sim is None:
            raise RuntimeError("Call reset() before step().")

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        searching = not self.victim_seen
        explored_before = self.memory.explored.sum()

        # Potential before executing the action.
        before = (
            state_potential(self.sim)
            if self.navigation_shaping else 0.0
        )

        command = self.ACTIONS[int(action)]

        result = self.sim.action_batch(
            [command],
            physics_ticks=1,
        )

        self.steps += 1

        terminated = bool(result["done"])
        truncated = self.steps >= self.max_steps and not terminated

        # Base rescue reward.
        reward = float(result["reward"])

        # Update observation and remembered memory
        observation = self._get_observation()
        newly_uncovered = max(0, int(self.memory.explored.sum() - explored_before))

        # Check auxiliary discovery task termination and reward
        if self.task_type == "discovery":
            victim_sighted = bool((observation["grid"][6] > 0).any())
            if victim_sighted and not self._discovery_completed:
                self._discovery_completed = True
                reward += self.discovery_reward
                terminated = True

        if self.reward_shaping:
            if self.reward_shaper is None:
                self.reward_shaper = RewardShaper()
            reward += self.reward_shaper.compute_reward(
                result["victims"],
                newly_uncovered_cells=newly_uncovered,
                current_position=(self.sim.agents[0]["r"], self.sim.agents[0]["c"]),
            )

        # New optional navigation reward.
        if self.navigation_shaping:
            after = (
                0.0 if terminated
                else state_potential(self.sim)
            )

            reward += potential_reward(before, after)

        if self.exploration_bonus and searching:
            reward += float(self.exploration_bonus * newly_uncovered)

        rescued = sum(
            int(v["rescued"]) for v in result["victims"]
        )

        task_success = (rescued == len(result["victims"])) or (
            self.task_type == "discovery" and self._discovery_completed
        )

        info = {
            "steps": self.steps,
            "victims_rescued": rescued,
            "success": task_success,
            "is_success": task_success,
            "timeout": truncated,
            "task_type": self.task_type,
            "discovery": self.victim_seen,
        }
        if self.task_type == "discovery":
            info["discovery_success"] = self._discovery_completed
        elif self.task_type == "return":
            info["return_success"] = (rescued == len(result["victims"]))

        return observation, reward, terminated, truncated, info

    def close(self):
        """Release the simulator's native resources."""

        if self.sim is not None:
            self.sim._core.close()
            self.sim = None
