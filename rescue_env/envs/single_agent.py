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
                 fire_suppression_enabled=False,
                 coverage_target=0.90, exploration_cell_bonus=0.05,
                 exploration_edge_bonus=0.02, max_exploration_reward=6.0,
                 exploration_step_penalty=0.001, oscillation_penalty=0.02,
                 spatial_memory_mode="none", recency_tau=40.0,
                 exploration_reward_mode="normalized_continuous"):
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

        if str(task_type).lower() not in ("full", "return", "discovery", "exploration_only", "branching_exploration"):
            raise ValueError(f"Unknown task_type: '{task_type}'. Valid: 'full', 'return', 'discovery', 'exploration_only', 'branching_exploration'")
        self.task_type = str(task_type).lower()
        self.discovery_reward = float(discovery_reward)
        self._discovery_completed = False

        self.branching_stage = "B1"
        self.unvisited_cell_bonus = 0.15
        self.unvisited_room_bonus = 1.0
        self.branch_entry_bonus = 1.0
        self.wait_streak_penalty = 0.005
        self.oscillation_penalty = 0.02
        self.step_penalty = 0.001

        self.coverage_target = float(coverage_target)
        self.exploration_cell_bonus = float(exploration_cell_bonus)
        self.exploration_edge_bonus = float(exploration_edge_bonus)
        self.max_exploration_reward = float(max_exploration_reward)
        self.exploration_step_penalty = float(exploration_step_penalty)
        self.oscillation_penalty = float(oscillation_penalty)

        self.spatial_memory_mode = str(spatial_memory_mode).lower()
        if self.spatial_memory_mode not in ("none", "visited_binary", "visit_recency"):
            raise ValueError(
                f"Unknown spatial_memory_mode: '{spatial_memory_mode}'. "
                f"Valid: 'none', 'visited_binary', 'visit_recency'"
            )
        self.recency_tau = float(recency_tau)

        self.exploration_reward_mode = str(exploration_reward_mode).lower()
        if self.exploration_reward_mode not in ("normalized_continuous", "phase7_capped"):
            raise ValueError(
                f"Unknown exploration_reward_mode: '{exploration_reward_mode}'. "
                f"Valid: 'normalized_continuous', 'phase7_capped'"
            )

        self.memory_channels = (
            10 if self.spatial_memory_mode in ("visited_binary", "visit_recency")
            else self.NUM_CHANNELS
        )

        self.fixed_seed = fixed_seed
        self.min_victim_distance = min_victim_distance
        self.memory = None

        if self.task_type != "exploration_only":
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
                shape=(self.memory_channels, self.max_height, self.max_width),
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
                if hasattr(self.memory, "visited"):
                    self.memory.visited[r, c] = 1.0

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

    def _replay_trajectory(self, path_cells: list[tuple[int, int]]):
        """Physically replay a sequence of cells to legitimately observe and mark visited cells."""
        if not path_cells:
            return
        e = self.sim
        for idx, (r, c) in enumerate(path_cells):
            e.agents[0]["r"] = r
            e.agents[0]["c"] = c
            e._core.set_people(e.victims, e.agents[:1])
            e.tick(0)
            raw = e.observe(agent_id=0, visibility_radius=self.visibility_radius)
            local_grid = self._extract_local_grid(raw, e.agents[0])
            self.memory.update(
                local_grid=local_grid,
                agent_position=(r, c),
                step_idx=idx,
            )
            if (r, c) in self._reachable_interior_cells:
                self._visited_reachable_cells.add((r, c))
            zone = e.zone[r, c]
            if zone >= 0:
                self._visited_rooms.add(zone)

        # Place agent at final cell of replay
        last_r, last_c = path_cells[-1]
        e.agents[0]["r"] = last_r
        e.agents[0]["c"] = last_c
        e._core.set_people(e.victims, e.agents[:1])
        e.tick(0)

    def _setup_branching_task(
        self,
        branching_stage: str = "B1",
        swap_visited_branch: bool = False,
        rng=None,
    ):
        """Configure scenario with controlled branching complexity and legitimate physical visit history.

        Stages:
        - B1 (Single Junction): 2 traversable branches connected to a common junction. One branch
          pre-visited, one unvisited. Agent starts at junction.
        - B2 (Multiple Junctions): 2 sequential junctions. Path through J1 to J2 and dead-end visited.
          Agent placed at J1 (or J2) requiring backtracking / choice of unvisited branch.
        - B3 (Multi-Room Navigation): 1-2 visited rooms via connecting corridors. 1-2 unvisited rooms
          beyond doorways. Agent placed in visited corridor/room.
        - B4 (Full Procedural Building): Full unknown 30x46 layout, normal start at exit.
        """
        if rng is None:
            rng = np.random.default_rng()

        sim = self.sim
        h, w = sim.h, sim.w
        exit_opening = next((o for o in sim.openings if o["kind"] == "exit" and o["open"] and not o.get("blocked")), None)
        if exit_opening is None:
            return

        start = sim._cells_for_opening(exit_opening, exit_opening["zone_a"])

        def get_nbrs(r, c):
            nbrs = []
            if r > 0 and sim.south[r - 1, c] >= 0.5 and not sim.debris[r - 1, c]:
                nbrs.append(((r - 1, c), 1))
            if c + 1 < w and sim.east[r, c] >= 0.5 and not sim.debris[r, c + 1]:
                nbrs.append(((r, c + 1), 2))
            if r + 1 < h and sim.south[r, c] >= 0.5 and not sim.debris[r + 1, c]:
                nbrs.append(((r + 1, c), 3))
            if c > 0 and sim.east[r, c - 1] >= 0.5 and not sim.debris[r, c - 1]:
                nbrs.append(((r, c - 1), 4))
            return nbrs

        # BFS from start
        queue = deque([start])
        parent = {start: None}
        dist = {start: 0}
        ordered_cells = []
        while queue:
            curr = queue.popleft()
            ordered_cells.append(curr)
            for nbr, act in get_nbrs(*curr):
                if nbr not in parent:
                    parent[nbr] = curr
                    dist[nbr] = dist[curr] + 1
                    queue.append(nbr)

        stage_str = str(branching_stage).upper()
        self.branching_stage = stage_str
        self.branching_decision_made = False
        self.branching_correct_decision = None
        self.branching_entered_new_branch = False
        self.branching_visited_action = 0
        self.branching_unvisited_action = 0
        self.branching_stem_action = 0
        self.branching_junction = start
        self.branching_visited_cells = set()
        self.branching_unvisited_cells = set()
        self._first_new_territory_step = None
        self._new_cells_after_step_50 = 0
        self._junction_exited = False
        self._branch_entry_rewarded = False
        self._rewarded_rooms = set(self._visited_rooms)

        if stage_str == "B1":
            cand_junctions = []
            for cell, d in dist.items():
                if 2 <= d <= 12:
                    nbrs = get_nbrs(*cell)
                    p = parent[cell]
                    non_p = [n for n in nbrs if n[0] != p]
                    if len(non_p) >= 2:
                        cand_junctions.append((cell, non_p, p))
            if not cand_junctions:
                for cell, d in dist.items():
                    nbrs = get_nbrs(*cell)
                    p = parent[cell]
                    non_p = [n for n in nbrs if n[0] != p]
                    if len(non_p) >= 2:
                        cand_junctions.append((cell, non_p, p))
                        break

            if cand_junctions:
                j_idx = int(rng.integers(len(cand_junctions)))
                junction, non_p_nbrs, parent_cell = cand_junctions[j_idx]
                curr = junction
                stem = []
                while curr is not None:
                    stem.append(curr)
                    curr = parent[curr]
                stem.reverse()

                perm = rng.permutation(len(non_p_nbrs))
                idx_a = perm[1] if swap_visited_branch else perm[0]
                idx_b = perm[0] if swap_visited_branch else perm[1]

                b_a_start, act_visited = non_p_nbrs[idx_a]
                b_b_start, act_unvisited = non_p_nbrs[idx_b]

                branch_a_cells = [b_a_start]
                b_curr = b_a_start
                b_visited = set(stem) | {b_a_start}
                branch_len = int(rng.integers(2, 5))
                for _ in range(branch_len):
                    b_nbrs = [n[0] for n in get_nbrs(*b_curr) if n[0] not in b_visited]
                    if not b_nbrs:
                        break
                    nxt = b_nbrs[int(rng.integers(len(b_nbrs)))]
                    branch_a_cells.append(nxt)
                    b_visited.add(nxt)
                    b_curr = nxt

                branch_b_cells = [b_b_start]
                b_curr = b_b_start
                b_seen = set(stem) | set(branch_a_cells) | {b_b_start}
                for _ in range(branch_len):
                    b_nbrs = [n[0] for n in get_nbrs(*b_curr) if n[0] not in b_seen]
                    if not b_nbrs:
                        break
                    nxt = b_nbrs[int(rng.integers(len(b_nbrs)))]
                    branch_b_cells.append(nxt)
                    b_seen.add(nxt)
                    b_curr = nxt

                stem_act = next((act for nbr, act in get_nbrs(*junction) if nbr == parent_cell), 0)
                replay_path = list(stem) + list(branch_a_cells) + list(reversed(branch_a_cells[:-1])) + [junction]

                self.branching_junction = junction
                self.branching_visited_action = int(act_visited)
                self.branching_unvisited_action = int(act_unvisited)
                self.branching_stem_action = int(stem_act)
                self.branching_visited_cells = set(branch_a_cells)
                self.branching_unvisited_cells = set(branch_b_cells)
                self.max_steps = 64

                self._replay_trajectory(replay_path)

        elif stage_str == "B2":
            cand_b2 = []
            for cell, d in dist.items():
                if 2 <= d <= 10:
                    nbrs = get_nbrs(*cell)
                    p = parent[cell]
                    non_p = [n for n in nbrs if n[0] != p]
                    if len(non_p) >= 2:
                        for nbr1, act1 in non_p:
                            q2 = deque([(nbr1, [nbr1])])
                            seen2 = {cell, nbr1}
                            while q2:
                                c2, path2 = q2.popleft()
                                nbrs2 = [n for n in get_nbrs(*c2) if n[0] not in seen2]
                                if len(nbrs2) >= 2 and len(path2) >= 2:
                                    cand_b2.append((cell, non_p, p, c2, path2, nbrs2))
                                    break
                                for n2, a2 in nbrs2:
                                    seen2.add(n2)
                                    q2.append((n2, path2 + [n2]))
                                if len(cand_b2) > 5:
                                    break
                if len(cand_b2) > 5:
                    break

            if cand_b2:
                j_idx = int(rng.integers(len(cand_b2)))
                j1, non_p1, p1, j2, path_j1_j2, nbrs_j2 = cand_b2[j_idx]
                curr = j1
                stem = []
                while curr is not None:
                    stem.append(curr)
                    curr = parent[curr]
                stem.reverse()

                cand_b1 = [n for n in non_p1 if n[0] != path_j1_j2[0]]
                b1_nbr, act_unvisited = cand_b1[0] if cand_b1 else non_p1[0]
                b2_nbr, act_visited_j2 = nbrs_j2[0]

                branch_a2 = [b2_nbr]
                b_seen = set(stem) | set(path_j1_j2) | {j2, b2_nbr}
                b_curr = b2_nbr
                for _ in range(int(rng.integers(2, 4))):
                    nb = [n[0] for n in get_nbrs(*b_curr) if n[0] not in b_seen]
                    if not nb:
                        break
                    nxt = nb[int(rng.integers(len(nb)))]
                    branch_a2.append(nxt)
                    b_seen.add(nxt)
                    b_curr = nxt

                replay_path = (
                    list(stem)
                    + list(path_j1_j2)
                    + list(branch_a2)
                    + list(reversed(branch_a2[:-1]))
                    + [j2]
                    + list(reversed(path_j1_j2[:-1]))
                    + [j1]
                )
                stem_act = next((act for nbr, act in get_nbrs(*j1) if nbr == p1), 0)
                act_visited = next((act for nbr, act in get_nbrs(*j1) if nbr == path_j1_j2[0]), 0)

                self.branching_junction = j1
                self.branching_visited_action = int(act_visited)
                self.branching_unvisited_action = int(act_unvisited)
                self.branching_stem_action = int(stem_act)
                self.branching_visited_cells = set(path_j1_j2) | set(branch_a2)
                self.branching_unvisited_cells = {b1_nbr}
                self.max_steps = 128

                self._replay_trajectory(replay_path)
            else:
                self._setup_branching_task(branching_stage="B1", swap_visited_branch=swap_visited_branch, rng=rng)

        elif stage_str == "B3":
            # Multi-room navigation: traverse into 2nd room, leave remaining rooms unvisited
            visited_zones = set()
            path_b3 = []
            for cell in ordered_cells:
                z = sim.zone[cell]
                visited_zones.add(z)
                if len(visited_zones) >= 2 and dist[cell] >= 8:
                    curr = cell
                    while curr is not None:
                        path_b3.append(curr)
                        curr = parent[curr]
                    path_b3.reverse()
                    break

            if not path_b3:
                path_b3 = ordered_cells[:min(15, len(ordered_cells))]

            self.branching_junction = path_b3[-1]
            self.branching_visited_cells = set(path_b3)
            self.branching_unvisited_cells = set()
            self.max_steps = 192

            self._replay_trajectory(path_b3)

        else:  # B4
            # Full procedural building exploration
            self.branching_junction = start
            self.branching_visited_cells = {start}
            self.branching_unvisited_cells = set()
            self.max_steps = 384
            self._replay_trajectory([start])

        self._rewarded_rooms = set(self._visited_rooms)

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

        if self.task_type in ("exploration_only", "branching_exploration"):
            mask[5] = False
        else:
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

    def _extract_local_grid(self, raw, person):
        side = 2 * self.visibility_radius + 1
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
        return local_grid

    def _get_observation(self):
        """Retrieve firefighter 0's clean 9-channel local and memory observation."""

        raw = self.sim.observe(
            agent_id=0,
            visibility_radius=self.visibility_radius,
        )

        person = self.sim.agents[0]
        local_grid = self._extract_local_grid(raw, person)

        carrying = any(
            v["carried_by"] == 0 and not v["dead"] and not v["rescued"]
            for v in self.sim.victims
        )

        self.victim_seen = self.victim_seen or bool((local_grid[6] > 0).any())
        agent = np.append(raw["agent"], float(carrying)).astype(np.float32)

        self.memory.update(
            local_grid=local_grid,
            agent_position=(person["r"], person["c"]),
            step_idx=self.steps,
        )

        memory = self.memory.get_observation(
            agent_position=(person["r"], person["c"]),
            step_idx=self.steps,
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
            if "task_type" in options:
                tt = str(options["task_type"]).lower()
                if tt not in ("full", "return", "discovery", "exploration_only", "branching_exploration"):
                    raise ValueError(f"Unknown task_type: '{tt}'. Valid: 'full', 'return', 'discovery', 'exploration_only', 'branching_exploration'")
                self.task_type = tt
            if "coverage_target" in options:
                self.coverage_target = float(options["coverage_target"])
            if "max_exploration_reward" in options:
                self.max_exploration_reward = float(options["max_exploration_reward"])
            if "exploration_cell_bonus" in options:
                self.exploration_cell_bonus = float(options["exploration_cell_bonus"])
            if "exploration_edge_bonus" in options:
                self.exploration_edge_bonus = float(options["exploration_edge_bonus"])
            if "max_steps" in options:
                self.max_steps = int(options["max_steps"])

        # Generate a reproducible episode seed
        episode_seed = int(
            episode_rng.integers(0, 2**31 - 1)
        )

        nvictims = 0 if self.task_type in ("exploration_only", "branching_exploration") else 1
        self.sim = Environment(
            size=4,
            resolution=2,
            seed=episode_seed,
            victims=nvictims,
            ignitions=0,
            params={
                "initial_blocked_percent": 0.0,
                "opening_open_percent": 100.0,
            },
            geometry_mode=self.geometry_mode,
        )
        self.sim._core.set_people(self.sim.victims, self.sim.agents[:1])
        self.sim.agents = self.sim.agents[:1]
        self.sim.tick(0)
        self.victim_exit_distance = None
        if self.task_type not in ("exploration_only", "branching_exploration"):
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
            if "discovery_reward" in options:
                self.discovery_reward = float(options["discovery_reward"])
            if "return_mode" in options:
                self.return_mode = str(options["return_mode"]).lower()
            if "return_level" in options:
                self.return_level = str(options["return_level"]).upper()
            if "spatial_memory_mode" in options:
                self.spatial_memory_mode = str(options["spatial_memory_mode"]).lower()
                self.memory_channels = (
                    10 if self.spatial_memory_mode in ("visited_binary", "visit_recency")
                    else self.NUM_CHANNELS
                )
            if "recency_tau" in options:
                self.recency_tau = float(options["recency_tau"])
            if "exploration_reward_mode" in options:
                self.exploration_reward_mode = str(options["exploration_reward_mode"]).lower()

        self._discovery_completed = False

        self.memory = MemoryMap(
            height=self.sim.h,
            width=self.sim.w,
            channels=self.memory_channels,
            max_height=self.max_height,
            max_width=self.max_width,
            spatial_memory_mode=self.spatial_memory_mode,
            recency_tau=self.recency_tau,
        )

        self._compute_reachable_graph()
        self._observed_reachable_cells = set()
        self._visited_reachable_cells = set()
        self._discovered_edges = set()
        self._visited_rooms = set()
        self._revisit_count = 0
        self._oscillation_steps = 0
        self._current_wait_streak = 0
        self._max_consecutive_waits = 0
        self._wait_streaks_gt5 = 0
        self._wait_streaks_gt10 = 0
        self._wait_streaks_gt50 = 0
        self._doorway_crossings = 0
        self._doorway_reversals = 0
        self._zone_history = []
        self._pos_history = []
        self._coverage_at_steps = {}
        self._total_exploration_reward = 0.0
        self._exploration_completed = False
        self._action_counts = np.zeros(self.action_space.n, dtype=np.int32)

        if self.task_type == "return":
            r_level = options.get("return_level") if (options and "return_level" in options) else getattr(self, "return_level", None)
            r_mode = options.get("return_mode") if (options and "return_mode" in options) else getattr(self, "return_mode", "legacy")
            self._setup_return_task(return_level=r_level, return_mode=r_mode, rng=episode_rng)
        elif self.task_type == "branching_exploration":
            b_stage = options.get("branching_stage", getattr(self, "branching_stage", "B1"))
            swap_b = bool(options.get("swap_visited_branch", False))
            self._setup_branching_task(branching_stage=b_stage, swap_visited_branch=swap_b, rng=episode_rng)
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
        self._update_exploration_tracking()

        denom_cells = len(self._reachable_interior_cells) if self._reachable_interior_cells else 1
        denom_edges = len(self._reachable_traversable_edges) if self._reachable_traversable_edges else 1
        observed_coverage = len(self._observed_reachable_cells) / denom_cells
        visited_coverage = len(self._visited_reachable_cells) / denom_cells
        edge_coverage = len(self._discovered_edges) / denom_edges

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
            "observed_coverage": observed_coverage,
            "visited_coverage": visited_coverage,
            "edge_coverage": edge_coverage,
            "reachable_interior_cells": len(self._reachable_interior_cells),
            "observed_reachable_cells": len(self._observed_reachable_cells),
            "visited_reachable_cells": len(self._visited_reachable_cells),
            "reachable_edges": len(self._reachable_traversable_edges),
            "discovered_edges": len(self._discovered_edges),
            "reachable_rooms": len(self._reachable_rooms),
            "unique_rooms_entered": len(self._visited_rooms),
            "revisits": self._revisit_count,
            "doorway_crossings": self._doorway_crossings,
            "doorway_reversals": self._doorway_reversals,
            "max_consecutive_waits": self._max_consecutive_waits,
            "wait_streaks_gt5": self._wait_streaks_gt5,
            "wait_streaks_gt10": self._wait_streaks_gt10,
            "wait_streaks_gt50": self._wait_streaks_gt50,
            "wait_action_fraction": 0.0,
            "oscillation_percentage": 0.0,
            "cells_per_100_steps": 0.0,
            "coverage_at_50": None,
            "coverage_at_100": None,
            "coverage_at_200": None,
            "coverage_at_384": None,
            "exploration_success": False,
            "branching_stage": getattr(self, "branching_stage", None),
            "correct_branch_selected": getattr(self, "branching_correct_decision", None),
            "entered_new_branch": getattr(self, "branching_entered_new_branch", False),
            "junction_pos": getattr(self, "branching_junction", None),
            "unvisited_branch_action": getattr(self, "branching_unvisited_action", None),
            "visited_branch_action": getattr(self, "branching_visited_action", None),
            "stem_action": getattr(self, "branching_stem_action", None),
            "junction_exited": False,
            "time_to_first_new_territory": None,
            "new_cells_after_step_50": 0,
        }

        return observation, info

    def _compute_reachable_graph(self):
        """BFS flood-fill from firefighter entrance across traversable edges.

        Calculates ground-truth reachable interior cells, traversable edges, and rooms
        for coverage denominators and evaluation metrics. Never exposed to policy.
        """
        agent_spawn = (
            (self.sim.agents[0]["r"], self.sim.agents[0]["c"])
            if len(self.sim.agents) > 0 else (0, 0)
        )
        reachable_cells = set()
        reachable_interior = set()
        reachable_edges = set()
        reachable_rooms = set()

        queue = deque([agent_spawn])
        seen = {agent_spawn}

        while queue:
            r, c = queue.popleft()
            reachable_cells.add((r, c))
            z = int(self.sim.zone[r, c])
            reachable_interior.add((r, c))
            reachable_rooms.add(z)

            # North: (r-1, c) via south[r-1, c]
            if r > 0 and self.sim.south[r - 1, c] >= 0.5 and self.sim.debris[r - 1, c] == 0:
                edge = ((min(r - 1, r), c), (max(r - 1, r), c))
                reachable_edges.add(edge)
                if (r - 1, c) not in seen:
                    seen.add((r - 1, c))
                    queue.append((r - 1, c))

            # South: (r+1, c) via south[r, c]
            if r < self.sim.h - 1 and self.sim.south[r, c] >= 0.5 and self.sim.debris[r + 1, c] == 0:
                edge = ((min(r, r + 1), c), (max(r, r + 1), c))
                reachable_edges.add(edge)
                if (r + 1, c) not in seen:
                    seen.add((r + 1, c))
                    queue.append((r + 1, c))

            # West: (r, c-1) via east[r, c-1]
            if c > 0 and self.sim.east[r, c - 1] >= 0.5 and self.sim.debris[r, c - 1] == 0:
                edge = ((r, min(c - 1, c)), (r, max(c - 1, c)))
                reachable_edges.add(edge)
                if (r, c - 1) not in seen:
                    seen.add((r, c - 1))
                    queue.append((r, c - 1))

            # East: (r, c+1) via east[r, c]
            if c < self.sim.w - 1 and self.sim.east[r, c] >= 0.5 and self.sim.debris[r, c + 1] == 0:
                edge = ((r, min(c, c + 1)), (r, max(c, c + 1)))
                reachable_edges.add(edge)
                if (r, c + 1) not in seen:
                    seen.add((r, c + 1))
                    queue.append((r, c + 1))

        self._reachable_cells = reachable_cells
        self._reachable_interior_cells = reachable_interior
        self._unobserved_reachable_interior_cells = set(reachable_interior)
        self._reachable_traversable_edges = reachable_edges
        self._reachable_rooms = reachable_rooms

    def _update_exploration_tracking(self):
        agent_pos = (self.sim.agents[0]["r"], self.sim.agents[0]["c"])
        ar, ac = agent_pos
        current_zone = int(self.sim.zone[ar, ac]) if (0 <= ar < self.sim.h and 0 <= ac < self.sim.w) else -1

        # 2-cell oscillation detection
        self._pos_history.append(agent_pos)
        if len(self._pos_history) > 4:
            self._pos_history.pop(0)
        is_oscillating = (
            len(self._pos_history) == 4
            and self._pos_history[3] == self._pos_history[1]
            and self._pos_history[2] == self._pos_history[0]
            and self._pos_history[3] != self._pos_history[2]
        )
        if is_oscillating:
            self._oscillation_steps += 1

        # Doorway crossings and reversals
        if len(self._zone_history) >= 1:
            prev_zone = self._zone_history[-1]
            if current_zone != prev_zone and current_zone >= 0 and prev_zone >= 0:
                self._doorway_crossings += 1
                if len(self._zone_history) >= 2 and self._zone_history[-2] == current_zone:
                    self._doorway_reversals += 1
        self._zone_history.append(current_zone)
        if len(self._zone_history) > 10:
            self._zone_history.pop(0)

        # Visited cells and rooms
        newly_visited_cells = []
        if agent_pos in self._reachable_interior_cells:
            if agent_pos in self._visited_reachable_cells:
                self._revisit_count += 1
            else:
                self._visited_reachable_cells.add(agent_pos)
                newly_visited_cells.append(agent_pos)
        if current_zone >= 0:
            self._visited_rooms.add(current_zone)

        # Newly observed reachable interior cells
        newly_observed_cells = []
        just_observed = [
            cell for cell in self._unobserved_reachable_interior_cells
            if self.memory.explored[cell[0], cell[1]] > 0.0
        ]
        for cell in just_observed:
            self._unobserved_reachable_interior_cells.remove(cell)
            self._observed_reachable_cells.add(cell)
            newly_observed_cells.append(cell)

        # Newly discovered reachable edges
        newly_discovered_edges = []
        for r, c in newly_observed_cells:
            if r > 0 and self.sim.south[r - 1, c] >= 0.5:
                edge = ((min(r - 1, r), c), (max(r - 1, r), c))
                if edge in self._reachable_traversable_edges and edge not in self._discovered_edges:
                    self._discovered_edges.add(edge)
                    newly_discovered_edges.append(edge)
            if r < self.sim.h - 1 and self.sim.south[r, c] >= 0.5:
                edge = ((min(r, r + 1), c), (max(r, r + 1), c))
                if edge in self._reachable_traversable_edges and edge not in self._discovered_edges:
                    self._discovered_edges.add(edge)
                    newly_discovered_edges.append(edge)
            if c > 0 and self.sim.east[r, c - 1] >= 0.5:
                edge = ((r, min(c - 1, c)), (r, max(c - 1, c)))
                if edge in self._reachable_traversable_edges and edge not in self._discovered_edges:
                    self._discovered_edges.add(edge)
                    newly_discovered_edges.append(edge)
            if c < self.sim.w - 1 and self.sim.east[r, c] >= 0.5:
                edge = ((r, min(c, c + 1)), (r, max(c, c + 1)))
                if edge in self._reachable_traversable_edges and edge not in self._discovered_edges:
                    self._discovered_edges.add(edge)
                    newly_discovered_edges.append(edge)

        return newly_observed_cells, newly_discovered_edges, is_oscillating, newly_visited_cells

    def step(self, action):
        """Execute one action in the simulation."""

        if self.sim is None:
            raise RuntimeError("Call reset() before step().")

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # Action tracking and stationary Wait streaks
        int_action = int(action)
        self._action_counts[int_action] += 1
        if int_action == 0:
            self._current_wait_streak += 1
            if self._current_wait_streak > self._max_consecutive_waits:
                self._max_consecutive_waits = self._current_wait_streak
        else:
            if self._current_wait_streak >= 50:
                self._wait_streaks_gt50 += 1
            elif self._current_wait_streak >= 10:
                self._wait_streaks_gt10 += 1
            elif self._current_wait_streak >= 5:
                self._wait_streaks_gt5 += 1
            self._current_wait_streak = 0

        searching = not self.victim_seen
        explored_before = self.memory.explored.sum()

        # Potential before executing the action.
        before = (
            state_potential(self.sim)
            if (self.navigation_shaping and self.task_type != "exploration_only") else 0.0
        )

        command = self.ACTIONS[int_action]

        result = self.sim.action_batch(
            [command],
            physics_ticks=1,
        )

        self.steps += 1

        if self.task_type == "exploration_only":
            terminated = bool(self._exploration_completed)
        elif self.task_type == "branching_exploration":
            terminated = False
        else:
            terminated = bool(result["done"])
        truncated = self.steps >= self.max_steps and not terminated

        # Base rescue reward.
        reward = float(result["reward"]) if self.task_type not in ("exploration_only", "branching_exploration") else 0.0

        # Update observation and remembered memory
        observation = self._get_observation()
        newly_uncovered = max(0, int(self.memory.explored.sum() - explored_before))

        new_cells, new_edges, is_oscillating, new_visited = self._update_exploration_tracking()

        denom_cells = len(self._reachable_interior_cells) if self._reachable_interior_cells else 1
        denom_edges = len(self._reachable_traversable_edges) if self._reachable_traversable_edges else 1
        observed_coverage = len(self._observed_reachable_cells) / denom_cells
        visited_coverage = len(self._visited_reachable_cells) / denom_cells
        edge_coverage = len(self._discovered_edges) / denom_edges

        if self.steps in (50, 100, 200, 384):
            self._coverage_at_steps[self.steps] = observed_coverage

        # Check exploration task completion and rewards
        if self.task_type == "exploration_only":
            if observed_coverage >= self.coverage_target:
                self._exploration_completed = True
                terminated = True
                truncated = False

            if self.reward_shaping:
                if self.exploration_reward_mode == "normalized_continuous":
                    shaping = -self.exploration_step_penalty
                    pos_potential = (
                        (len(new_cells) / denom_cells) * 8.0
                        + (len(new_edges) / denom_edges) * 4.0
                        + (len(new_visited) / denom_cells) * 4.0
                    )
                    shaping += pos_potential
                    self._total_exploration_reward += pos_potential
                    if is_oscillating:
                        shaping -= self.oscillation_penalty
                    reward = float(shaping)
                else:  # "phase7_capped"
                    shaping = -self.exploration_step_penalty
                    pos_potential = (
                        len(new_cells) * self.exploration_cell_bonus
                        + len(new_edges) * self.exploration_edge_bonus
                    )
                    if pos_potential > 0.0 and self._total_exploration_reward < self.max_exploration_reward:
                        actual_pos = min(
                            pos_potential,
                            self.max_exploration_reward - self._total_exploration_reward,
                        )
                        self._total_exploration_reward += actual_pos
                        shaping += actual_pos
                    if is_oscillating:
                        shaping -= self.oscillation_penalty
                    reward = float(shaping)
            else:
                reward = 0.0
        elif self.task_type == "branching_exploration":
            agent_pos = (self.sim.agents[0]["r"], self.sim.agents[0]["c"])
            if not getattr(self, "branching_decision_made", False) and action in (1, 2, 3, 4):
                self.branching_decision_made = True
                if action == getattr(self, "branching_unvisited_action", 0):
                    self.branching_correct_decision = True
                else:
                    self.branching_correct_decision = False

            if agent_pos != getattr(self, "branching_junction", None):
                self._junction_exited = True

            if (not getattr(self, "branching_entered_new_branch", False)) and (agent_pos in getattr(self, "branching_unvisited_cells", set())):
                self.branching_entered_new_branch = True

            if len(new_visited) > 0 and getattr(self, "_first_new_territory_step", None) is None:
                self._first_new_territory_step = self.steps

            if self.steps > 50 and len(new_visited) > 0:
                self._new_cells_after_step_50 = getattr(self, "_new_cells_after_step_50", 0) + len(new_visited)

            # Branching exploration runs for the full configured step budget (or until all reachable cells explored)
            if len(self._visited_reachable_cells) >= len(self._reachable_interior_cells):
                terminated = True

            # Reward shaping during training
            if self.reward_shaping:
                shaping = -self.step_penalty
                if len(new_visited) > 0:
                    shaping += len(new_visited) * self.unvisited_cell_bonus
                if getattr(self, "branching_entered_new_branch", False) and not getattr(self, "_branch_entry_rewarded", False):
                    self._branch_entry_rewarded = True
                    shaping += self.branch_entry_bonus
                current_zone = self.sim.zone[agent_pos[0], agent_pos[1]] if (0 <= agent_pos[0] < self.sim.h and 0 <= agent_pos[1] < self.sim.w) else -1
                if current_zone >= 0 and current_zone not in getattr(self, "_rewarded_rooms", set()):
                    if not hasattr(self, "_rewarded_rooms"):
                        self._rewarded_rooms = set()
                    self._rewarded_rooms.add(current_zone)
                    shaping += self.unvisited_room_bonus
                if self._current_wait_streak > 2 and action == 0:
                    shaping -= self.wait_streak_penalty * (self._current_wait_streak - 2)
                if is_oscillating:
                    shaping -= self.oscillation_penalty
                reward = float(shaping)
            else:
                reward = 0.0
        elif self.task_type == "discovery":
            victim_sighted = bool((observation["grid"][6] > 0).any())
            if victim_sighted and not self._discovery_completed:
                self._discovery_completed = True
                reward += self.discovery_reward
                terminated = True

        if self.task_type not in ("exploration_only", "branching_exploration"):
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

        if terminated or truncated:
            if self._current_wait_streak >= 50:
                self._wait_streaks_gt50 += 1
            elif self._current_wait_streak >= 10:
                self._wait_streaks_gt10 += 1
            elif self._current_wait_streak >= 5:
                self._wait_streaks_gt5 += 1

        rescued = sum(
            int(v["rescued"]) for v in result["victims"]
        )

        if self.task_type == "exploration_only":
            task_success = self._exploration_completed
        elif self.task_type == "discovery":
            task_success = self._discovery_completed
        elif self.task_type == "branching_exploration":
            if self.branching_stage in ("B1", "B2"):
                task_success = bool(self.branching_entered_new_branch)
            else:
                task_success = len(self._visited_rooms) >= 2 and len(self._visited_reachable_cells) >= 15
        else:
            task_success = len(result["victims"]) > 0 and (rescued == len(result["victims"]))

        cells_per_100_steps = (len(self._observed_reachable_cells) / (self.steps / 100.0)) if self.steps > 0 else 0.0
        wait_action_fraction = (self._action_counts[0] / self.steps) if self.steps > 0 else 0.0
        oscillation_pct = ((self._oscillation_steps / self.steps) * 100.0) if self.steps > 0 else 0.0

        info = {
            "steps": self.steps,
            "victims_rescued": rescued,
            "success": task_success,
            "is_success": task_success,
            "timeout": truncated,
            "task_type": self.task_type,
            "discovery": self.victim_seen,
            "observed_coverage": observed_coverage,
            "visited_coverage": visited_coverage,
            "edge_coverage": edge_coverage,
            "reachable_interior_cells": len(self._reachable_interior_cells),
            "observed_reachable_cells": len(self._observed_reachable_cells),
            "visited_reachable_cells": len(self._visited_reachable_cells),
            "reachable_edges": len(self._reachable_traversable_edges),
            "discovered_edges": len(self._discovered_edges),
            "reachable_rooms": len(self._reachable_rooms),
            "unique_rooms_entered": len(self._visited_rooms),
            "revisits": self._revisit_count,
            "doorway_crossings": self._doorway_crossings,
            "doorway_reversals": self._doorway_reversals,
            "max_consecutive_waits": self._max_consecutive_waits,
            "wait_streaks_gt5": self._wait_streaks_gt5,
            "wait_streaks_gt10": self._wait_streaks_gt10,
            "wait_streaks_gt50": self._wait_streaks_gt50,
            "wait_action_fraction": wait_action_fraction,
            "oscillation_percentage": oscillation_pct,
            "cells_per_100_steps": cells_per_100_steps,
            "coverage_at_50": self._coverage_at_steps.get(50, observed_coverage if self.steps >= 50 else None),
            "coverage_at_100": self._coverage_at_steps.get(100, observed_coverage if self.steps >= 100 else None),
            "coverage_at_200": self._coverage_at_steps.get(200, observed_coverage if self.steps >= 200 else (observed_coverage if terminated else None)),
            "coverage_at_384": self._coverage_at_steps.get(384, observed_coverage if self.steps >= 384 else (observed_coverage if terminated else None)),
            "exploration_success": self._exploration_completed,
            "branching_stage": getattr(self, "branching_stage", None),
            "correct_branch_selected": getattr(self, "branching_correct_decision", None),
            "entered_new_branch": getattr(self, "branching_entered_new_branch", False),
            "junction_exited": getattr(self, "_junction_exited", False),
            "time_to_first_new_territory": getattr(self, "_first_new_territory_step", None),
            "new_cells_after_step_50": getattr(self, "_new_cells_after_step_50", 0),
            "branching_junction": getattr(self, "branching_junction", None),
            "unvisited_branch_action": getattr(self, "branching_unvisited_action", None),
            "visited_branch_action": getattr(self, "branching_visited_action", None),
            "stem_action": getattr(self, "branching_stem_action", None),
        }
        if self.task_type == "discovery":
            info["discovery_success"] = self._discovery_completed
        elif self.task_type == "return":
            info["return_success"] = len(result["victims"]) > 0 and (rescued == len(result["victims"]))
        elif self.task_type == "exploration_only":
            info["exploration_reward"] = self._total_exploration_reward
        elif self.task_type == "branching_exploration":
            info["branching_success"] = task_success

        return observation, reward, terminated, truncated, info

    def close(self):
        """Release the simulator's native resources."""

        if self.sim is not None:
            self.sim._core.close()
            self.sim = None
