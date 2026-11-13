import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque

from rescue_env.flamebox.flamebox import Environment
from .memory_map import MemoryMap
from .reward_shaping import state_potential, potential_reward

class RescueEnv(gym.Env):
    """Gymnasium interface for one FlameBox firefighter."""

    ACTIONS = (0, 1, 2, 3, 4, 10, 5)
    SELECTED_CHANNELS = (2, 4, 5, 6, 8, 9, 10, 11, 13, 14)

    def __init__(self, max_steps=500, visibility_radius=7, victim_distance=3,
                 reward_shaping=False, fixed_seed=None, scenario_seeds=None,
                 min_victim_distance=1, navigation_shaping=False,
                 memory_agent_marker=False, exploration_bonus=0.0):
        super().__init__()

        self.max_steps = max_steps
        self.visibility_radius = visibility_radius
        self.victim_distance = victim_distance
        self.reward_shaping = reward_shaping
        self.navigation_shaping = navigation_shaping
        self.memory_agent_marker = memory_agent_marker
        if not np.isfinite(exploration_bonus) or exploration_bonus < 0:
            raise ValueError("exploration_bonus must be finite and nonnegative")
        self.exploration_bonus = exploration_bonus

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

        # Local observation: selected channels, square grid
        side = 2 * visibility_radius + 1

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(self.SELECTED_CHANNELS) + 1, side, side),
                dtype=np.float32,
            ),
            "memory": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(14 if memory_agent_marker else 13, 64, 64),
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
        mask[6] = True

        

        return mask

    def _get_observation(self):
        """Retrieve firefighter 0's local observation."""

        raw = self.sim.observe(
            agent_id=0,
            visibility_radius=self.visibility_radius,
        )

        exits = np.zeros_like(raw["grid"][0])
        grid = raw["grid"][list(self.SELECTED_CHANNELS)].astype(np.float32)
        person = self.sim.agents[0]
        for opening in self.sim.openings:
            if opening["kind"] != "exit":
                continue
            for row, col in self.sim._portal_cells(opening, opening["zone_a"]):
                r = row - person["r"] + self.visibility_radius
                c = col - person["c"] + self.visibility_radius
                if (0 <= r < exits.shape[0] and 0 <= c < exits.shape[1]
                        and raw["grid"][5, r, c] > 0):
                    exits[r, c] = 1
                    face = ((0 if opening["at"] == 0 else 2) if opening["orient"] == "H"
                            else (3 if opening["at"] == 0 else 1))
                    grid[4 + face, r, c] = bool(
                        opening["open"] and not opening.get("blocked") and not grid[1, r, c]
                    )

        carrying = any(
            v["carried_by"] == 0 and not v["dead"] and not v["rescued"]
            for v in self.sim.victims
        )

        grid = np.concatenate((grid, exits[None]), axis=0)
        self.victim_seen = self.victim_seen or bool((grid[3] > 0).any())
        agent = np.append(raw["agent"], carrying).astype(np.float32)
        

        self.memory.update(
            local_grid=grid,
            agent_position=(person["r"], person["c"]),
        )

        memory = self.memory.get_observation()
        if self.memory_agent_marker:
            marker = np.zeros((1, 64, 64), dtype=np.float32)
            r, c = person["r"], person["c"]
            # Grass positions stay outside the building map; agent coordinates remain available.
            if 0 <= r < self.sim.h and 0 <= c < self.sim.w:
                marker[0, r, c] = 1.0
            memory = np.concatenate((memory, marker), axis=0)

        return {
            "grid": grid,
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
        )
        self.sim._core.set_people(self.sim.victims, self.sim.agents[:1])
        self.sim.tick(0)
        self.victim_exit_distance = None
        self._place_victims(episode_rng)

        self.steps = 0
        self.rewarded_pickups = set()
        self.victim_seen = False

        self.memory = MemoryMap(
            height=self.sim.h,
            width=self.sim.w,
            channels=11,
        )

        observation = self._get_observation()
        info = {"seed": episode_seed, "scenario_seed": scenario_seed,
                "victim_exit_distance": self.victim_exit_distance}

        return observation, info

    
    def step(self, action):
        """Execute one action in the simulation."""

        if self.sim is None:
            raise RuntimeError("Call reset() before step().")

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        searching = not self.victim_seen
        explored_before = self.memory.explored.sum() if self.exploration_bonus else 0

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

        if self.reward_shaping:
            picked_up = {
                v["id"]
                for v in result["victims"]
                if v["carried_by"] == 0 or v["rescued"]
            }

            first_pickups = picked_up - self.rewarded_pickups
            self.rewarded_pickups.update(picked_up)

            reward += len(first_pickups) - 0.001

        # New optional navigation reward.
        if self.navigation_shaping:
            after = (
                0.0 if terminated
                else state_potential(self.sim)
            )

            reward += potential_reward(before, after)

        observation = self._get_observation()
        if self.exploration_bonus and searching:
            reward += float(self.exploration_bonus * (self.memory.explored.sum() - explored_before))

        rescued = sum(
            int(v["rescued"]) for v in result["victims"]
        )

        info = {
            "steps": self.steps,
            "victims_rescued": rescued,
            "success": rescued == len(result["victims"]),
            "is_success": rescued == len(result["victims"]),
            "timeout": truncated,
        }

        return observation, reward, terminated, truncated, info

    def close(self):
        """Release the simulator's native resources."""

        if self.sim is not None:
            self.sim._core.close()
            self.sim = None
