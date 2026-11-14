def shortest_distance(sim, start, targets):
    """Training-only distance on the same interior/grass graph as native movement."""
    distances = []
    for target in set(targets):
        path = sim.pathfind(start, target, world=True)
        if path:
            distances.append(len(path) - 1)
    return min(distances, default=None)


def state_potential(sim, max_distance=128):
    """Higher potential means closer to completing the rescue task.

    Continuous Potential Formulation (Policy-Invariance Preserving):
    - When carrying victim: distance from agent to exit portal
    - When searching (not carrying): distance from agent to victim + distance from victim to exit portal

    At the moment of pickup, distance(agent, victim) = 0, so:
    potential_before = -(0 + distance(victim, exit)) = -distance(victim, exit)
    potential_after  = -distance(agent, exit) = -distance(victim, exit)
    This guarantees mathematical continuity across the objective switch, eliminating
    the artificial penalty drops that previously punished distant pickups.
    """
    agent = sim.agents[0]
    position = (agent["r"], agent["c"])

    if agent["dead"]:
        return 0.0

    victims = [
        v for v in sim.victims
        if not v["dead"] and not v["rescued"]
    ]

    if not victims:
        return 0.0

    exit_targets = set()
    for opening in sim.openings:
        if (
            opening["kind"] == "exit"
            and opening["open"]
            and not opening.get("blocked", False)
        ):
            # Rescue occurs on the first grass cell, not the interior doorway.
            if opening["orient"] == "H":
                dr, dc = (-1, 0) if opening["at"] == 0 else (1, 0)
            else:
                dr, dc = (0, -1) if opening["at"] == 0 else (0, 1)
            exit_targets.update((r + dr, c + dc) for r, c in
                               sim._portal_cells(opening, opening["zone_a"]))

    carrying = any(v["carried_by"] == agent["id"] for v in victims)

    if carrying:
        distance = shortest_distance(sim, position, exit_targets)
        if distance is None:
            distance = max_distance
        return -float(min(distance, max_distance))
    else:
        total_distances = []
        for victim in victims:
            if victim["carried_by"] >= 0:
                continue

            vr, vc = victim["r"], victim["c"]
            targets = set()
            for r, c in (
                (vr, vc),
                (vr - 1, vc),
                (vr + 1, vc),
                (vr, vc - 1),
                (vr, vc + 1),
            ):
                inside = 0 <= r < sim.h and 0 <= c < sim.w
                if not inside or not sim.debris[r, c]:
                    targets.add((r, c))

            d_agent_v = shortest_distance(sim, position, targets)
            d_v_exit = shortest_distance(sim, (vr, vc), exit_targets)
            if d_agent_v is not None and d_v_exit is not None:
                total_distances.append(d_agent_v + d_v_exit)
            elif d_agent_v is not None:
                total_distances.append(d_agent_v + max_distance // 2)

        if not total_distances:
            distance = max_distance
        else:
            distance = min(total_distances)

        return -float(min(distance, max_distance))


continuous_state_potential = state_potential


def potential_reward(before, after, gamma=0.99, beta=0.1):
    return beta * (gamma * after - before)


class RewardShaper:
    """Training-only reward shaping for single-agent rescue.

    Components:
    - Step penalty: -0.001 per step
    - First victim pickup: +1.5 per victim
    - Frontier exploration: +0.05 per newly uncovered cell in memory_map (capped at +3.0 per episode)
    """

    def __init__(
        self,
        step_penalty: float = 0.001,
        pickup_reward: float = 1.5,
        exploration_bonus: float = 0.05,
        max_exploration_reward: float = 3.0,
    ):
        self.step_penalty = float(step_penalty)
        self.pickup_reward = float(pickup_reward)
        self.exploration_bonus = float(exploration_bonus)
        self.max_exploration_reward = float(max_exploration_reward)
        self.rewarded_pickups = set()
        self.total_exploration_reward = 0.0

    def reset(self):
        self.rewarded_pickups.clear()
        self.total_exploration_reward = 0.0

    def compute_reward(self, victims, newly_uncovered_cells: int = 0) -> float:
        shaping = 0.0

        # Step penalty
        shaping -= self.step_penalty

        # First victim pickup
        picked_up = {
            v["id"]
            for v in victims
            if v["carried_by"] == 0 or v["rescued"]
        }
        first_pickups = picked_up - self.rewarded_pickups
        if first_pickups:
            self.rewarded_pickups.update(first_pickups)
            shaping += self.pickup_reward * len(first_pickups)

        # Frontier exploration reward (+0.05/cell, capped at +3.0/episode)
        is_carrying = any(v["carried_by"] == 0 for v in victims)
        if not is_carrying and newly_uncovered_cells > 0 and self.total_exploration_reward < self.max_exploration_reward:
            bonus = newly_uncovered_cells * self.exploration_bonus
            remaining = self.max_exploration_reward - self.total_exploration_reward
            actual_bonus = min(bonus, remaining)
            self.total_exploration_reward += actual_bonus
            shaping += actual_bonus

        return float(shaping)

