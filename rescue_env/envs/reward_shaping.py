def shortest_distance(sim, start, targets):
    """Training-only distance on the same interior/grass graph as native movement."""
    distances = []
    for target in set(targets):
        path = sim.pathfind(start, target, world=True)
        if path:
            distances.append(len(path) - 1)
    return min(distances, default=None)


def state_potential(sim, max_distance=64):
    """Higher potential means closer to the current rescue goal."""

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

    carrying = any(v["carried_by"] == agent["id"] for v in victims)

    if carrying:
        targets = set()

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
                targets.update((r + dr, c + dc) for r, c in
                               sim._portal_cells(opening, opening["zone_a"]))

    else:
        # Pickup is possible on or next to a victim.
        targets = set()

        for victim in victims:
            if victim["carried_by"] >= 0:
                continue

            vr, vc = victim["r"], victim["c"]

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

    distance = shortest_distance(sim, position, targets)

    if distance is None:
        distance = max_distance

    return -float(min(distance, max_distance))


def potential_reward(before, after, gamma=0.99, beta=0.1):
    return beta * (gamma * after - before)
