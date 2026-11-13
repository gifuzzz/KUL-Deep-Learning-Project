"""Navigation memory built from visible cells and the firefighter's own position.

This is an explicit planner, not a learned policy or a source of hidden targets.
"""
from collections import Counter, defaultdict, deque

import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


class ObservedMap:
    def __init__(self, height, width):
        self.height, self.width = height, width
        self.known = set()
        self.edges = defaultdict(set)
        self.victims = set()
        self.exits = set()
        self.visits = Counter()
        self.waypoint = (0, 0)

    def observe(self, grid, position, carrying):
        radius = grid.shape[-1] // 2
        self.visits[position] += 1
        for r, c in np.argwhere(grid[2] > 0):
            cell = (position[0] + int(r) - radius, position[1] + int(c) - radius)
            self.known.add(cell)
            if grid[3, r, c] > 0 and not (carrying and cell == position):
                self.victims.add(cell)
            else:
                self.victims.discard(cell)
            for face, (dr, dc) in enumerate(DIRECTIONS):
                neighbor = (cell[0] + dr, cell[1] + dc)
                if grid[4 + face, r, c] >= 0.5 and not grid[1, r, c]:
                    self.edges[cell].add(neighbor)
                    self.edges[neighbor].add(cell)
                    if grid[-1, r, c] and not (
                        0 <= neighbor[0] < self.height and 0 <= neighbor[1] < self.width
                    ):
                        self.exits.add(neighbor)
                else:
                    self.edges[cell].discard(neighbor)
                    self.edges[neighbor].discard(cell)

        parents = {position: None}
        distances = {position: 0}
        queue = deque([position])
        while queue:
            cell = queue.popleft()
            for neighbor in sorted(self.edges[cell]):
                if neighbor not in parents:
                    parents[neighbor] = cell
                    distances[neighbor] = distances[cell] + 1
                    queue.append(neighbor)

        targets = (self.exits if carrying else self.victims) & parents.keys()
        kind = -1 if carrying else 1
        if not targets:
            kind = 0
            targets = parents.keys() - self.known - self.exits - {position}
        if not targets:
            targets = parents.keys() - self.exits - {position}
            goal = min(targets, key=lambda p: (self.visits[p], distances[p], p)) if targets else position
        else:
            goal = min(targets, key=lambda p: (distances[p], self.visits[p], p))
        path = [goal]
        while path[-1] != position:
            path.append(parents[path[-1]])
        self.waypoint = path[-2] if len(path) > 1 else position
        return np.array([self.waypoint[0] - position[0],
                         self.waypoint[1] - position[1], kind], np.float32)
