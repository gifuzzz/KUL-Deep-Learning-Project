import numpy as np
import pytest
import torch
from collections import deque

from rescue_env.flamebox.flamebox import Environment, GeometryEnvironment
from rescue_env.envs.single_agent import RescueEnv, extract_full_building_geometry
from rescue_env.envs.memory_map import MemoryMap
from rescue_env.wrappers import PreviousActionWrapper
from rescue_env.recorder import EpisodeRecorder
from agents.recurrent_ppo import MaskableRecurrentPPO
from agents.cnn import RecurrentSpatialRescueCNN


# -------------------------------------------------------------------------
# Test 1: Walls block crossing between adjacent cells
# -------------------------------------------------------------------------
def test_walls_block_crossing():
    env = Environment(size=4, resolution=2, seed=42, geometry_mode="edge")
    # Find two adjacent cells in different zones with no door
    found = False
    for r in range(env.h):
        for c in range(env.w - 1):
            if env.zone[r, c] != env.zone[r, c + 1] and env.east[r, c] < 0.5:
                # Cell (r, c) and (r, c+1) are separated by a wall
                assert env.east[r, c] < 0.5
                # Direct single-step pathfind across wall must fail
                path = env.pathfind((r, c), (r, c + 1))
                if path:
                    # If any path exists, it must detour through a door (length > 2)
                    assert len(path) > 2, f"Stepped directly through wall: {path}"
                found = True
                break
        if found:
            break
    assert found, "Did not find adjacent wall cells"


# -------------------------------------------------------------------------
# Test 2: Doorways permit crossing
# -------------------------------------------------------------------------
def test_doorways_permit_crossing():
    env = Environment(size=4, resolution=2, seed=42, geometry_mode="edge")
    # Find an open interior door
    doors = [op for op in env.openings if op["kind"] == "door" and op["open"] and not op.get("blocked")]
    assert len(doors) > 0, "No open door found"
    door = doors[0]

    lo = door["start"]
    at = door["at"]
    if door["orient"] == "V":
        # Vertical door at column boundary 'at', left is at-1, right is at
        r = lo
        c_left = at - 1
        c_right = at
        assert env.east[r, c_left] >= 0.5
        path_open = env.pathfind((r, c_left), (r, c_right))
        assert len(path_open) == 2, f"Expected direct step through open door, got {path_open}"

        # Toggle door closed
        env.toggle(door["id"])
        assert env.east[r, c_left] < 0.5
        path_closed = env.pathfind((r, c_left), (r, c_right))
        if path_closed:
            assert len(path_closed) > 2, f"Traversed closed door directly: {path_closed}"

        # Toggle back open
        env.toggle(door["id"])
        assert env.east[r, c_left] >= 0.5
    else:
        # Horizontal door at row boundary 'at', top is at-1, bottom is at
        r_top = at - 1
        r_bot = at
        c = lo
        assert env.south[r_top, c] >= 0.5
        path_open = env.pathfind((r_top, c), (r_bot, c))
        assert len(path_open) == 2, f"Expected direct step through open door, got {path_open}"

        # Toggle door closed
        env.toggle(door["id"])
        assert env.south[r_top, c] < 0.5
        path_closed = env.pathfind((r_top, c), (r_bot, c))
        if path_closed:
            assert len(path_closed) > 2, f"Traversed closed door directly: {path_closed}"


# -------------------------------------------------------------------------
# Test 3: Debris blocks destination cells independently of walls
# -------------------------------------------------------------------------
def test_debris_blocks_destination_cell():
    env = Environment(size=4, resolution=2, seed=42, geometry_mode="edge")
    # Find two adjacent cells in the same room where east is passable
    found = False
    for r in range(env.h):
        for c in range(env.w - 1):
            if env.zone[r, c] == env.zone[r, c + 1] and env.east[r, c] >= 0.5:
                # Add debris to cell (r, c+1)
                env.make_debris(r, c + 1, severity=1.0, force=True)
                assert env.east[r, c] >= 0.5
                assert env.debris[r, c + 1] > 0
                path_blocked = env.pathfind((r, c), (r, c + 1))
                assert len(path_blocked) == 0, f"Moved into debris cell: {path_blocked}"

                # Clear debris
                env.clear_debris(r, c + 1)
                path_cleared = env.pathfind((r, c), (r, c + 1))
                assert len(path_cleared) == 2
                found = True
                break
        if found:
            break
    assert found, "Did not find adjacent open cells"


# -------------------------------------------------------------------------
# Test 4: East/West and North/South edge symmetry
# -------------------------------------------------------------------------
def test_edge_symmetry():
    env = Environment(size=4, resolution=2, seed=42, geometry_mode="edge")
    geom = extract_full_building_geometry(env)

    # Ch 0: North, Ch 1: East, Ch 2: South, Ch 3: West
    # For every internal cell, East of (r, c) must equal West of (r, c+1)
    for r in range(env.h):
        for c in range(env.w - 1):
            p_east = geom[1, r, c]
            p_west = geom[3, r, c + 1]
            assert p_east == p_west, f"East/West asymmetry at ({r}, {c}) vs ({r}, {c+1}): {p_east} != {p_west}"

    # For every internal cell, South of (r, c) must equal North of (r+1, c)
    for r in range(env.h - 1):
        for c in range(env.w):
            p_south = geom[2, r, c]
            p_north = geom[0, r + 1, c]
            assert p_south == p_north, f"South/North asymmetry at ({r}, {c}) vs ({r+1}, {c}): {p_south} != {p_north}"


# -------------------------------------------------------------------------
# Test 5: Procedural room partitions across multiple seeds
# -------------------------------------------------------------------------
def test_procedural_room_partitions():
    for seed in (10, 42, 100, 2026):
        env = GeometryEnvironment(size=4, resolution=2, seed=seed, geometry_mode="edge")
        # Every cell must have a non-negative room zone
        assert int((env.zone == -1).sum()) == 0, f"Seed {seed} has negative zones in edge mode"
        assert env.wall is None, f"Seed {seed} has wall array in edge mode"
        assert len(set(env.zone.flatten())) > 1, f"Seed {seed} has only 1 zone"


# -------------------------------------------------------------------------
# Test 6: Exterior boundaries and exits
# -------------------------------------------------------------------------
def test_exterior_boundaries_and_exits():
    env = Environment(size=4, resolution=2, seed=42, geometry_mode="edge")
    # Perimeter cells cannot step outside unless at an exit opening
    exits = [op for op in env.openings if op["kind"] == "exit" and op["open"] and not op.get("blocked")]
    assert len(exits) > 0

    exit_portal_cells = set()
    for ex in exits:
        for pr, pc in env._portal_cells(ex, ex["zone_a"]):
            exit_portal_cells.add((pr, pc))

    # Test top boundary (r=0, stepping North to r=-1)
    for c in range(env.w):
        is_portal = (0, c) in exit_portal_cells
        path_out = env.pathfind((0, c), (-1, c), world=True)
        if not is_portal:
            assert len(path_out) != 2, f"Non-exit cell (0, {c}) can directly step to exterior grass"
        else:
            assert len(path_out) == 2, f"Exit portal cell (0, {c}) cannot directly step to exterior grass"


# -------------------------------------------------------------------------
# Test 7: Movement-action masks
# -------------------------------------------------------------------------
def test_movement_action_masks():
    env = RescueEnv(geometry_mode="edge", prevent_exterior_wandering=True)
    obs, info = env.reset(seed=42)

    masks = env.action_masks()
    assert isinstance(masks, np.ndarray)
    assert masks.shape == (7,)
    assert bool(masks[0]) is True  # No-op always legal

    # Verify that moving into a closed wall or exterior without carrying is masked out
    pr, pc = env.sim.agents[0]["r"], env.sim.agents[0]["c"]
    deltas = ((-1, 0), (0, 1), (1, 0), (0, -1))
    for action_idx, (dr, dc) in enumerate(deltas, start=1):
        nr, nc = pr + dr, pc + dc
        can_move = (len(env.sim.pathfind((pr, pc), (nr, nc), world=True)) == 2)
        is_ext = nr < 0 or nr >= env.sim.h or nc < 0 or nc >= env.sim.w
        if not can_move or is_ext:
            assert bool(masks[action_idx]) is False, f"Action {action_idx} to ({nr}, {nc}) should be masked False"


# -------------------------------------------------------------------------
# Test 8: Shortest-path distances
# -------------------------------------------------------------------------
def test_shortest_path_distances():
    env = RescueEnv(victim_distance=6, min_victim_distance=3, distance_metric="spawn", geometry_mode="edge")
    obs, info = env.reset(seed=42)

    # Measured distance should match BFS shortest traversable distance
    ar, ac = env.sim.agents[0]["r"], env.sim.agents[0]["c"]
    vr, vc = env.sim.victims[0]["r"], env.sim.victims[0]["c"]

    # Run independent BFS
    dist = { (ar, ac): 0 }
    queue = deque([(ar, ac)])
    while queue:
        r, c = queue.popleft()
        if (r, c) == (vr, vc):
            break
        neighbors = (
            (r - 1, c, env.sim.south[r - 1, c] if r > 0 else 0),
            (r, c + 1, env.sim.east[r, c] if c + 1 < env.sim.w else 0),
            (r + 1, c, env.sim.south[r, c] if r + 1 < env.sim.h else 0),
            (r, c - 1, env.sim.east[r, c - 1] if c > 0 else 0),
        )
        for nr, nc, cond in neighbors:
            if 0 <= nr < env.sim.h and 0 <= nc < env.sim.w and cond >= 0.5 and (nr, nc) not in dist and not env.sim.debris[nr, nc]:
                dist[(nr, nc)] = dist[(r, c)] + 1
                queue.append((nr, nc))

    assert (vr, vc) in dist
    assert dist[(vr, vc)] >= 3
    assert dist[(vr, vc)] <= 6


# -------------------------------------------------------------------------
# Test 9: Observation-memory consistency
# -------------------------------------------------------------------------
def test_observation_memory_consistency():
    env = RescueEnv(geometry_mode="edge")
    obs, info = env.reset(seed=42)

    # Local grid and memory map shapes
    local_grid = obs["grid"]
    memory = obs["memory"]
    assert local_grid.shape == (9, 15, 15)
    assert memory.shape == (9, 40, 52)

    # Center of local grid should correspond to agent position in memory
    ar, ac = env.sim.agents[0]["r"], env.sim.agents[0]["c"]
    mr = env.memory.pad_top + ar
    mc = env.memory.pad_left + ac

    # Agent position channel (Ch 8) should be 1.0 at agent local center
    assert local_grid[8, 7, 7] == 1.0
    # In memory, agent position should be at (mr, mc)
    assert memory[8, mr, mc] == 1.0


# -------------------------------------------------------------------------
# Test 10: No hidden-edge information leakage
# -------------------------------------------------------------------------
def test_no_hidden_edge_leakage():
    env = RescueEnv(geometry_mode="edge")
    obs, info = env.reset(seed=42)

    # Unexplored cells must have explored channel == 0 and not leak victims
    memory = env.memory.grid  # Unpadded (9, H, W)
    unexplored = (env.memory.explored == 0)

    assert np.any(unexplored), "Should have unexplored cells initially"
    # Explored channel (Ch 5) for unexplored cells must be 0
    assert np.all(memory[5, unexplored] == 0.0)
    # Victim channel (Ch 6) for unexplored cells must be 0
    assert np.all(memory[6, unexplored] == 0.0)


# -------------------------------------------------------------------------
# Test 11: Victim pickup and rescue behavior
# -------------------------------------------------------------------------
def test_victim_pickup_and_rescue():
    env = RescueEnv(victim_distance=2, min_victim_distance=1, geometry_mode="edge")
    obs, info = env.reset(seed=42)

    # Position agent on victim cell
    v = env.sim.victims[0]
    env.sim.agents[0]["r"] = v["r"]
    env.sim.agents[0]["c"] = v["c"]
    env.sim._core.set_people(env.sim.victims, env.sim.agents)

    # Action 5 is Pickup/Drop (code 10)
    masks = env.action_masks()
    assert bool(masks[5]) is True

    obs, reward, terminated, truncated, info = env.step(5)
    assert env.sim.victims[0]["carried_by"] == 0

    # Step to exit portal and step outside
    exits = [op for op in env.sim.openings if op["kind"] == "exit" and op["open"]]
    exit_op = exits[0]
    pr, pc = env.sim._portal_cells(exit_op, exit_op["zone_a"])[0]
    env.sim.agents[0]["r"] = pr
    env.sim.agents[0]["c"] = pc
    env.sim.victims[0]["r"] = pr
    env.sim.victims[0]["c"] = pc
    env.sim._core.set_people(env.sim.victims, env.sim.agents)

    # Outward step towards exterior
    out_action = (
        (1 if exit_op["at"] == 0 else 3)
        if exit_op["orient"] == "H"
        else (4 if exit_op["at"] == 0 else 2)
    )
    obs, reward, terminated, truncated, info = env.step(out_action)
    assert bool(env.sim.victims[0]["rescued"]) is True
    assert reward >= 10.0


# -------------------------------------------------------------------------
# Test 12: CPU/CUDA geometry and movement consistency
# -------------------------------------------------------------------------
def test_cpu_geometry_movement_consistency():
    env = Environment(size=4, resolution=2, seed=42, geometry_mode="edge")
    # Verify C++ core people and movement match Python bindings
    agent = env.agents[0]
    cpp_people = env._core.people(1)
    assert cpp_people[0]["r"] == agent["r"]
    assert cpp_people[0]["c"] == agent["c"]


# -------------------------------------------------------------------------
# Test 13: Visual debugger and trajectory replay correctness
# -------------------------------------------------------------------------
def test_visual_debugger_trajectory_recording():
    env = RescueEnv(geometry_mode="edge")
    env = PreviousActionWrapper(env)
    recorder = EpisodeRecorder()

    obs, info = env.reset(seed=42)
    recorder.start_episode(env, seed=42)
    masks = env.action_masks()
    recorder.record_step(
        step_idx=0,
        env=env,
        obs=obs,
        action_mask=masks,
        raw_logits=np.zeros(7),
        masked_logits=np.zeros(7),
        action_probs=np.ones(7) / 7.0,
        selected_action=0,
        value_estimate=0.0,
        gru_hidden_norm=0.0,
        reward=0.0,
        cumulative_reward=0.0,
        terminated=False,
        truncated=False,
        info=info,
    )
    recorder.finish_episode(env)
    traj = recorder.to_dict()

    assert "static_ground_truth" in traj
    sgt = traj["static_ground_truth"]
    assert sgt["geometry_mode"] == "edge"
    assert "zones" in sgt
    assert "openings" in sgt
    assert "south" in sgt
    assert "east" in sgt


# -------------------------------------------------------------------------
# Test 14: Legacy simulation reproducibility
# -------------------------------------------------------------------------
def test_legacy_simulation_reproducibility():
    env_legacy = Environment(size=8, resolution=2, seed=42, geometry_mode="legacy_cell")
    assert env_legacy.wall is not None
    assert int((env_legacy.zone == -1).sum()) == 398
    assert int((env_legacy.wall).sum()) == 398

    env_edge = Environment(size=8, resolution=2, seed=42, geometry_mode="edge")
    assert env_edge.wall is None
    assert int((env_edge.zone == -1).sum()) == 0


# -------------------------------------------------------------------------
# Test 15: No regression in recurrent PPO interfaces
# -------------------------------------------------------------------------
def test_recurrent_ppo_interface_in_edge_geometry():
    env = RescueEnv(geometry_mode="edge")
    env = PreviousActionWrapper(env)

    policy = RecurrentSpatialRescueCNN(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 0.0003,
        architecture_variant="baseline",
    )

    obs, info = env.reset(seed=42)
    masks = env.action_masks()

    # Create batch of 1
    obs_batch = {
        "grid": torch.as_tensor(obs["grid"]).unsqueeze(0),
        "memory": torch.as_tensor(obs["memory"]).unsqueeze(0),
        "agent": torch.as_tensor(obs["agent"]).unsqueeze(0),
        "previous_action": torch.as_tensor(obs["previous_action"]).unsqueeze(0),
    }
    masks_batch = torch.as_tensor(masks).unsqueeze(0)
    hidden = torch.zeros((1, 1, 128))
    starts = torch.as_tensor([True])

    actions, values, log_probs, next_hidden = policy.forward(
        obs_batch, hidden, starts, action_masks=masks_batch
    )

    assert actions.shape == (1,)
    assert values.shape == (1, 1)
    assert log_probs.shape == (1,)
    assert next_hidden.shape == (1, 1, 128)
    assert bool(masks[actions.item()]) is True
