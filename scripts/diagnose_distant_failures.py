#!/usr/bin/env python3
"""Diagnostic script for Phase 4: Distant Return Failure Analysis.

Evaluates checkpoints on the 200 matched distant scenarios (seeds 50000..50199):
- Unshaped native rescue only (reward_shaping=False, navigation_shaping=False)
- Fully unknown map (geometry_reveal_fraction=0.0, reveal_victim=False)
- Distant configuration: victim_distance=None, min_victim_distance=1, max_steps=384

Records per-episode metrics for all episodes where victim was discovered or picked up:
1. Discovery step t_disc
2. Pickup step t_pick
3. Remaining steps T_rem = 384 - t_pick
4. Shortest feasible return distance D_optimal to exit via offline ground-truth BFS
5. Actual steps and path length while carrying
6. Cell revisits and oscillations while carrying
7. Connected observed path to exit in memory map
8. Interaction failure check (reached portal/grass without completing rescue)
9. Mutually exclusive failure classification by precedence:
   - Time-infeasible (T_rem < D_optimal)
   - Interaction failure
   - Route-memory limitation
   - Navigation failure
   - Environment termination

Outputs:
- Machine-readable JSON summary and per-episode trajectory logs
- Comprehensive markdown report for docs/PHASE4_DISTANT_FAILURE_ANALYSIS.md
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from rescue_env.envs.single_agent import RescueEnv
from rescue_env.wrappers import PreviousActionWrapper
from agents.recurrent_ppo import MaskableRecurrentPPO
from agents.cnn import RecurrentSpatialRescueCNN
from training.train_ppo import load_checkpoint

OPPOSITE_ACTIONS = {
    1: 3,  # Up <-> Down
    3: 1,
    2: 4,  # Right <-> Left
    4: 2,
}


def compute_shortest_exit_distance(sim, start_pos: tuple[int, int]) -> int:
    """Compute shortest ground-truth path distance from start_pos to outside exit via open exits.

    Returns number of steps to reach exterior grass (portal_cell_distance + 1).
    """
    sr, sc = start_pos
    h, w = sim.h, sim.w

    # Target portal cells
    portal_targets = set()
    for op in sim.openings:
        if op["kind"] == "exit" and op["open"] and not op.get("blocked"):
            for pr, pc in sim._portal_cells(op, op["zone_a"]):
                if 0 <= pr < h and 0 <= pc < w:
                    portal_targets.add((pr, pc))

    if not portal_targets:
        return float("inf")

    if (sr, sc) in portal_targets:
        return 1  # 1 step through opening to exterior grass

    queue = deque([(sr, sc, 0)])
    visited = {(sr, sc)}

    while queue:
        r, c, d = queue.popleft()
        if (r, c) in portal_targets:
            return d + 1  # +1 step from portal cell onto exterior grass

        neighbors = (
            (r - 1, c, r > 0 and sim.south[r - 1, c] >= 0.5),
            (r, c + 1, c + 1 < w and sim.east[r, c] >= 0.5),
            (r + 1, c, r + 1 < h and sim.south[r, c] >= 0.5),
            (r, c - 1, c > 0 and sim.east[r, c - 1] >= 0.5),
        )
        for nr, nc, passable in neighbors:
            if passable and (nr, nc) not in visited and not sim.debris[nr, nc]:
                visited.add((nr, nc))
                queue.append((nr, nc, d + 1))

    return float("inf")


def check_memory_exit_connected(memory, start_pos: tuple[int, int]) -> bool:
    """Check if agent's memory map contains a connected explored path from start_pos to an exit portal."""
    sr, sc = start_pos
    h, w = memory.height, memory.width

    exit_cells = set(zip(*np.where(memory.grid[7] > 0.5)))
    if not exit_cells:
        return False

    if (sr, sc) in exit_cells:
        return True

    queue = deque([(sr, sc)])
    visited = {(sr, sc)}

    while queue:
        r, c = queue.popleft()
        if (r, c) in exit_cells:
            return True

        # Check 4 directions in memory map
        # North
        if r > 0 and memory.grid[0, r, c] >= 0.5 and memory.explored[r - 1, c] and not memory.grid[4, r - 1, c]:
            if (r - 1, c) not in visited:
                visited.add((r - 1, c))
                queue.append((r - 1, c))
        # East
        if c + 1 < w and memory.grid[1, r, c] >= 0.5 and memory.explored[r, c + 1] and not memory.grid[4, r, c + 1]:
            if (r, c + 1) not in visited:
                visited.add((r, c + 1))
                queue.append((r, c + 1))
        # South
        if r + 1 < h and memory.grid[2, r, c] >= 0.5 and memory.explored[r + 1, c] and not memory.grid[4, r + 1, c]:
            if (r + 1, c) not in visited:
                visited.add((r + 1, c))
                queue.append((r + 1, c))
        # West
        if c > 0 and memory.grid[3, r, c] >= 0.5 and memory.explored[r, c - 1] and not memory.grid[4, r, c - 1]:
            if (r, c - 1) not in visited:
                visited.add((r, c - 1))
                queue.append((r, c - 1))

    return False


def run_distant_diagnosis(
    model,
    n_episodes: int = 200,
    seed_start: int = 50000,
    deterministic: bool = False,
    device: str = "cpu",
) -> dict:
    env = RescueEnv(
        victim_distance=None,
        min_victim_distance=1,
        max_steps=384,
        reward_shaping=False,
        navigation_shaping=False,
        geometry_reveal_fraction=0.0,
        reveal_victim=False,
    )
    env = PreviousActionWrapper(env)

    episodes = []

    for ep in range(n_episodes):
        seed = seed_start + ep
        obs, _ = env.reset(seed=seed)
        if not deterministic:
            torch.manual_seed(seed)

        done = False
        step_count = 0

        # Episode tracking variables
        t_disc = None
        t_pick = None
        pickup_pos = None
        d_optimal_at_pickup = None
        memory_connected_at_pickup = False
        reached_portal_while_carrying = False

        pos_history = []
        action_history = []
        carrying_history = []
        carrying_pos_history = []

        gru_states = None
        episode_starts = np.ones((1,), dtype=bool)

        sim = env.unwrapped.sim
        initial_agent_pos = (sim.agents[0]["r"], sim.agents[0]["c"])
        pos_history.append(initial_agent_pos)

        # Check initial discovery
        if (obs["grid"][6] > 0).any():
            t_disc = 0

        while not done:
            action, gru_states = model.predict(
                obs,
                state=gru_states,
                episode_start=episode_starts,
                deterministic=deterministic,
                action_masks=env.action_masks(),
            )
            episode_starts = np.zeros((1,), dtype=bool)
            act_int = int(action)
            action_history.append(act_int)

            obs, reward, terminated, truncated, info = env.step(act_int)
            step_count += 1
            done = terminated or truncated

            agent_r, agent_c = sim.agents[0]["r"], sim.agents[0]["c"]
            agent_pos = (agent_r, agent_c)
            pos_history.append(agent_pos)

            # Check victim discovery
            if t_disc is None and (obs["grid"][6] > 0).any():
                t_disc = step_count

            # Check carrying state
            is_carrying = any(
                v["carried_by"] == 0 and not v["dead"] and not v["rescued"]
                for v in sim.victims
            )
            carrying_history.append(is_carrying)

            if is_carrying:
                carrying_pos_history.append(agent_pos)
                if t_pick is None:
                    t_pick = step_count
                    pickup_pos = agent_pos
                    d_optimal_at_pickup = compute_shortest_exit_distance(sim, pickup_pos)
                    memory_connected_at_pickup = check_memory_exit_connected(env.unwrapped.memory, pickup_pos)

                # Check if agent reached an exit portal cell while carrying
                for op in sim.openings:
                    if op["kind"] == "exit" and op["open"] and not op.get("blocked"):
                        for pr, pc in sim._portal_cells(op, op["zone_a"]):
                            if (agent_r, agent_c) == (pr, pc):
                                reached_portal_while_carrying = True

        is_success = bool(info.get("success", False))
        is_timeout = bool(info.get("timeout", False))
        agent_dead = bool(sim.agents[0]["dead"])
        victim_dead = bool(sim.victims[0]["dead"]) if sim.victims else False

        # If rescued, discovery and pickup must have happened
        if is_success:
            if t_disc is None:
                t_disc = t_pick if t_pick is not None else step_count
            if t_pick is None:
                t_pick = step_count
                pickup_pos = pos_history[-2] if len(pos_history) >= 2 else pos_history[-1]
                d_optimal_at_pickup = 1

        # Post-pickup analysis
        t_rem = (384 - t_pick) if t_pick is not None else None
        slack = (t_rem - d_optimal_at_pickup) if (t_rem is not None and d_optimal_at_pickup is not None) else None
        is_time_feasible = (slack >= 0) if slack is not None else None

        # Carrying movement metrics
        carrying_steps = len(carrying_pos_history)
        carrying_oscillations = 0
        carrying_revisits = 0
        if carrying_steps > 0:
            visited_counts = {}
            for p in carrying_pos_history:
                visited_counts[p] = visited_counts.get(p, 0) + 1
            carrying_revisits = sum(c - 1 for c in visited_counts.values())

            # Count 2-step oscillations during carrying
            for idx in range(2, carrying_steps):
                if carrying_pos_history[idx] == carrying_pos_history[idx - 2]:
                    carrying_oscillations += 1

        # Mutually exclusive failure classification
        failure_category = None
        if t_pick is not None and not is_success:
            # Classification precedence:
            # 1. Time-infeasible: T_rem < D_optimal
            # 2. Interaction failure: reached portal cell but did not complete rescue
            # 3. Route-memory limitation: memory lacks connected path to exit
            # 4. Navigation failure: connected route existed and time-feasible, but failed
            # 5. Environment termination: agent or victim dead
            if agent_dead or victim_dead:
                failure_category = "Environment termination"
            elif not is_time_feasible:
                failure_category = "Time-infeasible"
            elif reached_portal_while_carrying:
                failure_category = "Interaction failure"
            elif not memory_connected_at_pickup:
                # Also check final memory
                if not check_memory_exit_connected(env.unwrapped.memory, carrying_pos_history[-1]):
                    failure_category = "Route-memory limitation"
                else:
                    failure_category = "Navigation failure"
            else:
                failure_category = "Navigation failure"

        ep_record = {
            "seed": seed,
            "success": is_success,
            "timeout": is_timeout,
            "agent_dead": agent_dead,
            "victim_dead": victim_dead,
            "step_count": step_count,
            "discovered": t_disc is not None,
            "t_disc": t_disc,
            "picked_up": t_pick is not None,
            "t_pick": t_pick,
            "t_rem": t_rem,
            "pickup_pos": pickup_pos,
            "d_optimal": d_optimal_at_pickup,
            "slack": slack,
            "is_time_feasible": is_time_feasible,
            "memory_connected": memory_connected_at_pickup,
            "reached_portal": reached_portal_while_carrying,
            "carrying_steps": carrying_steps,
            "carrying_revisits": carrying_revisits,
            "carrying_oscillations": carrying_oscillations,
            "failure_category": failure_category,
        }
        episodes.append(ep_record)

    env.close()

    # Aggregate metrics
    n_total = len(episodes)
    n_disc = sum(1 for e in episodes if e["discovered"])
    n_pick = sum(1 for e in episodes if e["picked_up"])
    n_resc = sum(1 for e in episodes if e["success"])

    pickup_eps = [e for e in episodes if e["picked_up"]]
    n_feasible_pick = sum(1 for e in pickup_eps if e["is_time_feasible"])
    n_infeasible_pick = sum(1 for e in pickup_eps if not e["is_time_feasible"])

    # Failures among pickups
    failed_pickups = [e for e in pickup_eps if not e["success"]]
    failure_counts = {
        "Time-infeasible": sum(1 for e in failed_pickups if e["failure_category"] == "Time-infeasible"),
        "Interaction failure": sum(1 for e in failed_pickups if e["failure_category"] == "Interaction failure"),
        "Route-memory limitation": sum(1 for e in failed_pickups if e["failure_category"] == "Route-memory limitation"),
        "Navigation failure": sum(1 for e in failed_pickups if e["failure_category"] == "Navigation failure"),
        "Environment termination": sum(1 for e in failed_pickups if e["failure_category"] == "Environment termination"),
    }

    # Conditional success rates
    ret_given_pick = (n_resc / n_pick * 100.0) if n_pick > 0 else 0.0
    ret_given_feasible = (n_resc / n_feasible_pick * 100.0) if n_feasible_pick > 0 else 0.0

    # Timing metrics
    t_picks = [e["t_pick"] for e in pickup_eps]
    t_rems = [e["t_rem"] for e in pickup_eps]
    d_opts = [e["d_optimal"] for e in pickup_eps]
    slacks = [e["slack"] for e in pickup_eps]

    mean_t_pick = float(np.mean(t_picks)) if t_picks else 0.0
    mean_t_rem = float(np.mean(t_rems)) if t_rems else 0.0
    mean_d_opt = float(np.mean(d_opts)) if d_opts else 0.0
    mean_slack = float(np.mean(slacks)) if slacks else 0.0

    # Carrying oscillation rate
    total_carrying_steps = sum(e["carrying_steps"] for e in pickup_eps)
    total_carrying_osc = sum(e["carrying_oscillations"] for e in pickup_eps)
    carrying_osc_rate = (total_carrying_osc / max(1, total_carrying_steps)) * 100.0

    summary = {
        "mode": "Deterministic" if deterministic else "Stochastic",
        "total_episodes": n_total,
        "discovery_count": n_disc,
        "discovery_rate": (n_disc / n_total) * 100.0,
        "pickup_count": n_pick,
        "pickup_rate": (n_pick / n_total) * 100.0,
        "rescue_count": n_resc,
        "rescue_rate": (n_resc / n_total) * 100.0,
        "return_given_pickup_rate": ret_given_pick,
        "feasible_pickup_count": n_feasible_pick,
        "infeasible_pickup_count": n_infeasible_pick,
        "feasible_pickup_rate": (n_feasible_pick / n_total) * 100.0,
        "infeasible_pickup_rate": (n_infeasible_pick / n_total) * 100.0,
        "feasible_return_success_rate": ret_given_feasible,
        "mean_t_pick": mean_t_pick,
        "mean_t_rem": mean_t_rem,
        "mean_d_optimal": mean_d_opt,
        "mean_slack": mean_slack,
        "total_carrying_steps": total_carrying_steps,
        "carrying_oscillation_rate": carrying_osc_rate,
        "failure_counts": failure_counts,
        "failed_pickups_total": len(failed_pickups),
        "episodes": episodes,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Distant Return Failure Diagnosis")
    parser.add_argument(
        "--model",
        type=str,
        default="models/multitask_curriculum/production_250k_s42/final_model.zip",
        help="Path to model checkpoint",
    )
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes (default: 200)")
    parser.add_argument("--seed-start", type=int, default=50000, help="Start seed (default: 50000)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (default: cpu)")
    parser.add_argument("--out-dir", type=str, default="models/distant_failure_analysis", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {args.model}...")
    model = load_checkpoint(args.model, device=args.device)

    print(f"\n================================================================================")
    print(f"RUNNING DIAGNOSIS: 200 MATCHED DISTANT SCENARIOS (seeds {args.seed_start}..{args.seed_start + args.episodes - 1})")
    print(f"================================================================================")

    # 1. Stochastic mode
    print("Evaluating in STOCHASTIC mode...")
    stoch_summary = run_distant_diagnosis(
        model,
        n_episodes=args.episodes,
        seed_start=args.seed_start,
        deterministic=False,
        device=args.device,
    )

    # 2. Deterministic mode
    print("Evaluating in DETERMINISTIC mode...")
    det_summary = run_distant_diagnosis(
        model,
        n_episodes=args.episodes,
        seed_start=args.seed_start,
        deterministic=True,
        device=args.device,
    )

    # Print summary tables
    for s in [stoch_summary, det_summary]:
        print(f"\n--- {s['mode'].upper()} DISTANT DIAGNOSIS RESULTS ---")
        print(f"Total Episodes:        {s['total_episodes']}")
        print(f"Discovered:            {s['discovery_count']} ({s['discovery_rate']:.1f}%)")
        print(f"Picked Up:             {s['pickup_count']} ({s['pickup_rate']:.1f}%)")
        print(f"Rescued:               {s['rescue_count']} ({s['rescue_rate']:.1f}%)")
        print(f"Return | Pickup:       {s['return_given_pickup_rate']:.1f}%")
        print(f"Time-Feasible Pickups: {s['feasible_pickup_count']} (Mean T_rem={s['mean_t_rem']:.1f}, Mean D_opt={s['mean_d_optimal']:.1f}, Slack={s['mean_slack']:+.1f})")
        print(f"Time-Infeasible Picks: {s['infeasible_pickup_count']}")
        print(f"Feasible Return Rate:  {s['feasible_return_success_rate']:.1f}% ({s['rescue_count']}/{s['feasible_pickup_count']})")
        print(f"Carrying Oscillation:  {s['carrying_oscillation_rate']:.1f}%")
        print(f"Failure Breakdown (Failed Pickups = {s['failed_pickups_total']}):")
        for cat, cnt in s["failure_counts"].items():
            pct = (cnt / s["failed_pickups_total"] * 100.0) if s["failed_pickups_total"] > 0 else 0.0
            print(f"  - {cat:25s}: {cnt:3d} ({pct:5.1f}%)")

    # Save JSON files
    with open(out_dir / "distant_diagnosis_stoch.json", "w") as f:
        json.dump(stoch_summary, f, indent=2)
    with open(out_dir / "distant_diagnosis_det.json", "w") as f:
        json.dump(det_summary, f, indent=2)

    print(f"\nSaved diagnosis outputs to {out_dir}/")


if __name__ == "__main__":
    main()
