#!/usr/bin/env python3
"""Phase 11: Observation Equivalence Audit (Section 1).

Audits whether the local observation `(9, 15, 15)` can be reconstructed from or is
subsumed by the remembered map `(9, 40, 52)`.

Evaluates across 1,000 representative states sampled from diverse episodes:
- Channel-by-channel reconstruction fidelity (L1, L2, exact match %).
- Instantaneous line-of-sight (Channel 5) vs historical explored mask.
- Edge passabilities (Channels 0..3) within vs outside line-of-sight.
- Victim channel behavior (Channel 6) during discovery, carrying, and pickup.
- Exit portal channel (Channel 7).
- Agent marker alignment (Channel 8).
- Information loss analysis: Does local_grid contain any information absent from memory?
- Update timing analysis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from rescue_env.envs.single_agent import RescueEnv
from rescue_env.wrappers import PreviousActionWrapper

CHANNEL_NAMES = [
    "Passability North",
    "Passability East",
    "Passability South",
    "Passability West",
    "Debris / Obstacle",
    "Explored / Visibility",
    "Victim",
    "Exit Portal",
    "Agent Position",
]


def extract_memory_crop(memory: np.ndarray, agent_pos: tuple[int, int], pad_top: int, pad_left: int, crop_radius: int = 7) -> np.ndarray:
    """Extracts a (9, 2*radius+1, 2*radius+1) crop from memory centered on agent."""
    c, h, w = memory.shape
    crop_side = 2 * crop_radius + 1
    ar, ac = agent_pos
    center_r = pad_top + ar
    center_c = pad_left + ac

    r_min = center_r - crop_radius
    r_max = center_r + crop_radius + 1
    c_min = center_c - crop_radius
    c_max = center_c + crop_radius + 1

    crop = np.zeros((c, crop_side, crop_side), dtype=np.float32)

    # Intersection with memory bounds
    mem_r_min = max(0, r_min)
    mem_r_max = min(h, r_max)
    mem_c_min = max(0, c_min)
    mem_c_max = min(w, c_max)

    out_r_min = mem_r_min - r_min
    out_r_max = out_r_min + (mem_r_max - mem_r_min)
    out_c_min = mem_c_min - c_min
    out_c_max = out_c_min + (mem_c_max - mem_c_min)

    if mem_r_min < mem_r_max and mem_c_min < mem_c_max:
        crop[:, out_r_min:out_r_max, out_c_min:out_c_max] = memory[:, mem_r_min:mem_r_max, mem_c_min:mem_c_max]

    return crop


def run_observation_audit(n_target_states: int = 1000) -> dict:
    env = PreviousActionWrapper(RescueEnv(
        task_type="full",
        geometry_mode="edge",
        reward_shaping=False,
        distance_metric="spawn",
        spatial_memory_mode="none",
    ))

    states_collected = 0
    episodes_run = 0

    # Metrics accumulators
    channel_exact_matches_los = np.zeros(9, dtype=np.int64)
    channel_total_cells_los = np.zeros(9, dtype=np.int64)

    channel_exact_matches_all = np.zeros(9, dtype=np.int64)
    channel_total_cells_all = np.zeros(9, dtype=np.int64)

    channel_l1_los = np.zeros(9, dtype=np.float64)
    channel_l1_all = np.zeros(9, dtype=np.float64)

    agent_marker_match_count = 0
    memory_has_strictly_more_explored = 0
    local_has_unremembered_cells = 0
    victim_discrepancies = 0

    crop_side = 15
    crop_radius = 7

    seed = 42

    print(f"Collecting and auditing {n_target_states} representative states across episodes...")

    while states_collected < n_target_states:
        obs, info = env.reset(seed=seed)
        episodes_run += 1
        seed += 1

        done = False
        step = 0

        while not done and states_collected < n_target_states:
            person = env.sim.agents[0]
            agent_pos = (person["r"], person["c"])
            pad_top = env.memory.pad_top
            pad_left = env.memory.pad_left

            local_grid = obs["grid"]  # (9, 15, 15)
            memory_grid = obs["memory"]  # (9, 40, 52)

            crop = extract_memory_crop(memory_grid, agent_pos, pad_top, pad_left, crop_radius=crop_radius)

            los_mask = local_grid[5] > 0.0  # Line of sight cells

            # Check Channel 8 agent marker
            if crop[8, crop_radius, crop_radius] == 1.0 and local_grid[8, crop_radius, crop_radius] == 1.0:
                agent_marker_match_count += 1

            for ch in range(9):
                # Cells in current line of sight
                diff_los = np.abs(local_grid[ch, los_mask] - crop[ch, los_mask])
                channel_l1_los[ch] += np.sum(diff_los)
                channel_exact_matches_los[ch] += np.sum(diff_los < 1e-5)
                channel_total_cells_los[ch] += np.sum(los_mask)

                # All 15x15 cells
                diff_all = np.abs(local_grid[ch] - crop[ch])
                channel_l1_all[ch] += np.sum(diff_all)
                channel_exact_matches_all[ch] += np.sum(diff_all < 1e-5)
                channel_total_cells_all[ch] += crop_side * crop_side

            # Exploration difference
            mem_explored_in_crop = crop[5] > 0.0
            if np.sum(mem_explored_in_crop) > np.sum(los_mask):
                memory_has_strictly_more_explored += 1
            if np.any(los_mask & ~mem_explored_in_crop):
                local_has_unremembered_cells += 1

            # Victim channel check
            if not np.array_equal(local_grid[6, los_mask], crop[6, los_mask]):
                victim_discrepancies += 1

            states_collected += 1

            # Step environment with random legal action
            action_mask = env.action_masks()
            valid_actions = np.where(action_mask)[0]
            action = int(np.random.choice(valid_actions))

            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            done = terminated or truncated

    env.close()

    # Compile results
    audit_results = {
        "n_states_audited": states_collected,
        "n_episodes": episodes_run,
        "agent_marker_match_rate_pct": float(agent_marker_match_count / states_collected * 100.0),
        "memory_has_more_explored_rate_pct": float(memory_has_strictly_more_explored / states_collected * 100.0),
        "local_has_unremembered_cells_count": local_has_unremembered_cells,
        "victim_discrepancies_los_count": victim_discrepancies,
        "channel_metrics": {},
    }

    print("\n" + "=" * 90)
    print(f"AUDIT RESULTS ON {states_collected} REPRESENTATIVE STATES")
    print("=" * 90)
    print(f"{'Channel':25s} | {'LOS Exact Match %':18s} | {'LOS Mean L1':12s} | {'All 15x15 Match %':18s} | {'All Mean L1':12s}")
    print("-" * 90)

    for ch in range(9):
        name = CHANNEL_NAMES[ch]
        los_match = float(channel_exact_matches_los[ch] / max(1, channel_total_cells_los[ch]) * 100.0)
        los_l1 = float(channel_l1_los[ch] / max(1, channel_total_cells_los[ch]))
        all_match = float(channel_exact_matches_all[ch] / max(1, channel_total_cells_all[ch]) * 100.0)
        all_l1 = float(channel_l1_all[ch] / max(1, channel_total_cells_all[ch]))

        audit_results["channel_metrics"][name] = {
            "channel_idx": ch,
            "los_exact_match_pct": los_match,
            "los_mean_l1": los_l1,
            "all_cells_exact_match_pct": all_match,
            "all_cells_mean_l1": all_l1,
        }
        print(f"{name:25s} | {los_match:16.2f}% | {los_l1:12.6f} | {all_match:16.2f}% | {all_l1:12.6f}")

    print("=" * 90)
    print(f"Agent Marker Alignment at (7, 7): {audit_results['agent_marker_match_rate_pct']:.2f}%")
    print(f"States where Memory has MORE explored cells in crop than LOS: {audit_results['memory_has_more_explored_rate_pct']:.2f}%")
    print(f"States where Local has cells UNREMEMBERED in Memory: {audit_results['local_has_unremembered_cells_count']}")
    print(f"Victim discrepancies in LOS: {audit_results['victim_discrepancies_los_count']}")

    out_file = ROOT / "models" / "phase11_evaluations" / "observation_equivalence_audit.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\nSaved machine-readable audit to {out_file}")

    return audit_results


if __name__ == "__main__":
    run_observation_audit(1000)
