#!/usr/bin/env python3
"""Compiles all Phase 10 evaluation results into markdown tables."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "models" / "phase10_evaluations"

MODELS = [
    ("Baseline (Phase 6)", "eval_baseline_phase6.json"),
    ("Variant A (Control)", "eval_variant_a_control.json"),
    ("Variant B (No Osc)", "eval_variant_b_no_osc.json"),
    ("Variant C (Wait Pen)", "eval_variant_c_wait_penalty.json"),
    ("Variant D (Combined)", "eval_variant_d_combined.json"),
    ("Phase 9 (Best Nav)", "eval_phase9_best_nav.json"),
]

data = {}
for label, fname in MODELS:
    with open(EVAL_DIR / fname) as f:
        data[label] = json.load(f)

print("=" * 80)
print("1. NAVIGATION & EXPLORATION BENCHMARK (DETERMINISTIC, 200 EPS)")
print("=" * 80)
print(f"| Model | Visited Cov % | Unique Cells | Unique Rooms | Wait % | Max Wait Streak | Streaks >10 | Streaks >50 | Osc % | Doorway Rev | New after 50 |")
print(f"|---|---|---|---|---|---|---|---|---|---|---|")
for label, _ in MODELS:
    nav = data[label]["navigation_deterministic"]
    print(f"| {label} | {nav['mean_visited_coverage_pct']:.2f}% | {nav['mean_unique_cells_visited']:.1f} | {nav['mean_unique_rooms_entered']:.2f} | {nav['mean_wait_fraction_pct']:.1f}% | {nav['mean_max_wait_streak']:.1f} | {nav['episodes_wait_streak_gt10_pct']:.1f}% | {nav['episodes_wait_streak_gt50_pct']:.1f}% | {nav['mean_oscillation_pct']:.1f}% | {nav['mean_doorway_reversals']:.1f} | {nav['mean_new_cells_after_step_50']:.1f} |")

print("\n" + "=" * 80)
print("2. NAVIGATION & EXPLORATION BENCHMARK (STOCHASTIC, 200 EPS)")
print("=" * 80)
print(f"| Model | Visited Cov % | Unique Cells | Unique Rooms | Wait % | Max Wait Streak | Streaks >10 | Streaks >50 | Osc % | Doorway Rev | New after 50 |")
print(f"|---|---|---|---|---|---|---|---|---|---|---|")
for label, _ in MODELS:
    nav = data[label]["navigation_stochastic"]
    print(f"| {label} | {nav['mean_visited_coverage_pct']:.2f}% | {nav['mean_unique_cells_visited']:.1f} | {nav['mean_unique_rooms_entered']:.2f} | {nav['mean_wait_fraction_pct']:.1f}% | {nav['mean_max_wait_streak']:.1f} | {nav['episodes_wait_streak_gt10_pct']:.1f}% | {nav['episodes_wait_streak_gt50_pct']:.1f}% | {nav['mean_oscillation_pct']:.1f}% | {nav['mean_doorway_reversals']:.1f} | {nav['mean_new_cells_after_step_50']:.1f} |")

print("\n" + "=" * 80)
print("3. BRANCHING NAVIGATION BENCHMARK (100 EPS)")
print("=" * 80)
print(f"| Model | B1 Correct Branch % | B1 Wait % | B2 Entered New Branch % | B2 Wait % |")
print(f"|---|---|---|---|---|")
for label, _ in MODELS:
    b = data[label]["branching"]
    print(f"| {label} | {b['B1']['correct_branch_selection_rate_pct']:.1f}% | {b['B1']['mean_wait_fraction_pct']:.1f}% | {b['B2']['entered_new_branch_rate_pct']:.1f}% | {b['B2']['mean_wait_fraction_pct']:.1f}% |")

print("\n" + "=" * 80)
print("4. SPAWN-RELATIVE RESCUE LADDER (DETERMINISTIC, 200 EPS/TIER)")
print("=" * 80)
tiers = ["spawn_2_6", "spawn_7_12", "spawn_13_20", "spawn_21_35", "spawn_distant"]
header = "| Model | " + " | ".join(tiers) + " |"
print(header)
print("|---" * (len(tiers) + 1) + "|")
for label, _ in MODELS:
    row = f"| {label} | "
    for t in tiers:
        res = data[label]["spawn_ladder_deterministic"][t]["rescue_rate_pct"]
        row += f"{res:.1f}% | "
    print(row)

print("\n" + "=" * 80)
print("5. SPAWN-RELATIVE RESCUE LADDER (STOCHASTIC, 200 EPS/TIER)")
print("=" * 80)
print(header)
print("|---" * (len(tiers) + 1) + "|")
for label, _ in MODELS:
    row = f"| {label} | "
    for t in tiers:
        res = data[label]["spawn_ladder_stochastic"][t]["rescue_rate_pct"]
        row += f"{res:.1f}% | "
    print(row)

print("\n" + "=" * 80)
print("6. EXIT-RELATIVE RESCUE LADDER (DETERMINISTIC, 200 EPS/TIER)")
print("=" * 80)
exit_tiers = ["near", "medium", "bridge_a", "bridge_b", "bridge_c", "distant"]
header_exit = "| Model | " + " | ".join(exit_tiers) + " |"
print(header_exit)
print("|---" * (len(exit_tiers) + 1) + "|")
for label, _ in MODELS:
    row = f"| {label} | "
    for t in exit_tiers:
        res = data[label]["exit_ladder_deterministic"][t]["rescue_rate_pct"]
        row += f"{res:.1f}% | "
    print(row)

print("\n" + "=" * 80)
print("7. EXIT-RELATIVE RESCUE LADDER (STOCHASTIC, 200 EPS/TIER)")
print("=" * 80)
print(header_exit)
print("|---" * (len(exit_tiers) + 1) + "|")
for label, _ in MODELS:
    row = f"| {label} | "
    for t in exit_tiers:
        res = data[label]["exit_ladder_stochastic"][t]["rescue_rate_pct"]
        row += f"{res:.1f}% | "
    print(row)

print("\n" + "=" * 80)
print("8. FAILURE DECOMPOSITION (SPAWN LADDER DETERMINISTIC, 1,000 TOTAL EPS)")
print("=" * 80)
print("| Model | Rescued | Never Discovered | Discovered Not Picked | Carrier Timeout | Excess Wait Stall | Excess Oscillation | Other Timeout |")
print("|---|---|---|---|---|---|---|---|")
for label, _ in MODELS:
    f = data[label]["failure_decomposition_spawn_det"]
    print(f"| {label} | {f['rescued']} | {f['never_discovered']} | {f['discovered_not_picked']} | {f['carrier_timeout']} | {f['excessive_wait_stalling']} | {f['excessive_oscillation']} | {f['other_timeout']} |")
