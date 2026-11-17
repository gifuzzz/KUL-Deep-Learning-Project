#!/usr/bin/env python3
"""Phase 11: Compile Master Comparison Tables from Evaluation JSONs.

Reads evaluation results from `models/phase11_evaluations/` and reference
checkpoints from `models/phase10_evaluations/`, and outputs formatted markdown
tables for `docs/PHASE11_ARCHITECTURE_ABLATION_REPORT.md`.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    {
        "id": "variant_a",
        "name": "Variant A (Current)",
        "arch": "Local CNN + Memory CNN",
        "params": "1.43M (1,431,168)",
        "file": ROOT / "models/phase11_evaluations/eval_variant_a_current.json",
    },
    {
        "id": "variant_b",
        "name": "Variant B (Memory Only)",
        "arch": "Memory CNN Only",
        "params": "1.18M (1,179,128)",
        "file": ROOT / "models/phase11_evaluations/eval_variant_b_memory_only.json",
    },
    {
        "id": "variant_c",
        "name": "Variant C (Memory Crop)",
        "arch": "Memory CNN + Crop",
        "params": "1.43M (1,431,168)",
        "file": ROOT / "models/phase11_evaluations/eval_variant_c_memory_crop.json",
    },
    {
        "id": "ref_phase10_b",
        "name": "Phase 10 Var B (Base)",
        "arch": "Pretrained Source",
        "params": "1.43M (1,431,168)",
        "file": ROOT / "models/phase10_evaluations/eval_variant_b_no_osc.json",
    },
    {
        "id": "ref_phase6",
        "name": "Phase 6 Baseline",
        "arch": "Legacy Baseline",
        "params": "1.43M (1,431,168)",
        "file": ROOT / "models/phase10_evaluations/eval_baseline_phase6.json",
    },
]

SPAWN_TIERS = [
    ("spawn_2_6", "Spawn 2–6 (d=6, max=64)"),
    ("spawn_7_12", "Spawn 7–12 (d=12, max=128)"),
    ("spawn_13_20", "Spawn 13–20 (d=20, max=192)"),
    ("spawn_21_35", "Spawn 21–35 (d=35, max=288)"),
    ("spawn_distant", "Spawn Distant (max=384)"),
]

EXIT_TIERS = [
    ("near", "Near (d=2, max=64)"),
    ("medium", "Medium (d=6, max=128)"),
    ("bridge_a", "Bridge A (d=8, max=160)"),
    ("bridge_b", "Bridge B (d=10, max=192)"),
    ("bridge_c", "Bridge C (d=12, max=256)"),
    ("distant", "Distant (d=None, max=384)"),
]


def load_eval_data():
    data = {}
    for m in MODELS:
        fpath = m["file"]
        if not fpath.exists():
            print(f"Warning: {fpath} not found")
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            data[m["id"]] = json.load(f)
    return data


def format_table_1_executive(data):
    lines = [
        "### Table 1: Phase 11 Architecture Ablation — Executive Overview",
        "",
        "| Architecture Variant | Local View Input | Param Count | Visited Cov % (Det / Stoch) | Wait % (Det / Stoch) | Osc % (Det / Stoch) | Spawn Mean Rescue % (Det / Stoch) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for m in MODELS:
        mid = m["id"]
        if mid not in data:
            continue
        d = data[mid]
        ndet = d["navigation_deterministic"]
        nstoch = d["navigation_stochastic"]
        sdet = d["spawn_ladder_deterministic"]
        sstoch = d["spawn_ladder_stochastic"]

        sdet_mean = sum(t["rescue_rate_pct"] for t in sdet.values()) / len(sdet)
        sstoch_mean = sum(t["rescue_rate_pct"] for t in sstoch.values()) / len(sstoch)

        vis_cov = f"{ndet['mean_visited_coverage_pct']:.2f}% / {nstoch['mean_visited_coverage_pct']:.2f}%"
        wait = f"{ndet['mean_wait_fraction_pct']:.1f}% / {nstoch['mean_wait_fraction_pct']:.1f}%"
        osc = f"{ndet['mean_oscillation_pct']:.1f}% / {nstoch['mean_oscillation_pct']:.1f}%"
        rescue = f"{sdet_mean:.1f}% / {sstoch_mean:.1f}%"

        lines.append(f"| **{m['name']}** | {m['arch']} | {m['params']} | {vis_cov} | {wait} | {osc} | **{rescue}** |")

    return "\n".join(lines)


def format_table_2_spawn(data, stochastic=False):
    mode = "Stochastic" if stochastic else "Deterministic"
    key = "spawn_ladder_stochastic" if stochastic else "spawn_ladder_deterministic"
    lines = [
        f"### Table 2{'b' if stochastic else 'a'}: Spawn-Relative Rescue Ladder ({mode}, 200 eps/tier)",
        "",
        "| Spawn Distance Tier | Variant A (Current) | Variant B (Memory Only) | Variant C (Memory Crop) | Phase 10 Var B (Base) | Phase 6 Baseline |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for t_id, t_label in SPAWN_TIERS:
        row = [f"**{t_label}**"]
        for m in MODELS:
            mid = m["id"]
            if mid not in data:
                row.append("N/A")
                continue
            tier_data = data[mid][key].get(t_id, {})
            r_rate = tier_data.get("rescue_rate_pct", 0.0)
            row.append(f"{r_rate:.1f}%")
        lines.append("| " + " | ".join(row) + " |")

    # Add Average Row
    avg_row = ["**Mean Across Tiers**"]
    for m in MODELS:
        mid = m["id"]
        if mid not in data:
            avg_row.append("N/A")
            continue
        tiers = data[mid][key]
        avg = sum(t["rescue_rate_pct"] for t in tiers.values()) / len(tiers)
        avg_row.append(f"**{avg:.1f}%**")
    lines.append("| " + " | ".join(avg_row) + " |")

    return "\n".join(lines)


def format_table_3_exit(data, stochastic=False):
    mode = "Stochastic" if stochastic else "Deterministic"
    key = "exit_ladder_stochastic" if stochastic else "exit_ladder_deterministic"
    lines = [
        f"### Table 3{'b' if stochastic else 'a'}: Exit-Relative Rescue Ladder ({mode}, 200 eps/tier)",
        "",
        "| Exit Distance Tier | Variant A (Current) | Variant B (Memory Only) | Variant C (Memory Crop) | Phase 10 Var B (Base) | Phase 6 Baseline |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for t_id, t_label in EXIT_TIERS:
        row = [f"**{t_label}**"]
        for m in MODELS:
            mid = m["id"]
            if mid not in data:
                row.append("N/A")
                continue
            tier_data = data[mid][key].get(t_id, {})
            r_rate = tier_data.get("rescue_rate_pct", 0.0)
            row.append(f"{r_rate:.1f}%")
        lines.append("| " + " | ".join(row) + " |")

    avg_row = ["**Mean Across Tiers**"]
    for m in MODELS:
        mid = m["id"]
        if mid not in data:
            avg_row.append("N/A")
            continue
        tiers = data[mid][key]
        avg = sum(t["rescue_rate_pct"] for t in tiers.values()) / len(tiers)
        avg_row.append(f"**{avg:.1f}%**")
    lines.append("| " + " | ".join(avg_row) + " |")

    return "\n".join(lines)


def format_table_4_navigation(data):
    lines = [
        "### Table 4: Pure Navigation & Exploration Benchmark (200 eps, 384 max steps)",
        "",
        "| Metric | Variant A (Current) | Variant B (Memory Only) | Variant C (Memory Crop) | Phase 10 Var B (Base) | Phase 6 Baseline |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    metrics = [
        ("mean_visited_coverage_pct", "Visited Coverage % (Det)", "{:.2f}%", True),
        ("mean_visited_coverage_pct", "Visited Coverage % (Stoch)", "{:.2f}%", False),
        ("mean_observed_coverage_pct", "Observed Coverage % (Det)", "{:.2f}%", True),
        ("mean_unique_cells_visited", "Unique Cells Visited (Det)", "{:.1f}", True),
        ("mean_unique_cells_visited", "Unique Cells Visited (Stoch)", "{:.1f}", False),
        ("mean_unique_rooms_entered", "Unique Rooms Entered (Det)", "{:.2f}", True),
        ("mean_wait_fraction_pct", "Wait Fraction % (Det)", "{:.1f}%", True),
        ("mean_wait_fraction_pct", "Wait Fraction % (Stoch)", "{:.1f}%", False),
        ("episodes_wait_streak_gt10_pct", "Episodes with Wait Streak >10 % (Det)", "{:.1f}%", True),
        ("mean_oscillation_pct", "2-Cell Oscillation % (Det)", "{:.1f}%", True),
        ("mean_doorway_reversals", "Doorway Reversals (Det)", "{:.1f}", True),
        ("episodes_with_positive_discovery_after_step_50_pct", "Episodes with Info After Step 50 %", "{:.1f}%", True),
    ]

    for field, label, fmt, is_det in metrics:
        row = [f"**{label}**"]
        key = "navigation_deterministic" if is_det else "navigation_stochastic"
        for m in MODELS:
            mid = m["id"]
            if mid not in data:
                row.append("N/A")
                continue
            val = data[mid][key].get(field, 0.0)
            row.append(fmt.format(val))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def format_table_5_branching(data):
    lines = [
        "### Table 5: Branching Navigation Benchmark (Stages B1 & B2, 100 eps)",
        "",
        "| Benchmark Metric | Variant A (Current) | Variant B (Memory Only) | Variant C (Memory Crop) | Phase 10 Var B (Base) | Phase 6 Baseline |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    b_metrics = [
        ("B1", "correct_branch_selection_rate_pct", "Stage B1 Correct Branch Selection %", "{:.1f}%"),
        ("B1", "wait_action_fraction_pct", "Stage B1 Wait Action %", "{:.1f}%"),
        ("B2", "entered_new_branch_rate_pct", "Stage B2 Entered New Branch %", "{:.1f}%"),
        ("B2", "wait_action_fraction_pct", "Stage B2 Wait Action %", "{:.1f}%"),
    ]
    for stage, field, label, fmt in b_metrics:
        row = [f"**{label}**"]
        for m in MODELS:
            mid = m["id"]
            if mid not in data:
                row.append("N/A")
                continue
            b = data[mid]["branching"]
            val = b.get(stage, {}).get(field, 0.0)
            row.append(fmt.format(val))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main():
    data = load_eval_data()
    print(f"Loaded evaluation data for {len(data)} models.")
    if not data:
        return

    print("\n" + format_table_1_executive(data) + "\n")
    print("\n" + format_table_2_spawn(data, stochastic=False) + "\n")
    print("\n" + format_table_2_spawn(data, stochastic=True) + "\n")
    print("\n" + format_table_3_exit(data, stochastic=False) + "\n")
    print("\n" + format_table_3_exit(data, stochastic=True) + "\n")
    print("\n" + format_table_4_navigation(data) + "\n")
    print("\n" + format_table_5_branching(data) + "\n")


if __name__ == "__main__":
    main()
