#!/usr/bin/env python3
"""Phase 11: Compile Variant C Zero-Shot & Fine-Tuning Progression Tables.

Reads:
- models/phase11_evaluations/variant_c_zero_shot_audit.json
- models/phase11_evaluations/eval_variant_c_zero_shot.json (0 steps)
- models/phase11_evaluations/eval_variant_c_10k.json (10,000 steps)
- models/phase11_evaluations/eval_variant_c_25k.json (25,000 steps)
- models/phase11_evaluations/eval_variant_c_memory_crop.json (50,000 steps)
- models/phase10_evaluations/eval_variant_b_no_osc.json (Phase 10 Base)

Outputs markdown comparison tables answering:
1. Can Variant C reproduce the pretrained Phase 10 policy before fine-tuning?
2. Does fine-tuning help or cause forgetting?
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    {
        "id": "base_p10",
        "name": "Phase 10 Base (Var A)",
        "steps": "0 (Source)",
        "file": ROOT / "models/phase10_evaluations/eval_variant_b_no_osc.json",
    },
    {
        "id": "c_zero_shot",
        "name": "Variant C (0k Zero-Shot)",
        "steps": "0 steps",
        "file": ROOT / "models/phase11_evaluations/eval_variant_c_zero_shot.json",
    },
    {
        "id": "c_10k",
        "name": "Variant C (10k steps)",
        "steps": "10,000 steps",
        "file": ROOT / "models/phase11_evaluations/eval_variant_c_10k.json",
    },
    {
        "id": "c_25k",
        "name": "Variant C (25k steps)",
        "steps": "25,000 steps",
        "file": ROOT / "models/phase11_evaluations/eval_variant_c_25k.json",
    },
    {
        "id": "c_50k",
        "name": "Variant C (50k steps)",
        "steps": "50,000 steps",
        "file": ROOT / "models/phase11_evaluations/eval_variant_c_memory_crop.json",
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
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            data[m["id"]] = json.load(f)
    return data


def format_table_spawn(data, stochastic=False):
    mode = "Stochastic" if stochastic else "Deterministic"
    key = "spawn_ladder_stochastic" if stochastic else "spawn_ladder_deterministic"
    lines = [
        f"### Spawn-Relative Ladder Progression ({mode}, 200 eps/tier)",
        "",
        "| Spawn Distance Tier | Phase 10 Base | Variant C (0k Zero-Shot) | Variant C (10k) | Variant C (25k) | Variant C (50k) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for t_id, t_label in SPAWN_TIERS:
        row = [f"**{t_label}**"]
        for m in MODELS:
            mid = m["id"]
            if mid not in data:
                row.append("Pending...")
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
            avg_row.append("Pending...")
            continue
        tiers = data[mid][key]
        avg = sum(t["rescue_rate_pct"] for t in tiers.values()) / len(tiers)
        avg_row.append(f"**{avg:.1f}%**")
    lines.append("| " + " | ".join(avg_row) + " |")

    return "\n".join(lines)


def format_table_exit(data, stochastic=False):
    mode = "Stochastic" if stochastic else "Deterministic"
    key = "exit_ladder_stochastic" if stochastic else "exit_ladder_deterministic"
    lines = [
        f"### Exit-Relative Ladder Progression ({mode}, 200 eps/tier)",
        "",
        "| Exit Distance Tier | Phase 10 Base | Variant C (0k Zero-Shot) | Variant C (10k) | Variant C (25k) | Variant C (50k) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for t_id, t_label in EXIT_TIERS:
        row = [f"**{t_label}**"]
        for m in MODELS:
            mid = m["id"]
            if mid not in data:
                row.append("Pending...")
                continue
            tier_data = data[mid][key].get(t_id, {})
            r_rate = tier_data.get("rescue_rate_pct", 0.0)
            row.append(f"{r_rate:.1f}%")
        lines.append("| " + " | ".join(row) + " |")

    avg_row = ["**Mean Across Tiers**"]
    for m in MODELS:
        mid = m["id"]
        if mid not in data:
            avg_row.append("Pending...")
            continue
        tiers = data[mid][key]
        avg = sum(t["rescue_rate_pct"] for t in tiers.values()) / len(tiers)
        avg_row.append(f"**{avg:.1f}%**")
    lines.append("| " + " | ".join(avg_row) + " |")

    return "\n".join(lines)


def format_table_navigation(data):
    lines = [
        "### Pure Navigation & Exploration Benchmark Progression (200 eps, 384 max steps)",
        "",
        "| Metric | Phase 10 Base | Variant C (0k Zero-Shot) | Variant C (10k) | Variant C (25k) | Variant C (50k) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    metrics = [
        ("mean_visited_coverage_pct", "Visited Coverage % (Det)", "{:.2f}%", True),
        ("mean_visited_coverage_pct", "Visited Coverage % (Stoch)", "{:.2f}%", False),
        ("mean_unique_cells_visited", "Unique Cells Visited (Det)", "{:.1f}", True),
        ("mean_unique_cells_visited", "Unique Cells Visited (Stoch)", "{:.1f}", False),
        ("mean_wait_fraction_pct", "Wait Fraction % (Det)", "{:.1f}%", True),
        ("mean_wait_fraction_pct", "Wait Fraction % (Stoch)", "{:.1f}%", False),
        ("mean_oscillation_pct", "2-Cell Oscillation % (Det)", "{:.1f}%", True),
    ]

    for field, label, fmt, is_det in metrics:
        row = [f"**{label}**"]
        key = "navigation_deterministic" if is_det else "navigation_stochastic"
        for m in MODELS:
            mid = m["id"]
            if mid not in data:
                row.append("Pending...")
                continue
            val = data[mid][key].get(field, 0.0)
            row.append(fmt.format(val))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main():
    data = load_eval_data()
    print(f"Loaded progression data for {len(data)} / {len(MODELS)} models.")

    print("\n" + format_table_spawn(data, stochastic=False) + "\n")
    print("\n" + format_table_spawn(data, stochastic=True) + "\n")
    print("\n" + format_table_exit(data, stochastic=False) + "\n")
    print("\n" + format_table_exit(data, stochastic=True) + "\n")
    print("\n" + format_table_navigation(data) + "\n")


if __name__ == "__main__":
    main()
