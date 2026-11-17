#!/usr/bin/env python3
"""Aggregate and Display Complete Phase 9 Empirical Findings."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "models/phase9_evaluations"

def load_json(filename):
    p = EVAL_DIR / filename
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

def main():
    print("=" * 100)
    print("PHASE 9 EXPERIMENTAL RESULTS: COMPLETE COMPARATIVE ANALYSIS")
    print("=" * 100)

    # 1. Counterfactual Memory Test
    cf_init_a = load_json("initial_counterfactual_variant_a.json")
    cf_init_b = load_json("initial_counterfactual_variant_b.json")
    cf_train_a = load_json("trained_counterfactual_variant_a.json")
    cf_train_b = load_json("trained_counterfactual_variant_b.json")
    cf_train_b_zero = load_json("trained_counterfactual_variant_b_zero_ablated.json")

    print("\n--- 1. COUNTERFACTUAL MEMORY DIAGNOSTIC (100 matched pairs / 200 states) ---")
    print(f"{'Condition':<35} | {'Unvisited Pref %':<16} | {'Mean Logit Delta':<17} | {'Mean Prob Delta':<16} | {'P(Unvisited)':<12} | {'P(Visited)':<10}")
    print("-" * 115)
    for label, d in [
        ("Variant A Initial (9-ch Control)", cf_init_a),
        ("Variant B Initial (10-ch Binary)", cf_init_b),
        ("Variant A Trained (50k Control)", cf_train_a),
        ("Variant B Trained (50k Binary)", cf_train_b),
        ("Variant B Trained (Zero-Ablated Ch9)", cf_train_b_zero),
    ]:
        if d:
            print(f"{label:<35} | {d['unvisited_preference_rate_pct']:<16.2f}% | {d['mean_logit_delta']:<17.6f} | {d['mean_prob_delta']:<16.6f} | {d['mean_prob_unvisited']:<12.4f} | {d['mean_prob_visited']:<10.4f}")

    # 2. Navigation Benchmarks
    nav_init_a = load_json("initial_nav_variant_a.json")
    nav_init_b = load_json("initial_nav_variant_b.json")
    nav_train_a = load_json("trained_nav_variant_a.json")
    nav_train_b = load_json("trained_nav_variant_b.json")
    nav_train_b_zero = load_json("trained_nav_variant_b_zero_ablated.json")

    print("\n--- 2. NAVIGATION BENCHMARKS (Deterministic, 200 episodes/stage) ---")
    print(f"{'Condition':<28} | {'Stage':<5} | {'Correct %':<9} | {'NewBr %':<7} | {'Cells':<6} | {'Rooms':<6} | {'Wait %':<7} | {'Osc %':<6} | {'>50 %':<6}")
    print("-" * 95)
    for label, d in [
        ("Variant A Initial (Control)", nav_init_a),
        ("Variant B Initial (Binary)", nav_init_b),
        ("Variant A Trained (Control)", nav_train_a),
        ("Variant B Trained (Binary)", nav_train_b),
        ("Variant B Trained (Zero-Abl)", nav_train_b_zero),
    ]:
        if d:
            for st in ("B1", "B2", "B4"):
                m = d["stages"][st]["deterministic"]
                corr = f"{m['correct_branch_selection_rate']:.1f}%" if m['correct_branch_selection_rate'] is not None else "N/A"
                print(f"{label:<28} | {st:<5} | {corr:<9} | {m['new_branch_entry_rate']:<6.1f}% | {m['mean_unique_cells_visited']:<6.1f} | {m['mean_unique_rooms_entered']:<6.1f} | {m['mean_wait_fraction']*100:<6.1f}% | {m['mean_oscillation_pct']:<5.1f}% | {m['new_territory_after_step_50_rate']:<5.1f}%")
            print("-" * 95)

    # 3. Rescue Transfer Benchmarks
    res_a = load_json("transfer_rescue_variant_a.json")
    res_b = load_json("transfer_rescue_variant_b.json")
    res_b_zero = load_json("transfer_rescue_variant_b_zero_ablated.json")

    print("\n--- 3. FULL RESCUE TRANSFER: SPAWN LADDER (200 episodes/tier, Unshaped) ---")
    print(f"{'Tier':<15} | {'Variant A Trained':<22} | {'Variant B Trained':<22} | {'Variant B Zero-Ablated':<22}")
    print(f"{'':<15} | {'Rescue | Pickup | Ret':<22} | {'Rescue | Pickup | Ret':<22} | {'Rescue | Pickup | Ret':<22}")
    print("-" * 88)
    for tier in ("spawn_2_6", "spawn_7_12", "spawn_13_20", "spawn_21_35", "spawn_distant"):
        ra = res_a["spawn_ladder"][tier]
        rb = res_b["spawn_ladder"][tier]
        rbz = res_b_zero["spawn_ladder"][tier]
        sa = f"{ra['rescue_rate']:4.1f}% | {ra['pickup_rate']:4.1f}% | {ra['return_conditional']:4.1f}%"
        sb = f"{rb['rescue_rate']:4.1f}% | {rb['pickup_rate']:4.1f}% | {rb['return_conditional']:4.1f}%"
        sbz = f"{rbz['rescue_rate']:4.1f}% | {rbz['pickup_rate']:4.1f}% | {rbz['return_conditional']:4.1f}%"
        print(f"{tier:<15} | {sa:<22} | {sb:<22} | {sbz:<22}")

    print("\n--- 4. FULL RESCUE TRANSFER: EXIT LADDER (200 episodes/tier, Unshaped) ---")
    print(f"{'Tier':<15} | {'Variant A Trained':<22} | {'Variant B Trained':<22} | {'Variant B Zero-Ablated':<22}")
    print("-" * 88)
    for tier in ("near", "medium", "bridge_a", "bridge_b", "bridge_c", "distant"):
        ra = res_a["exit_ladder"][tier]
        rb = res_b["exit_ladder"][tier]
        rbz = res_b_zero["exit_ladder"][tier]
        sa = f"{ra['rescue_rate']:4.1f}% | {ra['pickup_rate']:4.1f}% | {ra['return_conditional']:4.1f}%"
        sb = f"{rb['rescue_rate']:4.1f}% | {rb['pickup_rate']:4.1f}% | {rb['return_conditional']:4.1f}%"
        sbz = f"{rbz['rescue_rate']:4.1f}% | {rbz['pickup_rate']:4.1f}% | {rbz['return_conditional']:4.1f}%"
        print(f"{tier:<15} | {sa:<22} | {sb:<22} | {sbz:<22}")

    # 4. Channel 9 Audit Summary
    ch9 = load_json("phase9_channel9_audit.json")
    if ch9:
        print("\n--- 5. CHANNEL 9 WEIGHT & GRADIENT AUDIT ---")
        w = ch9["weights_progression"]["Variant B (Binary Visited)"]
        print(f"Step 0 Weight Norm: {w['Step 0 (init)']['norm']:.6f} ({w['Step 0 (init)']['nonzero_count']}/288 nonzero)")
        print(f"Final Weight Norm : {w['Step 50k (final)']['norm']:.6f} ({w['Step 50k (final)']['nonzero_count']}/288 nonzero, {w['Step 50k (final)']['ratio_to_pretrained_norm_pct']:.2f}% of pretrained)")
        opt = ch9["optimizer_moments"]["Variant B (Binary Visited)"]["Step 50k (final)"]
        print(f"Adam Moment Norms : Momentum = {opt['first_moment_norm']:.4e} ({opt['nonzero_first_moment']}/288), Variance = {opt['second_moment_norm']:.4e} ({opt['nonzero_second_moment']}/288)")
        ag = ch9["active_gradients"]
        print(f"Active Grad Norm  : {ag['channel9_grad_norm']:.4f} ({ag['ratio_to_baseline_pct']:.2f}% of baseline Ch 0-8)")
        sens = ch9["logit_sensitivity"]["zero_ablation"]
        print(f"Logit Shift       : Mean = {sens['mean_max_logit_delta']:.6f}, Peak = {sens['peak_max_logit_delta']:.6f}")
        print(f"TVD Prob Shift    : Mean = {sens['mean_total_variation_prob_delta']:.6f}, Peak = {sens['peak_total_variation_prob_delta']:.6f}")
        print(f"Value Shift       : Mean = {sens['mean_value_delta']:.6f}, Peak = {sens['peak_value_delta']:.6f}")

if __name__ == "__main__":
    main()
