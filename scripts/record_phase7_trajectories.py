#!/usr/bin/env python3
"""Record representative Phase 7 trajectories for the Visual Debugger.

Required trajectories:
1. Pure exploration run for Variant B (seed 60000).
2. Pure exploration run for Variant C (seed 60000).
3. Distant rescue success for Variant B (seed 50093).
4. Distant rescue failure from doorway oscillation (seed 50057 on Baseline).
5. Distant rescue failure from stationary waiting (seed 50093 on Baseline).
6. Baseline vs. Variant B comparison on identical distant seed (seed 50093).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rescue_env.envs.single_agent import RescueEnv
from rescue_env.recorder import record_episode
from rescue_env.wrappers import PreviousActionWrapper
from training.train_ppo import load_checkpoint

TRAJECTORIES_DIR = ROOT / "trajectories" / "phase7"


def main():
    TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_baseline = ROOT / "models/phase6_spawn_curriculum/production_250k/stage3_spawn_curriculum/best_model.zip"
    ckpt_var_b = ROOT / "models/phase7_explore20/stage3_spawn_curriculum/best_model.zip"
    ckpt_var_c = ROOT / "models/phase7_explore40/stage3_spawn_curriculum/best_model.zip"

    print("Loading models...")
    model_baseline = load_checkpoint(ckpt_baseline, device="cpu")
    model_b = load_checkpoint(ckpt_var_b, device="cpu")
    model_c = load_checkpoint(ckpt_var_c, device="cpu")

    # 1. Pure Exploration Run for Variant B (Seed 60000)
    print("\n[1/6] Recording: Pure Exploration for Variant B (Seed 60000)...")
    env_exp_b = PreviousActionWrapper(RescueEnv(
        task_type="exploration_only",
        max_steps=384,
        reward_shaping=False,
        geometry_mode="edge",
    ))
    rec_exp_b = record_episode(
        model=model_b,
        env=env_exp_b,
        seed=60000,
        deterministic=True,
        model_path=str(ckpt_var_b.relative_to(ROOT)),
    )
    p1 = TRAJECTORIES_DIR / "pure_exploration_variant_b_seed60000.json.gz"
    rec_exp_b.save(p1)
    print(f"  Saved: {p1} ({rec_exp_b.summary['category']}, steps: {rec_exp_b.summary['total_steps']}, cov: {rec_exp_b.summary['explored_coverage']:.2f})")

    # 2. Pure Exploration Run for Variant C (Seed 60000)
    print("\n[2/6] Recording: Pure Exploration for Variant C (Seed 60000)...")
    env_exp_c = PreviousActionWrapper(RescueEnv(
        task_type="exploration_only",
        max_steps=384,
        reward_shaping=False,
        geometry_mode="edge",
    ))
    rec_exp_c = record_episode(
        model=model_c,
        env=env_exp_c,
        seed=60000,
        deterministic=True,
        model_path=str(ckpt_var_c.relative_to(ROOT)),
    )
    p2 = TRAJECTORIES_DIR / "pure_exploration_variant_c_seed60000.json.gz"
    rec_exp_c.save(p2)
    print(f"  Saved: {p2} ({rec_exp_c.summary['category']}, steps: {rec_exp_c.summary['total_steps']}, cov: {rec_exp_c.summary['explored_coverage']:.2f})")

    # 3. Distant Rescue Success for Variant B (Seed 50093)
    print("\n[3/6] Recording: Distant Rescue Success for Variant B (Seed 50093)...")
    env_dist = PreviousActionWrapper(RescueEnv(
        victim_distance=None,
        max_steps=384,
        reward_shaping=False,
        distance_metric="spawn",
        geometry_mode="edge",
    ))
    rec_succ_b = record_episode(
        model=model_b,
        env=env_dist,
        seed=50093,
        deterministic=True,
        model_path=str(ckpt_var_b.relative_to(ROOT)),
    )
    p3 = TRAJECTORIES_DIR / "distant_rescue_success_variant_b_seed50093.json.gz"
    rec_succ_b.save(p3)
    print(f"  Saved: {p3} ({rec_succ_b.summary['category']}, steps: {rec_succ_b.summary['total_steps']}, rescue: {rec_succ_b.summary['rescue_completed']})")

    # 4. Distant Rescue Failure from Doorway Oscillation (Seed 50057 on Baseline)
    print("\n[4/6] Recording: Distant Rescue Failure from Doorway Oscillation (Seed 50057 on Baseline)...")
    rec_osc_base = record_episode(
        model=model_baseline,
        env=env_dist,
        seed=50057,
        deterministic=True,
        model_path=str(ckpt_baseline.relative_to(ROOT)),
    )
    p4 = TRAJECTORIES_DIR / "distant_failure_doorway_oscillation_baseline_seed50057.json.gz"
    rec_osc_base.save(p4)
    print(f"  Saved: {p4} ({rec_osc_base.summary['category']}, osc: {rec_osc_base.summary['oscillation_count']}, steps: {rec_osc_base.summary['total_steps']})")

    # 5. Distant Rescue Failure from Stationary Waiting (Seed 50093 on Baseline)
    print("\n[5/6] Recording: Distant Rescue Failure from Stationary Waiting (Seed 50093 on Baseline)...")
    rec_wait_base = record_episode(
        model=model_baseline,
        env=env_dist,
        seed=50093,
        deterministic=True,
        model_path=str(ckpt_baseline.relative_to(ROOT)),
    )
    p5 = TRAJECTORIES_DIR / "distant_failure_stationary_wait_baseline_seed50093.json.gz"
    rec_wait_base.save(p5)
    print(f"  Saved: {p5} ({rec_wait_base.summary['category']}, steps: {rec_wait_base.summary['total_steps']}, timeout: {rec_wait_base.summary['timeout']})")

    # 6. Baseline vs Variant B on identical seed 50093
    print("\n[6/6] Confirming: Baseline vs. Variant B Matched Pair on Seed 50093...")
    p6_base = TRAJECTORIES_DIR / "identical_seed50093_baseline_failure.json.gz"
    p6_varb = TRAJECTORIES_DIR / "identical_seed50093_variant_b_success.json.gz"
    rec_wait_base.save(p6_base)
    rec_succ_b.save(p6_varb)
    print(f"  Saved Matched Pair:\n    Baseline: {p6_base} (Rescue: {rec_wait_base.summary['rescue_completed']}, Steps: {rec_wait_base.summary['total_steps']})\n    Variant B: {p6_varb} (Rescue: {rec_succ_b.summary['rescue_completed']}, Steps: {rec_succ_b.summary['total_steps']})")

    print("\nAll representative trajectories successfully recorded to:", TRAJECTORIES_DIR)


if __name__ == "__main__":
    main()
