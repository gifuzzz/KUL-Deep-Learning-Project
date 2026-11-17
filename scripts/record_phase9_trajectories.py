#!/usr/bin/env python3
"""Record representative Phase 9 trajectories for the Visual Debugger.

Required trajectories:
1. Stage B1 Branching for Variant B (Binary Visited, 10-ch) on Seed 60000.
2. Stage B1 Branching for Variant B with Channel 9 zero-ablated on Seed 60000.
3. Stage B1 Branching for Variant A (Control, 9-ch) on Seed 60000.
4. Stage B2 Multiple Junctions for Variant B on Seed 60000.
5. Stage B4 Full Exploration for Variant B on Seed 60000.
6. Distant Rescue Run for Variant B on Seed 50000.
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

TRAJECTORIES_DIR = ROOT / "trajectories" / "phase9"


def main():
    TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_a = ROOT / "models/phase9_control/best_navigation_model.zip"
    ckpt_b = ROOT / "models/phase9_binary_visited/best_navigation_model.zip"

    # Fallback to final if best doesn't exist
    if not ckpt_a.exists():
        ckpt_a = ROOT / "models/phase9_control/final_model.zip"
    if not ckpt_b.exists():
        ckpt_b = ROOT / "models/phase9_binary_visited/final_model.zip"

    print("Loading models...")
    model_a = load_checkpoint(ckpt_a, device="cpu")
    model_b = load_checkpoint(ckpt_b, device="cpu")

    # 1. Stage B1 Branching for Variant B (Seed 60000)
    print("\n[1/6] Recording: Stage B1 for Variant B (Seed 60000)...")
    env_b1_b = PreviousActionWrapper(RescueEnv(
        task_type="branching_exploration",
        spatial_memory_mode="visited_binary",
        reward_shaping=False,
    ))
    rec_b1_b = record_episode(
        model=model_b,
        env=env_b1_b,
        seed=60000,
        deterministic=True,
        options={"branching_stage": "B1"},
        model_path=str(ckpt_b.relative_to(ROOT)),
    )
    p1 = TRAJECTORIES_DIR / "variant_b_branching_b1_s60000.json.gz"
    rec_b1_b.save(p1)
    print(f"  Saved: {p1} (steps: {len(rec_b1_b.steps)}, correct: {rec_b1_b.summary.get('correct_branch_selected')})")

    # 2. Stage B1 Branching for Variant B with Channel 9 Zero-Ablated (Seed 60000)
    print("\n[2/6] Recording: Stage B1 for Variant B Zero-Ablated (Seed 60000)...")
    rec_b1_zero = record_episode(
        model=model_b,
        env=env_b1_b,
        seed=60000,
        deterministic=True,
        options={"branching_stage": "B1"},
        model_path=str(ckpt_b.relative_to(ROOT)),
        zero_ablate_ch9=True,
    )
    p2 = TRAJECTORIES_DIR / "variant_b_zero_ablated_b1_s60000.json.gz"
    rec_b1_zero.save(p2)
    print(f"  Saved: {p2} (steps: {len(rec_b1_zero.steps)}, correct: {rec_b1_zero.summary.get('correct_branch_selected')})")

    # 3. Stage B1 Branching for Variant A (Control, 9-ch) (Seed 60000)
    print("\n[3/6] Recording: Stage B1 for Variant A Control (Seed 60000)...")
    env_b1_a = PreviousActionWrapper(RescueEnv(
        task_type="branching_exploration",
        spatial_memory_mode="none",
        reward_shaping=False,
    ))
    rec_b1_a = record_episode(
        model=model_a,
        env=env_b1_a,
        seed=60000,
        deterministic=True,
        options={"branching_stage": "B1"},
        model_path=str(ckpt_a.relative_to(ROOT)),
    )
    p3 = TRAJECTORIES_DIR / "variant_a_branching_b1_s60000.json.gz"
    rec_b1_a.save(p3)
    print(f"  Saved: {p3} (steps: {len(rec_b1_a.steps)}, correct: {rec_b1_a.summary.get('correct_branch_selected')})")

    # 4. Stage B2 Multiple Junctions for Variant B (Seed 60000)
    print("\n[4/6] Recording: Stage B2 for Variant B (Seed 60000)...")
    rec_b2_b = record_episode(
        model=model_b,
        env=env_b1_b,
        seed=60000,
        deterministic=True,
        options={"branching_stage": "B2"},
        model_path=str(ckpt_b.relative_to(ROOT)),
    )
    p4 = TRAJECTORIES_DIR / "variant_b_branching_b2_s60000.json.gz"
    rec_b2_b.save(p4)
    print(f"  Saved: {p4} (steps: {len(rec_b2_b.steps)}, correct: {rec_b2_b.summary.get('correct_branch_selected')})")

    # 5. Stage B4 Full Exploration for Variant B (Seed 60000)
    print("\n[5/6] Recording: Stage B4 Full Exploration for Variant B (Seed 60000)...")
    rec_b4_b = record_episode(
        model=model_b,
        env=env_b1_b,
        seed=60000,
        deterministic=True,
        options={"branching_stage": "B4"},
        model_path=str(ckpt_b.relative_to(ROOT)),
    )
    p5 = TRAJECTORIES_DIR / "variant_b_exploration_b4_s60000.json.gz"
    rec_b4_b.save(p5)
    print(f"  Saved: {p5} (steps: {len(rec_b4_b.steps)})")

    # 6. Distant Rescue Run for Variant B (Seed 50000)
    print("\n[6/6] Recording: Distant Rescue for Variant B (Seed 50000)...")
    env_dist_b = PreviousActionWrapper(RescueEnv(
        task_type="full",
        victim_distance=None,
        min_victim_distance=1,
        max_steps=384,
        reward_shaping=False,
        distance_metric="spawn",
        geometry_mode="edge",
        spatial_memory_mode="visited_binary",
    ))
    rec_dist_b = record_episode(
        model=model_b,
        env=env_dist_b,
        seed=50000,
        deterministic=True,
        model_path=str(ckpt_b.relative_to(ROOT)),
    )
    p6 = TRAJECTORIES_DIR / "variant_b_distant_rescue_s50000.json.gz"
    rec_dist_b.save(p6)
    print(f"  Saved: {p6} (steps: {len(rec_dist_b.steps)}, rescue: {rec_dist_b.summary.get('rescue_completed')})")

    print(f"\nAll Phase 9 representative trajectories successfully recorded to: {TRAJECTORIES_DIR}")


if __name__ == "__main__":
    main()
