#!/usr/bin/env python3
"""Record representative Phase 8 trajectories for the Visual Debugger.

Required trajectories:
1. Pure exploration run for Variant A (Control, 9-ch) on Seed 60000.
2. Pure exploration run for Variant B (Binary Visited, 10-ch) on Seed 60000.
3. Pure exploration run for Variant C (Visit Recency, 10-ch) on Seed 60000.
4. Pure exploration run for Variant B with Channel 9 zero-ablated on Seed 60000.
5. Distant rescue run for Variant B (Binary Visited) on Seed 50000.
6. Distant rescue run for Variant C (Visit Recency) on Seed 50000.
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

TRAJECTORIES_DIR = ROOT / "trajectories" / "phase8"


def main():
    TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_a = ROOT / "models/phase8_control/stage3_spawn_curriculum/best_model.zip"
    ckpt_b = ROOT / "models/phase8_binary_visited/stage3_spawn_curriculum/best_model.zip"
    ckpt_c = ROOT / "models/phase8_recency/stage3_spawn_curriculum/best_model.zip"

    print("Loading models...")
    model_a = load_checkpoint(ckpt_a, device="cpu")
    model_b = load_checkpoint(ckpt_b, device="cpu")
    model_c = load_checkpoint(ckpt_c, device="cpu")

    # 1. Pure Exploration Run for Variant A (Control, 9-ch) - Seed 60000
    print("\n[1/6] Recording: Pure Exploration for Variant A (Control, Seed 60000)...")
    env_exp_a = PreviousActionWrapper(RescueEnv(
        task_type="exploration_only",
        max_steps=384,
        reward_shaping=False,
        geometry_mode="edge",
        spatial_memory_mode="none",
        exploration_reward_mode="normalized_continuous",
    ))
    rec_exp_a = record_episode(
        model=model_a,
        env=env_exp_a,
        seed=60000,
        deterministic=True,
        model_path=str(ckpt_a.relative_to(ROOT)),
    )
    p1 = TRAJECTORIES_DIR / "variant_a_exploration_s60000.json.gz"
    rec_exp_a.save(p1)
    print(f"  Saved: {p1} ({rec_exp_a.summary['category']}, steps: {rec_exp_a.summary['total_steps']}, obs_cov: {rec_exp_a.summary['explored_coverage']:.2f}, vis_cov: {rec_exp_a.summary.get('visited_coverage', 0.0):.2f})")

    # 2. Pure Exploration Run for Variant B (Binary Visited, 10-ch) - Seed 60000
    print("\n[2/6] Recording: Pure Exploration for Variant B (Binary Visited, Seed 60000)...")
    env_exp_b = PreviousActionWrapper(RescueEnv(
        task_type="exploration_only",
        max_steps=384,
        reward_shaping=False,
        geometry_mode="edge",
        spatial_memory_mode="visited_binary",
        exploration_reward_mode="normalized_continuous",
    ))
    rec_exp_b = record_episode(
        model=model_b,
        env=env_exp_b,
        seed=60000,
        deterministic=True,
        model_path=str(ckpt_b.relative_to(ROOT)),
    )
    p2 = TRAJECTORIES_DIR / "variant_b_exploration_s60000.json.gz"
    rec_exp_b.save(p2)
    print(f"  Saved: {p2} ({rec_exp_b.summary['category']}, steps: {rec_exp_b.summary['total_steps']}, obs_cov: {rec_exp_b.summary['explored_coverage']:.2f}, vis_cov: {rec_exp_b.summary.get('visited_coverage', 0.0):.2f})")

    # 3. Pure Exploration Run for Variant C (Visit Recency, 10-ch) - Seed 60000
    print("\n[3/6] Recording: Pure Exploration for Variant C (Visit Recency, Seed 60000)...")
    env_exp_c = PreviousActionWrapper(RescueEnv(
        task_type="exploration_only",
        max_steps=384,
        reward_shaping=False,
        geometry_mode="edge",
        spatial_memory_mode="visit_recency",
        recency_tau=40.0,
        exploration_reward_mode="normalized_continuous",
    ))
    rec_exp_c = record_episode(
        model=model_c,
        env=env_exp_c,
        seed=60000,
        deterministic=True,
        model_path=str(ckpt_c.relative_to(ROOT)),
    )
    p3 = TRAJECTORIES_DIR / "variant_c_exploration_s60000.json.gz"
    rec_exp_c.save(p3)
    print(f"  Saved: {p3} ({rec_exp_c.summary['category']}, steps: {rec_exp_c.summary['total_steps']}, obs_cov: {rec_exp_c.summary['explored_coverage']:.2f}, vis_cov: {rec_exp_c.summary.get('visited_coverage', 0.0):.2f})")

    # 4. Pure Exploration for Variant B with Channel 9 Zero-Ablated - Seed 60000
    print("\n[4/6] Recording: Pure Exploration for Variant B Zero-Ablated (Seed 60000)...")
    rec_exp_b_zero = record_episode(
        model=model_b,
        env=env_exp_b,
        seed=60000,
        deterministic=True,
        model_path=str(ckpt_b.relative_to(ROOT)),
        zero_ablate_ch9=True,
    )
    p4 = TRAJECTORIES_DIR / "variant_b_zero_ablated_s60000.json.gz"
    rec_exp_b_zero.save(p4)
    print(f"  Saved: {p4} ({rec_exp_b_zero.summary['category']}, steps: {rec_exp_b_zero.summary['total_steps']}, obs_cov: {rec_exp_b_zero.summary['explored_coverage']:.2f}, vis_cov: {rec_exp_b_zero.summary.get('visited_coverage', 0.0):.2f})")

    # 5. Distant Rescue Run for Variant B (Binary Visited) - Seed 50000
    print("\n[5/6] Recording: Distant Rescue for Variant B (Seed 50000)...")
    env_dist_b = PreviousActionWrapper(RescueEnv(
        victim_distance=None,
        max_steps=384,
        reward_shaping=False,
        distance_metric="spawn",
        geometry_mode="edge",
        spatial_memory_mode="visited_binary",
        exploration_reward_mode="normalized_continuous",
    ))
    rec_dist_b = record_episode(
        model=model_b,
        env=env_dist_b,
        seed=50000,
        deterministic=True,
        model_path=str(ckpt_b.relative_to(ROOT)),
    )
    p5 = TRAJECTORIES_DIR / "variant_b_distant_rescue_s50000.json.gz"
    rec_dist_b.save(p5)
    print(f"  Saved: {p5} ({rec_dist_b.summary['category']}, steps: {rec_dist_b.summary['total_steps']}, rescue: {rec_dist_b.summary['rescue_completed']})")

    # 6. Distant Rescue Run for Variant C (Visit Recency) - Seed 50000
    print("\n[6/6] Recording: Distant Rescue for Variant C (Seed 50000)...")
    env_dist_c = PreviousActionWrapper(RescueEnv(
        victim_distance=None,
        max_steps=384,
        reward_shaping=False,
        distance_metric="spawn",
        geometry_mode="edge",
        spatial_memory_mode="visit_recency",
        recency_tau=40.0,
        exploration_reward_mode="normalized_continuous",
    ))
    rec_dist_c = record_episode(
        model=model_c,
        env=env_dist_c,
        seed=50000,
        deterministic=True,
        model_path=str(ckpt_c.relative_to(ROOT)),
    )
    p6 = TRAJECTORIES_DIR / "variant_c_distant_rescue_s50000.json.gz"
    rec_dist_c.save(p6)
    print(f"  Saved: {p6} ({rec_dist_c.summary['category']}, steps: {rec_dist_c.summary['total_steps']}, rescue: {rec_dist_c.summary['rescue_completed']})")

    print(f"\nAll Phase 8 representative trajectories successfully recorded to: {TRAJECTORIES_DIR}")


if __name__ == "__main__":
    main()
