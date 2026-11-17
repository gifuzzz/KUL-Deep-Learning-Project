#!/usr/bin/env python3
"""Phase 10: Record Representative Trajectories for Visual Debugger (Section 11).

Records trajectories from each variant on representative failure and success modes:
1. Variant A (Control): Deadlock and repeated-wait stalling (Seed 50000, exploration_only)
2. Variant B (No Oscillation Penalty): Active oscillation without wait (Seed 50057, full task, spawn_13_20)
3. Variant C (Repeated-Wait Penalty): Non-waiting exploration failure / timeout (Seed 50000, exploration_only)
4. Variant D (Combined Correction): Best-performing navigation & escape (Seed 50000, exploration_only)
5. Variant D (Combined Correction): Full rescue on difficult spawn distance (Seed 50012, full task, spawn_13_20)

Saves all trajectories to `trajectories/phase10/` as `.json.gz` files compatible with visual debugger.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rescue_env.envs.single_agent import RescueEnv
from rescue_env.recorder import record_episode
from rescue_env.wrappers import PreviousActionWrapper
from agents.recurrent_ppo import load_checkpoint

TRAJECTORIES_DIR = ROOT / "trajectories" / "phase10"


def main():
    TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)

    dummy_env = PreviousActionWrapper(RescueEnv(reward_shaping=False, geometry_mode="edge"))
    
    ckpt_a = ROOT / "models/phase10_variant_a/final_model.zip"
    ckpt_b = ROOT / "models/phase10_variant_b/final_model.zip"
    ckpt_c = ROOT / "models/phase10_variant_c/final_model.zip"
    ckpt_d = ROOT / "models/phase10_variant_d/final_model.zip"

    print("Loading models for trajectory recording...")
    model_a = load_checkpoint(ckpt_a, env=dummy_env, device="cpu")
    model_b = load_checkpoint(ckpt_b, env=dummy_env, device="cpu")
    model_c = load_checkpoint(ckpt_c, env=dummy_env, device="cpu")
    model_d = load_checkpoint(ckpt_d, env=dummy_env, device="cpu")
    dummy_env.close()

    # 1. Variant A: Repeated-wait deadlock stall (Seed 50000, exploration_only)
    print("\n[1/5] Recording Variant A: Repeated-wait deadlock stall (Seed 50000)...")
    env_nav = PreviousActionWrapper(RescueEnv(
        task_type="exploration_only",
        geometry_mode="edge",
        reward_shaping=False,
        max_steps=384,
        distance_metric="spawn",
        spatial_memory_mode="none",
    ))
    rec_a = record_episode(
        model=model_a,
        env=env_nav,
        seed=50000,
        deterministic=True,
        model_path=str(ckpt_a.relative_to(ROOT)),
    )
    p_a = TRAJECTORIES_DIR / "variant_a_deadlock_wait_stall_s50000.json.gz"
    rec_a.save(p_a)
    print(f"  Saved: {p_a} (steps: {len(rec_a.steps)}, wait_fraction: {rec_a.summary.get('wait_action_fraction', 0.0):.2f})")

    # 2. Variant B: Oscillation without wait (Seed 50057, full task, spawn_13_20)
    print("\n[2/5] Recording Variant B: Oscillation without wait (Seed 50057)...")
    env_rescue = PreviousActionWrapper(RescueEnv(
        task_type="full",
        geometry_mode="edge",
        reward_shaping=False,
        distance_metric="spawn",
        spatial_memory_mode="none",
    ))
    rec_b = record_episode(
        model=model_b,
        env=env_rescue,
        seed=50057,
        deterministic=True,
        options={"tier": "spawn_13_20"},
        model_path=str(ckpt_b.relative_to(ROOT)),
    )
    p_b = TRAJECTORIES_DIR / "variant_b_oscillation_no_wait_s50057.json.gz"
    rec_b.save(p_b)
    print(f"  Saved: {p_b} (steps: {len(rec_b.steps)}, osc_pct: {rec_b.summary.get('oscillation_percentage', 0.0):.1f}%, wait_frac: {rec_b.summary.get('wait_action_fraction', 0.0):.2f})")

    # 3. Variant C: Non-waiting exploration failure / alternative timeout (Seed 50000, exploration_only)
    print("\n[3/5] Recording Variant C: Non-waiting exploration failure (Seed 50000)...")
    rec_c = record_episode(
        model=model_c,
        env=env_nav,
        seed=50000,
        deterministic=True,
        model_path=str(ckpt_c.relative_to(ROOT)),
    )
    p_c = TRAJECTORIES_DIR / "variant_c_non_waiting_exploration_s50000.json.gz"
    rec_c.save(p_c)
    print(f"  Saved: {p_c} (steps: {len(rec_c.steps)}, wait_frac: {rec_c.summary.get('wait_action_fraction', 0.0):.2f}, visited_cells: {rec_c.summary.get('visited_reachable_cells', 0)})")

    # 4. Variant D: Best-performing navigation & escape (Seed 50000, exploration_only)
    print("\n[4/5] Recording Variant D: Best-performing navigation (Seed 50000)...")
    rec_d_nav = record_episode(
        model=model_d,
        env=env_nav,
        seed=50000,
        deterministic=True,
        model_path=str(ckpt_d.relative_to(ROOT)),
    )
    p_d_nav = TRAJECTORIES_DIR / "variant_d_navigation_escape_s50000.json.gz"
    rec_d_nav.save(p_d_nav)
    print(f"  Saved: {p_d_nav} (steps: {len(rec_d_nav.steps)}, wait_frac: {rec_d_nav.summary.get('wait_action_fraction', 0.0):.2f}, visited_cells: {rec_d_nav.summary.get('visited_reachable_cells', 0)})")

    # 5. Variant D: Difficult spawn rescue (Seed 50012, full task, spawn_13_20)
    print("\n[5/5] Recording Variant D: Difficult spawn rescue (Seed 50012)...")
    rec_d_rescue = record_episode(
        model=model_d,
        env=env_rescue,
        seed=50012,
        deterministic=True,
        options={"tier": "spawn_13_20"},
        model_path=str(ckpt_d.relative_to(ROOT)),
    )
    p_d_rescue = TRAJECTORIES_DIR / "variant_d_rescue_spawn_13_20_s50012.json.gz"
    rec_d_rescue.save(p_d_rescue)
    print(f"  Saved: {p_d_rescue} (steps: {len(rec_d_rescue.steps)}, rescued: {rec_d_rescue.summary.get('victims_rescued', 0)})")

    env_nav.close()
    env_rescue.close()
    print("\nAll Phase 10 visual debugger trajectories recorded successfully!")


if __name__ == "__main__":
    main()
