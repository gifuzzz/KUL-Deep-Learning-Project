#!/usr/bin/env python3
"""Phase 11: Record Representative Trajectories for Visual Debugger.

Records matched trajectories comparing:
1. Variant A (Current: Local CNN + Memory CNN)
2. Variant B (Memory Only: Memory CNN only)
3. Variant C (Memory Crop: Memory CNN + Local Crop from Memory)

Matched scenarios:
- Matched Seed 50012: Full Rescue, Spawn 13-20 (Difficult multi-room corridor)
- Matched Seed 50057: Full Rescue, Spawn 7-12 (Medium search and return)
- Matched Seed 60000: Pure Exploration (Building coverage and loop avoidance)

Saves all trajectories to `trajectories/phase11/` as `.json.gz` files compatible with visual debugger.
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

TRAJECTORIES_DIR = ROOT / "trajectories" / "phase11"


def main():
    TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)

    dummy_env = PreviousActionWrapper(RescueEnv(reward_shaping=False, geometry_mode="edge"))

    ckpt_a = ROOT / "models/phase11_variant_a/final_model.zip"
    ckpt_b = ROOT / "models/phase11_variant_b/final_model.zip"
    ckpt_c = ROOT / "models/phase11_variant_c/final_model.zip"

    print("Loading Phase 11 models for trajectory recording...")
    model_a = load_checkpoint(ckpt_a, env=dummy_env, device="cpu")
    model_b = load_checkpoint(ckpt_b, env=dummy_env, device="cpu")
    model_c = load_checkpoint(ckpt_c, env=dummy_env, device="cpu")
    dummy_env.close()

    models = [
        ("variant_a_current", model_a, ckpt_a),
        ("variant_b_memory_only", model_b, ckpt_b),
        ("variant_c_memory_crop", model_c, ckpt_c),
    ]

    # Scenario 1: Spawn 13-20 on Seed 50012 (Difficult multi-room rescue)
    print("\n--- Recording Scenario 1: Spawn 13-20 (Seed 50012) ---")
    for name, model, ckpt in models:
        env = PreviousActionWrapper(RescueEnv(
            task_type="full",
            geometry_mode="edge",
            reward_shaping=False,
            distance_metric="spawn",
            spatial_memory_mode="none",
        ))
        rec = record_episode(
            model=model,
            env=env,
            seed=50012,
            options={"victim_distance": 20, "min_victim_distance": 13, "max_steps": 192},
            deterministic=True,
            model_path=str(ckpt.relative_to(ROOT)),
        )
        p = TRAJECTORIES_DIR / f"{name}_spawn13_20_s50012.json.gz"
        rec.save(p)
        print(f"  [{name}] Saved: {p.name} | Steps: {len(rec.steps)} | Rescued: {rec.summary.get('victim_rescued', False)}")
        env.close()

    # Scenario 2: Spawn 7-12 on Seed 50057 (Medium search and return)
    print("\n--- Recording Scenario 2: Spawn 7-12 (Seed 50057) ---")
    for name, model, ckpt in models:
        env = PreviousActionWrapper(RescueEnv(
            task_type="full",
            geometry_mode="edge",
            reward_shaping=False,
            distance_metric="spawn",
            spatial_memory_mode="none",
        ))
        rec = record_episode(
            model=model,
            env=env,
            seed=50057,
            options={"victim_distance": 12, "min_victim_distance": 7, "max_steps": 128},
            deterministic=True,
            model_path=str(ckpt.relative_to(ROOT)),
        )
        p = TRAJECTORIES_DIR / f"{name}_spawn7_12_s50057.json.gz"
        rec.save(p)
        print(f"  [{name}] Saved: {p.name} | Steps: {len(rec.steps)} | Rescued: {rec.summary.get('victim_rescued', False)}")
        env.close()

    # Scenario 3: Pure Exploration on Seed 60000 (Building coverage & loop behavior)
    print("\n--- Recording Scenario 3: Pure Exploration (Seed 60000) ---")
    for name, model, ckpt in models:
        env = PreviousActionWrapper(RescueEnv(
            task_type="exploration_only",
            geometry_mode="edge",
            reward_shaping=False,
            max_steps=384,
            distance_metric="spawn",
            spatial_memory_mode="none",
        ))
        rec = record_episode(
            model=model,
            env=env,
            seed=60000,
            deterministic=True,
            model_path=str(ckpt.relative_to(ROOT)),
        )
        p = TRAJECTORIES_DIR / f"{name}_exploration_s60000.json.gz"
        rec.save(p)
        print(f"  [{name}] Saved: {p.name} | Steps: {len(rec.steps)} | Visited: {rec.summary.get('unique_cells_visited', 0)}")
        env.close()

    print(f"\nAll representative trajectories saved to {TRAJECTORIES_DIR}")


if __name__ == "__main__":
    main()
