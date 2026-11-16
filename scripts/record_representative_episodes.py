"""Generate and record representative successful and failed episodes for the Visual Debugger.

Categories recorded:
1. Successful near rescue (Tier Near, d=2)
2. Successful medium rescue (Tier Medium, d=6)
3. Successful long-distance rescue (Tier Bridge B, d=10)
4. Successful distant multi-room rescue (Tier Distant, d=None)
5. Doorway oscillation failure (Carrier oscillation between two cells)
6. Carrier timeout failure (Picked up but ran out of time)
7. Discovered not picked up failure (Saw victim but failed to reach/pick up)
8. Exploration timeout failure (Failed to locate distant victim)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rescue_env.envs.single_agent import RescueEnv
from rescue_env.recorder import record_episode
from rescue_env.wrappers import PreviousActionWrapper
from training.evaluate import load_model

ROOT = Path(__file__).resolve().parents[1]
TRAJECTORIES_DIR = ROOT / "trajectories"


def generate_representative_trajectories(model_path: str = "models/phase4_return_curriculum/production_route_complexity_250k/final_model.zip"):
    TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)
    m_path = ROOT / model_path if not Path(model_path).is_absolute() else Path(model_path)
    print(f"Loading model: {m_path}")

    # 1. Successful Near Rescue (d=2, seed 42, deterministic)
    print("Recording: Successful Near Rescue...")
    env_near = PreviousActionWrapper(RescueEnv(victim_distance=2, max_steps=64, reward_shaping=False))
    model, _ = load_model(m_path, env=env_near, device="cpu")
    rec_near = record_episode(model, env_near, seed=42, deterministic=True, model_path=str(m_path.relative_to(ROOT)))
    rec_near.save(TRAJECTORIES_DIR / "successful_near_rescue_seed42_det.json.gz")
    print(f"  Saved: {TRAJECTORIES_DIR / 'successful_near_rescue_seed42_det.json.gz'} ({rec_near.summary['category']})")

    # 2. Successful Medium Rescue (d=6, seed 50001, deterministic)
    print("Recording: Successful Medium Rescue (Seed 50001)...")
    env_med = PreviousActionWrapper(RescueEnv(victim_distance=6, min_victim_distance=3, max_steps=128, reward_shaping=False))
    rec_med = record_episode(model, env_med, seed=50001, deterministic=True, model_path=str(m_path.relative_to(ROOT)))
    rec_med.save(TRAJECTORIES_DIR / "successful_medium_rescue_seed50001_det.json.gz")
    print(f"  Saved: {TRAJECTORIES_DIR / 'successful_medium_rescue_seed50001_det.json.gz'} ({rec_med.summary['category']})")

    # 3. Successful Long-Distance Rescue (Bridge B, d=10, seed 50006, stochastic)
    print("Recording: Successful Long-Distance Rescue (Bridge B, Seed 50006)...")
    env_bridge_b = PreviousActionWrapper(RescueEnv(victim_distance=10, min_victim_distance=7, max_steps=192, reward_shaping=False))
    rec_bridge = record_episode(model, env_bridge_b, seed=50006, deterministic=False, model_path=str(m_path.relative_to(ROOT)))
    rec_bridge.save(TRAJECTORIES_DIR / "successful_long_distance_rescue_bridge_b_seed50006_stoch.json.gz")
    print(f"  Saved: {TRAJECTORIES_DIR / 'successful_long_distance_rescue_bridge_b_seed50006_stoch.json.gz'} ({rec_bridge.summary['category']})")

    # 4. Successful Distant Multi-Room Rescue (Distant, d=None, seed 50068, deterministic)
    print("Recording: Successful Distant Multi-Room Rescue (Seed 50068)...")
    env_distant = PreviousActionWrapper(RescueEnv(victim_distance=None, max_steps=384, reward_shaping=False))
    rec_distant = record_episode(model, env_distant, seed=50068, deterministic=True, model_path=str(m_path.relative_to(ROOT)))
    rec_distant.save(TRAJECTORIES_DIR / "successful_distant_rescue_seed50068_det.json.gz")
    print(f"  Saved: {TRAJECTORIES_DIR / 'successful_distant_rescue_seed50068_det.json.gz'} ({rec_distant.summary['category']})")

    # 5. Successful Distant Multi-Room Rescue (Distant, d=None, seed 50094, deterministic)
    print("Recording: Successful Distant Multi-Room Rescue (Seed 50094)...")
    rec_distant_94 = record_episode(model, env_distant, seed=50094, deterministic=True, model_path=str(m_path.relative_to(ROOT)))
    rec_distant_94.save(TRAJECTORIES_DIR / "successful_distant_rescue_seed50094_det.json.gz")
    print(f"  Saved: {TRAJECTORIES_DIR / 'successful_distant_rescue_seed50094_det.json.gz'} ({rec_distant_94.summary['category']})")

    # 6. Doorway Oscillation Failure (Distant, seed 50057, deterministic, 341 oscillations)
    print("Recording: Doorway Oscillation Failure (Seed 50057)...")
    rec_osc = record_episode(model, env_distant, seed=50057, deterministic=True, model_path=str(m_path.relative_to(ROOT)))
    rec_osc.save(TRAJECTORIES_DIR / "doorway_oscillation_seed50057_det.json.gz")
    print(f"  Saved: {TRAJECTORIES_DIR / 'doorway_oscillation_seed50057_det.json.gz'} ({rec_osc.summary['category']})")

    # 7. Carrier Timeout Failure (Distant, seed 50027, deterministic, reached exit but waited)
    print("Recording: Carrier Timeout Failure (Seed 50027)...")
    rec_carrier = record_episode(model, env_distant, seed=50027, deterministic=True, model_path=str(m_path.relative_to(ROOT)))
    rec_carrier.save(TRAJECTORIES_DIR / "carrier_timeout_seed50027_det.json.gz")
    print(f"  Saved: {TRAJECTORIES_DIR / 'carrier_timeout_seed50027_det.json.gz'} ({rec_carrier.summary['category']})")

    # 8. Exploration Timeout Failure (Distant, seed 50012, deterministic, timeout without finding victim)
    print("Recording: Exploration Timeout Failure (Seed 50012)...")
    rec_explore = record_episode(model, env_distant, seed=50012, deterministic=True, model_path=str(m_path.relative_to(ROOT)))
    rec_explore.save(TRAJECTORIES_DIR / "exploration_timeout_seed50012_det.json.gz")
    print(f"  Saved: {TRAJECTORIES_DIR / 'exploration_timeout_seed50012_det.json.gz'} ({rec_explore.summary['category']})")

    # 9. Discovered Not Picked Up (Distant, seed 50045, deterministic)
    print("Recording: Discovered Not Picked Up (Seed 50045)...")
    rec_disc_not_pick = record_episode(model, env_distant, seed=50045, deterministic=True, model_path=str(m_path.relative_to(ROOT)))
    rec_disc_not_pick.save(TRAJECTORIES_DIR / "discovered_not_picked_up_seed50045_det.json.gz")
    print(f"  Saved: {TRAJECTORIES_DIR / 'discovered_not_picked_up_seed50045_det.json.gz'} ({rec_disc_not_pick.summary['category']})")

    print(f"\nAll representative trajectories successfully recorded to {TRAJECTORIES_DIR}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record representative episodes for Visual Debugger")
    parser.add_argument("--model", default="models/phase4_return_curriculum/production_route_complexity_250k/final_model.zip", help="Model path")
    args = parser.parse_args()
    generate_representative_trajectories(args.model)
