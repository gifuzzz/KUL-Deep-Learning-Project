"""Initialize Phase 8 transferred checkpoints with 10-channel spatial memory.

Copies weights from Phase 6 base model, strictly zero-initializes channel 9 in memory_cnn[0].weight,
and saves validated checkpoints for Variant B (visited_binary) and Variant C (visit_recency).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from rescue_env.envs.single_agent import RescueEnv
from rescue_env.wrappers import PreviousActionWrapper
from agents.cnn import RecurrentSpatialRescueCNN, transfer_spatial_memory_weights
from agents.recurrent_ppo import MaskableRecurrentPPO, save_checkpoint, load_checkpoint


def init_checkpoints():
    base_ckpt = Path("models/phase6_spawn_curriculum/production_250k/stage3_spawn_curriculum/best_model.zip")
    assert base_ckpt.exists(), f"Base checkpoint not found at {base_ckpt}"

    configs = [
        ("visited_binary", Path("models/phase8_binary_visited")),
        ("visit_recency", Path("models/phase8_recency")),
    ]

    for mode, save_dir in configs:
        save_dir.mkdir(parents=True, exist_ok=True)
        target_path = save_dir / "initial_transferred.zip"

        print(f"\n--- Initializing {mode} at {target_path} ---")
        env = PreviousActionWrapper(RescueEnv(spatial_memory_mode=mode, reward_shaping=False))
        model = MaskableRecurrentPPO(
            policy=RecurrentSpatialRescueCNN,
            env=env,
            architecture_variant="baseline",
            device="cpu",
        )

        summary = transfer_spatial_memory_weights(base_ckpt, model, zero_init_ch9=True)
        print(f"Transferred {summary['transferred_params']} parameters across {len(summary['transferred_keys'])} keys.")
        assert summary["zero_init_ch9"] is True

        save_checkpoint(
            model,
            target_path,
            metadata={
                "phase": 8,
                "variant": mode,
                "spatial_memory_mode": mode,
                "base_source": str(base_ckpt),
                "channel_9_init": "zero_initialized",
            },
        )
        print(f"Saved initial checkpoint to {target_path}")

        # Verification reload
        reloaded = load_checkpoint(target_path, env=env, device="cpu", expected_architecture="baseline")
        assert torch.all(reloaded.policy.memory_cnn[0].weight[:, 9, :, :] == 0.0)
        print(f"Reload verified: channel 9 strictly zero-initialized.")
        env.close()

    print("\n[SUCCESS] Phase 8 initial transferred checkpoints created successfully.")


if __name__ == "__main__":
    init_checkpoints()
