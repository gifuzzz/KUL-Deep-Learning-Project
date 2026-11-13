
from pathlib import Path
import pytest
import numpy as np
import torch

from sb3_contrib import MaskablePPO

from rescue_env.envs.single_agent import RescueEnv


def action_probabilities(model, observation, action_mask):
    """Return masked action probabilities for an observation."""

    with torch.no_grad():
        obs_tensor, _ = model.policy.obs_to_tensor(observation)

        distribution = model.policy.get_distribution(
            obs_tensor,
            action_masks=np.asarray(action_mask)[None, :],
        )

        return (
            distribution.distribution.probs
            .detach()
            .cpu()
            .numpy()[0]
        )


def test_exit_memory_ablation():
    torch.set_num_threads(1)

    env = RescueEnv(
        max_steps=64,
        victim_distance=3,
        reward_shaping=False,
        navigation_shaping=False,
    )

    checkpoint_path = Path("models/curriculum_d3/last_model.zip")
    if not checkpoint_path.exists():
        pytest.skip(f"Ablation model checkpoint {checkpoint_path} not found")

    model = MaskablePPO.load(
        str(checkpoint_path),
        device="cpu",
    )

    try:
        obs, _ = env.reset(seed=20_002)

        # Reproduce the pickup from the failed episode.
        pickup_actions = [
            2, 1, 4, 4, 4, 2, 4,
            3, 2, 2, 2, 4, 6, 2,
            4, 4, 6, 4, 3, 2, 5,
        ]

        for action in pickup_actions:
            obs, _, terminated, truncated, _ = env.step(action)
            assert not terminated
            assert not truncated

        # Verify the victim is being carried.
        carrying = any(
            v["carried_by"] == 0
            and not v["rescued"]
            and not v["dead"]
            for v in env.sim.victims
        )
        assert carrying

        # Recorded actions after pickup.
        return_actions = [
            1, 2, 3, 3, 6, 4, 6, 1, 3, 2,
            0, 4, 4, 4, 2, 4, 4, 4, 4, 1,
            1, 1, 2, 2, 4, 3, 3, 1, 6, 2,
            2, 2, 4, 4, 2, 2, 2, 4, 4, 1,
        ]

        found = False

        # Look for a carrying state with remembered,
        # but not currently visible, exits.
        for action in [None] + return_actions:

            local_exits = np.argwhere(
                obs["grid"][10] > 0.5
            )
            memory_exits = np.argwhere(
                obs["memory"][10] > 0.5
            )

            carrying = any(
                v["carried_by"] == 0
                and not v["rescued"]
                and not v["dead"]
                for v in env.sim.victims
            )

            if (
                carrying
                and len(local_exits) == 0
                and len(memory_exits) > 0
            ):
                found = True
                break

            if action is None:
                continue

            obs, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break

        # Inspect the final state reached.
        local_exits = np.argwhere(obs["grid"][10] > 0.5)
        memory_exits = np.argwhere(obs["memory"][10] > 0.5)

        print("\nState information")
        print("-----------------")
        print(
            "Agent position:",
            (env.sim.agents[0]["r"], env.sim.agents[0]["c"]),
        )
        print("Local exits:", local_exits.tolist())
        print("Remembered exits:", memory_exits.tolist())

        if not found:
            print(
                "\nNo memory-dependent state found."
                "\nThe chosen trajectory did not produce a carrying "
                "state with remembered but invisible exits."
            )
            return

        print("\nMemory-dependent state found!")

        action_mask = env.action_masks()

        original_probs = action_probabilities(
            model, obs, action_mask
        )

        # Remove remembered exits only.
        no_memory = {
            key: value.copy()
            for key, value in obs.items()
        }
        no_memory["memory"][10] = 0.0

        # Remove locally visible exits only.
        no_local = {
            key: value.copy()
            for key, value in obs.items()
        }
        no_local["grid"][10] = 0.0

        # Remove both exit channels.
        no_exits = {
            key: value.copy()
            for key, value in obs.items()
        }
        no_exits["memory"][10] = 0.0
        no_exits["grid"][10] = 0.0

        conditions = {
            "Original": obs,
            "No memory exits": no_memory,
            "No local exits": no_local,
            "No exits anywhere": no_exits,
        }

        print("\nExit information ablation")
        print("-------------------------")

        for name, observation in conditions.items():
            probs = action_probabilities(
                model, observation, action_mask
            )

            difference = np.abs(
                original_probs - probs
            ).sum()

            print(
                f"{name:<20} "
                f"Difference={difference:.6f} "
                f"Probs={np.round(probs, 4)}"
            )

            assert np.isfinite(probs).all()
            assert np.isclose(probs.sum(), 1.0)

    finally:
        env.close()
