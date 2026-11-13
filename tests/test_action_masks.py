import numpy as np

from rescue_env.envs.single_agent import RescueEnv


def test_initial_action_masks():
    env = RescueEnv()

    try:
        env.reset(seed=42)
        mask = env.action_masks()

        assert mask.shape == (env.action_space.n,)
        assert mask.dtype == np.bool_

        # Waiting is always permitted for a living firefighter.
        assert mask[0]

        # Water action should be available.
        assert mask[6]

        # Victim starts too far away to pick up.
        assert not mask[5]

    finally:
        env.close()


def test_movement_masks():
    env = RescueEnv()

    try:
        env.reset(seed=42)

        mask = env.action_masks()

        # Compare Gymnasium movement masks
        # with FlameBox's native movement masks.
        raw = env.sim.observe(
            agent_id=0,
            visibility_radius=env.visibility_radius,
        )

        np.testing.assert_array_equal(
            mask[1:5],
            raw["actions_mask"][1:5],
        )

    finally:
        env.close()


def test_pickup_mask():
    env = RescueEnv()

    try:
        env.reset(seed=42)

        # Known scenario:
        # Firefighter: (13, 0)
        # Victim: (13, 3)
        env.step(2)
        env.step(2)

        # Firefighter should now be adjacent.
        assert env.action_masks()[5]

        # Pick up victim.
        env.step(5)

        victim = env.sim.victims[0]
        assert victim["carried_by"] == 0

        # Pickup/drop action remains available
        # while carrying the victim.
        assert not env.action_masks()[5]

    finally:
        env.close()


def test_water_mask():
    env = RescueEnv()

    try:
        env.reset(seed=42)

        assert env.ACTIONS[6] == 5
        assert env.action_masks()[6]

        # Check that the action executes.
        obs, reward, terminated, truncated, info = env.step(6)

        assert env.observation_space.contains(obs)
        assert not terminated
        assert not truncated

    finally:
        env.close()
