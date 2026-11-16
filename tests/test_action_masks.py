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
    # 1. When prevent_exterior_wandering=False, matches raw FlameBox masks exactly
    env_raw = RescueEnv(prevent_exterior_wandering=False)
    try:
        env_raw.reset(seed=42)
        mask_raw = env_raw.action_masks()
        raw = env_raw.sim.observe(agent_id=0, visibility_radius=env_raw.visibility_radius)
        np.testing.assert_array_equal(mask_raw[1:5], raw["actions_mask"][1:5])
    finally:
        env_raw.close()

    # 2. When prevent_exterior_wandering=True (default), masks exterior grass when not carrying
    env = RescueEnv(prevent_exterior_wandering=True)
    try:
        env.reset(seed=42)
        mask = env.action_masks()
        # At seed 42, agent is at (13, 0) next to exit at col -1 (action 4 Left is exterior)
        assert bool(mask[1]) is True   # Up (12, 0)
        assert bool(mask[2]) is True   # Right (13, 1)
        assert bool(mask[3]) is True   # Down (14, 0)
        assert bool(mask[4]) is False  # Left (13, -1, exterior grass masked out)
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
