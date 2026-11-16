import pytest

from rescue_env.envs.reward_shaping import potential_reward
from rescue_env.envs.single_agent import RescueEnv


def test_potential_reward():
    # Moving closer: potential increases from -3 to -2.
    closer = potential_reward(-3, -2)

    # Moving farther away: potential decreases from -2 to -3.
    farther = potential_reward(-2, -3)

    assert closer > 0
    assert farther < 0


def test_environment_with_navigation_shaping():
    env = RescueEnv(
        max_steps=64,
        victim_distance=2,
        reward_shaping=True,
        navigation_shaping=True,
    )

    try:
        obs, _ = env.reset(seed=42)

        assert env.observation_space.contains(obs)

        # Test one valid movement.
        action = 2
        assert env.action_masks()[action]

        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(reward, float)
        assert env.observation_space.contains(obs)

    finally:
        env.close()

from rescue_env.envs.reward_shaping import state_potential


def test_navigation_reward_changes():
    env = RescueEnv(
        max_steps=64,
        victim_distance=2,
        reward_shaping=False,
        navigation_shaping=True,
    )

    try:
        env.reset(seed=42)

        before = state_potential(env.sim)

        # Move right toward the victim.
        _, reward, _, _, _ = env.step(2)

        after = state_potential(env.sim)

        expected = potential_reward(before, after)

        assert reward == pytest.approx(expected)

        print(
            f"\nPotential: {before} -> {after}"
            f"\nNavigation reward: {reward:.4f}"
        )

    finally:
        env.close()


def test_complete_rescue_with_shaping():
    env = RescueEnv(
        max_steps=64,
        victim_distance=3,
        reward_shaping=True,
        navigation_shaping=True,
    )

    try:
        env.reset(seed=42)

        # Known scenario: victim at (13, 3)
        # Move toward victim, pick up, return to exit.
        actions = [2, 2, 5, 4, 4, 4]

        total_reward = 0.0

        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(
                f"Action={action}, "
                f"Reward={reward:.4f}, "
                f"Position={env.sim.agents[0]['r'], env.sim.agents[0]['c']}"
            )

            if terminated or truncated:
                break

        assert info["success"]
        assert terminated
        assert info["victims_rescued"] == 1
        assert total_reward > 0

        print(f"Total reward: {total_reward:.4f}")

    finally:
        env.close()



def test_timeout_shaping():
    env = RescueEnv(
        max_steps=1,
        victim_distance=3,
        reward_shaping=False,
        navigation_shaping=True,
    )

    try:
        env.reset(seed=42)

        before = state_potential(env.sim)

        _, reward, terminated, truncated, _ = env.step(0)

        after = state_potential(env.sim)

        assert truncated
        assert not terminated

        expected = potential_reward(before, after)

        assert reward == pytest.approx(expected)

        print(f"\nPotential: {before} -> {after}")
        print(f"Timeout reward: {reward:.4f}")

    finally:
        env.close()


@pytest.mark.parametrize('seed', [20000, 20002])
def test_empty_handed_exit_is_farther_from_pickup(seed):
    env = RescueEnv(max_steps=64, victim_distance=3, navigation_shaping=True, prevent_exterior_wandering=False)
    try:
        env.reset(seed=seed)
        before = state_potential(env.sim)
        assert env.action_masks()[2]
        _, leaving_reward, terminated, truncated, _ = env.step(2)
        outside = state_potential(env.sim)
        assert not terminated and not truncated
        assert env.sim.agents[0]['c'] == env.sim.w
        assert outside == before - 1
        assert leaving_reward < 0
        _, returning_reward, _, _, _ = env.step(4)
        assert state_potential(env.sim) == before
        assert returning_reward > 0
    finally:
        env.close()


def test_carrying_potential_reaches_zero_on_rescue_grass():
    env = RescueEnv(max_steps=64, victim_distance=3, navigation_shaping=True)
    try:
        env.reset(seed=42)
        for action in [2, 2, 5, 4, 4]:
            _, _, terminated, truncated, _ = env.step(action)
            assert not terminated and not truncated
        assert env.sim.victims[0]['carried_by'] == 0
        assert state_potential(env.sim) == -1
        _, reward, terminated, _, info = env.step(4)
        assert terminated and info['success']
        assert state_potential(env.sim) == 0
        assert reward == pytest.approx(10.1)
    finally:
        env.close()
