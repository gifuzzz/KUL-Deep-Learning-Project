import numpy as np
import pytest

from rescue_env.envs.single_agent import RescueEnv


def test_exploration_rewards_only_new_observations_before_victim_seen():
    env = RescueEnv(victim_distance=8, min_victim_distance=5, exploration_bonus=.01)
    try:
        before, _ = env.reset(seed=10000)
        assert not env.victim_seen
        explored = before['memory'][11].sum()
        after, reward, _, _, _ = env.step(4)
        newly_seen = after['memory'][11].sum() - explored
        assert newly_seen > 0
        assert isinstance(reward, float)
        assert reward == pytest.approx(.01 * newly_seen)
        _, reward, _, _, _ = env.step(0)
        assert reward == 0
        reset, _ = env.reset(seed=10000)
        assert not env.victim_seen
        np.testing.assert_array_equal(reset['memory'][11], before['memory'][11])
    finally:
        env.close()


def test_visible_victim_disables_exploration_bonus():
    env = RescueEnv(victim_distance=3, exploration_bonus=.01)
    try:
        before, _ = env.reset(seed=42)
        assert env.victim_seen
        after, reward, _, _, _ = env.step(2)
        assert after['memory'][11].sum() > before['memory'][11].sum()
        assert reward == 0
    finally:
        env.close()


@pytest.mark.parametrize('bonus', [-.01, float('nan'), float('inf')])
def test_exploration_bonus_validation(bonus):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        RescueEnv(exploration_bonus=bonus)
