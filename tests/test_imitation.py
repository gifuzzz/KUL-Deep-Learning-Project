import random

import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO

from agents.cnn import RelativeSpatialRescueCNN
from rescue_env.envs.single_agent import RescueEnv
from training.train_imitation import ObservedTeacher, DemonstrationReplay, imitate

pytestmark = pytest.mark.skip(reason="Legacy 1.0 test: imitation learning teacher superseded by autonomous RL in 2.0")


@pytest.mark.parametrize('seed', [10000, 10001, 10002])
def test_observed_teacher_uses_valid_actions_and_completes_rescue(seed):
    env = RescueEnv(victim_distance=8, min_victim_distance=5, max_steps=1024)
    try:
        observation, _ = env.reset(seed=seed)
        teacher = ObservedTeacher(env.sim.h, env.sim.w)
        for _ in range(1024):
            before = {key: value.copy() for key, value in observation.items()}
            mask = env.action_masks()
            action = teacher.action(observation, mask)
            assert mask[action]
            for key in before:
                np.testing.assert_array_equal(observation[key], before[key])
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert info['success']
    finally:
        env.close()


def test_imitation_step_increases_expert_action_probability():
    torch.set_num_threads(1)
    env = RescueEnv(victim_distance=3)
    try:
        env.reset(seed=42)
        env.step(2)
        observation, *_ = env.step(2)
        mask = env.action_masks()
        assert mask[5]
        model = MaskablePPO('MultiInputPolicy', env, n_steps=8, batch_size=8,
                            seed=0, device='cpu',
                            policy_kwargs={'features_extractor_class': RelativeSpatialRescueCNN,
                                           'normalize_images': False})
        policy = model.policy
        policy.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        tensors, _ = policy.obs_to_tensor(observation)
        with torch.no_grad():
            before = float(policy.get_distribution(tensors, action_masks=mask).distribution.probs[0, 5])
        replay = [({key: value.astype(np.float16) for key, value in observation.items()}, 5, mask)] * 8
        for _ in range(3):
            assert np.isfinite(imitate(policy, replay))
        with torch.no_grad():
            after = float(policy.get_distribution(tensors, action_masks=mask).distribution.probs[0, 5])
        assert after > before
    finally:
        env.close()


def test_aggregate_replay_deduplicates_and_preserves_diversity():
    replay = DemonstrationReplay(capacity=16)
    rng = random.Random(1)
    mask = np.ones(7, dtype=bool)
    for i in range(1000):
        observation = {'grid': np.array([i], dtype=np.float32),
                       'agent': np.array([i / 1000], dtype=np.float32)}
        replay.add(observation, i % 7, mask, rng)
    assert len(replay.examples) == 16
    assert len(replay.seen) == 1000
    assert min(row[0]['grid'][0] for row in replay.examples) < 500
    for _ in range(100):
        replay.add(observation, 999 % 7, mask, rng)
    assert len(replay.seen) == 1000
    assert replay.conflicts == 0
    replay.add(observation, 0, mask, rng)
    assert replay.conflicts == 1
    batch = replay.batch(rng)
    assert len(batch) == 32
    assert all(row[0]['agent'].dtype == np.float32 for row in batch)
