import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO

from agents.cnn import RescueCNN
from rescue_env.envs.single_agent import RescueEnv


def test_marker_tracks_agent_without_changing_observed_contents():
    baseline = RescueEnv(max_steps=64, victim_distance=3)
    marked = RescueEnv(max_steps=64, victim_distance=3, memory_agent_marker=True)
    try:
        original, _ = baseline.reset(seed=42)
        observation, _ = marked.reset(seed=42)
        for action in [None, 2, 2, 5, 4, 4, 4]:
            if action is not None:
                original, *rest = baseline.step(action)
                observation, *marked_rest = marked.step(action)
                assert rest == marked_rest
            np.testing.assert_array_equal(observation['memory'][:13], original['memory'])
            np.testing.assert_array_equal(observation['grid'], original['grid'])
            np.testing.assert_array_equal(observation['agent'], original['agent'])
            np.testing.assert_array_equal(marked.action_masks(), baseline.action_masks())
            assert marked.observation_space.contains(observation)
            person = marked.sim.agents[0]
            marker = observation['memory'][13]
            inside = 0 <= person['r'] < marked.sim.h and 0 <= person['c'] < marked.sim.w
            assert marker.sum() == int(inside)
            if inside:
                assert marker[person['r'], person['c']] == 1
        # The final movement rescues outside; no clipped or stale location is marked.
        assert marked_rest[-1]['success']
        assert observation['memory'][13].sum() == 0
        reset, _ = marked.reset(seed=42)
        assert reset['memory'][13].sum() == 1
        np.testing.assert_array_equal(reset['memory'][:13], baseline.reset(seed=42)[0]['memory'])
    finally:
        baseline.close()
        marked.close()


def test_marker_cnn_and_checkpoint_space_check(tmp_path):
    torch.set_num_threads(1)
    baseline = RescueEnv()
    marked = RescueEnv(memory_agent_marker=True)
    try:
        observation, _ = marked.reset(seed=42)
        cnn = RescueCNN(marked.observation_space)
        features = cnn({key: torch.from_numpy(value)[None] for key, value in observation.items()})
        assert features.shape == (1, 128)
        assert torch.isfinite(features).all()
        model = MaskablePPO('MultiInputPolicy', baseline, n_steps=2, batch_size=2,
                            device='cpu', policy_kwargs={'features_extractor_class': RescueCNN,
                                                         'normalize_images': False})
        model.save(tmp_path / 'old')
        with pytest.raises(ValueError, match='Observation spaces do not match'):
            MaskablePPO.load(tmp_path / 'old', env=marked, device='cpu')
    finally:
        baseline.close()
        marked.close()


def test_grass_inside_padded_canvas_is_not_marked_as_building():
    env = RescueEnv(memory_agent_marker=True)
    try:
        env.reset(seed=20000)
        assert env.action_masks()[2]
        observation, _, terminated, truncated, _ = env.step(2)
        person = env.sim.agents[0]
        assert person['c'] == env.sim.w
        assert 0 <= person['r'] < 64 and 0 <= person['c'] < 64
        assert not terminated and not truncated
        assert observation['memory'][12, person['r'], person['c']] == 0
        assert observation['memory'][13].sum() == 0
        assert observation['agent'][1] == 1.0
    finally:
        env.close()
