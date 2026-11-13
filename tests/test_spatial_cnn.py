import numpy as np
import torch
import pytest
from sb3_contrib import MaskablePPO

from agents.cnn import SpatialRescueCNN
from rescue_env.envs.single_agent import RescueEnv


def test_spatial_extractor_round_trip(tmp_path):
    torch.set_num_threads(1)
    env = RescueEnv()
    try:
        obs, _ = env.reset(seed=42)
        model = MaskablePPO('MultiInputPolicy', env, n_steps=8, batch_size=8,
                            device='cpu', seed=1,
                            policy_kwargs={'features_extractor_class': SpatialRescueCNN,
                                           'normalize_images': False})
        tensors, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            local = model.policy.features_extractor.local_cnn(tensors['grid'])
            features = model.policy.features_extractor(tensors)
            expected = model.policy.get_distribution(tensors, action_masks=env.action_masks()).distribution.probs.numpy()
        assert local.shape == (1, 8 * 15 * 15)
        assert features.shape == (1, 128)
        assert torch.isfinite(features).all()
        model.save(tmp_path / 'spatial')
        restored = MaskablePPO.load(tmp_path / 'spatial', env=env, device='cpu')
        assert isinstance(restored.policy.features_extractor, SpatialRescueCNN)
        with torch.no_grad():
            actual = restored.policy.get_distribution(tensors, action_masks=env.action_masks()).distribution.probs.numpy()
        np.testing.assert_array_equal(expected, actual)
    finally:
        env.close()


def test_relative_memory_uses_known_position_and_preserves_original_channels():
    from agents.cnn import RelativeSpatialRescueCNN
    env = RescueEnv()
    captured = []
    try:
        obs, _ = env.reset(seed=20000)
        cnn = RelativeSpatialRescueCNN(env.observation_space)
        cnn.memory_cnn[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        for action in [None, 2]:
            if action is not None:
                obs, _, _, _, _ = env.step(action)
            tensors = {key: torch.from_numpy(value)[None] for key, value in obs.items()}
            with torch.no_grad():
                features = cnn(tensors)
            memory = captured[-1][0]
            np.testing.assert_array_equal(memory[:13].numpy(), obs['memory'])
            assert memory.shape == (15, 64, 64)
            assert features.shape == (1, 128)
            person = env.sim.agents[0]
            assert memory[13, 0, 0].item() == pytest.approx(-person['r'] / 64, abs=1e-6)
            assert memory[14, 0, 0].item() == pytest.approx(-person['c'] / 64, abs=1e-6)
            assert torch.count_nonzero(memory[13:, env.sim.h:]) == 0
            assert torch.count_nonzero(memory[13:, :, env.sim.w:]) == 0
    finally:
        env.close()


def test_detailed_memory_can_preserve_parent_features():
    from agents.cnn import RelativeSpatialRescueCNN, DetailedMemoryRescueCNN
    torch.set_num_threads(1)
    env = RescueEnv()
    try:
        observation, _ = env.reset(seed=42)
        parent = RelativeSpatialRescueCNN(env.observation_space)
        detailed = DetailedMemoryRescueCNN(env.observation_space)
        missing, unexpected = detailed.load_state_dict(parent.state_dict(), strict=False)
        assert not unexpected
        assert all(key.startswith(('memory_projection.', 'memory_correction.')) for key in missing)
        with torch.no_grad():
            detailed.memory_correction.weight.zero_()
            detailed.memory_correction.bias.zero_()
        tensors = {key: torch.from_numpy(value)[None] for key, value in observation.items()}
        with torch.no_grad():
            np.testing.assert_array_equal(parent(tensors).numpy(), detailed(tensors).numpy())
        detailed(tensors).sum().backward()
        assert detailed.memory_correction.weight.grad.abs().sum() > 0
    finally:
        env.close()


def test_remembered_local_crop_is_centered_even_outside():
    from agents.cnn import RememberedLocalRescueCNN
    env = RescueEnv()
    try:
        observation, _ = env.reset(seed=20000)
        cnn = RememberedLocalRescueCNN(env.observation_space)
        crops = []
        cnn.local_cnn.register_forward_pre_hook(lambda module, args: crops.append(args[0].detach().clone()))
        for action in [None, 2]:
            if action is not None:
                observation, *_ = env.step(action)
            tensors = {key: torch.from_numpy(value)[None] for key, value in observation.items()}
            with torch.no_grad():
                features = cnn(tensors)
            assert features.shape == (1, 128)
            assert torch.isfinite(features).all()
            crop = crops[-1][0].numpy()
            person = env.sim.agents[0]
            expected = np.zeros((11, 15, 15), dtype=np.float32)
            for r in range(15):
                for c in range(15):
                    gr, gc = person['r'] + r - 7, person['c'] + c - 7
                    if 0 <= gr < 64 and 0 <= gc < 64:
                        expected[:, r, c] = observation['memory'][:11, gr, gc]
            np.testing.assert_array_equal(crop, expected)
    finally:
        env.close()
