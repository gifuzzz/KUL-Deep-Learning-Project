import pytest
import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO

from agents.cnn import RecurrentSpatialRescueCNN, SpatialRescueFeaturesExtractor
from rescue_env.envs.single_agent import RescueEnv


def test_recurrent_spatial_cnn_architecture_and_forward():
    """Verify Phase 2 observation dictionary forward pass, dimensions, and output shapes."""
    batch_size = 4
    obs_space = spaces.Dict({
        "grid": spaces.Box(low=-np.inf, high=np.inf, shape=(9, 15, 15), dtype=np.float32),
        "memory": spaces.Box(low=-np.inf, high=np.inf, shape=(9, 40, 52), dtype=np.float32),
        "agent": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
        "previous_action": spaces.Box(low=0, high=6, shape=(), dtype=np.float32),
    })
    policy = RecurrentSpatialRescueCNN(obs_space, spaces.Discrete(7))

    # Verify layer architectures
    # Local Branch: 8 * 15 * 15 = 1,800
    assert policy.local_features_dim == 1800
    # Memory Branch: 16 * 20 * 26 = 8,320 (No MaxPool2d)
    assert policy.memory_features_dim == 8320
    assert not any(isinstance(m, torch.nn.MaxPool2d) for m in policy.memory_cnn.modules())
    # Fusion: 1800 + 8320 + 5 + 7 = 10,132 -> 128
    assert policy.fusion.in_features == 10132
    assert policy.fusion.out_features == 128
    # Recurrence: GRU(128, 128)
    assert policy.gru.input_size == 128
    assert policy.gru.hidden_size == 128
    # Policy and Value heads
    assert policy.action_net.out_features == 7
    assert policy.value_net.out_features == 1

    # Observation tensors from Phase 2 contract
    obs = {
        "grid": torch.randn(batch_size, 9, 15, 15, dtype=torch.float32),
        "memory": torch.randn(batch_size, 9, 40, 52, dtype=torch.float32),
        "agent": torch.randn(batch_size, 5, dtype=torch.float32),
        "previous_action": torch.randint(0, 7, (batch_size,), dtype=torch.long),
    }

    # 1. Forward returning 3-tuple (actions, values, log_prob)
    actions, values, log_prob = policy(obs)
    assert actions.shape == (batch_size, 1) or actions.shape == (batch_size,)
    assert values.shape == (batch_size, 1)
    assert log_prob.shape == (batch_size,)

    # 2. Forward returning 4-tuple with explicit hidden state
    h0 = torch.zeros(1, batch_size, 128)
    actions_rec, values_rec, log_prob_rec, h1 = policy(obs, lstm_states=h0, return_hidden_state=True)
    assert h1.shape == (1, batch_size, 128)
    assert not torch.allclose(h0, h1)

    # 3. Predict values
    val = policy.predict_values(obs)
    assert val.shape == (batch_size, 1)


def test_action_masking_blocks_illegal_actions():
    """Verify illegal actions have -1e8 logits, zero probability, and are never selected."""
    batch_size = 2
    obs_space = spaces.Dict({
        "grid": spaces.Box(low=-np.inf, high=np.inf, shape=(9, 15, 15), dtype=np.float32),
        "memory": spaces.Box(low=-np.inf, high=np.inf, shape=(9, 40, 52), dtype=np.float32),
        "agent": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
        "previous_action": spaces.Box(low=0, high=6, shape=(), dtype=np.float32),
    })
    policy = RecurrentSpatialRescueCNN(obs_space, spaces.Discrete(7))

    obs = {
        "grid": torch.randn(batch_size, 9, 15, 15, dtype=torch.float32),
        "memory": torch.randn(batch_size, 9, 40, 52, dtype=torch.float32),
        "agent": torch.randn(batch_size, 5, dtype=torch.float32),
        "previous_action": torch.tensor([0, 1], dtype=torch.long),
    }

    # Mask: For batch item 0, only action 2 and 4 are legal.
    # For batch item 1, only action 0 is legal.
    masks = torch.tensor([
        [False, False, True, False, True, False, False],
        [True, False, False, False, False, False, False],
    ], dtype=torch.bool)

    # Get distribution with masks
    dist = policy.get_distribution(obs, action_masks=masks)
    probs = dist.distribution.probs

    # Check probabilities of masked actions are strictly 0.0
    assert torch.all(probs[0, [0, 1, 3, 5, 6]] == 0.0)
    assert torch.all(probs[1, 1:] == 0.0)
    assert probs[1, 0].item() == pytest.approx(1.0, abs=1e-5)

    # Check logits of illegal actions are very negative (-1e8)
    masked_logits = policy.get_masked_logits(obs, action_masks=masks)
    assert torch.all(masked_logits[0, [0, 1, 3, 5, 6]] <= -1e7)
    assert torch.all(masked_logits[1, 1:] <= -1e7)

    # Sample actions 100 times, assert illegal actions are never chosen
    for _ in range(100):
        actions, _, _ = policy(obs, action_masks=masks)
        a0 = int(actions[0].item())
        a1 = int(actions[1].item())
        assert a0 in (2, 4), f"Illegal action {a0} was selected!"
        assert a1 == 0, f"Illegal action {a1} was selected!"


def test_hidden_state_updates_and_episode_resets():
    """Verify hidden states update on transition and reset strictly at episode boundaries."""
    batch_size = 4
    obs_space = spaces.Dict({
        "grid": spaces.Box(low=-np.inf, high=np.inf, shape=(9, 15, 15), dtype=np.float32),
        "memory": spaces.Box(low=-np.inf, high=np.inf, shape=(9, 40, 52), dtype=np.float32),
        "agent": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
    })
    policy = RecurrentSpatialRescueCNN(obs_space, spaces.Discrete(7))

    obs_step1 = {
        "grid": torch.randn(batch_size, 9, 15, 15, dtype=torch.float32),
        "memory": torch.randn(batch_size, 9, 40, 52, dtype=torch.float32),
        "agent": torch.randn(batch_size, 5, dtype=torch.float32),
    }

    # Step 1: Initial state (zeros)
    h0 = torch.zeros(1, batch_size, 128)
    _, _, _, h1 = policy(obs_step1, lstm_states=h0, return_hidden_state=True)
    assert not torch.allclose(h0, h1)

    obs_step2 = {
        "grid": torch.randn(batch_size, 9, 15, 15, dtype=torch.float32),
        "memory": torch.randn(batch_size, 9, 40, 52, dtype=torch.float32),
        "agent": torch.randn(batch_size, 5, dtype=torch.float32),
    }

    # Step 2: Envs 0 and 2 are new episodes (episode_starts = 1.0), Envs 1 and 3 continue
    episode_starts = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    _, values2, _, h2 = policy(
        obs_step2,
        lstm_states=h1,
        episode_starts=episode_starts,
        return_hidden_state=True,
    )

    # Check env 0: Because episode_starts[0] == 1.0, hidden state was reset to zero before stepping.
    # Therefore, stepping env 0 from h0[:, 0:1] must yield identical output and hidden state.
    obs_env0 = {k: v[0:1] for k, v in obs_step2.items()}
    _, val_env0, _, h_env0 = policy(obs_env0, lstm_states=h0[:, 0:1], return_hidden_state=True)
    assert torch.allclose(h2[:, 0:1], h_env0, atol=1e-5)
    assert torch.allclose(values2[0:1], val_env0, atol=1e-5)

    # Check env 1: Because episode_starts[1] == 0.0, hidden state was preserved from h1.
    obs_env1 = {k: v[1:2] for k, v in obs_step2.items()}
    _, val_env1, _, h_env1 = policy(obs_env1, lstm_states=h1[:, 1:2], return_hidden_state=True)
    assert torch.allclose(h2[:, 1:2], h_env1, atol=1e-5)
    assert torch.allclose(values2[1:2], val_env1, atol=1e-5)


def test_sb3_maskable_ppo_end_to_end(tmp_path):
    """Verify integration with MaskablePPO, policy rollouts, learning, and save/load."""
    env = RescueEnv(max_steps=32, victim_distance=2)
    try:
        model = MaskablePPO(
            RecurrentSpatialRescueCNN,
            env,
            n_steps=8,
            batch_size=8,
            device="cpu",
            verbose=0,
        )

        obs, _ = env.reset(seed=42)
        action, state = model.predict(obs, action_masks=env.action_masks())
        assert env.action_space.contains(int(action))
        assert env.action_masks()[int(action)]
        assert state is not None
        assert state.shape == (1, 1, 128)

        # Train a few steps
        model.learn(total_timesteps=16)

        # Save and restore
        save_file = tmp_path / "recurrent_rescue.zip"
        model.save(save_file)
        restored = MaskablePPO.load(save_file, env=env, device="cpu")
        assert isinstance(restored.policy, RecurrentSpatialRescueCNN)

        action_restored, state_restored = restored.predict(obs, state=state, action_masks=env.action_masks())
        assert env.action_space.contains(int(action_restored))
    finally:
        env.close()


def test_spatial_features_extractor_standalone():
    """Verify SpatialRescueFeaturesExtractor produces 128-dim features."""
    env = RescueEnv()
    try:
        extractor = SpatialRescueFeaturesExtractor(env.observation_space)
        obs, _ = env.reset(seed=42)
        tensor_obs = {
            k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()
        }
        with torch.no_grad():
            features = extractor(tensor_obs)
        assert features.shape == (1, 128)
        assert torch.isfinite(features).all()
    finally:
        env.close()


def test_gradients_flow_through_all_branches():
    """Verify gradients backpropagate through local_cnn, memory_cnn, fusion, gru, and heads."""
    batch_size = 2
    obs_space = spaces.Dict({
        "grid": spaces.Box(low=-np.inf, high=np.inf, shape=(9, 15, 15), dtype=np.float32),
        "memory": spaces.Box(low=-np.inf, high=np.inf, shape=(9, 40, 52), dtype=np.float32),
        "agent": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
        "previous_action": spaces.Box(low=0, high=6, shape=(), dtype=np.float32),
    })
    policy = RecurrentSpatialRescueCNN(obs_space, spaces.Discrete(7))

    obs = {
        "grid": torch.randn(batch_size, 9, 15, 15, requires_grad=True),
        "memory": torch.randn(batch_size, 9, 40, 52, requires_grad=True),
        "agent": torch.randn(batch_size, 5, requires_grad=True),
        "previous_action": torch.tensor([1, 4]),
    }
    actions = torch.tensor([1, 4])
    masks = torch.ones((batch_size, 7), dtype=torch.bool)

    values, log_prob, entropy = policy.evaluate_actions(obs, actions, action_masks=masks)
    loss = values.mean() - log_prob.mean() - 0.01 * entropy.mean()
    loss.backward()

    # Check gradients exist and are non-zero across branches
    assert policy.local_cnn[0].weight.grad is not None and policy.local_cnn[0].weight.grad.abs().sum() > 0
    assert policy.local_cnn[4].weight.grad is not None and policy.local_cnn[4].weight.grad.abs().sum() > 0
    assert policy.memory_cnn[0].weight.grad is not None and policy.memory_cnn[0].weight.grad.abs().sum() > 0
    assert policy.memory_cnn[4].weight.grad is not None and policy.memory_cnn[4].weight.grad.abs().sum() > 0
    assert policy.fusion.weight.grad is not None and policy.fusion.weight.grad.abs().sum() > 0
    assert policy.gru.weight_ih_l0.grad is not None and policy.gru.weight_ih_l0.grad.abs().sum() > 0
    assert policy.action_net.weight.grad is not None and policy.action_net.weight.grad.abs().sum() > 0
    assert policy.value_net.weight.grad is not None and policy.value_net.weight.grad.abs().sum() > 0


def test_backward_compatibility_missing_previous_action():
    """Verify policy handles observations where previous_action is omitted cleanly."""
    batch_size = 2
    obs_space = spaces.Dict({
        "grid": spaces.Box(low=-np.inf, high=np.inf, shape=(9, 15, 15), dtype=np.float32),
        "memory": spaces.Box(low=-np.inf, high=np.inf, shape=(9, 40, 52), dtype=np.float32),
        "agent": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
    })
    policy = RecurrentSpatialRescueCNN(obs_space, spaces.Discrete(7))

    obs = {
        "grid": torch.randn(batch_size, 9, 15, 15),
        "memory": torch.randn(batch_size, 9, 40, 52),
        "agent": torch.randn(batch_size, 5),
    }

    actions, values, log_prob = policy(obs)
    assert actions.shape == (batch_size, 1) or actions.shape == (batch_size,)
    assert values.shape == (batch_size, 1)
    assert log_prob.shape == (batch_size,)

