"""Benchmark rescue algorithms, feature extractors, and reward variants."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from rescue_env.envs.single_agent import RescueEnv


REWARD_CONFIGS = {
    "sparse": dict(reward_shaping=False, navigation_shaping=False, exploration_bonus=0.0),
    "potential": dict(reward_shaping=True, navigation_shaping=False, exploration_bonus=0.0),
    "navigation": dict(reward_shaping=True, navigation_shaping=True, exploration_bonus=0.0),
    "exploration": dict(reward_shaping=True, navigation_shaping=True, exploration_bonus=0.01),
}


class SizedRescueCNN(BaseFeaturesExtractor):
    """Configurable CNN used to compare capacity and spatial preservation."""

    def __init__(self, observation_space, width=32, features_dim=128, spatial=False):
        super().__init__(observation_space, features_dim=features_dim)
        local_channels = observation_space["grid"].shape[0]
        memory_channels = observation_space["memory"].shape[0]
        agent_dim = observation_space["agent"].shape[0]
        local_out = max(8, width // 4) if spatial else width * 2
        self.spatial = spatial
        self.local_cnn = nn.Sequential(
            nn.Conv2d(local_channels, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.ReLU(),
            *(
                [nn.Conv2d(width * 2, local_out, 1), nn.ReLU(), nn.Flatten()]
                if spatial else [nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten()]
            ),
        )
        self.memory_cnn = nn.Sequential(
            nn.Conv2d(memory_channels, width, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        height, grid_width = observation_space["grid"].shape[1:]
        local_features = local_out * height * grid_width if spatial else width * 2 * 3 * 3
        memory_features = width * 2 * 4 * 4
        self.fusion = nn.Sequential(
            nn.Linear(local_features + memory_features + agent_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.fusion(torch.cat((
            self.local_cnn(observations["grid"]),
            self.memory_cnn(observations["memory"]),
            observations["agent"],
        ), dim=1))


MODEL_SIZES = {
    "small": (16, 96),
    "base": (32, 128),
    "large": (48, 192),
}


class SmallRescueCNN(SizedRescueCNN):
    def __init__(self, observation_space):
        super().__init__(observation_space, width=16, features_dim=96)


class BaseRescueCNN(SizedRescueCNN):
    def __init__(self, observation_space):
        super().__init__(observation_space, width=32, features_dim=128)


class LargeRescueCNN(SizedRescueCNN):
    def __init__(self, observation_space):
        super().__init__(observation_space, width=48, features_dim=192)


class SmallSpatialRescueCNN(SizedRescueCNN):
    def __init__(self, observation_space):
        super().__init__(observation_space, width=16, features_dim=96, spatial=True)


class BaseSpatialRescueCNN(SizedRescueCNN):
    def __init__(self, observation_space):
        super().__init__(observation_space, width=32, features_dim=128, spatial=True)


class LargeSpatialRescueCNN(SizedRescueCNN):
    def __init__(self, observation_space):
        super().__init__(observation_space, width=48, features_dim=192, spatial=True)


def extractor_for(config):
    classes = {
        ("small", "rescue"): SmallRescueCNN,
        ("base", "rescue"): BaseRescueCNN,
        ("large", "rescue"): LargeRescueCNN,
        ("small", "spatial"): SmallSpatialRescueCNN,
        ("base", "spatial"): BaseSpatialRescueCNN,
        ("large", "spatial"): LargeSpatialRescueCNN,
    }
    return classes[(config["model_size"], config["extractor"])]


def make_env(reward_config, args, seeds=None):
    return Monitor(RescueEnv(
        max_steps=args.max_steps,
        visibility_radius=args.visibility_radius,
        victim_distance=args.victim_distance,
        min_victim_distance=1,
        reward_shaping=reward_config["reward_shaping"],
        navigation_shaping=reward_config["navigation_shaping"],
        exploration_bonus=reward_config["exploration_bonus"],
        scenario_seeds=seeds,
    ))


def masked_dqn_action(model, observation, mask):
    tensor, _ = model.policy.obs_to_tensor(observation)
    with torch.no_grad():
        q_values = model.policy.q_net(tensor).cpu().numpy()[0]
    q_values[~mask] = -np.inf
    return int(np.argmax(q_values))


def masked_policy_action(model, observation, mask):
    tensor, _ = model.policy.obs_to_tensor(observation)
    with torch.no_grad():
        probabilities = model.policy.get_distribution(tensor).distribution.probs
    probabilities = probabilities.cpu().numpy()[0]
    probabilities[~mask] = -np.inf
    return int(np.argmax(probabilities))


def evaluate(model, algorithm, args, reward_config, seeds):
    env = RescueEnv(
        max_steps=args.max_steps,
        visibility_radius=args.visibility_radius,
        victim_distance=args.victim_distance,
        min_victim_distance=1,
        **reward_config,
        scenario_seeds=seeds,
    )
    successes = pickups = 0
    try:
        for _ in seeds:
            observation, _ = env.reset()
            done = False
            picked_up = False
            while not done:
                if algorithm == "dqn":
                    action = masked_dqn_action(model, observation, env.action_masks())
                elif algorithm == "a2c":
                    action = masked_policy_action(model, observation, env.action_masks())
                else:
                    action, _ = model.predict(
                        observation,
                        action_masks=env.action_masks(),
                        deterministic=True,
                    )
                    action = int(action)
                observation, _, terminated, truncated, info = env.step(action)
                picked_up |= any(
                    victim["carried_by"] >= 0 or victim["rescued"]
                    for victim in env.sim.victims
                )
                done = terminated or truncated
            successes += int(info["success"])
            pickups += int(picked_up)
    finally:
        env.close()
    count = len(seeds)
    return {
        "episodes": count,
        "success_rate": successes / count,
        "pickup_rate": pickups / count,
    }


def train_one(config, args, output):
    reward_config = REWARD_CONFIGS[config["reward"]]
    train_env = make_env(reward_config, args)
    policy_kwargs = {
        "features_extractor_class": extractor_for(config),
        "normalize_images": False,
    }
    try:
        if config["algorithm"] == "dqn":
            model = DQN(
                "MultiInputPolicy",
                train_env,
                policy_kwargs=policy_kwargs,
                learning_starts=min(2_000, max(100, args.steps // 10)),
                buffer_size=50_000,
                batch_size=64,
                train_freq=4,
                gradient_steps=1,
                exploration_fraction=.25,
                exploration_final_eps=.05,
                learning_rate=1e-4,
                device=args.device,
                seed=args.seed,
                verbose=0,
            )
        elif config["algorithm"] == "a2c":
            model = A2C(
                "MultiInputPolicy",
                train_env,
                policy_kwargs=policy_kwargs,
                n_steps=128,
                learning_rate=1e-4,
                gamma=.99,
                ent_coef=.01,
                device=args.device,
                seed=args.seed,
                verbose=0,
            )
        else:
            model = MaskablePPO(
                "MultiInputPolicy",
                train_env,
                policy_kwargs=policy_kwargs,
                n_steps=1024,
                batch_size=64,
                n_epochs=4,
                learning_rate=1e-4,
                gamma=.99,
                ent_coef=.01,
                device=args.device,
                seed=args.seed,
                verbose=0,
            )
        model.learn(total_timesteps=args.steps)
        model.save(output / "model")
        validation = evaluate(
            model,
            config["algorithm"],
            args,
            reward_config={**reward_config, "reward_shaping": False,
                           "navigation_shaping": False, "exploration_bonus": 0.0},
            seeds=list(range(args.validation_start, args.validation_start + args.episodes)),
        )
        return validation
    finally:
        train_env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--validation-start", type=int, default=30_000)
    parser.add_argument("--victim-distance", type=int, default=2)
    parser.add_argument("--visibility-radius", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument("--configs", nargs="+")
    args = parser.parse_args()
    if args.steps < 1 or args.episodes < 1 or args.output.exists():
        parser.error("steps and episodes must be positive; output must be new")
    if args.device == "auto":
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(1)
    names = args.configs or [
        "ppo_rescue_sparse",
        "ppo_spatial_navigation",
        "dqn_rescue_sparse",
        "dqn_spatial_navigation",
        "dqn_spatial_exploration",
    ]
    configs = []
    for name in names:
        parts = name.split("_")
        if len(parts) < 3 or parts[0] not in ("ppo", "a2c", "dqn"):
            parser.error(f"invalid config name: {name}")
        if parts[1] in MODEL_SIZES:
            if len(parts) < 4 or parts[2] not in ("rescue", "spatial"):
                parser.error(f"invalid model configuration: {name}")
            model_size, extractor, reward_parts = parts[1], parts[2], parts[3:]
        else:
            model_size, extractor, reward_parts = "base", parts[1], parts[2:]
            if extractor not in ("rescue", "spatial"):
                parser.error(f"invalid extractor in config: {name}")
        reward = "_".join(reward_parts)
        if reward not in REWARD_CONFIGS:
            parser.error(f"invalid reward in config: {name}")
        configs.append({"name": name, "algorithm": parts[0], "model_size": model_size,
                        "extractor": extractor, "reward": reward})
    args.output.mkdir(parents=True)
    results = []
    for config in configs:
        result_dir = args.output / config["name"]
        result_dir.mkdir()
        result = train_one(config, args, result_dir)
        row = {**config, **result}
        (result_dir / "result.json").write_text(json.dumps(row, indent=2))
        results.append(row)
        print(json.dumps(row), flush=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
