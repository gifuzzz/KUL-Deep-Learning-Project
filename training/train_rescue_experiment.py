"""Train or continue the rescue CNN with fixed validation and a new output directory."""
import argparse
import json
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.cnn import RescueCNN, SpatialRescueCNN
from rescue_env.envs.single_agent import RescueEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/no_drop/best_model.zip")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30_720)
    parser.add_argument("--navigation-shaping", action="store_true")
    parser.add_argument("--exploration-bonus", type=float, default=0.0)
    parser.add_argument("--fresh", action="store_true", help="Train from scratch instead of loading --model")
    parser.add_argument("--memory-agent-marker", action="store_true")
    parser.add_argument("--spatial-local", action="store_true")
    parser.add_argument("--victim-distance", type=int, default=3)
    parser.add_argument("--min-victim-distance", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ent-coef", type=float)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--training-scenarios", type=int, nargs="+")
    parser.add_argument("--validation-scenarios", type=int, nargs="+")
    parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="auto")
    args = parser.parse_args()
    if args.steps < 1 or args.max_steps < 1 or args.output.exists():
        parser.error("steps must be positive and output must be a new directory")
    if args.n_envs < 1 or 1024 % args.n_envs or args.learning_rate <= 0:
        parser.error("n-envs must divide 1024 and learning-rate must be positive")
    if args.ent_coef is not None and args.ent_coef < 0:
        parser.error("ent-coef must be nonnegative")
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(1)
    def make_train_env(rank):
        scenarios = args.training_scenarios
        if scenarios:
            offset = rank % len(scenarios)
            scenarios = scenarios[offset:] + scenarios[:offset]
        return Monitor(RescueEnv(max_steps=args.max_steps, victim_distance=args.victim_distance,
                                min_victim_distance=args.min_victim_distance, reward_shaping=True,
                                navigation_shaping=args.navigation_shaping,
                                memory_agent_marker=args.memory_agent_marker,
                                exploration_bonus=args.exploration_bonus,
                                scenario_seeds=scenarios))

    train = (make_train_env(0) if args.n_envs == 1 else
             DummyVecEnv([lambda rank=i: make_train_env(rank) for i in range(args.n_envs)]))
    validation_scenarios = args.validation_scenarios or list(range(10_000, 10_050))
    validation = Monitor(RescueEnv(max_steps=args.max_steps, victim_distance=args.victim_distance,
                                   min_victim_distance=args.min_victim_distance,
                                   scenario_seeds=validation_scenarios,
                                   memory_agent_marker=args.memory_agent_marker))
    try:
        # Checks observation/action spaces; the source must also use current action semantics.
        if args.fresh:
            model = MaskablePPO(
                "MultiInputPolicy", train,
                policy_kwargs={"features_extractor_class": SpatialRescueCNN if args.spatial_local else RescueCNN,
                               "normalize_images": False},
                learning_rate=args.learning_rate, n_steps=1024 // args.n_envs,
                batch_size=64, n_epochs=4,
                gamma=0.99, ent_coef=0.01 if args.ent_coef is None else args.ent_coef,
                seed=args.seed, device=device, verbose=1,
                tensorboard_log=str(args.output / "tensorboard"),
            )
        else:
            load_options = {"learning_rate": args.learning_rate}
            if args.n_envs > 1:
                load_options["n_steps"] = 1024 // args.n_envs
            if args.ent_coef is not None:
                load_options["ent_coef"] = args.ent_coef
            model = MaskablePPO.load(args.model, env=train, device=device,
                                    tensorboard_log=str(args.output / "tensorboard"),
                                    **load_options)
        model.set_random_seed(args.seed)
        args.output.mkdir(parents=True, exist_ok=False)
        config = dict(vars(args), output=str(args.output), device=device,
                      victim_distance=args.victim_distance, max_steps=args.max_steps,
                      validation_seeds=args.validation_scenarios or [10_000, 10_050],
                      actions=list(RescueEnv.ACTIONS))
        config["validation_scenarios"] = validation_scenarios
        config["model"] = None if args.fresh else args.model
        config["parameters"] = sum(p.numel() for p in model.policy.parameters())
        config["extractor"] = model.policy.features_extractor.__class__.__name__
        config["spatial_local"] = isinstance(model.policy.features_extractor, SpatialRescueCNN)
        config["ent_coef"] = model.ent_coef
        (args.output / "config.json").write_text(json.dumps(config, indent=2))
        callback = MaskableEvalCallback(validation, best_model_save_path=str(args.output),
                                        log_path=str(args.output),
                                        eval_freq=max(1, 5120 // args.n_envs),
                                        n_eval_episodes=len(validation_scenarios), deterministic=True)
        model.learn(total_timesteps=args.steps, callback=callback, reset_num_timesteps=False)
        model.save(args.output / "last_model")
    finally:
        train.close()
        validation.close()


if __name__ == "__main__":
    main()
