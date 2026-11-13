
import torch

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor

from agents.cnn import RescueCNN
from rescue_env.envs.single_agent import RescueEnv


def make_env(reward_shaping=False, navigation_shaping=False):
    return Monitor(
        RescueEnv(
            max_steps=64,
            victim_distance=2,
            reward_shaping=reward_shaping,
            navigation_shaping=navigation_shaping,
        ),
        info_keywords=("success", "timeout"),
    )


def main():
    torch.set_num_threads(1)

    train_env = make_env(
        reward_shaping=True,
        navigation_shaping=True,
    )

    eval_env = make_env(
        reward_shaping=False,
        navigation_shaping=False,
    )

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path="./models/no_drop/",
        log_path="./logs/no_drop_eval/",
        eval_freq=5000,
        n_eval_episodes=20,
        deterministic=True,
    )

    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=train_env,
        policy_kwargs={
            "features_extractor_class": RescueCNN,
            "normalize_images": False,
        },
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        device="mps",
        tensorboard_log="./logs/",
        seed=42,
    )

    try:
        model.learn(
            total_timesteps=100_000,
            callback=eval_callback,
            tb_log_name="CNN_Memory_",
        )

        model.save("./models/no_drop/last_model")

    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
