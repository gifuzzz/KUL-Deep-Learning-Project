import torch

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor

from rescue_env.envs.single_agent import RescueEnv


def make_env(reward_shaping=False):
    return Monitor(
        RescueEnv(
            max_steps=64,
            victim_distance=3,
            min_victim_distance=1,
            reward_shaping=reward_shaping,
            navigation_shaping=False,
        ),
        info_keywords=("success", "timeout"),
    )


def main():
    torch.set_num_threads(1)

    train_env = make_env(reward_shaping=True)
    eval_env = make_env(reward_shaping=False)

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path="./models/curriculum_d3/",
        log_path="./logs/curriculum_d3_eval/",
        eval_freq=5000,
        n_eval_episodes=20,
        deterministic=True,
    )

    model = MaskablePPO.load(
        "models/no_drop/best_model.zip",
        env=train_env,
        device="mps",
        tensorboard_log="./logs/",
    )

    try:
        model.learn(
            total_timesteps=100_000,
            callback=eval_callback,
            tb_log_name="CNN_Memory_Curriculum_D3",
            reset_num_timesteps=False,
        )

        model.save("./models/curriculum_d3/last_model")

    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()

