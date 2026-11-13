from gymnasium.utils.env_checker import check_env

from rescue_env.envs.single_agent import RescueEnv


def main():
    env = RescueEnv()

    try:
        # Verify Gymnasium interface
        check_env(env, skip_render_check=True)

        obs, info = env.reset(seed=42)

        print("Grid shape:", obs["grid"].shape)
        print("Agent shape:", obs["agent"].shape)
        print("Valid observation:",
              env.observation_space.contains(obs))

        for i in range(20):
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            print(
                f"Step {i + 1}: "
                f"action={action}, "
                f"reward={reward}, "
                f"rescued={info['victims_rescued']}"
            )

            if terminated or truncated:
                print("Episode finished!")
                obs, info = env.reset()

        print("Environment test completed!")

    finally:
        env.close()


if __name__ == "__main__":
    main()
