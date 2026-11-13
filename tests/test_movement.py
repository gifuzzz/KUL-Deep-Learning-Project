from rescue_env.envs.single_agent import RescueEnv

env = RescueEnv()
obs, info = env.reset(seed=42)

print("Starting position:", (env.sim.agents[0]["r"], env.sim.agents[0]["c"]))

for action in range(env.action_space.n):
    obs, reward, terminated, truncated, info = env.step(action)

    agent = env.sim.agents[0]

    print(
        f"Action {action}: "
        f"Position=({agent['r']}, {agent['c']}), "
        f"Reward={reward}, "
        f"Done={terminated or truncated}"
    )

    if terminated or truncated:
        break

env.close()

