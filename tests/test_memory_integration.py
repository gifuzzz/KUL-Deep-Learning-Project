from rescue_env.envs.single_agent import RescueEnv

env = RescueEnv()
obs, info = env.reset(seed=42)

print("Observation keys:", obs.keys())
print("Local grid:", obs["grid"].shape)
print("Memory:", obs["memory"].shape)
print("Agent:", obs["agent"].shape)

print("Explored cells:", obs["memory"][-2].sum())
print("Matches declared space:", env.observation_space.contains(obs))

# Move the firefighter and observe the memory update.
obs, reward, terminated, truncated, info = env.step(2)

print("\nAfter moving:")
print("Explored cells:", obs["memory"][-2].sum())
print("Matches declared space:", env.observation_space.contains(obs))

env.close()

