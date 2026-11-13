from rescue_env.envs.single_agent import RescueEnv

env = RescueEnv()
obs, info = env.reset(seed=42)

print("Action space:", env.action_space)
print("Action mask:", env.action_masks())

print("Water action mapping:", env.ACTIONS[6])

obs, reward, terminated, truncated, info = env.step(6)

print("Water action executed")
print("Reward:", reward)
print("Firefighter:", env.sim.agents[0])

env.close()

