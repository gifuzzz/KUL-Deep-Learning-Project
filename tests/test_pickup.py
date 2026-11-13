
from rescue_env.envs.single_agent import RescueEnv

env = RescueEnv()
obs, info = env.reset(seed=42)

# Move toward victim.
for action in [2, 2]:
    env.step(action)

# Pick up victim.
env.step(5)

print("After pickup:", env.sim.victims[0])
print("Agent position:", (env.sim.agents[0]["r"], env.sim.agents[0]["c"]))

# Return toward the original entry point.
for action in [4, 4]:
    env.step(action)

print("\nAfter returning:")
print("Agent:", env.sim.agents[0])
print("Victim:", env.sim.victims[0])
print("Action mask:", env.action_masks())

# Attempt rescue if permitted.
if env.action_masks()[6]:
    obs, reward, terminated, truncated, info = env.step(6)
    print("\nRescue reward:", reward)
    print("Success:", info["success"])
    print("Episode terminated:", terminated)
else:
    print("\nRescue not yet available.")
    print("We need to determine how the firefighter exits the building.")

env.close()
