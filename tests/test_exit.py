from rescue_env.envs.single_agent import RescueEnv

env = RescueEnv()
env.reset(seed=42)

# Reach the victim and pick them up.
for action in [2, 2, 5, 4, 4]:
    env.step(action)

print("Current position:", env.sim.agents[0]["r"], env.sim.agents[0]["c"])

print("\nExits:")
for opening in env.sim.openings:
    if opening["kind"] == "exit":
        print(opening)
        print("Exit cells:", env.sim._portal_cells(opening, opening["zone_a"]))

print("\nAction mask before trying to exit:", env.action_masks())

# Try moving left from column zero.
obs, reward, terminated, truncated, info = env.step(4)

print("\nAfter moving left:")
print("Agent:", env.sim.agents[0])
print("Victim:", env.sim.victims[0])
print("Action mask:", env.action_masks())
print("Reward:", reward)
print("Terminated:", terminated)

env.close()

