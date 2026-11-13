import numpy as np

from rescue_env.envs.memory_map import MemoryMap


memory = MemoryMap(height=20, width=20)

# Simulate a 5x5 local observation
local = np.zeros((11, 5, 5), dtype=np.float32)

# Mark the center cell as visible
local[2, 2, 2] = 1.0

# Put a victim at the visible center cell
local[3, 2, 2] = 1.0

memory.update(local, agent_position=(10, 10))

print("Victim remembered:", memory.grid[3, 10, 10])
print("Cell explored:", memory.explored[10, 10])

# Simulate moving away with no new visible cells
empty = np.zeros((11, 5, 5), dtype=np.float32)
memory.update(empty, agent_position=(15, 15))

print("Victim still remembered:", memory.grid[3, 10, 10])
print("Observation shape:", memory.get_observation().shape)

# Verify that reset clears the memory
memory.reset()
print("After reset:", memory.grid[3, 10, 10])

