
import torch

from agents.cnn import RescueCNN
from rescue_env.envs.single_agent import RescueEnv


env = RescueEnv()
obs, _ = env.reset(seed=42)

cnn = RescueCNN(env.observation_space)

# Convert observations to PyTorch tensors
# Add a batch dimension because the CNN expects batches
tensor_obs = {
    key: torch.tensor(value, dtype=torch.float32).unsqueeze(0)
    for key, value in obs.items()
}

with torch.no_grad():
    features = cnn(tensor_obs)

print("Local grid:", tensor_obs["grid"].shape)
print("Memory:", tensor_obs["memory"].shape)
print("Agent:", tensor_obs["agent"].shape)
print("CNN output:", features.shape)

assert features.shape == (1, 128)

print("CNN test passed!")

env.close()
