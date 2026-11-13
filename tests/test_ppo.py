from sb3_contrib import MaskablePPO

from agents.cnn import RescueCNN
from rescue_env.envs.single_agent import RescueEnv


env = RescueEnv(
    max_steps=64,
    victim_distance=2,
)

model = MaskablePPO(
    "MultiInputPolicy",
    env,
    policy_kwargs={
        "features_extractor_class": RescueCNN,
        "normalize_images": False,
    },
    device="cpu",
    verbose=0,
)

obs, _ = env.reset(seed=42)

action, _ = model.predict(
    obs,
    deterministic=False,
    action_masks=env.action_masks(),
)

print("Predicted action:", action)
print("Action mask:", env.action_masks())

assert env.action_space.contains(int(action))
assert env.action_masks()[int(action)]

obs, reward, terminated, truncated, info = env.step(int(action))

print("Step executed successfully!")
print("Reward:", reward)

env.close()

