# Phase 3 Contract: Recurrent Spatial Policy Architecture

## Classes & Constructor Arguments
- `RecurrentSpatialRescueCNN(observation_space, action_space=None, lr_schedule=None, hidden_size=128, ...)`
  Subclasses `MaskableActorCriticPolicy` for SB3-contrib `MaskablePPO` with a 128-unit GRU.
- `SpatialRescueFeaturesExtractor(observation_space, features_dim=128)`
  Subclasses `BaseFeaturesExtractor` (unpooled local + stride-2 zero-maxpool memory).

## Architectural Specs
- Local: `Conv(9,32,k3,p1)->ReLU->Conv(32,64,k3,p1)->ReLU->Conv(64,8,k1)->Flat` (1,800).
- Memory: `Conv(9,32,k3,p1)->ReLU->Conv(32,32,k3,s2,p1)->ReLU->Conv(32,16,k1)->Flat` (8,320).
- Fusion & GRU: `Concat(1800+8320+5+7=10132)->Linear(...,128)->GRU(128,128)`.
- Heads: 7-dim masked policy logits (`action_net`) & 1-dim value scalar (`value_net`).

## Usage Snippet
```python
from sb3_contrib import MaskablePPO
from agents.cnn import RecurrentSpatialRescueCNN
from rescue_env.envs.single_agent import RescueEnv

env = RescueEnv()
model = MaskablePPO(RecurrentSpatialRescueCNN, env, n_steps=128, batch_size=64)
obs, _ = env.reset(seed=42)
# Single step rollout / inference with masking and recurrent state
action, state = model.predict(obs, action_masks=env.action_masks())
obs, reward, done, _, _ = env.step(int(action))
```
