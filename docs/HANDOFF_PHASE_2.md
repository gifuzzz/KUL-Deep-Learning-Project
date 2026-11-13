# Phase 2 Contract: Environment, Observations & Rewards

## 1. Observation Space (`gym.spaces.Dict`, all `np.float32`)
- `"grid"`: `(9, 15, 15)` — Channel-first local window (visibility radius = 7).
- `"memory"`: `(9, 40, 52)` — Fixed canvas, centered tightly on building footprint.
- `"agent"`: `(5,)` — `[row_norm, col_norm, health, role_norm, carrying]`.

## 2. Channel Ordering (Channels 0–8 for `"grid"` and `"memory"`)
- Ch 0: Passability North (1.0 if North face traversable / flux >= 0.5, else 0.0)
- Ch 1: Passability East (1.0 if East face traversable / flux >= 0.5, else 0.0)
- Ch 2: Passability South (1.0 if South face traversable / flux >= 0.5, else 0.0)
- Ch 3: Passability West (1.0 if West face traversable / flux >= 0.5, else 0.0)
- Ch 4: Debris / Obstacle (1.0 if cell blocked by collapse/debris, else 0.0)
- Ch 5: Explored Mask (1.0 if previously explored / in line-of-sight, else 0.0)
- Ch 6: Victim (>0.0 normalized type if live victim present, else 0.0)
- Ch 7: Exit (1.0 if exit portal present/remembered, else 0.0)
- Ch 8: Agent Position (1.0 at current firefighter cell, else 0.0; transient)

## 3. Action Space & Action Mask
- `action_space`: `gym.spaces.Discrete(7)` — `(0: Wait, 1: N, 2: E, 3: S, 4: W, 5: Pickup, 6: Water)`.
- Action mask: `env.action_masks()` returns `np.ndarray(shape=(7,), dtype=bool)`.

## 4. Reward Shaping (`reward_shaping=True`; 0.0 when `False` in evaluation)
- Step penalty: `-0.001` per step.
- First victim pickup: `+1.5` for first live pickup.
- Frontier exploration: `+0.05` per newly uncovered cell in `memory_map` (cap: `+3.0`/ep).
