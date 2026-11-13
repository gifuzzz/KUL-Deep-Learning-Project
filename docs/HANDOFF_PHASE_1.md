# Phase 1 Specification: Observation Channels & Shapes

## 1. Exact 9-Channel Mapping (Channels 0–8)
| Ch | Name | Source in 1.0 (`single_agent` / `memory_map`) | Description |
|---:|---|---|---|
| 0 | Passability North | `raw[8]` / `grid[4]` | 1.0 if face to North is traversable (flux >= 0.5), else 0.0 |
| 1 | Passability East | `raw[9]` / `grid[5]` | 1.0 if face to East is traversable (flux >= 0.5), else 0.0 |
| 2 | Passability South | `raw[10]` / `grid[6]` | 1.0 if face to South is traversable (flux >= 0.5), else 0.0 |
| 3 | Passability West | `raw[11]` / `grid[7]` | 1.0 if face to West is traversable (flux >= 0.5), else 0.0 |
| 4 | Debris / Obstacle | `raw[4]` / `grid[1]` | 1.0 if full cell blocked by debris/collapse, else 0.0 |
| 5 | Explored Mask | `memory[11]` (`self.explored`) | 1.0 if cell was previously in line-of-sight, else 0.0 |
| 6 | Victim | `raw[6]` / `grid[3]` | >0 (normalized type) if live victim present, else 0.0 |
| 7 | Exit | `grid[10]` (`exits`) | 1.0 if opening kind == "exit", else 0.0 |
| 8 | Agent Position | `memory[13]` (marker) | 1.0 at agent current (r, c) interior cell, else 0.0 |
*Channels 0–4 represent the 5 wall/passability layers; 5–8 represent active task channels.*

## 2. Array Shapes & Dimensions
- **Local Grid Shape:** `(9, 15, 15)` (Channel-first `(C, H, W)` with `H = W = 2 * 7 + 1 = 15`).
- **Memory Canvas Shape:** `(9, 64, 64)` (Fixed canvas padded up to `64 x 64`).
- **Cropped Memory Footprint:** `(9, 15, 15)` (15x15 window centered on agent position).

## 3. Agent Scalar Feature Indices (Shape: `(5,)`)
| Index | Feature | Source & Normalization |
|---:|---|---|
| 0 | `row_norm` | `agent_r / sim.h` (Normalized row coordinate) |
| 1 | `col_norm` | `agent_c / sim.w` (Normalized column coordinate) |
| 2 | `health` | Firefighter health in `[0.0, 1.0]` |
| 3 | `role_norm` | `role / 3.0` (0.0 for firefighter) |
| 4 | `carrying_victim` | `1.0` if carrying a live unrescued victim, else `0.0` |
