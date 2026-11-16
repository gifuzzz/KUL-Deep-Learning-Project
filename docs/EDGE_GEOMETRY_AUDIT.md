# Edge-Based Wall Geometry Audit: Project 2.0 vs Upstream FlameBox

**Date:** November 16, 2026  
**Auditor:** Gemini Orchestrator  
**Workspaces Audited:**
- Target Workspace: `fire-rescue-project-2.0` (`/Users/luigi/uni/Master/Semester 1/Deep Learning/project/fire-rescue-project-2.0`)
- Upstream Reference: `FlameBoxWorkspace 2` (`/Users/luigi/Downloads/FlameBoxWorkspace 2`)

---

## 1. Executive Verdict

**The discrepancy between full-cell walls and edge-based walls is NOT merely visual.**

`fire-rescue-project-2.0` inherited an artificial Python-level modification in `rescue_env/flamebox/flamebox.py` (lines 401–420) from `fire-rescue-project-1.0`:
```python
# Walls occupy cells rather than being represented only by blocked faces.
self.wall = np.zeros((H, W), dtype=bool)
...
self.zone[self.wall] = -1
```

In a standard $44 \times 52$ building (size=8, resolution=2, seed=42):
- **Current `fire-rescue-project-2.0`:** **398 cells** (17.4% of the building footprint) are explicitly converted into `zone = -1`, turning them into impassable obstacle cells.
- **Upstream `FlameBoxWorkspace 2`:** **0 cells** have `zone = -1`. The `self.wall` array **does not exist**. Every single cell inside $[0, H) \times [0, W)$ belongs to a valid room ($z \ge 0$).

The native C++ simulation kernel (`sim.cpp` / `sim_cpu.cpp`) **never used full-cell walls**. As stated on line 2 of `sim.cpp`:
> `// Cell graph: dense 4-connected lattice with per-face conductance; zones are node metadata.`

The full-cell wall behavior was solely introduced by the Python wrapper carving out perimeter cells and setting their zone IDs to `-1`.

---

## 2. In-Depth System Audit

### 2.1 Native CPU/CUDA Kernels (`sim.cpp` vs `sim_cpu.cpp`)
- **Node Representation:** Nodes are cells $i = r \cdot W + c$. Cell coordinates $0 \le r < H$ and $0 \le c < W$ define interior valid cells (`valid(r, c)`).
- **Edge Representation:** Edges are stored as per-cell East and South face conductances (`flux_e`, `flux_s`).
  - East face of cell $i$ connects to $i + 1$.
  - South face of cell $i$ connects to $i + W$.
- **Movement Legality (`traversable(i, j)`):**
  ```cpp
  bool traversable(int i, int j) const {
    if (debris[j]) return false;
    if (j == i + 1) return flux_e[i] >= 0.5f;
    if (j == i - 1) return flux_e[j] >= 0.5f;
    if (j == i + W) return flux_s[i] >= 0.5f;
    if (j == i - W) return flux_s[j] >= 0.5f;
    return false;
  }
  ```
- **Conclusion:** The native kernel is 100% edge-conductance based. It natively supports movement between any adjacent cells as long as the connecting face has conductance $\ge 0.5$ and the destination is free of debris.

### 2.2 Procedural Building Generation (`flamebox.py`)
- **Room & Corridor Partitioning:** `_layout()` partitions $[0, H_{\text{base}}) \times [0, W_{\text{base}})$ into rectangular rooms and corridors. Scaling by resolution $q$ defines `self.zone` for all $(r, c) \in [0, H) \times [0, W)$.
- **Discrepancy:**
  - Upstream: Zone boundaries naturally have differing room IDs (`zone[r, c] != zone[r, c+1]`). `_apply_openings()` initializes `east` and `south` to 0.0 across different zones, and sets door spans to 1.0 (open) or 0.012 (closed).
  - Project 2.0: Scans adjacent cells with different zones and marks the *left* and *top* cells as `self.wall = True`, then sets `self.zone[self.wall] = -1`. This asymmetrically destroys room floor space and creates artificial 1-cell dead zones.

### 2.3 Wall and Doorway Representations
- **Upstream:**
  - Wall: A boundary face with conductance $< 0.5$ (typically 0.0). Both cells adjacent to the wall remain valid floor spaces that agents and victims can occupy.
  - Doorway: A span of face conductance on the boundary set to 1.0 (open) or 0.012 (closed).
- **Project 2.0:**
  - Wall: An entire grid cell with `zone = -1`.
  - Doorway: A gap in the carved wall cells, forcing agents through an artificial 1-cell corridor.

### 2.4 Movement and Collision Rules
- **Upstream:** Movement checks `flux_e`/`flux_s` between cells $A$ and $B$, plus destination obstacle `debris[B] == 0`.
- **Project 2.0:** Because wall cells had `zone = -1`, `flux_e` and `flux_s` entering them were 0.0. Furthermore, `prevent_exterior_wandering` masked out movements where `zone[nr, nc] < 0`, conflating interior walls with exterior grass.

### 2.5 Observation Generation (`single_agent.py` & `sim.cpp`)
- Local observation is $(9, 15, 15)$:
  - Ch 0: North Passability (`flux_s[idx - W] >= 0.5`)
  - Ch 1: East Passability (`flux_e[idx] >= 0.5`)
  - Ch 2: South Passability (`flux_s[idx] >= 0.5`)
  - Ch 3: West Passability (`flux_e[idx - 1] >= 0.5`)
  - Ch 4: Debris (`debris[idx] > 0`)
  - Ch 5: Explored Mask (`line_of_sight(ag, target)`)
  - Ch 6: Victim (`v.kind`)
  - Ch 7: Exit Portal
  - Ch 8: Agent Position
- **Channel Semantics Already Match Edges!** The local grid channels 0–3 have *always* encoded directional face conductances. However, with full-cell walls, cells adjacent to walls observed 0.0 passability entering the wall cell, and the wall cell itself was unobservable/impassable. In edge geometry, channels 0–3 encode whether the agent can step North, East, South, or West across the cell boundary.

### 2.6 Remembered-Map Encoding (`memory_map.py`)
- Memory map is $(9, 40, 52)$, tightly centered on the building footprint.
- Channels 0–3 store remembered N, E, S, W edge passabilities.
- When unobserved, all channels are 0.0 and explored (Ch 5) is 0.0.
- When observed (`vis_mask > 0`), the observed directional passabilities are copied into memory, and Ch 5 is set to 1.0.
- **Symmetry Opportunity:** When observing cell $A$ with East passability $P_{E}(A)$, the adjacent cell $B = A + (0, 1)$ must have West passability $P_{W}(B) = P_{E}(A)$. Currently, memory updates each cell strictly within the visible mask. In edge geometry, edge symmetry can be enforced directly.

### 2.7 Legal-Action Masking
- Actions: 0: No-op, 1: North, 2: East, 3: South, 4: West, 5: Toggle, 6: Pickup/Drop.
- Upstream C++ `sim.move_mask()` evaluates `move_possible(r, c, nr, nc)`:
  - North: checks `flux_s` of cell above $\ge 0.5$ and `!debris`.
  - East: checks `flux_e` of current cell $\ge 0.5$ and `!debris`.
  - South: checks `flux_s` of current cell $\ge 0.5$ and `!debris`.
  - West: checks `flux_e` of cell to left $\ge 0.5$ and `!debris`.
- `prevent_exterior_wandering` in `single_agent.py`:
  - When carrying a victim: movement through exterior exits onto grass is legal and triggers rescue.
  - When empty-handed: movement onto grass is masked out.
  - **Correction required:** In edge geometry, exterior cells are strictly outside the building envelope (`nr < 0 or nr >= H or nc < 0 or nc >= W`). No interior cell has `zone < 0`.

### 2.8 Distance Calculations (`single_agent.py`, `diagnose_curriculum.py`)
- Shortest paths use grid-BFS checking edge conductances $\ge 0.5$ and `!debris`.
- With full-cell walls, BFS path lengths were lengthened because agents had to route around 398 carved wall cells.
- In edge geometry, rooms have their full contiguous floor space, leading to shorter, more natural shortest-path trajectories.

### 2.9 Rescue and Exterior Movement
- Exterior grass is $r < 0$, $r \ge H$, $c < 0$, or $c \ge W$.
- Exit openings lie on the boundary edges ($c = 0$, $c = W$, $r = 0$, $r = H$).
- Moving onto grass with a victim triggers `v.rescued = 1` in native C++.
- This behavior is identical in both modes; edge geometry simply restores the perimeter floor cells immediately adjacent to exterior walls.

### 2.10 Visual Debugger Rendering (`training/viewer/app.js`)
- Lines 504–508:
  ```javascript
  const isExt = zCur < 0;
  const fill = isExt ? "#0f172a" : (zCur === 0 ? "#1e293b" : roomColors[(zCur - 1) % roomColors.length]);
  worldCtx.fillRect(offsetX + c * cellSize, offsetY + r * cellSize, cellSize, cellSize);
  ```
  Because Project 2.0 set `zone[self.wall] = -1`, the viewer filled all 398 wall cells with `#0f172a` (exterior black background), visibly rendering walls as full solid cells.
- Lines 518–550 also drew edge lines on top of the black cells.
- In edge geometry: all interior cells have `zCur >= 0` and are rendered as rooms/corridors. Walls are rendered exclusively as edge lines where `south == 0` or `east == 0`!

---

## 3. Discrepancy Matrix

| Feature | Legacy Cell Geometry (`legacy_cell`) | Corrected Edge Geometry (`edge`) |
|---|---|---|
| Wall representation | Full impassable grid cells (`zone = -1`) | Zero-conductance face boundaries (`flux < 0.5`) |
| Internal cells with `zone == -1` | 398 cells (17.4% of building) | **0 cells** (100% of building is valid floor) |
| Wall thickness | 1 full physical grid cell | 0 thickness (infinitesimal boundary edge) |
| Room perimeter cells | Carved out and destroyed | Fully accessible floor space |
| Doorways | 1-cell opening punched through wall cell | Conductance set to 1.0 on boundary face |
| Debris | Blocks full cell | Blocks full cell (unchanged) |
| Exterior definition | Out of bounds OR `zone < 0` | Strictly out of bounds (`r<0, r>=H, c<0, c>=W`) |
| Action mask checks | C++ edge check + `zone[dest] < 0` exterior mask | C++ edge check + boundary exterior mask |
| Ground truth visualizer | Paints 398 cells solid black | Paints all rooms to boundary; walls as edge lines |
| Policy tensor shapes | `(9, 15, 15)` and `(9, 40, 52)` | **Identical** `(9, 15, 15)` and `(9, 40, 52)` |
| Policy channel semantics | Ch 0–3: N, E, S, W passability | **Identical** Ch 0–3: N, E, S, W passability |

---

## 4. Architectural & Checkpoint Compatibility Assessment

1. **Tensor Shape Invariance:**
   - Input shapes to the neural network remain strictly `grid: (9, 15, 15)`, `memory: (9, 40, 52)`, `agent: (5,)`, `previous_action: (7,)`.
   - Action space remains 7 discrete actions.
   - GRU hidden dimension remains 128 (or 256 for Variant C).
2. **Channel Semantic Invariance:**
   - Channels 0–3 are already defined as North, East, South, West passabilities across face boundaries.
   - The neural network's convolution kernels expect values of 1.0 (passable) and 0.0 (blocked) in these channels.
   - In edge geometry, the agent will see fewer false 0.0s caused by artificial wall cells, but the semantic meaning of "1.0 means I can walk this way, 0.0 means a wall blocks me" is **100% identical**.
3. **Weight Transfer Feasibility:**
   - Pretrained checkpoints (`final_model.zip`, `phase5_pilot`, etc.) can be loaded **directly without changing network shapes or parameter names**.
   - Because the semantic conventions are identical, **zero-shot transfer** is immediately testable.

---

## 5. Migration Plan

1. **Parameterize Geometry Mode in `Environment` (`flamebox.py`):**
   - Add `geometry_mode: str = "edge"` (options: `"edge"`, `"legacy_cell"`).
   - In `"edge"` mode: completely omit the `self.wall` block and preserve all rooms and corridors at full dimensions.
   - In `"legacy_cell"` mode: retain the legacy wall carving for reproducing historical benchmarks.
2. **Update `single_agent.py`:**
   - Support `geometry_mode` parameter in `RescueEnv`.
   - Update `prevent_exterior_wandering` so in `"edge"` mode, exterior is strictly `nr < 0 or nr >= H or nc < 0 or nc >= W`.
   - Update BFS shortest-path distance routines to navigate edge-connected floor cells.
3. **Update Visual Debugger (`training/viewer/app.js`):**
   - Check `geometry_mode`: in `"edge"` mode, render room floor tiles up to the boundary and render walls cleanly as boundary edges.
4. **Validation Suite:**
   - Verify edge symmetry ($P_E(r, c) = P_W(r, c+1)$, $P_S(r, c) = P_N(r+1, c)$).
   - Verify doors open/close dynamically and toggle passability across edges.
   - Verify zero-shot evaluation on pretrained checkpoints.
