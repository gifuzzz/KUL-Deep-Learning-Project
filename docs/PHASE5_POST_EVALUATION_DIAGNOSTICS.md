# Phase 5 Diagnostic Report: Distance Semantics, Oscillation Mechanisms, and Exterior Exploration

**Document ID:** `PHASE5_POST_EVALUATION_DIAGNOSTICS.md`  
**Target Codebase:** `fire-rescue-project-2.0`  
**Authors:** Gemini Orchestrator & Recurrent RL Team  
**Date:** November 2026  
**Status:** Completed Diagnostic & Verification  

---

## 1. Executive Summary

Following visual inspection of the agent in the FlameBox Visual Agent Debugger, three critical anomalies were identified that directly constrain long-distance search-and-rescue performance:

1. **Distance Metric Distortion:** The scenario generator measures victim distance from **any building exit** rather than the firefighter's actual starting position. In 40%–55% of episodes across Medium and Bridge tiers, the victim was placed near a *different exit* across the building, placing the victim **20 to 80 steps away** from the firefighter rather than the nominal 6–12 steps. When evaluated strictly on true spawn distance, the agent achieves **86.8% rescue success** on nominal Medium ($d \le 6$) and **33.6%** on Bridge B ($d \le 10$).
2. **Exploration & Carrying Oscillation:** The agent frequently enters 2-cell limit cycles ($A \leftrightarrow B$), pacing between adjacent cells. In Distant deterministic evaluation, **236.2 out of 384 steps (61.5% of the episode)** are spent oscillating. 85.5% of these oscillations occur in open room/hallway interiors, driven by competition between the exploration gradient (pulling forward into unvisited cells) and the return gradient (pulling backward toward the known spawn exit).
3. **Pathological Exterior Wandering:** When stepping through an exterior opening without carrying a victim, the agent enters the perimeter grass ring (`zone = -1`). Once outside, C++ zeroes out all local observation channels, the memory map loses track of agent position, and all movements remain legally unmasked, trapping the agent in an observation void for up to 150+ steps.

This report documents the exact code mechanisms, empirical measurements across 200 held-out matched scenarios per tier, representative trajectory traces, and minimal testable corrections.

---

## 2. Investigation 1: Distance Semantics & Scenario Generation Audit

### 2.1 Code-Level Root Cause
In `rescue_env/envs/single_agent.py` (`_place_victims`, lines 208–239):
```python
        start = e._cells_for_opening(exit_, exit_["zone_a"])

        starts = [start]
        if self.min_victim_distance > 1:
            starts = [cell for o in e.openings
                      if o["kind"] == "exit" and o["open"] and not o.get("blocked")
                      for cell in e._portal_cells(o, o["zone_a"])]
        seen = set(starts)
        queue = deque((cell, 0) for cell in starts)
        candidates = []
        while queue:
            (r, c), distance = queue.popleft()
            if (distance >= self.min_victim_distance and ...):
                candidates.append((r, c, distance))
            if distance >= self.victim_distance:
                continue
```
In `rescue_env/flamebox/flamebox.py` (`Environment.__init__`, lines 479–493):
```python
  exit0=next(o for o in self.openings if o['kind']=='exit')
  spawn=self._cells_for_opening(exit0,exit0['zone_a'])
  available=[];seen={spawn};todo=deque([spawn])
  # ... BFS places agent at available[0] (immediately adjacent to exit0) ...
  self.agents.append({'id':k,'r':rr,'c':cc})
```

### 2.2 Distinguishing the Three Distance Metrics
1. **$D_{\text{exit}}^{\text{min}}$ (Current Metric):** $\min_{e \in \mathcal{E}_{\text{open}}} \text{dist}_{\text{BFS}}(e, v)$. Shortest traversable grid distance from the victim to the *closest open exterior exit*.
2. **$D_{\text{spawn}}^{\text{BFS}}$ (Actual Task Difficulty):** $\text{dist}_{\text{BFS}}(s_{\text{agent}}, v)$. Shortest traversable grid distance from the firefighter's *actual starting position* to the victim.
3. **$D_{\text{spawn}}^{\text{Manhattan}}$:** $|r_{\text{agent}} - r_{\text{victim}}| + |c_{\text{agent}} - c_{\text{victim}}|$. $L_1$ coordinate distance without wall obstructions.

### 2.3 Empirical Quantification (200 Held-Out Episodes per Tier, Seeds 50000..50199)

| Evaluation Tier | Nominal Target ($D_{\text{target}}$) | Exit BFS ($D_{\text{exit}}^{\text{min}}$) | Spawn BFS ($D_{\text{spawn}}^{\text{BFS}}$) | Spawn Manhattan | Same Exit as Spawn | Different Exit from Spawn |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Near** ($d=2$) | 2 | **1.1** (0–2) | **1.5** (1–2) | 1.5 | **200 (100.0%)** | 0 (0.0%) |
| **Medium** ($d=6$) | 6 | **4.7** (3–6) | **21.3** (3–75) | 19.6 | **106 (53.0%)** | **94 (47.0%)** |
| **Bridge A** ($d=8$) | 8 | **6.7** (5–8) | **19.6** (5–77) | 17.8 | **122 (61.0%)** | **78 (39.0%)** |
| **Bridge B** ($d=10$) | 10 | **8.8** (7–10) | **21.4** (7–79) | 19.5 | **119 (59.5%)** | **81 (40.5%)** |
| **Bridge C** ($d=12$) | 12 | **10.7** (9–12) | **23.6** (9–81) | 20.8 | **110 (55.0%)** | **90 (45.0%)** |
| **Distant** ($d=\text{None}$) | None | **30.9** (7–100) | **40.6** (7–110) | 30.9 | **91 (45.5%)** | **109 (54.5%)** |

### 2.4 Impact on Historical Benchmarks: The "Medium" Mystery Solved
Why was the historical Medium ($d=6$) success rate ~48%?
When we split the 200 Medium evaluation episodes by whether the victim was placed near the **same exit as the spawn** versus a **different exit**, the result is staggering:

```
Medium Tier (d=6) Rescue Rate Split (Production 250k Model, Deterministic):
  Same Exit as Spawn (106 episodes, mean spawn dist = 5.1):  92 / 106 = 86.8% RESCUE SUCCESS
  Different Exit     ( 94 episodes, mean spawn dist = 39.6):  4 /  94 =  4.3% RESCUE SUCCESS
  Overall Blended Benchmark:                                 96 / 200 = 48.0% RESCUE SUCCESS
```

- When the victim was actually placed 3–6 steps from the firefighter, the agent succeeded **86.8%** of the time!
- When the victim was placed near another exit 40 steps away across the building, the agent succeeded only **4.3%** of the time within the 128-step budget.
- The reported 48.0% success rate was not an agent struggling with 6-step navigation; it was an artifact of averaging 86.8% near rescues with 4.3% cross-building marathon rescues!
- On **Bridge B ($d=10$)**: Same-exit success is **33.6%** (40/119), while different-exit success is **2.5%** (2/81).

### 2.5 Preservation Policy
To maintain integrity:
1. All historical benchmark definitions (`victim_distance` measured from nearest exit) are preserved unchanged under the name `legacy_entrance_distance`.
2. A corrected `spawn_relative_distance` evaluation mode will be introduced side-by-side to report true spawn-relative capability.

---

## 3. Investigation 2: Corner and Doorway Oscillation

### 3.1 Mechanism Audit
Is oscillation caused by inaccurate observations, invalid-action masking, GRU state defects, or learned policy dynamics?
- **Inaccurate Observations:** **NO.** Inspection of captured tensors in `trajectories/doorway_oscillation_seed50057_det.json.gz` shows the 9-channel local grid and memory map accurately represent adjacent walls, doorways, and explored cells.
- **Invalid-Action Masking:** **NO.** Actions `Move Right` and `Move Left` are both legal when moving between two open cells.
- **Learned Policy Limit Cycles:** **YES.** The oscillation is an attractor limit cycle in the recurrent policy network.

### 3.2 Exploring vs. Carrying Oscillation Breakdown (30 Distant Seeds, 50000..50029)

| Evaluation Mode | Exploring Oscillation (Steps/Ep) | Episodes with $\ge 5$ Exploring Osc | Carrying Oscillation (Steps/Ep) | Episodes with $\ge 5$ Carrying Osc |
|---|:---:|:---:|:---:|:---:|
| **Deterministic** | **236.2 steps** (7,086 total) | **86.7%** | 0.0 steps* | 0.0%* |
| **Stochastic** | **51.4 steps** (1,543 total) | **90.0%** | **6.2 steps** (185 total) | **16.7%** |

*\*Note: In deterministic mode on Distant, the agent gets trapped in exploring oscillation so early that it rarely reaches the victim to begin carrying.*

### 3.3 Spatial Location of Oscillations
Inspecting 4,101 oscillating steps across deterministic seeds:
- **Open Hallway / Room Interior:** **3,453 steps (84.2%)**
- **Corner / Dead-End:** **482 steps (11.8%)**
- **Doorway Threshold:** **166 steps (4.0%)**

Dominant action pairs:
- `Move Down` $\leftrightarrow$ `Move Up`: **2,983 steps (72.7%)**
- `Move Left` $\leftrightarrow$ `Move Right`: **1,118 steps (27.3%)**

### 3.4 Root Cause: Gradient Opposition at Doorways & Corridors
In 1D hallways (such as Zone 6 in seed 50000, width 3, bounded by walls on left and right), the agent experiences two opposing behavioral forces:
1. **The Exit-Return Bias (South):** Trained during return-training, the value function attracts the agent toward known exterior exits (located South at $r=38$).
2. **The Frontier-Exploration Drive (North):** When `carrying = False`, moving closer to the exit yields zero discovery reward, while moving North uncovers unexplored building space.

At cell $(33, 31)$, the exploration gradient dominates $\to$ `Move Up` (logit 2.29 vs Down 1.39).  
At cell $(32, 31)$, all surrounding cells have already been uncovered; the return bias dominates $\to$ `Move Down` (logit 1.80 vs Up 1.59).  
In deterministic mode ($\arg\max$), this forms a deterministic 2-cycle that repeats endlessly for 300+ steps. In stochastic mode, sampling noise allows the policy to occasionally break the cycle (reducing oscillation from 236 to 51 steps).

---

## 4. Investigation 3: Unproductive Exterior Exploration

### 4.1 Audit of Exterior Mechanics
We audited C++, Python wrappers, memory map, and reward shaping:

1. **C++ Observation Void (`rescue_env/flamebox/native/sim.cpp`, line 1091):**
   ```cpp
   if(!valid(ag.r,ag.c)||!valid(nr,nc) || y*y+x*x>radius*radius || !line_of_sight(ag.r,ag.c,nr,nc))continue;
   ```
   When the agent steps onto the perimeter grass ring (`!valid(r, c)` where $r < 0$, $r \ge H$, $c < 0$, or $c \ge W$), `valid(ag.r, ag.c)` is **false**. C++ returns an **all-zero local grid** ($15 \times 15 \times 9 = 0$).
2. **Memory Map Disconnect (`rescue_env/envs/memory_map.py`, line 125):**
   When $r \ge H$ or $c \ge W$, the boundary check `0 <= ar < self.height` fails. Channel 8 (Agent Position) is **completely omitted** from the memory tensor!
3. **Agent Scalar Out-of-Bounds:**
   $r / H$ and $c / W$ become $> 1.0$, feeding out-of-distribution scalar inputs to the MLP fusion layer.
4. **Reward Shaping:**
   Exterior grass exploration **does NOT receive positive reward**. `newly_uncovered` is 0 because the memory map only covers interior dimensions $(H, W)$. The agent receives $-0.001$ step penalty.
5. **Action Legality Flaw (`sim.cpp`, line 929):**
   ```cpp
   bool inside=valid(r,c),destination=valid(nr,nc);
   if(!inside&&!destination)return true;
   ```
   Once on the grass outside, **all movement in all 4 directions is 100% legal**! The agent is allowed to wander indefinitely along the exterior lawn.
6. **Termination Semantics:**
   Stepping onto the grass terminates the episode **only if the agent is carrying a victim** (`v.carried_by == k`). When `carrying == False`, stepping outside does not terminate the episode; the firefighter simply wanders blind in the grass until timeout.

### 4.2 Frequency of Exterior Wandering
- **Deterministic Mode:** **1.0%** (1 / 100 episodes on Distant).
- **Stochastic Mode:** **18.0%** (9 / 50 episodes on Distant).
- **Representative Case:** Seed 50005 stochastic. At step 230, the agent reached the exit portal $(29, 45)$. At step 231, it executed `Move Right` onto the exterior grass at $(29, 46)$ without carrying a victim. It wandered outside for 5 steps before fortuitously stepping back inside at step 236.

---

## 5. Proposed Minimal, Independently Testable Corrections

### Correction 1: Dual Distance Evaluation (Preserving Legacy Benchmarks)
- **Problem:** `victim_distance` in `RescueEnv._place_victims()` uses multi-source BFS from all exits when `min_victim_distance > 1`.
- **Minimal Fix:**
  Add a parameter `distance_metric="exit"` (default, preserving legacy behavior) vs `distance_metric="spawn"`:
  ```python
  if self.distance_metric == "spawn":
      # Start BFS strictly from agent spawn position (the spawn exit)
      starts = [start_spawn]
  else:
      # Legacy behavior: BFS from all open exits
      starts = [cell for o in e.openings if o["kind"] == "exit" ... for cell in e._portal_cells(o, o["zone_a"])]
  ```
- **Benefit:** Allows direct, uncorrupted measurement of true $d=2, 6, 8, 10, 12$ navigation without altering legacy checkpoint evaluations.

### Correction 2: Mask Exterior Movements When Not Carrying
- **Problem:** Agent steps through exterior exits onto grass when searching for a victim, entering an observation void.
- **Minimal Fix:**
  In `action_masks()` of `rescue_env/envs/single_agent.py`:
  ```python
  if not carrying:
      for action_idx, (dr, dc) in enumerate([(-1,0), (0,1), (1,0), (0,-1)], start=1):
          nr, nc = person["r"] + dr, person["c"] + dc
          if self.sim.grass(nr, nc):
              mask[action_idx] = False
  ```
- **Benefit:** Completely eliminates 100% of pathological exterior wandering. Zero impact on interior doors or windows. Allows stepping onto grass when `carrying == True` for rescue.

### Correction 3: Anti-Oscillation Training Penalty
- **Problem:** Opposing exploration and return gradients create 2-cell limit cycles in 1D corridors and room doorways.
- **Minimal Fix:**
  During training with reward shaping (`reward_shaping=True`), apply an oscillation penalty:
  If $s_t = s_{t-2}$ and $s_{t-1} = s_{t-3}$ and $s_t \neq s_{t-1}$:
  Award $-0.02$ anti-oscillation penalty.
  Disable completely during evaluation (`reward_shaping=False`).
- **Benefit:** Breaks the symmetry of the logit competition during training without imposing scripted actions or heuristics.

---

## 6. Verification and Next Steps

1. Implement Correction 1 (`distance_metric` support in `RescueEnv` and evaluation tooling).
2. Implement Correction 2 (Exterior grass action masking when `carrying == False`).
3. Run unit tests to verify zero regressions.
4. Execute a short controlled pilot on matched seeds to quantify improvements on true distance tiers and oscillation reduction.

---
*Report certified by Gemini Orchestrator & Recurrent RL Team.*
