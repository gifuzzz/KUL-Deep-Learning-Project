# Phase 11 — Observation Equivalence Audit

**Author:** Autonomous Agent (Antigravity)  
**Date:** November 17, 2026  
**Environment:** `fire-rescue-project-2.0`  
**Dataset:** 1,000 representative environment states sampled across multi-room edge-wall episodes  
**Telemetry File:** [`models/phase11_evaluations/observation_equivalence_audit.json`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/models/phase11_evaluations/observation_equivalence_audit.json)  
**Audit Script:** [`scripts/audit_phase11_observation_equivalence.py`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/scripts/audit_phase11_observation_equivalence.py)

---

## 1. Executive Summary

Before implementing the architecture ablation, we conducted an exhaustive mathematical and empirical audit across 1,000 representative states to determine whether the 9-channel local observation `(9, 15, 15)` can be reconstructed from or is subsumed by the remembered global map `(9, 40, 52)`.

### Core Findings:
1. **Zero Information Loss within Line-of-Sight:**  
   Across all cells within the agent's current line-of-sight (`local_grid[5] > 0`), the agent-centered crop from the remembered map achieves **100.00% bit-exact equivalence** with `local_grid` across **all 9 channels** (Mean L1 error $= 0.000000$).
2. **Strict Information Subsumption:**  
   In **0 out of 1,000 states** did `local_grid` contain any information absent from `memory`. The remembered map strictly subsumes all spatial information in the local observation.
3. **Memory Crop Outperforms Local View in Occluded Areas:**  
   In **97.40% of audited states**, the memory crop contains **strictly more explored geometry** than `local_grid`. Because `local_grid` is an instantaneous raycasted observation, cells behind interior walls within the $15 \times 15$ window are zeroed out (fog of war). The remembered map retains the previously mapped walls, doorways, and victims in those occluded cells.
4. **Zero Update Latency:**  
   In [`rescue_env/envs/single_agent.py`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/rescue_env/envs/single_agent.py#L1005-L1015), `self.memory.update()` is called immediately upon receiving the simulator's raw observation, before `self.memory.get_observation()`. Consequently, the remembered map is completely synchronous with the environment step.

---

## 2. Channel-by-Channel Reconstruction Fidelity (1,000 States)

A crop of shape `(9, 15, 15)` was extracted from `obs["memory"]` centered at `(pad_top + agent_r, pad_left + agent_c)`.

| Channel Index | Semantic Meaning | LOS Exact Match % | LOS Mean L1 Error | All $15 \times 15$ Match % | All $15 \times 15$ Mean L1 | Reconstruction Status |
|---|---|---|---|---|---|---|
| **Ch 0** | Passability North | **100.00%** | $0.000000$ | 87.73% | 0.122720 | Bit-exact in LOS; Memory retains occluded walls |
| **Ch 1** | Passability East | **100.00%** | $0.000000$ | 86.53% | 0.134680 | Bit-exact in LOS; Memory retains occluded walls |
| **Ch 2** | Passability South | **100.00%** | $0.000000$ | 87.64% | 0.123587 | Bit-exact in LOS; Memory retains occluded walls |
| **Ch 3** | Passability West | **100.00%** | $0.000000$ | 86.46% | 0.135382 | Bit-exact in LOS; Memory retains occluded walls |
| **Ch 4** | Debris / Obstacle | **100.00%** | $0.000000$ | **100.00%** | $0.000000$ | Bit-exact everywhere |
| **Ch 5** | Explored / Visibility | **100.00%** | $0.000000$ | 86.33% | 0.136720 | Bit-exact in LOS; Memory retains explored history |
| **Ch 6** | Victim Presence | **100.00%** | $0.000000$ | 99.99% | 0.000027 | Bit-exact in LOS; Discrepancy only upon victim removal |
| **Ch 7** | Exit Portal | **100.00%** | $0.000000$ | 99.96% | 0.000360 | Bit-exact in LOS; Memory retains discovered exits |
| **Ch 8** | Agent Marker | **100.00%** | $0.000000$ | **100.00%** | $0.000000$ | Exact center alignment at coordinate $(7, 7)$ |

---

## 3. Structural Differences & Semantic Nuances

### 3.1 Instantaneous Visibility vs. Accumulated Explored Mask (Channel 5)
- In `local_grid[5]`, Channel 5 is the instantaneous line-of-sight mask computed by the simulator's raycaster (`raw["grid"][5] > 0`). Any cell behind a closed door or solid wall is `0.0`.
- In `memory[5]`, Channel 5 is the accumulated explored mask. Once a cell has been observed, it remains `1.0` indefinitely.
- **Architectural Implication:** In an agent-centered crop from memory, Channel 5 indicates which local cells are known vs unknown. In `local_grid`, Channel 5 indicates what is actively visible right now.

### 3.2 Occluded Geometry Retention (Channels 0–4)
- When an agent stands in a doorway or near an interior partition, `local_grid` cannot see behind the wall; those cells have passability `0.0`.
- If the agent previously explored the adjacent room, the crop from memory retains the passability of those occluded cells.
- This explains why the match rate across all $15 \times 15$ cells is $\sim 86.5\% - 87.7\%$: the memory crop provides **richer, more complete local spatial context** by integrating past observations with the current view.

### 3.3 Victim State Persistence (Channel 6)
- In `local_grid`, an undiscovered victim is invisible until line-of-sight is established. When picked up, the victim transitions to `carried_by == 0`.
- In `memory`, once a victim is observed, `memory[6]` retains their location until the agent arrives and picks them up.
- In our audit, across all 1,000 states, there were zero discrepancies within line-of-sight.

### 3.4 Spatial Coordinate Alignment (Channel 8)
- In `local_grid`, Channel 8 has a single `1.0` at $(7, 7)$ (the exact center).
- In the crop from `memory`, the crop is taken with radius 7 around $(pad\_top + agent\_r, pad\_left + agent\_c)$. The agent marker in `memory[8]` falls precisely at $(7, 7)$ with **100.00% accuracy**.

---

## 4. Conclusion & Architecture Recommendation

The observation audit conclusively demonstrates that:
1. The separate local observation `obs["grid"]` is **informationally redundant**: all of its contents are already present in `obs["memory"]`.
2. An agent-centered crop derived directly from `obs["memory"]` preserves the agent-centric coordinate frame of the local CNN while actually **retaining occluded geometry** that the raw raycaster blacks out.
3. Therefore, testing **Variant B (Memory Only)** and **Variant C (Memory + Local Crop derived from Memory)** represents a sound, principled ablation of architectural redundancy.
