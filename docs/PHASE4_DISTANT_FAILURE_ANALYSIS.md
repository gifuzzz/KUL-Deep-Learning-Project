# Phase 4: Long-Distance Return Failure Analysis

**Benchmark Target:** `models/multitask_curriculum/production_250k_s42/final_model.zip`  
**Evaluation Set:** 200 matched distant scenarios (seeds `50000..50199`)  
**Evaluation Protocol:** Fully unknown geometry (`geometry_reveal_fraction=0.0`), hidden victim (`reveal_victim=False`), native unshaped rescue (+10 only), $T_{\max} = 384$ steps.  
**Evaluator Script:** [`scripts/diagnose_distant_failures.py`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/scripts/diagnose_distant_failures.py)  
**Machine-Readable Data:** [`models/distant_failure_analysis/distant_diagnosis_stoch.json`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/models/distant_failure_analysis/distant_diagnosis_stoch.json) and [`models/distant_failure_analysis/distant_diagnosis_det.json`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/models/distant_failure_analysis/distant_diagnosis_det.json)

---

## 1. Executive Summary & Core Diagnostic Finding

The Multi-Task Curriculum (Phase 3) demonstrated that training an agent on discovery, return, and full rescue tasks doubled distant discovery to **15.5%–17.5%** and tripled distant pickup to **14.0%–14.5%**. However, complete distant rescues remained bottlenecked at **1.5%–3.5%**.

To identify the root cause of failures after victim pickup, a comprehensive trajectory and graph diagnostic was conducted across all 200 matched distant evaluation scenarios in both Stochastic and Deterministic modes.

### Key Empirical Findings:
1. **Temporal Infeasibility is NOT the bottleneck (0.0% of failures):**
   - In **100.0%** of pickup episodes (28/28 stochastic, 17/17 deterministic), the agent had more than sufficient remaining time to return.
   - Mean steps remaining at pickup ($T_{\text{rem}}$): **318.9 steps** (stochastic) and **330.5 steps** (deterministic) out of 384 total steps.
   - Mean optimal BFS return distance ($D_{\text{optimal}}$): **22.6 steps** (stochastic) and **21.2 steps** (deterministic).
   - Mean temporal slack margin ($T_{\text{rem}} - D_{\text{optimal}}$): **+296.4 steps** (stochastic) and **+309.2 steps** (deterministic).
   - Every single pickup occurred with at least +62 to +368 steps of spare time.

2. **Route-Memory Limitations are NOT the bottleneck (0.0% of failures):**
   - In **100.0%** of pickup episodes, the remembered map contained a contiguous, fully connected, explored path between the victim pickup cell and the exit portal.
   - Because the agent entered through the exit opening and explored inwards to find the victim, its physical line-of-sight and visited cells guaranteed a complete return corridor in memory.

3. **Interaction Failures are NOT the bottleneck (0.0% of failures):**
   - The agent never reached an exit portal cell and failed to step outside. Every carrier that reached the exit portal cell successfully evacuated.

4. **100.0% of Failures are NAVIGATION POLICY FAILURES:**
   - In deterministic mode, **91.4% of carrying steps are 2-step back-and-forth oscillations** ($A \leftrightarrow B$). The agent becomes trapped pacing between two adjacent cells for up to 376 consecutive steps until timeout.
   - In stochastic mode, sampling stochasticity helps break the 2-step deadlock in some cases, raising the return success rate conditional on pickup to **25.0%** (7 rescues out of 28 pickups), but **41.7% of carrying steps remain oscillations**, and 21 out of 28 carriers pace until timing out.

---

## 2. Quantitative Diagnostic Results

### 200-Episode Matched Distant Performance Summary

| Metric | Stochastic Mode | Deterministic Mode |
|---|---:|---:|
| Total Evaluated Episodes | 200 | 200 |
| **Distant Discovery Count (Rate %)** | **31 (15.5%)** | **24 (12.0%)** |
| **Distant Pickup Count (Rate %)** | **28 (14.0%)** | **17 (8.5%)** |
| **Complete Distant Rescues (Rate %)** | **7 (3.5%)** | **1 (0.5%)** |
| **Return Success Conditional on Pickup** | **25.0% (7/28)** | **5.9% (1/17)** |
| Time-Feasible Pickups ($T_{\text{rem}} \ge D_{\text{optimal}}$) | 28 (100.0%) | 17 (100.0%) |
| Time-Infeasible Pickups ($T_{\text{rem}} < D_{\text{optimal}}$) | 0 (0.0%) | 0 (0.0%) |
| **Feasible Return Success Rate** | **25.0% (7/28)** | **5.9% (1/17)** |
| Mean Discovery Step ($t_{\text{disc}}$) | 59.8 | 39.4 |
| Mean Pickup Step ($t_{\text{pick}}$) | 65.1 | 53.5 |
| Mean Steps Remaining at Pickup ($T_{\text{rem}}$) | 318.9 | 330.5 |
| Mean Optimal Return Distance ($D_{\text{optimal}}$) | 22.6 | 21.2 |
| Mean Temporal Slack ($T_{\text{rem}} - D_{\text{optimal}}$) | +296.4 | +309.2 |
| Carrying Oscillation Rate | **41.7%** | **91.4%** |
| Mean Revisits per Carrier Episode | 221.4 | 309.6 |

---

## 3. Mutually Exclusive Failure Classification

Using the strict classification precedence defined in the Phase 4 Directive:
1. **Time-infeasible:** $T_{\text{rem}} < D_{\text{optimal}}$ (mathematically impossible to return in time).
2. **Interaction failure:** Reached exit portal cell while carrying but failed to step outside.
3. **Route-memory limitation:** Memory map lacks a connected explored path to the exit.
4. **Navigation failure:** Connected route exists in memory and $T_{\text{rem}} \ge D_{\text{optimal}}$, but policy wandered or timed out.
5. **Environment termination:** Agent or victim died.

### Failure Breakdown Across Failed Pickups

| Failure Category | Stochastic Mode ($N=21$) | Deterministic Mode ($N=16$) | Precedence Level |
|---|---:|---:|:---:|
| **Time-infeasible** | 0 (0.0%) | 0 (0.0%) | 1 |
| **Interaction failure** | 0 (0.0%) | 0 (0.0%) | 2 |
| **Route-memory limitation** | 0 (0.0%) | 0 (0.0%) | 3 |
| **Navigation failure** | **21 (100.0%)** | **16 (100.0%)** | **4** |
| **Environment termination** | 0 (0.0%) | 0 (0.0%) | 5 |

---

## 4. Representative Trajectory Case Studies

### Case 1: Extreme Deterministic Oscillation Trap (Seed 50068)
- **Outcome:** Timeout Failure while carrying (384 steps).
- **Initial Discovery & Pickup:** Agent navigated swiftly from entrance, discovered victim, and picked up at **step 9**.
- **Remaining Time:** $T_{\text{rem}} = 375$ steps.
- **Optimal Exit Distance:** $D_{\text{optimal}} = 7$ steps (exit was 7 steps away!).
- **Slack:** $+368$ steps.
- **Carrying Behavior:**
  - Carrying steps: 376 steps.
  - 2-step oscillations: **373 oscillations** (99.2% of all carrying steps).
  - The policy alternated between cell $(14, 4)$ and cell $(14, 5)$ for 373 consecutive steps until the 384-step timeout terminated the episode.

### Case 2: Multi-Room Carrier Oscillation (Seed 50027)
- **Outcome:** Timeout Failure while carrying (384 steps).
- **Initial Discovery & Pickup:** Picked up victim at **step 21** in Room 2.
- **Remaining Time:** $T_{\text{rem}} = 363$ steps.
- **Optimal Exit Distance:** $D_{\text{optimal}} = 17$ steps.
- **Slack:** $+346$ steps.
- **Carrying Behavior:**
  - Carrying steps: 364 steps.
  - Re 一visited cells: 359 times.
  - Oscillations: 359 steps (98.6%).
  - At the doorway between Room 2 and the main corridor, the policy repeatedly stepped forward into the doorway, then immediately stepped backward into Room 2, failing to traverse the doorway.

### Case 3: Successful Deterministic Return Navigation (Seed 50187)
- **Outcome:** **Complete Rescue at Step 72!**
- **Initial Discovery & Pickup:** Discovered at step 19, picked up at **step 57**.
- **Remaining Time:** $T_{\text{rem}} = 327$ steps.
- **Optimal Exit Distance:** $D_{\text{optimal}} = 13$ steps.
- **Slack:** $+314$ steps.
- **Carrying Behavior:**
  - Carrying steps: **15 steps** (optimal was 13; path efficiency = 86.7%).
  - 2-step oscillations: **0 oscillations (0.0%)**.
  - Cell revisits: **0 revisits**.
  - The policy followed a direct, monotonic path through the corridor and stepped through the exit portal at step 72.

### Case 4: Stochastic Breakout & Rescue (Seed 50118)
- **Outcome:** **Complete Rescue at Step 189!**
- **Initial Discovery & Pickup:** Deep search; victim picked up at **step 170**.
- **Remaining Time:** $T_{\text{rem}} = 214$ steps.
- **Optimal Exit Distance:** $D_{\text{optimal}} = 16$ steps.
- **Carrying Behavior:**
  - Carrying steps: **19 steps** (near-optimal).
  - Oscillations: 2 oscillations.
  - The agent moved through two intermediate rooms directly to the exit portal.

---

## 5. Root Cause: Why Did Phase 3 Return Training Fail to Generalize?

Inspection of `RescueEnv._setup_return_task()` in Phase 3 reveals the structural reason for this navigation failure:

```python
# In Phase 3 _setup_return_task:
start_cell = self.sim._cells_for_opening(exit_opening, exit_opening["zone_a"])
path = self._find_path_to_victim(start_cell, (vr, vc))
...
self._populate_path_memory(path, exit_opening, (vr, vc))
```

1. **Artificial Single-Corridor Simplicity:**
   - In Phase 3 return training, the environment initialized the agent at the victim location and populated the memory map along the **single shortest BFS path** between the exit and the victim.
   - The memory map contained only the corridor cells along that single path. There were **no branching corridors, no dead ends, no adjacent explored rooms, and no alternative doorways**.
   - As a result, the return policy learned a trivial feature correlation: "follow the only non-zero corridor in memory."

2. **Failure in Realistic Multi-Room Scenarios:**
   - In an actual full-rescue episode, before finding the victim, the agent explores multiple rooms, hallway intersections, and dead-ends.
   - Consequently, when the agent picks up the victim, the memory map contains **rich, branching geometry with multiple open doorways and intersections**.
   - When faced with multiple explored corridors in its remembered map, the policy's attention mechanism fails to orient toward the exit portal, causing symmetric logits between forward and reverse actions that degenerate into the $A \leftrightarrow B$ oscillation trap.

3. **Doorway Negotiation Bottlenecks:**
   - Doorways represent narrow 1-cell bottlenecks between rooms. In actual exploration, moving through a doorway often changes the local observation abruptly.
   - Without explicit training on returning through multiple doorways and intersections, the agent hesitates at room boundaries and reverses its direction.

---

## 6. Actionable Blueprint for Phase 2 & Phase 3 (Route-Complexity Curriculum)

Because **100% of failures are navigation failures caused by doorway/branching oscillation**, the solution must directly address return navigation through realistic, multi-room, branching geometry:

1. **Implement Route-Complexity Levels (R1–R4):**
   - **R1 (Short Simple):** Distance $d \in [2, 4]$, direct corridor to exit.
   - **R2 (Longer Corridor):** Distance $d \in [6, 8]$, 2–3 turns, multiple doorways.
   - **R3 (Branching Multi-Room):** Distance $d \in [10, 12]$, multiple intersections, visited dead-end side rooms in memory, agent must choose correct branch toward exit.
   - **R4 (Realistic Distant Pickups):** Distance $d \in [12, 16, \text{None}]$, complex paths through 3+ rooms, realistic explored memory containing multiple explored branches, realistic remaining time budget ($T_{\text{rem}} \in [64, 256]$).

2. **Branching Memory Generation:**
   - Instead of revealing only the single shortest path, generate realistic memory exploration prefixes: include visited side rooms and alternate branches that were "explored" during search, forcing the agent to identify and follow the true exit route among multiple explored options.

3. **Curriculum Progression:**
   - Progressively mix R1 $\rightarrow$ R2 $\rightarrow$ R3 $\rightarrow$ R4 within the 35% return-training task allocation, ensuring the agent learns to reject false turns and navigate doorways while carrying.
