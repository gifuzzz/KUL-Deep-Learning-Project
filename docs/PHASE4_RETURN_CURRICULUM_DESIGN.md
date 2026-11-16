# Phase 4: Route-Complexity Return Curriculum Design

**Document ID:** `PHASE4_RETURN_CURRICULUM_DESIGN.md`  
**Target Codebase:** `fire-rescue-project-2.0`  
**Authors:** Gemini Orchestrator & Recurrent RL Team  
**Date:** November 2026  

---

## 1. Executive Summary & Design Rationale

In Phase 1 diagnostic benchmarking ([`docs/PHASE4_DISTANT_FAILURE_ANALYSIS.md`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/docs/PHASE4_DISTANT_FAILURE_ANALYSIS.md)), evaluating the multi-task 250k production checkpoint on 200 matched distant scenarios revealed an unambiguous root cause:
- **0.0% of failures were temporal infeasibilities:** In 100% of pickup episodes, the agent had substantial remaining time ($T_{\text{rem}} = 318.9$ steps on average vs $D_{\text{optimal}} = 22.6$ steps, with $+296.4$ steps of slack).
- **0.0% of failures were route-memory limitations:** In 100% of pickup episodes, the remembered map contained a continuous explored path between the victim and the exit portal.
- **100.0% of failures were navigation policy failures:** When carrying the victim, the policy entered severe 2-step back-and-forth oscillation loops ($A \leftrightarrow B$). In deterministic mode, **91.4% of carrying steps were oscillations**, pacing endlessly until the 384-step timeout terminated the episode.

### The Mechanism of Generalization Failure in Phase 3 Return Training
In Phase 3 multi-task training, the auxiliary return task initialized the agent carrying the victim, but populated memory along only the **single shortest BFS corridor** from the exit to the victim. The remembered map contained no branching corridors, no alternative doorways, and no explored side rooms. Consequently:
1. The policy learned a trivial heuristic: "move into whichever single open corridor is visible in memory."
2. In real full-rescue episodes, however, the agent explores multiple rooms, hallway intersections, and dead ends before finding the victim.
3. At pickup time in real episodes, the remembered map contains **multiple explored branching paths**.
4. Faced with branching corridors and narrow doorways, the Phase 3 policy suffered from symmetric forward/reverse logits, falling into perpetual $A \leftrightarrow B$ oscillation loops at doorways.

To resolve this, the **Route-Complexity Return Curriculum** systematically trains the policy on realistic, branching, multi-room return trajectories with physically consistent explored side branches.

---

## 2. Route-Complexity Levels (R1–R4)

Instead of varying victim distance alone, return scenarios are structured into four progressive difficulty levels based on topological complexity, doorway negotiations, and branching decisions:

| Level | Level Name | Distance Tier | Victim Distance ($d$) | Inbound Turns | Doorways | Explored Side Branches | Branch Depth | Remaining Budget ($T_{\text{rem}}$) |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **R1** | Short, Simple Returns | Near | $d \in [2, 4]$ | 1–2 | 1 (Exit) | 0 branches | 0 | 64 |
| **R2** | Longer Corridor Returns | Medium | $d \in [6, 8]$ | 2–3 | 1–2 | 1 branch | 4 | 128 |
| **R3** | Branching Multi-Room | Bridge B | $d \in [10, 12]$ | 3–5 | 2–3 | 2 branches | 6 | 192 |
| **R4** | Realistic Distant Pickups | Distant | $d \ge 12$ / None | 5+ | 3+ | 3 branches | 8 | 256–384 |

### Detailed Specification of Levels

#### Level R1 — Short, Simple Returns
- **Objective:** Master basic carrier movement and direct exit portal traversal without hesitation.
- **Topological Structure:** Direct, unobstructed corridor connecting the victim to the exit portal.
- **Memory Map:** Single clean corridor without branches.
- **Target Skill:** Immediate orientation toward the exit and 1-step exterior grass transition.

#### Level R2 — Longer Corridor Returns
- **Objective:** Navigate multi-turn hallways and cross the first interior doorway while carrying.
- **Topological Structure:** 2–3 turns, 1 interior doorway separating two rooms/corridors.
- **Explored Branches:** 1 side branch (depth 4) simulating a dead-end alcove checked during search.
- **Target Skill:** Maintain heading through hallway bends and ignore the short dead-end alcove.

#### Level R3 — Branching Multi-Room Returns
- **Objective:** Negotiate multiple doorways and select the correct hallway at true 3-way/4-way intersections.
- **Topological Structure:** Multi-room layout with 2–3 interior doorways and hallway junctions.
- **Explored Branches:** 2 side branches (depth 6) into adjacent rooms that were explored and abandoned.
- **Target Skill:** Disambiguate between the true exit branch and dead-end side rooms in spatial memory.

#### Level R4 — Realistic Distant Pickup States
- **Objective:** Robust, oscillation-free long-distance evacuation across 3+ rooms.
- **Topological Structure:** Deep multi-room building layout with multiple room transitions.
- **Explored Branches:** 3 deep side branches (depth 8) providing rich, realistic exploration history.
- **Target Skill:** Long-horizon path execution across multiple doorways without oscillation traps.

---

## 3. Physical Consistency & Observation Integrity

A strict requirement of the Gemini Orchestrator Directive is that the return generator must **never leak future observations or unvisited geometry**.

### Physical Consistency Guarantees:
1. **Traversable Connectivity:**
   Every cell $(r, c)$ in `branch_cells` is generated via breadth-first search from a candidate fork cell along `main_path`. Every edge traversed in BFS requires simulator conductance $\ge 0.5$ and `debris == 0`. Thus, every explored cell is physically reachable from the entrance.
2. **Strictly Ground-Truth Local Observations:**
   Channels 0–3 (passability North, East, South, West), Channel 4 (debris), and Channel 5 (explored mask) are populated strictly using the simulator's true wall geometry (`sim.south`, `sim.east`, `sim.debris`).
3. **No Unrevealed Rooms Leaked:**
   Rooms and corridors that were neither on the `main_path` nor in the sampled `branch_cells` (nor their immediate line-of-sight neighbors) remain strictly `0.0`. Total explored cells in R3/R4 typically comprise 15%–35% of the building, keeping 65%–85% of the building completely unknown.
4. **Simulator Rule Adherence:**
   The victim is attached as carried (`victim['carried_by'] = 0`), the agent is positioned at `(victim['r'], victim['c'])`, and people states are synchronized with the C++ simulator core (`sim._core.set_people(...)`).

---

## 4. Multi-Task Curriculum Integration

In accordance with Phase 4 Directive Section 6, the task distribution is maintained at:
- **Full Rescue:** 40%
- **Return Training:** 35%
- **Discovery Training:** 25%

Within the 35% Return Training allocation:
- **Variant A (Control):** Uses the Phase 3 legacy return generator (single direct corridor, sampled across distance tiers).
- **Variant B (Route-Complexity):** Uses the new route-complexity generator, sampling across R1–R4:
  - $P(\text{R1}) = 15\%$
  - $P(\text{R2}) = 25\%$
  - $P(\text{R3}) = 30\%$
  - $P(\text{R4}) = 30\%$

This distribution ensures that the policy maintains low-distance grounding (40% R1/R2) while dedicating 60% of return training to the critical multi-room branching scenarios (R3/R4) where oscillation traps occur.

---

## 5. Controlled Pilot Protocol (Phase 5)

### Experimental Setup
- **Base Checkpoint:** `models/multitask_curriculum/production_250k_s42/final_model.zip`
- **Architecture:** Unchanged 1.43M-parameter CNN + 128-unit GRU (`baseline`)
- **Action Space:** 7 primitive actions, strictly masked
- **Pilot Budget:** 50,000 environment steps per variant
- **Hyperparameters:**
  - Learning rate: $3.0 \times 10^{-4}$
  - Batch size: 64
  - Rollout steps: 128 (8 parallel environments)
  - PPO epochs: 4
  - Target KL: 0.02
  - Entropy coefficient: 0.01

### Controlled Comparison Variants:
1. **Variant A (Control):** `--return-mode legacy` (unchanged Phase 3 return distribution)
2. **Variant B (Route-Complexity):** `--return-mode route_complexity` (R1–R4 with branching memory)

### Benchmark Protocol:
Both variants are evaluated on the identical 200-episode matched 6-tier benchmark (`seeds 50000..50199`) under fully unknown geometry and unshaped native rescue (+10 only).
Key metrics tracked:
1. Complete distant rescue success rate (%)
2. Distant return success conditional on pickup (%)
3. Feasible return success rate (%)
4. Carrying oscillation frequency (%)
5. Retention of Near ($d=2$) and Medium ($d=6$) performance.
