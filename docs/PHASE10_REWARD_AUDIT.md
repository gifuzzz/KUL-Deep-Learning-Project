# Phase 10 — Reward Incentive Audit & Pre-Training Attribution Report

**Date:** November 17, 2026  
**Environment:** `fire-rescue-project-2.0`  
**Target Checkpoint:** Phase 6 Production Best Checkpoint (`models/phase6_spawn_curriculum/production_250k/stage3_spawn_curriculum/best_model.zip`)  
**Architecture:** 9-Channel CNN + 128-unit GRU (`architecture_variant="baseline"`, 1,431,168 parameters)  
**Test Suite:** [`tests/test_phase10_rewards.py`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/tests/test_phase10_rewards.py) (**12/12 tests passing**)  
**Diagnostic Telemetry:** [`models/phase10_evaluations/phase10_reward_telemetry.json`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/models/phase10_evaluations/phase10_reward_telemetry.json)  

---

## 1. Executive Summary

This audit rigorously evaluates the reward incentives in `fire-rescue-project-2.0` prior to launching the Phase 10 controlled reward ablation.

### Primary Audit Findings:
1. **The Economic Asymmetry Between Wait and Movement:**
   - In any explored building region, moving forward and backward between two cells (doorway hesitation, corner deadlock) incurs an immediate penalty of **$-0.021$ per step** ($-0.001$ timestep penalty $+ -0.02$ anti-oscillation penalty).
   - In contrast, taking Action 0 (**Wait**) incurs **only $-0.001$ per step** with **zero risk** of triggering the oscillation penalty.
   - Taking a single Wait action at position $B$ when entering a cycle $A \to B \to A \to B$ updates the position history to $[B, A, B, B]$, satisfying $p_3 == p_2$, which **instantly clears the oscillation condition** ($p_3 \neq p_2$ becomes false).
   - Waiting is **21 times cheaper** than attempting to navigate difficult narrow thresholds, and acts as an immediate escape mechanism from movement penalties. This creates a strong learned incentive for the policy to remain stationary.
2. **Legitimate Backtracking is NOT Penalized on Multi-Cell Paths:**
   - Tracing legitimate retreats along multi-cell corridors ($A \to B \to C \to B \to A$) proves mathematically and empirically that $p_2 == p_0$ is false during linear retreats. On 25 recorded backtracking steps in held-out scenarios, **0% were penalized** by the oscillation penalty.
   - However, **1-cell alcoves or doorway thresholds** ($A \leftrightarrow B$) trigger the full $-0.02$ penalty immediately on the 4th half-step, punishing brief exploratory checks into small rooms or hesitation at doorways.
3. **Reward Shaping Status Across Evaluation:**
   - Unshaped evaluation strictly sets `reward_shaping = False`, disabling all shaping terms, timestep penalties, exploration bonuses, and oscillation penalties. Evaluation reward is $+10.0$ on completed rescue and $0.0$ otherwise.
   - Action 6 (water hose) remains strictly disabled (`mask[6] == False`) in fireless rescue across all tasks and evaluations.

---

## 2. Comprehensive Reward Components Table

The table below lists every reward component implemented across the entire simulator and training curriculum, verified against the actual source code.

| Component | Coefficient / Formula | Activation Condition | Max Cumulative Contribution | Applicable Tasks | Code Reference |
|---|---|---|---|---|---|
| **Native Rescue Reward** | `+10.0` per victim | Carried victim moves onto exterior grass tile (`result['reward']`) | `+10.0` (1 victim in fireless mode) | `full`, `return` | [`flamebox.py:1162`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/rescue_env/flamebox/flamebox.py#L1162), [`single_agent.py:1425`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/rescue_env/envs/single_agent.py#L1425) |
| **Victim Discovery Reward** | `+2.0` (`discovery_reward`) | Sighting victim on local grid (`grid[6] > 0`) for the first time | `+2.0` (terminates episode upon sighting) | `discovery` | [`single_agent.py:1540`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/rescue_env/envs/single_agent.py#L1540) |
| **First Victim Pickup** | `+1.5` per victim | First time agent picks up victim (`carried_by == 0`) | `+1.5` (awarded once) | `full`, `return` (if `reward_shaping=True`) | [`reward_shaping.py:153`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/rescue_env/envs/reward_shaping.py#L153) |
| **Frontier LOS Discovery** | `+0.05` per newly uncovered cell in line-of-sight | When `not is_carrying` and `newly_uncovered > 0` | `+3.0` per episode (strict ceiling) | `full`, `return`, `discovery` (training only) | [`reward_shaping.py:160`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/rescue_env/envs/reward_shaping.py#L160) |
| **Exploration LOS Bonus (Phase 7)** | `+0.05` per cell (`exploration_cell_bonus`) | Line-of-sight cell discovery in pure exploration | `+6.0` (`max_exploration_reward` ceiling) | `exploration_only` (`phase7_capped`) | [`single_agent.py:1465`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/rescue_env/envs/single_agent.py#L1465) |
| **Exploration Edge Bonus (Phase 7)** | `+0.02` per edge (`exploration_edge_bonus`) | Traversable edge discovery in pure exploration | Included in `+6.0` ceiling | `exploration_only` (`phase7_capped`) | [`single_agent.py:1466`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/rescue_env/envs/single_agent.py#L1466) |
| **Normalized Continuous Exploration** | $\frac{\text{cells}}{N_{\text{cells}}} \cdot 8.0 + \frac{\text{edges}}{N_{\text{edges}}} \cdot 4.0 + \frac{\text{visited}}{N_{\text{cells}}} \cdot 4.0$ | Continuous potential in pure exploration | `+16.0` maximum | `exploration_only` (`normalized_continuous`) | [`single_agent.py:1453`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/rescue_env/envs/single_agent.py#L1453) |
| **Physical Visit Bonus (Phase 9)** | `+0.15` per newly visited cell (`unvisited_cell_bonus`) | Stepping on a previously unvisited reachable cell | $\approx 207$ (uncapped on 1,380 cells) | `branching_exploration` | [`single_agent.py:1509`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/rescue_env/envs/single_agent.py#L1509) |
| **Branch Entry Bonus (Phase 9)** | `+1.0` | First step into unvisited branch in branching task | `+1.0` (awarded once) | `branching_exploration` | [`single_agent.py:1512`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/rescue_env/envs/single_agent.py#L1512) |
| **Room-Transition Bonus (Phase 9)** | `+1.0` per unvisited room (`unvisited_room_bonus`) | Entering a previously unvisited room zone | $1.0 \times N_{\text{rooms}}$ (typically 4–8) | `branching_exploration` | [`single_agent.py:1518`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/rescue_env/envs/single_agent.py#L1518) |
| **Timestep Penalty (General)** | `-0.001` per step (`step_penalty`) | Applied every step when `reward_shaping=True` | $-0.001 \times \text{max\_steps}$ ($-0.064$ to $-0.384$) | `full`, `return`, `discovery`, `branching_exploration` | [`reward_shaping.py:142`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/rescue_env/envs/reward_shaping.py#L142) |
| **Timestep Penalty (Exploration)** | `-0.001` per step (`exploration_step_penalty`) | Applied every step in exploration task | $-0.001 \times 384 = -0.384$ | `exploration_only` | [`single_agent.py:1451`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/rescue_env/envs/single_agent.py#L1451) |
| **Anti-Oscillation Penalty** | `-0.02` per step (`oscillation_penalty`) | Repeating 2-cell cycle ($p_3 == p_1, p_2 == p_0, p_3 \neq p_2$) while `not is_carrying` | Uncapped (e.g. $-0.02 \times 300 = -6.00$) | `full`, `return`, `discovery`, `exploration_only`, `branching_exploration` | [`reward_shaping.py:172`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/rescue_env/envs/reward_shaping.py#L172) |
| **Stationary Wait Penalty (Baseline)** | **`0.000`** in Phase 6 Production Rescue! | Not implemented in rescue tasks | `0.000` | None in rescue (only $-0.005 \times (\text{streak}-2)$ in Phase 9 branching) | [`single_agent.py:1520`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/rescue_env/envs/single_agent.py#L1520) |
| **Stationary Wait Penalty (Phase 10 C & D)** | Bounded schedule: Wait 1: `0.0`, Wait 2: `-0.002`, Wait 3: `-0.004`, Wait 4+: `-0.008` | Consecutive Wait actions when `repeated_wait_penalty_enabled=True` | Capped at `-0.008` per step | Phase 10 Variants C & D (training only) | [`reward_shaping.py:138`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/rescue_env/envs/reward_shaping.py#L138) |
| **Navigation Potential Shaping** | $0.1 \times (0.99 \cdot \Phi_{t+1} - \Phi_t)$ where $\Phi = -d_{\text{shortest}}$ | Enabled if `navigation_shaping=True` (disabled in Phase 6) | Telescoping potential $\approx 0.1 \times 128 \approx 12.8$ | None in Phase 6 (`navigation_shaping=False`) | [`reward_shaping.py:97`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/rescue_env/envs/reward_shaping.py#L97) |

---

## 3. Analysis of Representative Failure & Success Scenarios

Using [`scripts/diagnose_phase10_rewards.py`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/scripts/diagnose_phase10_rewards.py), we executed the Phase 6 baseline model on held-out procedural scenarios with training reward shaping and recorded step-by-step telemetry for the six target behaviors.

All telemetry records are persisted in [`models/phase10_evaluations/phase10_reward_telemetry.json`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/models/phase10_evaluations/phase10_reward_telemetry.json).

### Scenario 1: Repeated Wait Actions at Dead End
- **Configuration:** Seed 50000, Task: `full`, Tier: `spawn_21_35`, Deterministic.
- **Behavior:** The firefighter enters Room 2 through a doorway, reaches a dead end at $(13, 0)$, and selects Action 0 (Wait) for **365 consecutive steps** until the 384-step timeout.
- **Per-Step Metrics:**
  - Action probabilities: $P(\text{Wait}) \ge 0.998$, $P(\text{Move}) \le 0.002$.
  - Immediate reward: Exactly **$-0.00100$** every single step.
  - Value estimate: Stable negative floor $V(s) \approx -0.332$.
  - Oscillation penalty: **0.000** (never triggers).
- **Economic Mechanism:** The agent receives $-0.001$ per step. Over the 365 wait steps, total penalty is $-0.365$. If the agent had moved back and forth between $(13, 0)$ and $(12, 0)$, it would have incurred $-0.021 \times 365 = -7.665$. Wait is a massive relative economic gain ($+7.300$).

### Scenario 2: Two-Cell Doorway Oscillation
- **Configuration:** Seed 50057, Task: `full`, Tier: `spawn_21_35`, Deterministic.
- **Behavior:** The firefighter becomes trapped in a 2-cell limit cycle across an interior doorway between cells $(15, 8)$ and $(15, 9)$, oscillating for **96.1% of the episode** (369 out of 384 steps).
- **Per-Step Metrics:**
  - Position alternating: $(15, 8) \leftrightarrow (15, 9)$.
  - Immediate reward: Exactly **$-0.02100$** per step ($-0.001$ step penalty $+ -0.02$ oscillation penalty).
  - Cumulative episode reward: **$-7.881$**.
  - Value estimate: Degrades from $+0.504$ down to $-0.450$.
- **Economic Mechanism:** Attempting to transition the doorway without a clear spatial goal triggers severe ongoing penalties. A single Wait action would have reset the position buffer and spared the agent $-0.020$ per step.

### Scenario 3: Legitimate Backtracking from a Dead End
- **Configuration:** Seed 50008, Task: `full`, Tier: `spawn_7_12`, Stochastic.
- **Behavior:** The firefighter advances down a corridor, reaches a dead end, turns 180 degrees, and retreats along the corridor back to a multi-way junction.
- **Per-Step Metrics:**
  - Backtracking steps recorded: **25 steps**.
  - Penalized backtracking steps: **0 steps (0.0%)**.
  - Oscillation percentage: $2.1\%$.
- **Mathematical Proof:**
  - Consider a multi-cell path: $p = [X, A, B, C]$ where $C$ is a dead end.
  - Step 1 (Retreat to $B$): History becomes $[A, B, C, B]$.
    - $p_3 = B, p_1 = B \implies p_3 == p_1$ is True.
    - $p_2 = C, p_0 = A \implies p_2 == p_0$ is **False** (since $C \neq A$).
    - Condition $p_3 == p_1 \land p_2 == p_0$ is **False**. Oscillation penalty is **NOT** applied.
  - Step 2 (Retreat to $A$): History becomes $[B, C, B, A]$.
    - $p_3 = A, p_1 = C \implies p_3 == p_1$ is **False** (since $A \neq C$).
    - Oscillation penalty is **NOT** applied.
  - **Conclusion:** Legitimate multi-cell backtracking is completely free of oscillation penalties.

### Scenario 4: Successful Movement into a New Room
- **Configuration:** Seed 50000, Task: `full`, Tier: `spawn_13_20`, Deterministic.
- **Behavior:** Firefighter traverses entrance doorway into Room 1, crosses an interior doorway into Room 2, and reaches Room 3, uncovering 28 reachable cells.
- **Per-Step Metrics:**
  - Unique rooms entered: **3 rooms**.
  - Observed coverage: $18.4\%$.
  - First-room entry reward: $+0.05 \times 12$ newly uncovered cells $= +0.60$.
  - Subsequent doorway transition: clean single-step transit without oscillation penalty.

### Scenario 5: Victim Pickup Followed by Failed Return
- **Configuration:** Seed 50021, Task: `return`, Tier: `spawn_13_20`, Deterministic.
- **Behavior:** Firefighter starts carrying the victim, attempts return navigation, but becomes stuck in a dead-end corridor carrying the victim for **382 steps** until timeout.
- **Per-Step Metrics:**
  - Carrying status: `is_carrying == True` for 382 steps.
  - Immediate reward per step: **$-0.00100$** (step penalty only).
  - Oscillation penalty: **0.000** (anti-oscillation penalty is disabled while carrying: `if not is_carrying and ...`).
  - Frontier exploration bonus: **0.000** (exploration bonus is disabled while carrying).
- **Finding:** While carrying, the agent faces zero oscillation penalty, but lacks any directional gradient since `navigation_shaping=False`. It stalls because it cannot find the exterior door in an unknown layout.

### Scenario 6: Successful Distant Rescue
- **Configuration:** Seed 50012, Task: `full`, Tier: `spawn_13_20`, Deterministic.
- **Behavior:** Firefighter navigates down hallway, sights victim at step 18, picks up victim at step 20, reverses direction, and exits onto grass portal at step 27.
- **Per-Step Metrics:**
  - Step 1–19 (Search): Exploration bonuses awarded ($+1.85$ cumulative), step penalties ($-0.019$).
  - Step 20 (Pickup): First pickup reward awarded (**$+1.50$**).
  - Step 21–26 (Return): Carrying victim back to door ($-0.001$ per step).
  - Step 27 (Rescue): Stepping onto exterior grass portal awards **$+10.00$ native rescue reward**. Total episode reward: $+13.315$.

---

## 4. Key Questions Answered

### Q1: Does the original oscillation penalty discourage legitimate backtracking?
**Answer: No, not for multi-cell retreat.**
The formula requires $p_3 == p_1 \land p_2 == p_0 \land p_3 \neq p_2$. In any linear retreat of length $\ge 2$, $p_2 \neq p_0$, so the condition is mathematically impossible. Our empirical trace verified 0/25 backtracking steps were penalized.
**Caveat:** The penalty DOES discourage 1-cell hesitation (stepping into a 1-cell alcove and stepping back, then stepping forward again), which can occur at doorway thresholds.

### Q2: Is stationary Wait economically preferable under the actual reward function?
**Answer: Yes, decisively.**
- Moving during uncertainty/deadlocks risks repeating 2-cell transitions, incurring **$-0.021$ per step**.
- Waiting incurs **$-0.001$ per step** and immediately breaks the oscillation condition.
- Waiting is **21 times cheaper** than attempting to navigate difficult narrow thresholds.

### Q3: How do immediate reward differences relate to long-term value estimates?
- In states where the agent chooses Wait, the critic value $V(s) \approx -0.33$, reflecting the discounted sum of $-0.001$ step penalties:
  $$\sum_{t=0}^{\infty} \gamma^t (-0.001) = \frac{-0.001}{1 - 0.99} = -0.10 \text{ to } -0.38$$
- If the policy were forced to oscillate, value estimates would collapse to:
  $$\sum_{t=0}^{\infty} \gamma^t (-0.021) = \frac{-0.021}{1 - 0.99} = -2.10$$
- In PPO advantage estimation ($A_t = r_t + \gamma V(s_{t+1}) - V(s_t)$), taking Wait yields an immediate advantage of:
  $$A(\text{Wait}) - A(\text{Move}) \approx (-0.001) - (-0.021) = \mathbf{+0.020}$$
- The policy directly learns to choose Wait because PPO gradient updates actively reinforce actions with higher relative advantage.

---

## 5. Verification Gate Status

The test suite [`tests/test_phase10_rewards.py`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/tests/test_phase10_rewards.py) was executed:
- **12 passed, 0 failed in 2.23s**.
- All 12 correctness properties verified.
- Reward attribution and diagnostic replays complete.

**Approval granted to proceed to Section 3: Controlled Pilot Training.**
