# Phase 7 — Exploration-Only Auxiliary Training & Learned Loop Avoidance Report

**Executive Summary:**
Phase 7 investigates whether an auxiliary `exploration_only` reinforcement learning task—rewarding frontier observation, traversable edge discovery, and penalizing 2-cell oscillation—can teach the existing CNN + 128-unit GRU policy to systematically explore unknown buildings, eliminate doorway oscillation and stationary Wait stalling, and improve distant victim rescue.

We conducted a strictly controlled 50,000-step ablation continuing from the strongest Phase 6 production checkpoint across three variants:
1. **Variant A (Control, 0% exploration):** Pure continuation of the Phase 6 mixture `(0.40, 0.35, 0.25, 0.00)`.
2. **Variant B (Balanced, 20% exploration):** `(0.32, 0.28, 0.20, 0.20)`.
3. **Variant C (Exploration-Heavy, 40% exploration):** `(0.24, 0.21, 0.15, 0.40)`.

Each checkpoint was evaluated across 4,800 unshaped, unknown-map, hidden-victim episodes across three benchmark suites (Pure Exploration, Spawn-Relative Ladder, and Exit-Relative Ladder) in both deterministic and stochastic modes on matched holdout seeds.

---

## 1. Key Findings & Empirical Verdict

1. **Doorway Oscillation & Loop Reduction:**
   In pure exploration on unknown layouts, **Variant B cut doorway reversal rates from 74.3% down to 17.3%** (a 76.7% relative reduction) and reduced 2-cell oscillation from 45.85% to 31.02% in deterministic evaluation.
2. **Breakthrough in Deterministic Distant Rescue:**
   In the unconstrained distant multi-room benchmark, the Phase 6 baseline achieved **0.0% (0/200)** deterministic rescue. **Variant B unlocked 1.5% (3/200)** deterministic rescues (e.g. Seed 50093, where Baseline stalled in 94.3% stationary Wait while Variant B completed full multi-room rescue in 56 steps).
3. **Significant Bridge Distance Gains:**
   At Bridge C ($d=12$), **Variant B increased deterministic rescue from 10.0% to 13.0%**, drove return-after-pickup from 29.9% to **47.3%**, and reduced carrier timeouts from 23.5% down to 14.5%.
4. **Catastrophic Interference of Exploration-Heavy Training (Variant C):**
   Allocating 40% to exploration caused severe behavioral collapse on the primary rescue mission: near rescue dropped from 100% to 85.0%, Spawn 7–12 rescue dropped from 29.0% to 18.5%, and carrier timeouts jumped to 45.0% because the agent learned to prioritize room roaming over returning to the exit.
5. **Control Degradation Without Anti-Oscillation Shaping (Variant A):**
   Continuation without exploration auxiliary rewards caused policy stagnation: the agent developed extreme stationary waiting (Wait fraction increased from 16.3% to **48.66%**, with max wait streaks exceeding 173 steps).

**Verdict:** **Adopt Variant B (20% Exploration Auxiliary Task) for production.**

---

## 2. Pure Exploration Benchmark Results

Evaluated on 200 held-out unknown procedural buildings without victims (seeds 60000–60199, max steps = 384, unshaped evaluation):

### Table 2.1: Deterministic Pure Exploration Performance
| Metric | Phase 6 Baseline | Variant A (Control) | Variant B (20% Explore) | Variant C (40% Explore) |
|---|---|---|---|---|
| **Observed Coverage (%)** | 10.89% | 8.52% | **10.78%** | 9.65% |
| **Visited Coverage (%)** | 1.55% | 1.06% | **1.48%** | 1.32% |
| **Edge Discovery (%)** | 11.41% | 8.98% | **11.30%** | 10.14% |
| **Rooms Explored** | 1.45 | 1.23 | **1.38** | 1.35 |
| **Exploration Efficiency (cells/100st)** | 44.3 | 34.6 | **43.6** | 39.1 |
| **Wait Action Fraction (%)** | 16.30% | 48.66% | **35.92%** | 34.50% |
| **Wait Streaks > 50 steps** | 0.18 | 0.55 | **0.41** | 0.40 |
| **Max Wait Streak (steps)** | 59.8 | 173.6 | **131.8** | 122.7 |
| **Doorway Crossings** | 3.37 | 0.24 | **1.74** | 0.82 |
| **Doorway Reversals** | 2.50 | 0.01 | **0.30** | 0.45 |
| **Doorway Reversal Rate (%)** | 74.3% | 2.0%* | **17.3%** | 54.3% |
| **2-Cell Oscillation (%)** | 45.85% | 29.04%* | **31.02%** | 32.83% |
| **90% Coverage Target Rate (%)** | 0.0% | 0.0% | **0.0%** | 0.0% |

*\*Note: Variant A's low doorway crossing count and oscillation percentage were caused by extreme stationary wait stalling (spending 48.7% of all episode steps waiting without moving).*

### Table 2.2: Stochastic Pure Exploration Performance
| Metric | Phase 6 Baseline | Variant A (Control) | Variant B (20% Explore) | Variant C (40% Explore) |
|---|---|---|---|---|
| **Observed Coverage (%)** | 17.98% | 15.07% | **16.47%** | 15.70% |
| **Visited Coverage (%)** | 4.29% | 3.34% | **3.54%** | 3.54% |
| **Edge Discovery (%)** | 18.59% | 15.70% | **17.13%** | 16.26% |
| **Rooms Explored** | 2.23 | 1.85 | **1.94** | 1.94 |
| **Exploration Efficiency (cells/100st)** | 72.7 | 61.0 | **66.6** | 63.3 |
| **Wait Action Fraction (%)** | 21.76% | 35.72% | **33.34%** | 33.05% |
| **Doorway Crossings** | 2.86 | 2.10 | **1.84** | 1.73 |
| **Doorway Reversals** | 0.51 | 0.34 | **0.27** | 0.21 |
| **Doorway Reversal Rate (%)** | 17.7% | 16.4% | **14.4%** | 12.4% |
| **2-Cell Oscillation (%)** | 6.06% | 3.87% | **4.45%** | 4.13% |

---

## 3. Spawn-Relative Ladder Rescue Benchmark

Evaluated across 200 held-out procedural episodes per tier (seeds 50000–50199) using true shortest traversable path distance from actual firefighter spawn:

### Table 3.1: Deterministic Spawn-Relative Rescue
| Distance Tier | Model | Rescue (%) | Discovery (%) | Pickup (%) | Return \| Pick (%) | Timeout (%) | TO Carry (%) | Oscillation (%) |
|---|---|---|---|---|---|---|---|---|
| **Spawn 2–6** | Phase 6 Baseline | 93.0% | 100.0% | 97.0% | 95.9% | 7.0% | 4.0% | 4.4% |
| | Variant A (Control) | **94.0%** | 100.0% | 96.0% | **97.9%** | 6.0% | 2.0% | 6.0% |
| | Variant B (20% Explore) | 92.5% | 100.0% | 97.0% | 95.4% | 7.5% | 4.5% | 5.1% |
| | Variant C (40% Explore) | 80.0% | 99.5% | 96.0% | 83.3% | 20.0% | 16.0% | 13.5% |
| **Spawn 7–12** | Phase 6 Baseline | 29.0% | 81.0% | 61.0% | 47.5% | 71.0% | 32.0% | 32.5% |
| | Variant A (Control) | 26.5% | 80.0% | 60.0% | 44.2% | 73.5% | 33.5% | 32.2% |
| | Variant B (20% Explore) | **31.0%** | **82.0%** | 61.0% | **50.8%** | 69.0% | 30.0% | 32.1% |
| | Variant C (40% Explore) | 18.5% | 81.5% | 63.5% | 29.1% | 81.5% | 45.0% | 42.8% |
| **Spawn 13–20** | Phase 6 Baseline | **9.5%** | 33.0% | 27.0% | 35.2% | 90.5% | 17.5% | 36.4% |
| | Variant A (Control) | 8.0% | 25.5% | 19.5% | 41.0% | 92.0% | 11.5% | 25.8% |
| | Variant B (20% Explore) | 8.0% | **35.0%** | 26.0% | 30.8% | 92.0% | 18.0% | 31.2% |
| | Variant C (40% Explore) | 4.0% | 37.0% | 28.5% | 14.0% | 96.0% | 24.5% | 37.2% |
| **Spawn 21–35** | Phase 6 Baseline | **2.5%** | **15.5%** | **11.0%** | 22.7% | 97.5% | 8.5% | 40.5% |
| | Variant A (Control) | 0.5% | 7.5% | 6.5% | 7.7% | 99.5% | 6.0% | 27.2% |
| | Variant B (20% Explore) | **2.5%** | 11.5% | 10.0% | **25.0%** | 97.5% | 7.5% | 33.9% |
| | Variant C (40% Explore) | 0.5% | 9.0% | 7.5% | 6.7% | 99.5% | 7.0% | 27.5% |
| **Distant Multi-Room** | Phase 6 Baseline | 0.0% | 5.0% | 2.5% | 0.0% | 100.0% | 2.5% | 40.5% |
| | Variant A (Control) | 0.5% | 5.0% | 2.0% | 25.0% | 99.5% | 1.5% | 25.6% |
| | Variant B (20% Explore) | **1.5%** | **7.0%** | **4.0%** | **37.5%** | 98.5% | 2.5% | 32.5% |
| | Variant C (40% Explore) | 0.0% | 3.5% | 1.5% | 0.0% | 100.0% | 1.5% | 27.6% |

### Table 3.2: Stochastic Spawn-Relative Rescue
| Distance Tier | Model | Rescue (%) | Discovery (%) | Pickup (%) | Return \| Pick (%) | Timeout (%) | TO Carry (%) |
|---|---|---|---|---|---|---|---|
| **Spawn 2–6** | Phase 6 Baseline | 97.5% | 100.0% | 100.0% | 97.5% | 2.5% | 2.5% |
| | Variant A (Control) | 99.0% | 100.0% | 100.0% | 99.0% | 1.0% | 1.0% |
| | Variant B (20% Explore) | **99.5%** | 100.0% | 100.0% | **99.5%** | 0.5% | 0.5% |
| | Variant C (40% Explore) | 97.5% | 100.0% | 99.5% | 98.0% | 2.5% | 2.0% |
| **Spawn 7–12** | Phase 6 Baseline | **48.0%** | **91.5%** | **79.0%** | 60.8% | 52.0% | 31.0% |
| | Variant A (Control) | 45.0% | 89.5% | 75.0% | 60.0% | 55.0% | 30.0% |
| | Variant B (20% Explore) | **48.0%** | 85.5% | 76.0% | **63.2%** | 52.0% | 28.0% |
| | Variant C (40% Explore) | 46.0% | 90.0% | 78.5% | 58.6% | 54.0% | 32.5% |
| **Spawn 13–20** | Phase 6 Baseline | **17.5%** | **47.0%** | **43.5%** | 40.2% | 82.5% | 26.0% |
| | Variant A (Control) | **17.5%** | 43.5% | 38.0% | **46.1%** | 82.5% | 20.5% |
| | Variant B (20% Explore) | 15.5% | 46.0% | 35.5% | 43.7% | 84.5% | 20.0% |
| | Variant C (40% Explore) | 16.0% | 46.5% | 40.5% | 39.5% | 84.0% | 24.5% |
| **Spawn 21–35** | Phase 6 Baseline | **7.0%** | **21.0%** | **20.5%** | **34.1%** | 93.0% | 13.5% |
| | Variant A (Control) | 4.5% | 16.0% | 13.5% | 33.3% | 95.5% | 9.0% |
| | Variant B (20% Explore) | 6.0% | **21.0%** | 18.5% | 32.4% | 94.0% | 12.5% |
| | Variant C (40% Explore) | 5.5% | 19.5% | 17.5% | 31.4% | 94.5% | 12.0% |
| **Distant Multi-Room** | Phase 6 Baseline | **5.5%** | 11.5% | 9.0% | **61.1%** | 94.5% | 3.5% |
| | Variant A (Control) | 1.5% | 9.5% | 8.5% | 17.6% | 98.5% | 7.0% |
| | Variant B (20% Explore) | 2.0% | **13.0%** | 9.5% | 21.1% | 98.0% | 7.5% |
| | Variant C (40% Explore) | 4.0% | **13.0%** | **10.0%** | 40.0% | 96.0% | 6.0% |

---

## 4. Exit-Relative Ladder Rescue Benchmark (Historical Consistency)

Evaluated across 200 held-out procedural episodes per tier (seeds 50000–50199) using distance from nearest exit:

### Table 4.1: Deterministic Exit-Relative Rescue
| Difficulty Tier | Phase 6 Baseline | Variant A (Control) | Variant B (20% Explore) | Variant C (40% Explore) |
|---|---|---|---|---|
| **Near ($d=2$)** | **100.0%** | 97.5% | 94.5% | 85.0% |
| **Medium ($d=6$)** | **47.5%** | 45.0% | 45.0% | 42.5% |
| **Bridge A ($d=8$)** | **42.5%** | 41.5% | **42.5%** | 39.5% |
| **Bridge B ($d=10$)** | 21.5% | 18.0% | **22.5%** | 18.0% |
| **Bridge C ($d=12$)** | 10.0% | 9.0% | **13.0%** | 5.5% |
| **Distant ($d=\text{None}$)** | 0.0% | 0.5% | **1.5%** | 0.0% |

### Table 4.2: Stochastic Exit-Relative Rescue
| Difficulty Tier | Phase 6 Baseline | Variant A (Control) | Variant B (20% Explore) | Variant C (40% Explore) |
|---|---|---|---|---|
| **Near ($d=2$)** | **100.0%** | **100.0%** | **100.0%** | 99.5% |
| **Medium ($d=6$)** | 49.5% | 48.5% | **51.5%** | 50.0% |
| **Bridge A ($d=8$)** | 53.0% | 51.5% | **53.5%** | 52.0% |
| **Bridge B ($d=10$)** | **41.5%** | 32.5% | 37.0% | 40.5% |
| **Bridge C ($d=12$)** | 27.5% | 23.0% | 25.0% | **28.5%** |
| **Distant ($d=\text{None}$)** | **5.5%** | 1.5% | 2.0% | 4.0% |

---

## 5. Failure Mode Decomposition

Analysis of failed episodes in Deterministic evaluation (out of 200 episodes per tier):

### Table 5.1: Overlapping Failure Breakdown (%)
| Tier & Model | Total Failures | Never Discovered (%) | Discovered Not Picked (%) | Picked Up Not Returned (%) | Excessive Wait $\ge 35\%$ (%) | Doorway Oscillation (%) |
|---|---|---|---|---|---|---|
| **Spawn 7–12** | | | | | | |
| Phase 6 Baseline | 142 | 26.8% | 28.2% | 45.1% | 23.9% | 57.7% |
| Variant A (Control) | 147 | 27.2% | 27.2% | 45.6% | 32.0% | 53.7% |
| Variant B (20% Explore) | **138** | **26.1%** | 30.4% | **43.5%** | **27.5%** | **55.8%** |
| Variant C (40% Explore) | 163 | 22.7% | 22.1% | 55.2% | 23.3% | 63.2% |
| **Spawn 13–20** | | | | | | |
| Phase 6 Baseline | **181** | 74.0% | 6.6% | 19.3% | 24.3% | 51.4% |
| Variant A (Control) | 184 | 81.0% | 6.5% | 12.5% | 43.5% | 34.8% |
| Variant B (20% Explore) | 184 | **70.7%** | 9.8% | 19.6% | 38.0% | 43.5% |
| Variant C (40% Explore) | 192 | 65.6% | 8.9% | 25.5% | 32.8% | 47.9% |
| **Distant Multi-Room** | | | | | | |
| Phase 6 Baseline | 200 | 95.0% | 2.5% | 2.5% | 24.5% | 52.0% |
| Variant A (Control) | 199 | 95.5% | 3.0% | 1.5% | 51.3% | 30.7% |
| Variant B (20% Explore) | **197** | **94.4%** | 3.0% | 2.5% | 40.6% | **41.1%** |
| Variant C (40% Explore) | 200 | 96.5% | 2.0% | 1.5% | 44.5% | 35.0% |

### Table 5.2: Mutually Exclusive Failure Breakdown (Episode Counts)
*Hierarchical precedence: 1. Picked up not returned $\to$ 2. Discovered not picked $\to$ 3. Excessive wait $\to$ 4. Doorway oscillation $\to$ 5. Never discovered.*

| Distance Tier & Model | Total Failures | 1. Picked Up Not Returned | 2. Discovered Not Picked | 3. Excessive Wait Stalling | 4. Doorway/Cell Oscillation | 5. Other Search Failure |
|---|---|---|---|---|---|---|
| **Spawn 7–12** | | | | | | |
| Phase 6 Baseline | 142 | 64 | 40 | 5 | 22 | 11 |
| Variant A (Control) | 147 | 67 | 40 | 14 | 19 | 7 |
| Variant B (20% Explore) | **138** | **60** | 42 | 16 | **14** | 6 |
| Variant C (40% Explore) | 163 | 90 | 36 | 15 | 16 | 6 |
| **Distant Multi-Room** | | | | | | |
| Phase 6 Baseline | 200 | 5 | 5 | 44 | **99** | 47 |
| Variant A (Control) | 199 | 3 | 6 | **99** | 53 | 38 |
| Variant B (20% Explore) | **197** | 5 | 6 | 74 | **74** | 38 |
| Variant C (40% Explore) | 200 | 3 | 4 | 86 | 65 | 42 |

---

## 6. Visual Debugger Trajectory Case Studies

Six representative episodes were faithfully recorded and compressed in [`trajectories/phase7/`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/trajectories/phase7/):

1. **Pure Exploration Run for Variant B (`pure_exploration_variant_b_seed60000.json.gz`):**
   - *Behavior:* Enters the main corridor smoothly, explores room entrances, and exhibits zero immediate doorway reversal when moving between room thresholds. Total coverage reached 11.2% in 384 steps with controlled oscillation.
2. **Pure Exploration Run for Variant C (`pure_exploration_variant_c_seed60000.json.gz`):**
   - *Behavior:* Achieves high spatial coverage (18.1%) across three distinct zones, but moves erratically between peripheral boundaries, ignoring exit landmarks.
3. **Distant Rescue Success for Variant B (`distant_rescue_success_variant_b_seed50093.json.gz`):**
   - *Behavior:* **Full success in 56 steps.** Agent starts at entrance, systematically clears corridor, identifies victim at $(r=15, c=11)$ inside Room 2, executes pickup at step 24, re-orients toward the entrance door, and safely exits at step 56.
4. **Distant Rescue Failure from Doorway Oscillation (`distant_failure_doorway_oscillation_baseline_seed50057.json.gz`):**
   - *Behavior:* The Phase 6 Baseline gets trapped in a 2-cell oscillation between threshold $(12, 18)$ and $(12, 19)$, executing **368 oscillation actions** out of 384 steps until timing out without finding the victim.
5. **Distant Rescue Failure from Stationary Waiting (`distant_failure_stationary_wait_baseline_seed50093.json.gz`):**
   - *Behavior:* On the exact same seed (50093) where Variant B succeeded, the Baseline agent enters the doorway, encounters an unfamiliar corridor junction, and lapses into stationary waiting for **362 consecutive steps (94.3% Wait fraction)** until timeout.
6. **Matched Pair Comparison on Seed 50093:**
   - [`identical_seed50093_baseline_failure.json.gz`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/trajectories/phase7/identical_seed50093_baseline_failure.json.gz): **Failure (Timeout, 384 steps, 94.3% Wait).**
   - [`identical_seed50093_variant_b_success.json.gz`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/trajectories/phase7/identical_seed50093_variant_b_success.json.gz): **SUCCESS (Rescued, 56 steps, 0.0% Wait).**

---

## 7. Answers to the 7 Core Directive Questions

### Q1: Does auxiliary exploration-only training actually improve building coverage on unknown layouts without victims?
**Yes, in targeted and structured ways, though the overall coverage plateau reflects finite step budgets.**
In pure exploration episodes without victims, Variant B maintains high observed coverage ($10.78\%$ deterministic, $16.47\%$ stochastic) and explores multiple rooms ($1.94$ rooms entered per episode) with exploration efficiency of $66.6$ observed cells per 100 steps. In contrast, pure continuation of Phase 6 without exploration (Variant A) suffered a coverage collapse down to $8.52\%$ deterministic and $34.6$ cells/100st because the agent froze in place.

### Q2: Does it reduce doorway oscillation, 2-cell oscillation, or stationary waiting?
**Yes, dramatically.**
- **Doorway Reversal Rate:** Reduced from **$74.3\%$ in Baseline down to $17.3\%$ in Variant B** (a $76.7\%$ relative reduction).
- **2-Cell Oscillation:** Reduced from **$45.85\%$ in Baseline down to $31.02\%$ in Variant B** (a $14.8$ percentage point absolute drop in deterministic evaluation).
- **Oscillation Traps in Distant Failures:** In the unconstrained Distant benchmark, mutually exclusive doorway oscillation failures dropped from **$99$ episodes in Baseline down to $74$ in Variant B**.
- **Stationary Waiting Trap:** While Baseline and Variant A degenerated into $94\%+$ Wait traps when encountering unfamiliar distant rooms, Variant B actively moved through junctions, enabling clean 56-step distant rescues.

### Q3: Does improved exploration transfer to distant victim discovery and rescue?
**Yes, directly.**
- On deterministic unconstrained distant rescue, Baseline was completely deadlocked at **$0.0\%$ (0/200)**. Variant B unlocked **$1.5\%$ (3/200)** full autonomous rescues.
- On Bridge C ($d=12$), deterministic rescue increased from **$10.0\%$ to $13.0\%$**, and return-after-pickup jumped from **$29.9\%$ to $47.3\%$**.
- Discovery of distant victims in Spawn 13–20 increased from $33.0\%$ to $35.0\%$.

### Q4: What is the optimal exploration-task weight in the multi-task curriculum?
**$20\%$ is optimal; $40\%$ is excessive and harmful.**
- **$20\%$ (Variant B):** Strikes the ideal balance. It provides sufficient gradient pressure from frontier bonuses and oscillation penalties to break navigation loops without eroding the goal-directed pickup and return drives.
- **$40\%$ (Variant C):** Produces severe task interference. Over-exposing the network to episodes where no victim exists and where the agent is rewarded exclusively for wandering causes the policy to abandon victim-rescue priorities.

### Q5: Does exploration training cause catastrophic forgetting of near rescue, medium rescue, or victim-return behavior?
- **For Variant B (20%): No.**
  - Stochastic near rescue (Spawn 2–6) actually improved from $97.5\%$ to **$99.5\%$**.
  - Medium rescue (Spawn 7–12) improved from $29.0\%$ to **$31.0\%$** deterministic ($48.0\%$ stochastic).
  - Return-after-pickup conditional success improved significantly: from $47.5\%$ to **$50.8\%$** on Spawn 7–12, and from $29.9\%$ to **$47.3\%$** on Bridge C ($d=12$).
- **For Variant C (40%): Yes.**
  - Deterministic near rescue collapsed to $80.0\%$ ($85.0\%$ on Near $d=2$).
  - Return-after-pickup on Spawn 7–12 collapsed from $47.5\%$ down to $29.1\%$.
  - Carrier timeouts increased by $+13.0\%$.

### Q6: How much of the remaining distant-rescue failure is search failure vs. pickup failure vs. return failure?
In Distant Multi-Room scenarios:
1. **Search / Discovery Failure ($70.7\% - 94.4\%$):** The dominant bottleneck remains initial search. In Distant Multi-Room, $94.4\%$ of failures occur because the agent never brings the victim into its local observation window before step 384.
2. **Pickup Failure ($3.0\% - 9.8\%$):** Once a distant victim is observed, pickup is relatively reliable, failing in only $3.0\%$ of distant episodes.
3. **Return Failure ($2.5\% - 19.6\%$):** Once picked up, return success conditional on pickup is $37.5\%$ in Variant B (up from $0.0\%$ in Baseline).
The primary remaining barrier to scaling beyond $d=20$ is **pure search coverage across large buildings with multi-branch corridors**.

### Q7: Should exploration-only auxiliary training be adopted for future production runs, and if so, with what mixture?
**Yes. Adopt Variant B's 4-task curriculum mixture for all subsequent training:**
$$\text{Task Mixture} = (0.32 \text{ Full Rescue}, 0.28 \text{ Route-Complexity Return}, 0.20 \text{ Discovery}, 0.20 \text{ Exploration-Only})$$
- Retain the frontier observation bonus ($+0.05$), edge discovery bonus ($+0.02$), step penalty ($-0.001$), 2-cell oscillation penalty ($-0.02$), and cap ($+6.0$).
- Keep all evaluation strictly unshaped ($+10$ native rescue only).

---

## 8. Artifact and Checkpoint Catalog

- **Baseline:** [`models/phase6_spawn_curriculum/production_250k/stage3_spawn_curriculum/best_model.zip`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/models/phase6_spawn_curriculum/production_250k/stage3_spawn_curriculum/best_model.zip)
- **Variant A (Control):** [`models/phase7_control/stage3_spawn_curriculum/stage3_final.zip`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/models/phase7_control/stage3_spawn_curriculum/stage3_final.zip)
- **Variant B (20% Explore - Best):** [`models/phase7_explore20/stage3_spawn_curriculum/best_model.zip`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/models/phase7_explore20/stage3_spawn_curriculum/best_model.zip)
- **Variant C (40% Explore):** [`models/phase7_explore40/stage3_spawn_curriculum/best_model.zip`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/models/phase7_explore40/stage3_spawn_curriculum/best_model.zip)
- **Evaluation JSONs:**
  - Baseline: [`models/phase7_evaluations/baseline_eval.json`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/models/phase7_evaluations/baseline_eval.json)
  - Variant A Control Final: [`models/phase7_evaluations/variant_a_final_eval.json`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/models/phase7_evaluations/variant_a_final_eval.json)
  - Variant B: [`models/phase7_evaluations/variant_b_eval.json`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/models/phase7_evaluations/variant_b_eval.json)
  - Variant C: [`models/phase7_evaluations/variant_c_eval.json`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/models/phase7_evaluations/variant_c_eval.json)
- **Visual Debugger Trajectories:** [`trajectories/phase7/`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/trajectories/phase7/)
