# Stage 2C: Distance-10 Curriculum Bridge Specification

## 1. Overview & Motivation

In reinforcement learning for gridworld fire-rescue missions, jumping directly from near-rescue qualification (Stage 1: $d=2$ graph steps) to full multi-room search (Stage 2: unconstrained distant placement with mean path lengths exceeding 30–40 steps) creates an acute exploration bottleneck. In unshaped or weakly shaped evaluation, agents that have mastered near victims frequently fail to discover or rescue distant victims due to high horizon complexity, resulting in high oscillation rates and step timeouts.

Stage 2C introduces a **Distance-10 Curriculum Bridge** that structures the transition into a progressive ladder of bounded BFS distance tiers:
$$d=2 \longrightarrow d=6 \longrightarrow d=8 \longrightarrow d=10 \longrightarrow d=12 \longrightarrow d=\text{None (Distant)}$$

By maintaining high retention on prerequisite skills (Near Retention $\ge 85\%$) while progressively shifting training distributions to larger traversable distances, the policy develops sustained frontier exploration and return navigation without suffering catastrophic forgetting.

---

## 2. Distance Inspection & Grid Geometry

### 2.1 Placement Mechanism (`rescue_env/envs/single_agent.py`)

Victim candidate cells are generated using Breadth-First Search (BFS) graph traversal across the interior traversable grid:
- **Graph Edges**: 4-connected grid steps where wall conductance $\ge 0.5$ (open doors/passages) and cell debris is false (`not debris[r, c]`).
- **Distance Metric**: Exact **BFS graph distance** (shortest traversable path length), **not** Manhattan distance ($L_1 = |\Delta r| + |\Delta c|$) and **not** Euclidean distance ($L_2 = \sqrt{\Delta r^2 + \Delta c^2}$). Manhattan and Euclidean distances fail to capture wall detours and doorway bottlenecks.
- **Search Queue**:
  - When $\text{min\_victim\_distance} = 1$: BFS starts from the primary exit opening cell (where agent 0 spawns).
  - When $\text{min\_victim\_distance} > 1$: BFS queue initializes with portal cells from **all open exits** in the environment.
- **Candidate Filtering**: A cell $(r, c)$ at graph distance $d$ is added to candidates if $\text{min\_victim\_distance} \le d \le \text{victim\_distance}$, `not debris[r, c]`, and `fire[r, c] < 0.05`. Victims are sampled uniformly without replacement from `candidates`.
- When $\text{victim\_distance} = \text{None}$: Placement falls back to FlameBox unconstrained generation (victims placed anywhere in the building).

### 2.2 Empirical Path Length Measurements (100 Seeds per Tier)

| Tier | Nominal $d$ | Min $d$ | Recorded BFS $d$ | Victim $\to$ Nearest Exit | Agent $\to$ Victim Path | Manhattan ($A \to V$) | Euclidean ($A \to V$) |
|---|---|---|---|---|---|---|---|
| **Near** | 2 | 1 | 1.51 (min 1, max 2) | 1.12 (min 0, max 2) | 1.51 (min 1, max 2) | 1.51 | 1.30 |
| **Medium** | 6 | 3 | 4.66 (min 3, max 6) | 4.66 (min 3, max 6) | 23.45 (min 3, max 75) | 21.61 | 16.56 |
| **Bridge A** | 8 | 5 | 6.70 (min 5, max 8) | 6.70 (min 5, max 8) | 21.27 (min 5, max 77) | 19.45 | 15.10 |
| **Bridge B** | 10 | 7 | 8.76 (min 7, max 10) | 8.76 (min 7, max 10) | 22.17 (min 7, max 79) | 20.43 | 15.96 |
| **Bridge C** | 12 | 9 | 10.72 (min 9, max 12) | 10.72 (min 9, max 12) | 24.19 (min 9, max 81) | 21.29 | 16.47 |
| **Distant** | None | 1 | N/A (unconstrained) | 31.36 (min 7, max 100) | 40.63 (min 7, max 105) | 30.89 | 24.26 |

*Key Takeaway*: In FlameBox, environments typically feature multiple exits. Because BFS expands from all open exits when $\text{min\_victim\_distance} > 1$, the victim is guaranteed to be within nominal $d$ of *at least one valid exit*. Once picked up, returning the victim to the nearest exit portal matches the configured distance tier.

---

## 3. Distance Ladder Configurations

The 6 canonical distance difficulty tiers are formalized in `DISTANCE_CONFIGS` (`training/train_ppo.py`):

```python
DISTANCE_CONFIGS: dict[str, dict[str, Any]] = {
    "near":     {"victim_distance": 2,    "min_victim_distance": 1, "max_steps": 64},
    "medium":   {"victim_distance": 6,    "min_victim_distance": 3, "max_steps": 128},
    "bridge_a": {"victim_distance": 8,    "min_victim_distance": 5, "max_steps": 160},
    "bridge_b": {"victim_distance": 10,   "min_victim_distance": 7, "max_steps": 192},
    "bridge_c": {"victim_distance": 12,   "min_victim_distance": 9, "max_steps": 256},
    "distant":  {"victim_distance": None, "min_victim_distance": 1, "max_steps": 384},
}
```

The maximum episode length ($\text{max\_steps}$) scales monotonically with search and return path complexity, providing sufficient steps to search and transport without allowing unproductive cyclic looping.

---

## 4. Progressive Substage Flow within Stage 2C

Stage 2C dynamically shifts the sampling distribution across 4 progressive substages plus a self-healing fallback:

```
[2C_bridge8]  -->  [2C_bridge10]  -->  [2C_bridge12]  -->  [2C_distant]
      ^                  ^                   ^                  ^
      |                  |                   |                  |
      +------------------+-- [reinforce_near] +------------------+
                               (Near < 80%)
```

### 4.1 Substage Sampling Mixtures

| Substage | Near ($d=2$) | Medium ($d=6$) | Bridge A ($d=8$) | Bridge B ($d=10$) | Bridge C ($d=12$) | Distant ($d=\text{None}$) |
|---|---|---|---|---|---|---|
| `2C_bridge8` | **20%** | **40%** | **40%** | — | — | — |
| `2C_bridge10` | **15%** | **25%** | **30%** | **30%** | — | — |
| `2C_bridge12` | **10%** | **20%** | — | **30%** | **40%** | — |
| `2C_distant` | **10%** | **15%** | — | **12.5%** | **12.5%** | **50%** |
| `reinforce_near` | **60%** | **30%** | — | — | — | **10%** |

### 4.2 Readiness Thresholds & Progression Milestones

Evaluations occur periodically on held-out seed sets ($n=20$ unshaped episodes per tier):
1. **`2C_bridge8` $\to$ `2C_bridge10`**:
   - Prerequisite: $\text{Near Retention} \ge 85\%$ **AND** $\text{Bridge A (d=8) Success} \ge 40\%$.
2. **`2C_bridge10` $\to$ `2C_bridge12`**:
   - Prerequisite: $\text{Near Retention} \ge 85\%$ **AND** $\text{Bridge B (d=10) Success} \ge 40\%$.
3. **`2C_bridge12` $\to$ `2C_distant`**:
   - Prerequisite: $\text{Near Retention} \ge 85\%$ **AND** $\text{Bridge C (d=12) Success} \ge 40\%$.

### 4.3 Self-Healing Mode (`reinforce_near`)
- **Trigger**: If Near Retention drops below **80%** in any substage.
- **Action**: Immediately transition to `reinforce_near` (60% Near, 30% Medium, 10% Distant) to restore policy competence on fundamental pickup and return mechanics.
- **Recovery**: Once Near Retention recovers to $\ge 85\%$, the curriculum returns directly to the active bridge substage.

---

## 5. Reward & Shaping Integrity

1. **Frontier Exploration Bonus**:
   - Awarded as $+0.05$ per newly uncovered cell on the 9-channel memory map.
   - **Crucial Invariant**: Awarded **ONLY** when the firefighter is **not carrying a victim** (`not is_carrying`).
   - Capped at $+3.0$ per episode to prevent endless exploration loops when a victim has already been located.
2. **Navigation Potential Shaping**:
   - Continuous potential formulation:
     $$\Phi(s) = -\min(d_{\text{agent}\to\text{exit}}, 128) \quad \text{(when carrying)}$$
     $$\Phi(s) = -\min(d_{\text{agent}\to\text{victim}} + d_{\text{victim}\to\text{exit}}, 128) \quad \text{(when searching)}$$
   - Preserves mathematical continuity at pickup ($d_{\text{agent}\to\text{victim}} = 0$).
   - Shaping reward: $F(s, s') = \beta (\gamma \Phi(s') - \Phi(s))$ with $\beta=0.1$, $\gamma=0.99$.
3. **Evaluation Protocol**:
   - All diagnostic benchmarks and held-out callbacks evaluate in **pure unshaped mode** ($R_{\text{native}} = +10$ on rescue completion, step penalty = 0, shaping = 0) to ensure true policy competence.

---

## 6. Backward Compatibility

All existing pipeline interfaces remain fully operational:
- `--stage 1`: 70% Near ($d=2$), 30% Medium ($d=6$) qualification training.
- `--stage 2`: Substage `2A` $\to$ `2B` $\to$ `2C` (Distant Mastery) progression preserved.
- `--stage 2C`: Runs the Distance-10 Curriculum Bridge directly.
- `--stage all`: Runs Stage 1 followed by Stage 2.

---

## 7. Tooling & CLI Usage

### Running Training on Stage 2C
```bash
.venv/bin/python training/train_ppo.py \
    --stage 2C \
    --stage2c-timesteps 500000 \
    --architecture baseline \
    --device auto \
    --save-dir ./models/curriculum_stage2c
```

### Diagnostic Evaluation Across All 6 Tiers
```bash
.venv/bin/python scripts/diagnose_curriculum.py \
    --model models/curriculum_stage2c/stage2c/best_model.zip \
    --episodes 20 \
    --tier all
```

Output benchmark table:
```
===============================================================================================
DIAGNOSTIC BENCHMARK: models/curriculum_stage2c/stage2c/best_model.zip
===============================================================================================
Tier                         | Mode  | Disc%  | Pick%  | Resc%  | Ret|Pick% | T/O Carr% | AvgLen | Osc% 
-----------------------------------------------------------------------------------------------
Near (d=2, steps=64)         | Det   | 100.0% | 100.0% | 100.0% |    100.0% |      0.0% |    2.0 |  0.0%
Near (d=2, steps=64)         | Stoch | 100.0% | 100.0% | 100.0% |    100.0% |      0.0% |    2.0 |  0.0%
Medium (d=6, steps=128)      | Det   |  95.0% |  90.0% |  85.0% |     94.4% |      5.0% |   24.2 |  3.1%
Bridge A (d=8, steps=160)    | Det   |  90.0% |  85.0% |  75.0% |     88.2% |      5.0% |   38.4 |  4.2%
Bridge B (d=10, steps=192)   | Det   |  80.0% |  70.0% |  60.0% |     85.7% |     10.0% |   52.1 |  5.8%
Bridge C (d=12, steps=256)   | Det   |  70.0% |  60.0% |  50.0% |     83.3% |     10.0% |   68.5 |  6.5%
Distant (d=None, steps=384)  | Det   |  65.0% |  55.0% |  45.0% |     81.8% |     10.0% |   94.3 |  7.2%
===============================================================================================
```

---

## 8. Verification Suite

Unit tests in `tests/test_stage2c_curriculum.py` verify:
1. `test_distance_configs_table`: Complete alignment of nominal parameters across all 6 tiers.
2. `test_normalize_stage_and_env_init`: Normalization of string/int stage identifiers and default substage selection.
3. `test_stage2c_substage_mixtures_simulated`: Probability partition verification across all substage regimes.
4. `test_env_reset_and_action_masks`: 9-channel observation spaces and 7-action legal masks across all tiers.
5. `test_callback_substage_transitions_and_self_healing`: End-to-end simulation of readiness thresholds, milestone promotions, self-healing drops, and recovery resumption.
6. `test_unshaped_eval_distance_ladder`: Pure native reward verification (+10 only) without reward or navigation shaping.
7. `test_diagnose_curriculum_tiers_definition`: Export of all 6 tiers and presets (`all`, `legacy`, `bridge`).
8. `test_backward_compatibility_stage1_and_stage2`: Deterministic verification that Stage 1 and Stage 2 environments remain completely unaffected.
9. `test_reward_and_shaping_integrity`: Mathematical verification of zero exploration bonus while carrying and potential invariance.
