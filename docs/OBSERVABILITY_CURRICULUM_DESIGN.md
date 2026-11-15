# Observability Curriculum Design & Experimental Specification (A/B Study)

**Project:** `fire-rescue-project-2.0`  
**Date:** 2026-11-15  
**Authors:** Antigravity Autonomous Orchestrator & Research Team  
**Status:** IMPLEMENTED / VERIFIED  

---

## 1. Executive Summary & Research Question

Having completed the architecture capacity ablation—confirming that the compact 1.43M-parameter recurrent CNN baseline (`baseline`, 128-unit GRU) outperforms larger models (2.76M and 5.78M) on sparse-reward multi-room navigation—we investigate an orthogonal hypothesis:

> **Central Research Question:**  
> *Does pretraining a recurrent reinforcement-learning agent with complete building geometry and known victim locations improve its ability to navigate, explore, and rescue victims when the environment subsequently transitions to complete partial observability?*

To answer this conclusively without confounding variables, we design a strictly controlled **A/B Experiment**:
- **Experiment A (Baseline / Tabula Rasa):**  
  The agent trains from scratch under standard partial observability from step 0:
  - Building geometry is initially completely unknown.
  - Victims are hidden until physically discovered in line-of-sight.
  - Memory map is populated solely from legitimate local observations.
  - Neural network controls all 7 primitive actions.
- **Experiment B (Observability Curriculum / Full Info $\to$ Unknown):**  
  The identical neural architecture is trained using a staged observability curriculum:
  - **Stage B1 (Full Information):** Complete building geometry revealed + victim location known in memory.
  - **Stage B2 (Full Blueprint, Hidden Victim):** Complete building geometry revealed, but victim location is hidden until physically discovered.
  - **Stage B3 (Progressive Geometry Fade):** Topologically contiguous regions of geometry are revealed (75% $\to$ 50% $\to$ 25%), while victims remain hidden.
  - **Stage B4 (Fully Unknown):** Geometry is 0% revealed and victims are hidden (identical to Experiment A).

Both agents are evaluated under **identical, strictly unshaped (+10 native rescue only), fully unknown-map conditions**.

---

## 2. Observation Semantics & The 9-Channel Representation

The observation tensor dimensions are strictly preserved across all experiments and curriculum stages:
- **Local Observation:** `grid` $\in \mathbb{R}^{9 \times 15 \times 15}$ (visibility radius $R=7$ centered on the agent)
- **Global Memory Map:** `memory` $\in \mathbb{R}^{9 \times 40 \times 52}$ (centered on building footprint, zero-padded)
- **Agent Scalar Features:** `agent` $\in \mathbb{R}^{5}$ (health, carrying status, orientation/speed)
- **Previous Action:** `previous_action` $\in \{0..6\}$ (one-hot encoded)

### Precise 9-Channel Semantic Definitions

| Channel | Semantic Name | Meaning in Local Grid | Meaning in Global Memory Map | Unknown Cell Value |
| :---: | :--- | :--- | :--- | :---: |
| **Ch 0** | **Passability North** | 1.0 if face to North is passable (flux $\ge 0.5$) | 1.0 if wall/portal to North is known passable | **0.0** |
| **Ch 1** | **Passability East** | 1.0 if face to East is passable (flux $\ge 0.5$) | 1.0 if wall/portal to East is known passable | **0.0** |
| **Ch 2** | **Passability South** | 1.0 if face to South is passable (flux $\ge 0.5$) | 1.0 if wall/portal to South is known passable | **0.0** |
| **Ch 3** | **Passability West** | 1.0 if face to West is passable (flux $\ge 0.5$) | 1.0 if wall/portal to West is known passable | **0.0** |
| **Ch 4** | **Debris / Obstacle** | 1.0 if cell blocked by debris/collapse | 1.0 if cell is known to be blocked by debris | **0.0** |
| **Ch 5** | **Explored Mask** | 1.0 if cell is in current line-of-sight | 1.0 if cell has been **physically scanned** in LOS | **0.0** |
| **Ch 6** | **Victim** | Normalized victim value ($>0$) if in LOS | 1.0 if victim known at cell; 0.0 otherwise | **0.0** |
| **Ch 7** | **Exit** | 1.0 if exit portal cell visible in local grid | 1.0 if exit portal cell is known | **0.0** |
| **Ch 8** | **Agent Position** | 1.0 at local grid center $(7, 7)$ | 1.0 at agent current coordinates $(r, c)$ | **0.0** |

### Critical Separation: Known Blueprint vs. Physically Explored

A common pitfall in map pretraining is conflating **blueprint knowledge** with **physical exploration**:
1. When privileged geometry is pre-filled in Stages B1, B2, and B3, it is written exclusively to channels **0, 1, 2, 3, 4, 7** (passability, debris, exits) and optionally channel **6** (victim in B1).
2. **Channel 5 (`explored_mask`) and internal tracker `self.memory.explored` are left strictly at `0.0`.**
3. `self.memory.explored` is incremented **only** when the agent executes steps and brings cells into its physical line-of-sight (`vis_mask > 0`).
4. **Reward Shaping Isolation:** The training-time frontier exploration reward ($+0.05$ per newly uncovered cell, capped at $+3.0$) computes:
   $$\Delta_{\text{uncovered}} = \max(0, \sum \text{explored}_{t} - \sum \text{explored}_{t-1})$$
   Because prefilled geometry leaves `explored` as 0, **privileged geometry never triggers false exploration bonuses**.
5. **Policy Disambiguation:** In the memory map, the neural network can distinguish:
   - Passable $+ \text{Explored}=0$: A known hallway or room that has **not yet been searched**.
   - Passable $+ \text{Explored}=1$: A hallway or room that has **already been searched**.
   - Impassable $+ \text{Explored}=0$: Unmapped darkness / uncharted exterior.
   - Impassable $+ \text{Explored}=1$: Confirmed solid interior wall.

---

## 3. Coherent Partial-Geometry Generation (Grid-BFS)

To avoid unphysical "salt-and-pepper" random pixel revealing, partial geometry in Stage B3 is generated via **Breadth-First Search (BFS) flood-fill across traversable grid passages starting from the agent's initial entrance position**:

```python
def get_connected_revealed_mask(sim, start_pos, fraction: float) -> np.ndarray:
    if fraction >= 1.0:
        return (sim.zone >= 0)  # All interior building cells
    if fraction <= 0.0:
        return np.zeros((sim.h, sim.w), dtype=bool)

    total_interior = int(np.count_nonzero(sim.zone >= 0))
    target_count = int(round(fraction * total_interior))

    queue = deque([start_pos])
    revealed = np.zeros((sim.h, sim.w), dtype=bool)
    revealed[start_pos] = True
    count = 1

    while queue and count < target_count:
        r, c = queue.popleft()
        neighbors = (
            (r - 1, c, r > 0 and sim.south[r - 1, c] >= 0.5),
            (r, c + 1, c + 1 < sim.w and sim.east[r, c] >= 0.5),
            (r + 1, c, r + 1 < sim.h and sim.south[r, c] >= 0.5),
            (r, c - 1, c > 0 and sim.east[r, c - 1] >= 0.5),
        )
        for nr, nc, passable in neighbors:
            if passable and not revealed[nr, nc] and not sim.debris[nr, nc]:
                revealed[nr, nc] = True
                count += 1
                queue.append((nr, nc))
                if count >= target_count:
                    break
    return revealed
```

### Properties of Grid-BFS Generation:
- **Physical Contiguity:** Every revealed cell is physically connected by valid corridors and open doors to the entrance.
- **Accurate Granularity:**
  - $f = 0.75$: The entrance wing, central corridor, and first two rooms are revealed.
  - $f = 0.50$: The entrance wing and central hallway are revealed.
  - $f = 0.25$: Only the immediate entrance corridor is revealed.
  - $f = 0.00$: Zero geometry revealed (fully unknown).

---

## 4. Experimental Control Protocol

To ensure valid scientific comparison, both experiments adhere to strict ceteris paribus controls:

| Factor | Experiment A | Experiment B | Control Status |
| :--- | :---: | :---: | :--- |
| **Model Architecture** | `baseline` (1.43M params, 128-unit GRU) | `baseline` (1.43M params, 128-unit GRU) | **Identical** |
| **Input Dimensions** | Grid $(9, 15, 15)$, Memory $(9, 40, 52)$, Agent $(5,)$ | Grid $(9, 15, 15)$, Memory $(9, 40, 52)$, Agent $(5,)$ | **Identical** |
| **Primitive Actions** | 7 discrete actions (Stay, N, E, S, W, Pick, Drop) | 7 discrete actions (Stay, N, E, S, W, Pick, Drop) | **Identical** |
| **Action Masking** | Legal-action masking via `MaskablePPO` | Legal-action masking via `MaskablePPO` | **Identical** |
| **Total Step Budget** | 1,000,000 environment steps | 1,000,000 environment steps | **Identical** |
| **Scenario Seeds** | Fixed PRNG schedule | Matched identical PRNG schedule | **Identical** |
| **Navigation Potential Shaping** | **DISABLED** (`navigation_shaping=False`) | **DISABLED** (`navigation_shaping=False`) | **Controlled** |
| **Exploration Bonus** | $+0.05$ per physical cell uncovered | $+0.05$ per physical cell uncovered | **Identical** |
| **Pickup Reward** | $+1.5$ first pickup | $+1.5$ first pickup | **Identical** |
| **Evaluation Protocol** | Strictly unshaped (+10 native rescue only), unknown map | Strictly unshaped (+10 native rescue only), unknown map | **Identical** |

### Elimination of Navigation Shaping Confound
In earlier legacy experiments, `navigation_shaping` computed state potential using the true simulator victim coordinates:
$$\Phi(s) = -d_{\text{shortest}}(\text{agent}, \text{victim})$$
If left enabled, this would act as a hidden compass guiding the agent toward the victim even when the victim was supposedly "hidden". In both Experiment A and Experiment B, **`navigation_shaping` is strictly disabled**, ensuring that reward signals provide zero privileged information about victim coordinates.

---

## 5. Staged Curriculum Schedule (Experiment B)

To prevent catastrophic forgetting when transitioning between visibility stages, Experiment B uses a staged progression that gradually phases out privileged information while mixing in unknown-map scenarios:

### 1,000,000-Step Production Schedule

| Stage | Step Range | Cumulative Steps | Primary Mode | Geometry Reveal | Victim Known | Mixed Unknown (B4) |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **B1** | 0 – 200,000 | 200,000 | Full Info | 100% | **Yes** | 20% |
| **B2** | 200,000 – 400,000 | 400,000 | Full Blueprint | 100% | **No** (Hidden) | 20% |
| **B3a** | 400,000 – 550,000 | 550,000 | 75% Fade | 75% | **No** (Hidden) | 25% |
| **B3b** | 550,000 – 700,000 | 700,000 | 50% Fade | 50% | **No** (Hidden) | 25% |
| **B3c** | 700,000 – 800,000 | 800,000 | 25% Fade | 25% | **No** (Hidden) | 30% |
| **B4** | 800,000 – 1,000,000 | 1,000,000 | Fully Unknown | **0%** | **No** (Hidden) | **100%** |

### 50,000-Step Diagnostic Pilot Schedule

| Stage | Step Range | Primary Mode | Geometry Reveal | Victim Known |
| :---: | :---: | :--- | :---: | :---: |
| **Pilot B1** | 0 – 15,000 | Full Info | 100% | Yes (80% / 20% B4) |
| **Pilot B2** | 15,000 – 30,000 | Full Blueprint | 100% | No (70% / 30% B4) |
| **Pilot B3** | 30,000 – 40,000 | Partial Blueprint | 50% | No (60% / 40% B4) |
| **Pilot B4** | 40,000 – 50,000 | Fully Unknown | 0% | No (100% B4) |

---

## 6. Mandatory Automated Verification Suite

Before launching production training, twelve automated test cases must pass:

1. **`test_b1_full_information`**: Verifies B1 reveals complete building geometry (100% of passability, debris, and exits) and sets victim channel at true victim position.
2. **`test_b2_full_geometry_hidden_victim`**: Verifies B2 reveals complete geometry, but victim channel is strictly 0.0 at reset.
3. **`test_b3_partial_geometry_fractions`**: Verifies B3 correctly reveals contiguous connected regions matching 75%, 50%, and 25% fractions.
4. **`test_b4_observation_equivalence`**: Verifies B4 produces identical observations to Experiment A.
5. **`test_unrevealed_cells_impassable`**: Verifies unrevealed cells have 0.0 passability in all 4 directions.
6. **`test_no_false_exploration_bonuses`**: Verifies prefilling geometry leaves `explored_mask` at 0.0 and yields 0 exploration reward at reset.
7. **`test_observation_shape_invariance`**: Verifies observation shapes remain `(9, 15, 15)`, `(9, 40, 52)`, `(5,)`, `(1,)`.
8. **`test_action_mask_integrity`**: Verifies action masking is identical regardless of visibility stage.
9. **`test_recurrent_bptt_continuity`**: Verifies GRU hidden states update and propagate gradients across time steps.
10. **`test_evaluation_strictly_unknown`**: Verifies evaluation runs strictly with 0% geometry, hidden victims, and native +10 reward.
11. **`test_zero_information_leakage`**: Verifies hidden victim coordinates do not leak into A or B4 observations.
12. **`test_backward_compatibility`**: Verifies existing checkpoints can be loaded without error.
