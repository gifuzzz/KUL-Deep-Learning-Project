# Phase 8 — Exploration Reward Mechanism & Saturation Audit

**Author:** Antigravity Autonomous Agent  
**Date:** November 17, 2026  
**Scope:** `rescue_env/envs/single_agent.py` exploration-only task reward formulation, saturation dynamics, and empirical verification.

---

## 1. Executive Summary

During Phase 7, empirical diagnostics demonstrated that in **99% of deterministic episodes** (and 81% of stochastic episodes), the policy received **zero positive exploration reward after step 50**.

This audit investigates the exact mathematical mechanics behind this phenomenon. We confirm that **the Phase 7 exploration reward suffered from an artificial per-episode hard cap (`max_exploration_reward = 6.0`)** combined with high initial line-of-sight gains ($+0.05$ per observed cell, $+0.02$ per edge). Because entering an initial room reveals 80–120 cells and edges within 25–40 steps, the entire $+6.0$ budget was exhausted immediately. From step 40 to 384, **the environment awarded 0.0 for discovering new rooms and corridors**, reducing the reward function to a pure penalty-minimization problem ($-0.001$ per step, $-0.02$ for oscillation). Under this regime, the mathematically optimal action was to freeze in place (`Wait`), directly causing the extreme stationary wait stalling and failure to explore distant wings.

---

## 2. Code-Level Mechanics in Phase 7

In `rescue_env/envs/single_agent.py` (lines 1108–1125), exploration rewards were calculated at each step as follows:

```python
if self.task_type == "exploration_only":
    if self.reward_shaping:
        shaping = -self.exploration_step_penalty  # -0.001
        pos_potential = (
            len(new_cells) * self.exploration_cell_bonus   # len(new_cells) * 0.05
            + len(new_edges) * self.exploration_edge_bonus # len(new_edges) * 0.02
        )
        if pos_potential > 0.0 and self._total_exploration_reward < self.max_exploration_reward:
            actual_pos = min(
                pos_potential,
                self.max_exploration_reward - self._total_exploration_reward,  # Cap at 6.0
            )
            self._total_exploration_reward += actual_pos
            shaping += actual_pos
        if is_oscillating:
            shaping -= self.oscillation_penalty  # -0.02
        reward = float(shaping)
    else:
        reward = 0.0
```

---

## 3. Detailed Audit Questions & Findings

### Q1: How are cell-discovery and edge-discovery rewards calculated?
- **Cell Discovery:** When the agent steps into a location, `self.memory.explored` updates with all cells currently within its $15 \times 15$ line-of-sight cone (radius = 7). Cells in `self._unobserved_reachable_interior_cells` that transition from `explored == 0.0` to `explored > 0.0` are added to `new_cells`. Each new cell is credited $+0.05$.
- **Edge Discovery:** For every newly observed cell $(r, c)$, adjacent edges to reachable non-debris cells are checked. Any edge not previously in `self._discovered_edges` is added to `new_edges` and credited $+0.02$.

### Q2: Is information rewarded once or repeatedly?
- **Information is rewarded strictly ONCE.**
- Once a cell is added to `self._observed_reachable_cells`, it is removed from `self._unobserved_reachable_interior_cells`. Re-observing or revisiting the cell produces `len(new_cells) == 0`.
- Once an edge is in `self._discovered_edges`, it cannot be re-added.
- Therefore, stationary waiting, oscillating, or looping through previously observed rooms yields $0.0$ positive potential.

### Q3: Is the reward cap per-step or per-episode?
- **The cap (`max_exploration_reward = 6.0`) is strictly PER EPISODE.**
- `self._total_exploration_reward` is initialized to $0.0$ at `reset()`. It monotonically accumulates every positive reward awarded during the episode.
- Once `self._total_exploration_reward >= 6.0`, no positive reward can ever be granted again in that episode, regardless of how many steps remain or how many new rooms are discovered.

### Q4: Do large initial line-of-sight reveals consume most of the budget?
- **YES. The initial reveal consumes virtually 100% of the budget.**
- In FlameBox, visibility radius is 7. A single open doorway into Room 1 or the central corridor instantly reveals a field of view containing 40 to 90 reachable cells and 80 to 160 traversable edges:
  $$\text{Initial Potential} = (80 \times 0.05) + (140 \times 0.02) = 4.00 + 2.80 = 6.80$$
- Because $6.80 > 6.00$, the cap is fully saturated within the first 15–35 steps of entering the building!

### Q5: How does observed-cell discovery differ from physically entering a new cell?
- `new_cells` tracks **line-of-sight observation**, NOT physical traversal.
- Standing at an entrance doorway marks the entire visible room as "observed" ($1.0$ in `memory.explored`).
- The agent was therefore heavily rewarded for *peeking* into rooms from afar, but received **zero incremental reward for physically walking through those rooms**.

### Q6: Can newly observed geometry behind doorways generate rewards after step 50?
- **Mechanically: YES, the BFS code supports it.** If `self._total_exploration_reward < 6.0`, discovering a new room at step 100 or 200 properly generates positive reward.
- **Practically: NO, because the cap is already 100% full.** By step 50, `self._total_exploration_reward` was already at 5.5–6.0. Any geometry uncovered behind a second or third doorway yielded `min(pos_potential, 6.0 - 6.0) = 0.0`!

---

## 4. Why Saturation Incentivizes Waiting and Oscillation

When `self._total_exploration_reward == 6.0` at step 40, the reward function for the remaining 344 steps becomes:

$$R_t = -0.001 - 0.02 \cdot \mathbb{I}(\text{oscillating})$$

1. **Active Exploration:**
   - Walking down an unknown corridor yields $-0.001$ per step (zero positive reward, because cap is exhausted).
   - If the agent hits a dead end or doorway and turns around, it risks triggering a 2-cell oscillation penalty ($-0.02$).
2. **Stationary Waiting (`Wait` Action 0):**
   - Standing completely still yields $-0.001$ per step.
   - It is guaranteed to NEVER trigger the $-0.02$ oscillation penalty.
3. **PPO Value Collapse:**
   The value network quickly learns that $V(s) \approx -0.001 \times (T - t)$ everywhere after step 50. The policy gradient actively discourages venturing into narrow doorways where reversals risk severe negative shaping.

---

## 5. Proposed Reward Correction for Phase 8

To sustain an exploration incentive throughout all 384 steps without early saturation or unbounded reward explosion:

1. **Normalized Coverage-Potential Reward:**
   Instead of fixed unnormalized $+0.05$ per cell and $+0.02$ per edge with a hard ceiling, tie positive rewards directly to fractional progress toward the total reachable capacity:
   $$R_{\text{pos}}(t) = W_{\text{cell}} \cdot \frac{|\text{new\_cells}_t|}{|\text{Reachable Interior Cells}|} + W_{\text{edge}} \cdot \frac{|\text{new\_edges}_t|}{|\text{Reachable Edges}|}$$
   - Where $W_{\text{cell}} = 8.0$ and $W_{\text{edge}} = 4.0$.
   - **Theoretical Maximum Episode Reward:** If the agent uncovers 100% of the building's reachable interior and edges, the total cumulative reward is strictly $8.0 + 4.0 = 12.0$ (commensurate with the $+10.0$ native rescue reward).
   - **Zero Premature Cliff:** Uncovering 10% of the building at step 20 awards $+1.20$. Uncovering another 10% behind a distant doorway at step 150 awards another $+1.20$. Discovery remains **linearly and continuously rewarded across all 384 steps**.
2. **Physical Visit Progress Bonus:**
   To encourage physically entering rooms rather than merely peeking from doorways, award a small bonus for newly *physically visited* reachable interior cells:
   $$R_{\text{visit}}(t) = W_{\text{visit}} \cdot \frac{|\text{newly\_visited\_cells}_t|}{|\text{Reachable Interior Cells}|}, \quad W_{\text{visit}} = 4.0$$
   - Total cumulative visit progress reward bounded at $+4.0$ for 100% visited building.
3. **Preserve Training-Only Anti-Oscillation & Step Penalty:**
   - Step penalty: $-0.001$ per step.
   - Oscillation penalty: $-0.02$ for 2-cell oscillation.
4. **Evaluation Integrity:**
   - Evaluation on rescue benchmarks remains **strictly unshaped** (+10 native rescue only).
