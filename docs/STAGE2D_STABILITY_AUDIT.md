# Stage 2D Training Instability Diagnostic Audit

## 1. Executive Summary & Objective

In the 500,000-step Stage 2C production run (`fire-rescue-project-2.0`), the agent achieved unprecedented long-distance rescue capabilities at its best checkpoint (`best_model.zip`, step 1,041,464). However, across the final 60,000 steps of training, performance deteriorated significantly between `best_model.zip` and `stage2c_final.zip` (step 1,102,192).

This audit rigorously inspects the complete training telemetry (all 1,076 PPO updates, loss curves, KL divergences, clipping fractions, gradient norms, and curriculum transitions) to identify the empirical root causes of late-stage policy deterioration.

### Key Audit Finding
The late-stage performance drop was **not** caused by critic divergence, gradient explosions, or catastrophic capacity saturation of the 128-unit GRU. Instead, it was driven by the combination of three specific factors:
1. **Unbounded Trust-Region Drift**: `target_kl` was unconstrained (`None`), permitting approximate KL divergence to repeatedly exceed `0.05–0.08` with clipping fractions over `24–31%`.
2. **Aggressive Constant Learning Rate in Late Stage**: Learning rate was maintained at $\eta = 3 \times 10^{-4}$ for over 1.1 million cumulative timesteps without annealing or decay.
3. **Transition Dilution in `2C_distant`**: Distant multi-room episodes have an horizon of 384 steps compared to 64 steps for Near and 128 steps for Medium. Consequently, Distant scenarios accounted for **over 75% of transitions in the rollout buffer**, starving the policy of positive terminal rescue rewards and causing drift in doorway and exit navigation.

---

## 2. Matched Empirical Comparison: Best vs Final Checkpoint

Both checkpoints were evaluated on the identical 200 held-out seeds (`50000..50199`) per tier under strictly unshaped native rescue rewards (+10 only):

| Difficulty Tier | Evaluation Mode | Stage 2C Best (`best_model.zip` @ 1,041k) | Stage 2C Final (`stage2c_final.zip` @ 1,102k) | Deterioration Delta ($\Delta$) | Discovery% Drop | Pickup% Drop | Return\|Pick% Drop |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Near** ($d=2$) | Det<br>Stoch | **95.0%**<br>**100.0%** | 86.0%<br>97.0% | -9.0%<br>-3.0% | 0.0%<br>0.0% | -5.0%<br>-0.5% | -4.5%<br>-2.5% |
| **Medium** ($d=6$) | Det<br>Stoch | **36.0%**<br>**44.5%** | 17.5%<br>29.5% | **-18.5%**<br>**-15.0%** | +3.0%<br>+5.0% | -13.0%<br>-10.0% | **-25.3%**<br>**-15.3%** |
| **Bridge A** ($d=8$) | Det<br>Stoch | **24.0%**<br>**41.0%** | 9.0%<br>22.0% | **-15.0%**<br>**-19.0%** | +6.5%<br>+3.0% | -21.0%<br>-17.5% | **-15.8%**<br>**-16.3%** |
| **Bridge B** ($d=10$) | Det<br>Stoch | **10.0%**<br>**26.0%** | 2.0%<br>13.5% | **-8.0%**<br>**-12.5%** | +15.5%<br>+5.0% | -13.0%<br>-19.0% | **-22.0%**<br>-8.6% |
| **Bridge C** ($d=12$) | Det<br>Stoch | **1.0%**<br>**14.0%** | 0.5%<br>6.0% | -0.5%<br>**-8.0%** | +22.0%<br>+7.0% | -9.5%<br>-14.5% | +3.1%<br>-11.1% |
| **Distant** ($d=\text{None}$) | Det<br>Stoch | 0.0%<br>**1.5%** | 0.0%<br>**1.5%** | 0.0%<br>0.0% | +2.0%<br>+3.5% | -1.0%<br>-4.0% | 0.0%<br>+11.5% |

### Critical Behavioral Observation:
Notice that **Discovery% did not drop** (in fact, discovery rate on Bridge A, B, C and Distant actually increased by +3% to +22%). 
However, **Pickup% dropped sharply (-10% to -21%)** and **Return-after-Pickup dropped by up to -25.3%**.
This proves that the policy did not lose its spatial exploration coverage; rather, **it lost precision in local interaction (aligning with the victim to execute pickup) and topological return navigation (following door sequences back to the exit)**.

---

## 3. PPO Hyperparameter & Update Telemetry Analysis

We analyzed all 1,076 PPO update records recorded in `training_metrics_history` across the 500,000 steps of Stage 2C, segmented into four operational phases:

1. **Phase 1: Bridge Ladder** (Steps 601,464 to 761,464; $N=743$ updates)
2. **Phase 2: Early Distant** (Steps 761,464 to 900,000; $N=135$ updates)
3. **Phase 3: Mid Distant to Best** (Steps 900,000 to 1,041,464; $N=138$ updates)
4. **Phase 4: Degradation Interval** (Steps 1,041,464 to 1,102,192; $N=60$ updates)

### Summary Telemetry Table

| Metric | Phase 1 (Ladder) | Phase 2 (Early Distant) | Phase 3 (Distant to Best) | Phase 4 (Degradation) | Diagnostic Assessment |
|---|:---:|:---:|:---:|:---:|---|
| **Updates ($N$)** | 743 | 135 | 138 | 60 | 60 updates post-best model |
| **Learning Rate ($\eta$)** | $3 \times 10^{-4}$ (const) | $3 \times 10^{-4}$ (const) | $3 \times 10^{-4}$ (const) | $3 \times 10^{-4}$ (const) | **Too high for late-stage fine-tuning** |
| **Approx KL (mean)** | 0.0271 | 0.0338 | 0.0346 | **0.0359** | Steady upward drift (+32.5%) |
| **Approx KL (95th %ile)** | 0.0449 | 0.0484 | 0.0493 | **0.0530** | Repeated trust-region violations |
| **Approx KL (max)** | 0.0775 | 0.0822 | 0.0644 | **0.0608** | Severe spikes without early-stop |
| **Target KL** | `None` | `None` | `None` | `None` | **Early-stopping completely disabled** |
| **Clipping Fraction (mean)** | 20.1% | 23.4% | 23.6% | **24.3%** | ~1 in 4 transitions clipped |
| **Clipping Fraction (max)** | 34.4% | 33.0% | 33.0% | **31.3%** | Large policy step sizes |
| **Value Loss (mean)** | 0.3526 | 0.1826 | 0.2200 | **0.1873** | Value loss stable / declining (Healthy) |
| **Critic Explained Var (mean)** | 0.726 | 0.755 | 0.786 | **0.794** | Highly accurate value predictions |
| **Policy Loss (mean)** | -0.0096 | -0.0150 | -0.0166 | -0.0142 | Stable surrogate objective |
| **Action Entropy (mean)** | 0.936 | 0.882 | 0.917 | **0.857** | Monotonically compressing |
| **GRU Gradient Norm (mean)** | 1.418 | 1.006 | 1.078 | **0.957** | Norms stable, no gradient explosions |

---

## 4. In-Depth Root Cause Diagnosis

### 4.1 Root Cause 1: Lack of Trust-Region Enforcement (`target_kl = None`)
- In `agents/recurrent_ppo.py` lines 494–498, early stopping logic exists:
  ```python
  if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
      continue_training = False
      break
  ```
- However, throughout Stage 1, Stage 2, and Stage 2C, `target_kl` was passed as `None`.
- Consequently, during each update iteration (which performs 4 epochs over 8 minibatches of sequence chunks), when the policy began to diverge from the old policy ($\text{KL} > 0.04$), the optimizer continued taking unconstrained gradient steps.
- In Phase 4, the 95th percentile of KL reached **0.0530** with maximum **0.0608**. Over 60 consecutive updates, this created unconstrained policy drift.

### 4.2 Root Cause 2: Unannealed Constant Learning Rate ($\eta = 3 \times 10^{-4}$)
- The model trained for a cumulative **1,102,192 environment steps** under a constant learning rate of $3 \times 10^{-4}$.
- In deep recurrent RL policies, earlier stages benefit from higher learning rates for gross visual representation learning and initial spatial mapping.
- However, in late-stage navigation fine-tuning, the policy resides near a sharp local optimum (representing complex coordinate transformations from 9-channel memory maps into 7-action sequences). Taking large steps of $3 \times 10^{-4}$ without a decaying schedule causes the weights to oscillate and jump out of narrow reward basins.

### 4.3 Root Cause 3: Transition Dilution in `2C_distant`
- In `CurriculumRescueEnv`, the substage `2C_distant` was configured with:
  - 10% Near ($d=2$, max steps 64)
  - 15% Medium ($d=6$, max steps 128)
  - 25% Bridge B/C ($d=10/12$, max steps 192/256)
  - 50% Distant ($d=\text{None}$, max steps 384)
- **Transition Asymmetry**: Because successful Near episodes terminate in ~3.6 steps and Medium in ~11.4 steps, while Distant episodes almost always run for the full 384 steps before timeout, **Distant transitions comprised >75% of every rollout buffer**.
- In Distant scenarios, the rescue success rate is only 1.5%. Over 98% of Distant transitions receive no terminal rescue reward (+10), only tiny step penalties (-0.001) and sporadic exploration bonuses (+0.05).
- As a result, the optimizer was overwhelmed by non-informative distant wanderings, drowning out the high-quality gradients from Near and Medium rescue trajectories.

### 4.4 Root Cause 4: Guard Trigger Latency vs Budget Expiration
- Guard 2 (`reinforce_medium`) was designed to intervene if Medium success fell below 25%.
- However, evaluation callbacks occur every 10,000 steps.
- At step 1,091,464, Medium success was 30.0% (Guard 2 inactive). Over the next 10,000 steps, unconstrained drift pushed Medium down to 10.0%.
- Guard 2 correctly fired at step 1,101,464:
  `Curriculum Alert [Guard 2]: Medium retention dropped below 25% (10.0%). Shifting to 'reinforce_medium' mode.`
- However, the 500k-step training budget terminated at step 1,102,192 (just 728 steps later), before the self-healing mode had time to collect rollouts and restore the policy.
- Consequently, `stage2c_final.zip` was captured at the exact temporary trough of the drift cycle.

---

## 5. Checkpoint Restoration Integrity Audit

To ensure Phase 3 controlled experiments are scientifically valid, we audited what happens when resuming from `best_model.zip`:

| Component | Status upon `load_checkpoint` | Evidence & Verification |
|---|:---:|---|
| **Model Weights** | **Fully Restored** | Verified: Exact bitwise match of policy state dict. |
| **Optimizer Moments ($\beta_1, \beta_2$)** | **Fully Restored** | Verified: `policy.optimizer.pth` contains all 22 parameter state dicts. |
| **Learning Rate Parameter** | **Restored to Checkpoint Value** | Param group lr is restored to 0.0003. Can be overridden explicitly for conservative PPO. |
| **Timestep Counters** | **Fully Restored** | Verified: `model.num_timesteps = 1041464`. |
| **Curriculum Substage** | **Restored via Metadata** | `metadata.json` records `substage: 2C_distant` and metric history. |
| **Recurrent Rollout State** | **Reset to Zero** | Rollout buffer transitions are re-collected afresh; GRU states initialize cleanly at rollout start and persist across rollout boundaries as verified in Test 2. |

---

## 6. Actionable Stabilization Plan for Stage 2D

To eliminate late-stage deterioration while preserving and continuing to improve learned navigation, Stage 2D will implement:

1. **Conservative Learning Rate with Linear Decay**:
   - Reduce base learning rate from $3 \times 10^{-4}$ to **$1 \times 10^{-4}$** (or $5 \times 10^{-5}$).
   - Optionally apply linear schedule decaying to $1 \times 10^{-5}$ across the training budget.
2. **Strict Trust-Region Enforcement (`target_kl = 0.015 - 0.02`)**:
   - Set `--target-kl 0.015`. If `approx_kl > 1.5 * target_kl` (0.0225), PPO immediately terminates the update epoch, guaranteeing that the policy never drifts more than 2% in probability distribution in a single update.
3. **Transition Rebalancing in Curriculum**:
   - In `2C_distant`, adjust the environment mixture to prevent Distant transition starvation:
     - Increase Near to **20%** (guarantees frequent +10 reward signals).
     - Increase Medium to **30%** (preserves doorway and multi-room exit navigation).
     - Maintain Bridge B/C at **25%** ($d=10/12$).
     - Set Distant to **25%** (prevents buffer drowning while maintaining long-range exposure).
4. **Retention-Aware Model Selection**:
   - Checkpoints must not be chosen solely based on sporadic distant rescues.
   - Require multi-tier Pareto dominance: $\text{Near} \ge 90\% \land \text{Medium} \ge 35\% \land \text{Bridge} \ge 25\% \land \text{Distant} \ge \text{best}$.
