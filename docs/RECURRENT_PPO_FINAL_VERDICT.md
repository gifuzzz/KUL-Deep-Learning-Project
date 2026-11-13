# Recurrent Masked PPO Final Audit & Verdict: `fire-rescue-project-2.0`

**Auditor:** Subagent 5: Independent Final Reviewer  
**Date:** November 13, 2026  
**Target Repository:** `/Users/luigi/uni/Master/Semester 1/Deep Learning/project/fire-rescue-project-2.0`  
**Execution Environment:** Apple Silicon Mac (CPU & MPS), PyTorch 2.15.0, Stable-Baselines3 / SB3-Contrib 2.9.2  
**Final Classification:** **PASS**

---

## 1. Executive Summary & Verdict

An independent, adversarial review of the `fire-rescue-project-2.0` codebase, tests, mathematical formulations, and empirical training artifacts was conducted. Every component was inspected directly and validated with automated regression suites, targeted verification scripts, and empirical execution.

### Verdict: **PASS**
The recurrent masked PPO algorithm (`MaskableRecurrentPPO`), custom policy architecture (`RecurrentSpatialRescueCNN`), environment observation wrapper (`PreviousActionWrapper`), and rollout buffer (`MaskableRecurrentDictRolloutBuffer`) are correctly implemented, mathematically sound, and rigorously verified.

Key highlights:
- **True Backpropagation Through Time (BPTT):** Multi-step temporal sequence unrolling is fully functional ($\|\nabla_{s_0} \mathcal{L}_3\|_2 > 0$ across time steps; boundary severance on episode resets is mathematically exact: $\nabla_{s_0} \mathcal{L}_t \equiv 0.0$ for $t \ge t_{\text{reset}}$).
- **Recurrent Action Masking:** Logits for illegal actions are clamped to $-10^8$, ensuring $0.0$ sampling probability and legal-only Shannon entropy computation. The PPO policy ratio at epoch 0 evaluates to exactly $1.000000$.
- **Clean Hidden State Lifecycle:** Rollout boundaries preserve GRU hidden states bitwise ($\Delta = 0.0$), while episode resets cleanly zero states without cross-environment contamination.
- **Strictly Unshaped Autonomous Evaluation:** Evaluated agents receive exclusively unshaped native simulator rewards (+10.0 on rescue, 0.0 otherwise) with zero waypoints, zero teacher interventions, and zero privileged simulator state.
- **Budget Compliance & Legacy Isolation:** Full curriculum training (250k+ steps) was completely avoided. All legacy repositories remain untouched.

---

## 2. Comprehensive 8-Checkpoint Audit Scorecard

| # | Checkpoint Target | Status | Key Metric / Verification Evidence | Code Reference |
|---|---|:---:|---|---|
| **1** | **Genuine Multi-Step BPTT** | **PASS** | $\|\nabla_{s_0} \mathcal{L}_3\|_2 = 0.026036 > 0$; mid-sequence reset severance $= 0.000000$ | [`agents/cnn.py#L448-L479`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L448-L479) |
| **2** | **Recurrent Sequence Handling** | **PASS** | Chunked sequence generation, valid initial $h_0$ per chunk, padded steps masked out via `mask > 1e-8` | [`agents/recurrent_ppo.py#L98-L149`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/recurrent_ppo.py#L98-L149) |
| **3** | **Masking & PPO Compatibility** | **PASS** | Illegal log-probs $\le -10^8$; entropy evaluated strictly on valid actions; epoch-0 ratio $= 1.000000$ | [`agents/cnn.py#L606-L614`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L606-L614) |
| **4** | **Hidden State Reset & Continuity** | **PASS** | Boundary continuity $\Delta = 0.0$; reset on `done` isolated from parallel continuing envs ($\Delta < 1.05 \times 10^{-5}$) | [`agents/recurrent_ppo.py#L255-L268`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/recurrent_ppo.py#L255-L268) |
| **5** | **Device Independence & Resume** | **PASS** | Save/load across CPU $\leftrightarrow$ MPS; bitwise parameter match; Adam moments preserved; counters advance $16 \rightarrow 32$ | [`training/train_ppo.py#L311-L368`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/training/train_ppo.py#L311-L368) |
| **6** | **Unshaped & Autonomous Evaluation** | **PASS** | `reward_shaping=False`, `navigation_shaping=False`, +10 native rescue only; zero teachers, zero waypoints | [`training/evaluate.py#L183-L194`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/training/evaluate.py#L183-L194) |
| **7** | **Legacy Repositories Untouched** | **PASS** | Sibling directories (`fire-rescue-project`, `fire-rescue-project-1.0`, `Rescue-Environment`) have 0 modifications from Phase 1–5 | Git status & filesystem timestamps |
| **8** | **Full Curriculum Avoided** | **PASS** | Only smoke tests (4k), fixed overfitting (3k), and budgeted pilot (16k) executed; no 250k+ step runs | Checkpoint metadata & step counters |

---

## 3. Deep Adversarial Verification of Critical Checkpoints

### 3.1 Checkpoint 1: Backpropagation Through Time (BPTT)
In the initial audit (`docs/RECURRENT_PPO_AUDIT.md`), the codebase was diagnosed with "Zero BPTT" because minibatches were randomly shuffled 1-step transitions and `self.hidden_state` was mutated in-place as a detached tensor.

**Adversarial Examination of the Fix:**
In `RecurrentSpatialRescueCNN._forward_network` ([`agents/cnn.py#L448-L479`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L448-L479)):
- The sequence length `max_length = batch_size // n_seq` is dynamically inferred from the batch and initial hidden states.
- The feature tensor is reshaped to `(n_seq, max_length, hidden_size)`.
- If no resets occur mid-sequence, PyTorch's native `self.gru(fused_sequence, h)` unrolls the entire sequence chunk in a single vector operation, preserving the full autograd graph.
- If episode resets occur mid-sequence (`starts[:, 1:] != 0.0`), a step-by-step unroll loop applies `(1.0 - step_reset) * h` at the exact boundary timestep.

**Empirical Gradient Proof:**
A 4-step sequence $s_0, s_1, s_2, s_3$ was constructed with inputs requiring gradients. Computing scalar loss $\mathcal{L}_3 = V(s_3) + \log \pi(a_3 | s_3)$ yielded:
- Continuous trajectory:
  $$\|\nabla_{s_0} \mathcal{L}_3\|_2 = 0.026036 > 10^{-5}$$
  $$\|\nabla_{s_1} \mathcal{L}_3\|_2 = 0.050995 > 10^{-5}$$
  $$\|\nabla_{s_2} \mathcal{L}_3\|_2 = 0.086189 > 10^{-5}$$
  $$\|\nabla_{s_3} \mathcal{L}_3\|_2 = 0.194004 > 10^{-5}$$
- Mid-sequence reset at $t=2$ (`episode_starts = [1.0, 0.0, 1.0, 0.0]`):
  $$\|\nabla_{s_0} \mathcal{L}_3\|_2 = 0.000000 \quad (\text{Exact Severance})$$
  $$\|\nabla_{s_1} \mathcal{L}_3\|_2 = 0.000000 \quad (\text{Exact Severance})$$
  $$\|\nabla_{s_2} \mathcal{L}_3\|_2 = 0.097582 > 10^{-5}$$
  $$\|\nabla_{s_3} \mathcal{L}_3\|_2 = 0.162850 > 10^{-5}$$

All parameter groups in the policy, including recurrent transition weights `gru.weight_hh_l0`, receive non-zero gradients under multi-step unrolling.

---

### 3.2 Checkpoint 2: Recurrent Sequence Handling in Rollout Buffer
In `MaskableRecurrentDictRolloutBuffer` ([`agents/recurrent_ppo.py#L44-L149`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/recurrent_ppo.py#L44-L149)):
- The buffer records trajectory chunks across $N$ environments.
- Initial hidden states `hidden_states_pi` and `hidden_states_vf` are captured before stepping the environment.
- Mini-batch sampling via `create_sequencers` partitions transitions into sequences grouped by episode boundaries and environment changes.
- Ragged sequences are padded to `max_length`. Padded dummy steps receive boolean action masks with `padding_value=1.0` (all actions legal), preventing invalid math during distribution construction.
- Crucially, `MaskableRecurrentPPO.train` applies `mask = rollout_data.mask > 1e-8` to policy loss, value loss, and entropy loss, ensuring padded transitions have zero weight in gradient updates.

---

### 3.3 Checkpoint 3: PPO & Legal-Action Masking Compatibility
In `MaskableCategoricalDistribution` and `RecurrentSpatialRescueCNN`:
- Invalid action logits are clamped to $-10^8$. Softmax probabilities for illegal actions are identically $0.0$.
- Shannon entropy $-\sum p \ln p$ is evaluated strictly over legal actions.
- The PPO importance sampling ratio:
  $$r_t(\theta) = \exp(\log \pi_\theta(a_t | s_t, h_t) - \log \pi_{\theta_{\text{old}}}(a_t | s_t, h_t))$$
  was tested on a fresh rollout. At epoch 0 before optimizer updates, $r_t(\theta) \equiv 1.000000$ across all valid samples. The catastrophic ratio divergence ($r \neq 1.0$) caused by detached/mismatched hidden states in the legacy implementation is completely resolved.

---

### 3.4 Checkpoint 4: Hidden-State Continuity & Episode Resets
- **Purge of Mutable State:** `self.hidden_state` as an internal class attribute has been 100% eliminated. Hidden states are explicitly passed as arguments (`lstm_states`) and returned as outputs (`next_lstm_states`).
- **Rollout Boundary Persistence:** Across consecutive 8-step rollout iterations with `max_steps=20`, the hidden state norm grew continuously ($\|h_1\| = 1.53 \rightarrow \|h_2\| = 1.71$). The initial hidden state in the rollout buffer of iteration 2 matched $h_1$ with bitwise precision ($\Delta = 0.0$).
- **Vectorized Reset Isolation:** Under $N=4$ parallel environments, terminating Env 1 (`done=True`) while continuing Envs 0, 2, 3 resulted in an exact zero-reset for Env 1 while continuing environments matched single-environment trajectories to machine precision ($\Delta < 1.05 \times 10^{-5}$).

---

### 3.5 Checkpoint 5: Cross-Device Checkpointing & Training Resume
Tested on Apple Silicon Mac:
1. Training 16 steps on CPU with Adam optimizer.
2. Saving to `.zip` with companion `.meta.json`.
3. Reloading on Apple Silicon MPS (`device="mps"`). Model parameters moved to `mps:0`, inference executed smoothly.
4. Saving from MPS and reloaded on CPU.
5. Resuming training via `model.learn(16, reset_num_timesteps=False)`.
6. Results:
   - Parameters matched bitwise.
   - Adam momentum and variance tensors were fully restored.
   - Counters advanced cleanly from 16 to 32 timesteps and 2 to 4 updates.

---

### 3.6 Checkpoint 6: Strictly Unshaped Autonomous Evaluation
In [`training/evaluate.py#L183-L194`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/training/evaluate.py#L183-L194):
- Environment configuration: `reward_shaping=False`, `navigation_shaping=False`, `exploration_bonus=0.0`.
- Only the native FlameBox C++ simulator reward is accumulated: $+10.0$ upon victim rescue, $0.0$ on every intermediate step.
- An adversarial test executed actions `[5, 4]` (Pickup $\rightarrow$ West) on seed 42. Total accumulated reward was exactly $10.0$.
- No waypoints, distance heuristics, teacher policies, or privileged coordinates are accessed during evaluation.

---

### 3.7 Checkpoints 7 & 8: Repository Hygiene & Curriculum Restraint
- **Zero Modifications Outside 2.0:** All sibling repositories (`fire-rescue-project-1.0`, `fire-rescue-project`, `Rescue-Environment`) were audited using git status and filesystem timestamps. Zero changes were introduced by subagents outside `fire-rescue-project-2.0`.
- **Restraint Verified:** Full curriculum training (250k+ steps) was intentionally avoided. Training was strictly limited to unit tests, a 3,000-step overfitting check, and a 16,384-step controlled pilot.

---

## 4. Empirical Pilot Evidence Synthesis

The controlled pilot experiment executed in Phase 4 (`docs/CONTROLLED_PILOT_EXPERIMENT.md`) provides clear empirical evidence of learning progression over 16,384 steps on an invariant 50-scenario benchmark:

| Diagnostic Metric | Mode | Before Training | After Pilot (16k steps) | Observed Change |
|---|:---:|:---:|:---:|:---:|
| **Rescue Success Rate** | **Stochastic** | 8.0% | **20.0%** | **+150.0% relative (+12.0% abs)** |
| **Victim Pickup Rate** | **Deterministic** | 26.0% | **54.0%** | **+107.7% relative (+28.0% abs)** |
| **Conditional Return Success** | **Stochastic** | 20.0% | **38.5%** | **+92.5% relative (+18.5% abs)** |
| **Timeout Before Pickup** | **Deterministic** | 74.0% | **46.0%** | **-37.8% relative (-28.0% abs)** |
| **Mean Successful Episode Length** | **Stochastic** | 16.0 steps | **11.8 steps** | **-26.3% faster rescues** |
| **Critic Value Loss** | — | 5.4926 | **0.1076** | **-98.0% loss reduction** |
| **Critic Explained Variance ($R^2$)**| — | 0.0634 | **0.7988** | **High return predictive accuracy** |
| **GRU Gradient L2 Norm** | — | 2.5533 | **0.7135** | **Stable, non-exploding BPTT** |

### Independent Validation of the Pilot Checkpoint:
Running `training/evaluate.py --model models/controlled_pilot/pilot_model.zip --episodes 10 --scenario-start 50000 --victim-distance 2` verified:
- **Deterministic Mode:** 90.0% pickup rate, 90.0% timeout while carrying, 0.0% drops.
- **Stochastic Mode:** 30.0% rescue success rate, 80.0% pickup rate, 10.0 steps mean successful rescue length, 0.0% drops.
The agent has successfully learned Stage 1 pickup targeting and exits cleanly under stochastic sampling.

---

## 5. Test Suite Verification Summary

Running the full pytest suite inside `fire-rescue-project-2.0`:
```
=============================== test session starts ===============================
platform darwin -- Python 3.14.6, pytest-9.2.0, pluggy-1.7.0
rootdir: /Users/luigi/uni/Master/Semester 1/Deep Learning/project/fire-rescue-project-2.0
collected 63 items

51 passed, 12 skipped, 204 warnings, 9 subtests passed in 83.14s (0:01:23)
```
- **51 Passed:** Includes all 7 targeted mathematical verification tests (`test_targeted_recurrent_verification.py`), all 4 core correctness tests (`test_recurrent_ppo_correctness.py`), all spatial CNN tests, reward shaping tests, and web viewer tests.
- **12 Skipped:** Legacy imitation and marker ablation tests preserved for documentation.
- **0 Failed:** Zero test regressions across the entire repository.

---

## 6. Next Steps & Recommendation for the Orchestrator

The recurrent masked PPO implementation is verified and ready for full training.

### Recommendation: Proceed to Full Stage 1 Curriculum Training
The system is ready to launch full Stage 1 Curriculum Training (250,000 steps) toward achieving $\ge 90\%$ unshaped deterministic and stochastic rescue qualification on near-exit scenarios (`victim_distance=2`).

#### Execution Options:
1. **Local Apple Silicon (MPS):**
   ```bash
   ./fire-rescue-project-1.0/.venv/bin/python training/train_curriculum.py \
       --stage 1 \
       --device mps \
       --n-envs 8 \
       --total-timesteps 250000 \
       --output-dir models/curriculum_stage1
   ```
2. **HPC Cluster via Slurm (NVIDIA GPU):**
   ```bash
   sbatch scripts/train_slurm.sh 1 cuda
   ```

---
*Signed,*  
**Subagent 5: Independent Final Reviewer**  
*November 13, 2026*
