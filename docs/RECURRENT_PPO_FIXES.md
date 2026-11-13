# Recurrent PPO Correctness Fixes and Architectural Specification

**Date:** November 13, 2026  
**Project:** `fire-rescue-project-2.0`  
**Status:** Completed and Verified  
**Target Environment:** Gymnasium + Stable-Baselines3 / SB3-Contrib (PyTorch)  
**Hardware Platforms:** Apple Silicon (CPU / MPS) & HPC Clusters (NVIDIA CUDA via Slurm)  

---

## 1. Executive Summary

A comprehensive architectural audit (`docs/RECURRENT_PPO_AUDIT.md`) of `fire-rescue-project-2.0` identified critical failure modes preventing effective temporal credit assignment and policy optimization:
1. **Broken Recurrent Rollout & Gradient Flow (Zero BPTT):** Standard PPO buffers flattened multi-step temporal transitions into isolated steps ($L=1$). State transitions in the policy relied on a mutable instance attribute (`self.hidden_state`), completely severing the PyTorch autograd computation graph across consecutive steps ($\nabla_{s_t} \mathcal{L}_{t+1} = 0$).
2. **Incompatible Masking and Recurrence Libraries:** Stable-Baselines3 separates recurrent policies (`sb3-contrib.RecurrentPPO`) from masked discrete policies (`sb3-contrib.MaskablePPO`). No native algorithm supported simultaneous recurrent sequence chunking and dynamic action masking.
3. **Missing Previous Action Tracking:** The policy required `previous_action` in the observation space, but the environment did not track past action indices.
4. **Hardware Portability & Evaluation Divergence:** Checkpoints contained hardware-bound state dictionaries, and evaluation scripts lacked recurrent state propagation and proper observation wrappers.

All issues have been resolved by implementing a unified, mathematically rigorous recurrent masked RL pipeline. All verification benchmarks, regression suites, and training runs pass with zero regressions (44 passed, 12 skipped, 0 failed).

---

## 2. Core Architecture: `MaskableRecurrentPPO` & `MaskableRecurrentDictRolloutBuffer`

### 2.1 The Need for a Unified Algorithm
In `fire-rescue-project-2.0`, the rescue agent must:
- Navigate a partially observable environment with a 9-channel local window, 9-channel remembered canvas, and scalar agent status.
- Maintain memory across long horizons via a 128-unit GRU cell.
- Obey spatial and physical rules (walls, carrying capacity, water reserves) strictly enforced via 7-dimensional discrete action masks.
- Optimize policy and value networks with Backpropagation Through Time (BPTT).

Standard `sb3-contrib.RecurrentPPO` does not support action masks (it uses standard `CategoricalDistribution`). Standard `sb3-contrib.MaskablePPO` does not support recurrent buffers or sequence chunking.

We designed and implemented `MaskableRecurrentPPO` in [`agents/recurrent_ppo.py`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/recurrent_ppo.py), extending `sb3_contrib.RecurrentPPO` and subclassing `MaskableRecurrentDictRolloutBuffer` from `RecurrentDictRolloutBuffer`.

### 2.2 Buffer Architecture (`MaskableRecurrentDictRolloutBuffer`)
The rollout buffer stores trajectory chunks of length `n_steps` across `n_envs`:
- **Action Mask Storage:** Stores boolean action masks of shape `(buffer_size, n_envs, 7)`.
- **Initial Hidden State Tracking:** Captures initial GRU states `(n_layers=1, n_envs, hidden_dim=128)` at the start of each rollout segment.
- **Bootstrapping on Truncated Horizons:** Handles time-limit truncation properly by computing terminal value bootstrapping $V(s_{last})$ rather than treating timeouts as terminal zero-value states.
- **Padded Sequence Generator:** Generates mini-batch chunks grouped by environment sequences, padding ragged ends with mask vectors and setting `episode_starts` flags to reset recurrent states cleanly at episode boundaries.

```python
class MaskableRecurrentDictRolloutBufferSamples(NamedTuple):
    observations: Dict[str, torch.Tensor]
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    lstm_states: RNNStates
    episode_starts: torch.Tensor
    mask: torch.Tensor
    action_masks: torch.Tensor  # Dynamic action masks (batch_size, 7)
```

### 2.3 Loss Formulation with Action Masking
During the PPO update epoch, mini-batch sequence chunks $(s_{t:t+L}, a_{t:t+L}, m_{t:t+L})$ are unrolled through the policy:
$$\mathcal{L}_{PPO}(\theta) = \hat{\mathbb{E}}_{t} \left[ \mathcal{L}_{CLIP}(\theta) - c_1 \mathcal{L}_{VF}(\theta) + c_2 S[\pi_\theta](s_t, m_t) \right]$$

Where:
- **Masked Policy Ratio:**
  $$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t, m_t)}{\pi_{\theta_{old}}(a_t \mid s_t, m_t)}$$
- **Masked Categorical Distribution:** For logits $z \in \mathbb{R}^7$ and boolean mask $m \in \{0, 1\}^7$:
  $$z'_i = \begin{cases} z_i & \text{if } m_i = 1 \\ -\infty & \text{if } m_i = 0 \end{cases}$$
  $$p_i = \frac{e^{z'_i}}{\sum_{j: m_j=1} e^{z'_j}}$$
- **Restricted Entropy:** Entropy $S[\pi_\theta](s_t, m_t) = -\sum_{i: m_i=1} p_i \ln p_i$ is evaluated strictly over the support of valid actions, preventing entropy bonuses from promoting illegal actions.

---

## 3. Policy Architecture: `RecurrentSpatialRescueCNN`

Located in [`agents/cnn.py`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py).

### 3.1 Elimination of Mutable Instance Side-Effects
In the previous codebase, `self.hidden_state` was mutated in-place during `.forward()` or `.predict()`:
```python
# BROKEN (Previous):
self.hidden_state = new_hidden_state.detach()  # Severed computation graph!
```
This prohibited multi-step unrolling during backward passes and caused silent race conditions between environment stepping and training mini-batches.

**The Fix:**
- Completely removed `self.hidden_state`.
- Policy methods (`forward`, `evaluate_actions`, `get_distribution`, `predict_values`, `predict`) now explicitly accept and return `(lstm_states, episode_starts)`.
- Follows the functional recurrent contract of PyTorch:
  $$h_{t} = \text{GRUCell}(x_t, h_{t-1} \cdot (1 - d_t))$$

### 3.2 Dual Class Hierarchy via ABC Registration
SB3's type checks enforce:
- `RecurrentPPO` requires `isinstance(policy, RecurrentActorCriticPolicy)`.
- `MaskablePPO` utilities require `isinstance(policy, MaskableActorCriticPolicy)`.

Multiple inheritance (`class RecurrentSpatialRescueCNN(RecurrentActorCriticPolicy, MaskableActorCriticPolicy)`) caused severe MRO initialization conflicts (`TypeError: got unexpected keyword argument 'squash_output'`).

**Solution:** Subclass `RecurrentActorCriticPolicy` directly, and register with Python's Abstract Base Class mechanism:
```python
class RecurrentSpatialRescueCNN(RecurrentActorCriticPolicy):
    ...

# Register as virtual subclass of MaskableActorCriticPolicy without MRO conflicts
MaskableActorCriticPolicy.register(RecurrentSpatialRescueCNN)
```
This guarantees `isinstance(policy, RecurrentActorCriticPolicy)` and `isinstance(policy, MaskableActorCriticPolicy)` both evaluate to `True`.

### 3.3 Multi-Step Temporal Sequence Unrolling and Autograd BPTT
In `_forward_network(obs, lstm_states, episode_starts)`:
1. Feature extraction maps spatial and scalar inputs to flat representation vectors $x \in \mathbb{R}^{B \times 128}$.
2. If $B = N_{\text{seq}} \times L$ with sequence length $L > 1$, the feature tensor is reshaped to $(L, N_{\text{seq}}, 128)$.
3. Sequences with episode reset flags ($d_t = 1$) reset hidden states at the exact timestep boundary.
4. The GRU unrolls over time:
   $$h_{1:L}, h_L = \text{GRU}(x_{1:L}, h_0)$$
5. The full unrolled computational graph is retained, allowing gradients to flow backward through time:
   $$\frac{\partial \mathcal{L}_{t+1}}{\partial s_t} = \frac{\partial \mathcal{L}_{t+1}}{\partial h_{t+1}} \frac{\partial h_{t+1}}{\partial h_t} \frac{\partial h_t}{\partial s_t} \neq 0$$

---

## 4. Observation Pipeline: `PreviousActionWrapper`

Located in [`rescue_env/wrappers.py`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/rescue_env/wrappers.py).

### 4.1 Mechanics and Contract
`PreviousActionWrapper(gym.Wrapper)` augments the environment observation dict:
- Adds `'previous_action': gym.spaces.Discrete(7)` to `observation_space.spaces`.
- **At `reset()`:** Initializes `previous_action = 0` (NOOP).
- **At `step(action)`:** 
  1. Executes `action` in the underlying environment.
  2. Injects the executed `action` into the returned `obs["previous_action"]`.
  3. Records `previous_action` into `info["terminal_observation"]` upon episode termination.
  4. Stores the executed action for the next step.
- **Action Masking Pass-Through:** Delegates `action_masks()` directly to `self.env.action_masks()`.
- **Attribute Proxying:** Defines `__getattr__(self, name)` to ensure underlying environment attributes (e.g., `env.sim`, `env.agent_pos`, `env.victim_pos`) remain accessible without wrapper peeling.

### 4.2 Complete Observation Specification
| Key | Shape | Type | Description |
| :--- | :--- | :--- | :--- |
| `local_map` | `(9, 7, 7)` | `float32` | 7x7 ego-centric spatial crop with 9 semantic channels |
| `remembered_map` | `(9, 32, 32)` | `float32` | Global canvas memory with 9 semantic channels |
| `agent_status` | `(5,)` | `float32` | Carrying victim, water level, step fraction, etc. |
| `previous_action` | `()` (Discrete) | `int64` | Action index $a_{t-1} \in \{0, \dots, 6\}$ |

---

## 5. Hardware Compatibility & HPC Integration

### 5.1 Device-Independent Checkpointing
To prevent CUDA-bound deserialization errors when moving checkpoints between HPC nodes and local development machines (e.g. `RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda is not available`):
1. Checkpoint loading specifies `custom_objects={"device": device}` to map tensors cleanly to CPU, MPS, or CUDA.
2. Metadata is archived directly inside the checkpoint ZIP (`metadata.json`) and mirrored in a human-readable companion file (`<checkpoint>.meta.json`).
3. Captured metadata attributes:
   - `stage`: Curriculum stage index.
   - `total_timesteps`: Cumulative training steps.
   - `device`: Training hardware used.
   - `git_commit`: Active git hash for reproducibility.
   - `hyperparameters`: Learning rate, batch size, n_steps, clip range, etc.

### 5.2 Standalone HPC Runner (`scripts/train_hpc.sh`)
An executable shell script for bare-metal HPC nodes or interactive GPU allocations:
- Automatically detects Python environment (`.venv` or conda).
- Inspects PyTorch CUDA availability and GPU model (`nvidia-smi`).
- Configures thread count (`OMP_NUM_THREADS=4`) to prevent CPU core over-subscription.
- Auto-generates log and checkpoint directories.

```bash
chmod +x scripts/train_hpc.sh
./scripts/train_hpc.sh --stage 1 --device cuda
```

### 5.3 Slurm Batch Submission (`scripts/train_slurm.sh`)
An executable submission script with Slurm resource directives:
- Requests GPU partition with 1 GPU, 4 CPU cores, and 32GB RAM.
- Exports execution logs to `slurm_logs/train_%j.log`.
- Executes curriculum training cleanly across batch queues.

```bash
sbatch scripts/train_slurm.sh 1 cuda
```

---

## 6. Evaluation Framework Alignment (`training/evaluate.py`)

The evaluation script has been updated to support both recurrent and feedforward policies:
- Detects policy recurrence and initializes `(lstm_states, episode_starts)`.
- Automatically wraps the evaluation environment in `PreviousActionWrapper`.
- Evaluates policy deterministically using strictly unshaped native rewards (+10.0 for successful rescue, 0.0 otherwise).
- Outputs comprehensive rescue diagnostics:
  - Success rate
  - Pickup rate
  - Rescue rate after pickup
  - Timeout rate without pickup / while carrying
  - Drop rates and mean successful episode lengths

---

## 7. Verification Results

All correctness criteria specified in the audit were implemented as automated unit tests in [`tests/test_recurrent_ppo_correctness.py`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/tests/test_recurrent_ppo_correctness.py):

| Test Property | Criterion | Result | Evidence / Metric |
| :--- | :--- | :--- | :--- |
| **Property A: BPTT Gradient Flow** | Temporal gradient $\nabla_{s_t} \mathcal{L}_{t+1} \neq 0$ across consecutive steps | **PASSED** | Grad norm = **$0.0856 > 0$** |
| **Property B: Hidden State Propagation** | Rollout boundaries maintain continuity; episode resets clear hidden states | **PASSED** | State carried across iterations; reset on `done` |
| **Property C: Action Masking Enforcement** | Illegal actions have probability $0$; masked entropy equals legal entropy | **PASSED** | Log-prob of masked action $= -\infty$; $p_{illegal} = 0.0$ |
| **Property D: Checkpoint Round-Trip** | Model saves and reloads across CPU/MPS/CUDA with identical outputs and metadata | **PASSED** | Action logits difference $= 0.0$; metadata preserved |

### Full Test Suite Status
Running `pytest tests/ -v`:
```
======================= 44 passed, 12 skipped, 204 warnings in 13.24s =======================
```
- **Zero regressions** across existing environment, reward shaping, spatial CNN, and web viewer tests.
- Training dry-run (`training/train_ppo.py`) runs iterations, logs value/policy losses, and saves checkpoints cleanly.
- Evaluation run (`training/evaluate.py`) executes end-to-end with zero errors.

---

## 8. Summary of Modified and Created Files

- **`agents/recurrent_ppo.py`** *(New)*: Implements `MaskableRecurrentDictRolloutBufferSamples`, `MaskableRecurrentDictRolloutBuffer`, and `MaskableRecurrentPPO`.
- **`agents/cnn.py`** *(Updated)*: Refactored `RecurrentSpatialRescueCNN` to remove mutable attributes, implement multi-step sequence unrolling BPTT, register as a virtual subclass of `MaskableActorCriticPolicy`, and support masked distributions.
- **`rescue_env/wrappers.py`** *(New)*: Implements `PreviousActionWrapper` with Gymnasium attribute delegation.
- **`rescue_env/__init__.py`** *(Updated)*: Exports `PreviousActionWrapper`.
- **`training/train_ppo.py`** *(Updated)*: Uses `MaskableRecurrentPPO`, wraps environments in `PreviousActionWrapper`, adds CUDA support, and saves multi-format metadata.
- **`training/evaluate.py`** *(Updated)*: Supports recurrent model loading, wraps environments in `PreviousActionWrapper`, and tracks unshaped rescue metrics.
- **`scripts/train_hpc.sh`** *(New)*: Executable entrypoint for HPC environments.
- **`scripts/train_slurm.sh`** *(New)*: Slurm submission script for cluster training.
- **`tests/test_recurrent_ppo_correctness.py`** *(New)*: Verification test suite for properties A through D.
- **`docs/RECURRENT_PPO_FIXES.md`** *(New)*: This technical specification document.
