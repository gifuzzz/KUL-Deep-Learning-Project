# Architecture Capacity Audit Report: `fire-rescue-project-2.0`

**Auditor:** Subagent 1 — Architecture Capacity Auditor  
**Date:** November 14, 2026  
**Target Repository:** `/Users/luigi/uni/Master/Semester 1/Deep Learning/project/fire-rescue-project-2.0`  
**Execution Environment:** macOS, Python 3.14.6, PyTorch 2.15.0, Stable-Baselines3 2.9.2, sb3_contrib 2.9.2  
**Target Policy:** `RecurrentSpatialRescueCNN` in [`agents/cnn.py`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py)  
**Algorithm:** `MaskableRecurrentPPO` in [`agents/recurrent_ppo.py`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/recurrent_ppo.py)  
**Verification Status:** 15/15 Targeted Verification Unit Tests Passing  

---

## 1. Executive Summary

This audit rigorously evaluates the neural network architecture, parameter distribution, tensor transformations, recurrent state dynamics, and representational capacity of `RecurrentSpatialRescueCNN`.

### Key Findings:
1. **Total Trainable Parameters:** **1,431,168** parameters across 7 modules.
2. **Extreme Parameter Concentration:** The fusion layer (`self.fusion`) accounts for **1,297,024** parameters, which is exactly **90.627%** of the entire network's weights.
3. **Zero Non-Linear Depth in Fusion:** Despite consuming 90.6% of the network capacity, `self.fusion` is implemented as a single affine linear transformation (`nn.Linear(10132, 128)`) with **zero non-linear activation functions** (no ReLU, GELU, LayerNorm, or multi-layer depth).
4. **Severe Spatial Bottleneck:** Flattened unpooled spatial features from the local visual field ($1,800$) and global memory map ($8,320$), combined with agent state ($5$) and previous action one-hot ($7$), total **10,132 dimensions**. This vector is compressed directly down to **128 dimensions** in a single linear step—an instantaneous **79.16x spatial compression ratio** (or **162.16x** relative to the raw 20,757 observation values).
5. **Temporal Bottleneck over Long Horizons:** The recurrent core consists of a single-layer GRU with hidden dimension **128**. While 128 hidden units are demonstrably sufficient for short episodes (Near $d=2$: 3.15–4.3 steps, 100% success; Medium $d=6$: 7.3–20.7 steps, 80–85% success), it suffers representational exhaustion over 384-step search horizons in $40 \times 52$ buildings (Distant $d=\text{None}$: 0.0% rescue success, 85.5% oscillation frequency).
6. **Correct Dynamic Integration:** Recurrent state propagation (multi-step BPTT chunking, rollout persistence, and asynchronous environment reset zeroing), previous action one-hot injection, and legal-action masking are verified to be mathematically and behaviorally correct.

---

## 2. Complete Architectural Blueprint & Exact Formulas

The policy network `RecurrentSpatialRescueCNN` inherits from `RecurrentActorCriticPolicy` ([`agents/cnn.py#L170`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L170)). It processes a multi-modal observation dictionary containing:
- `"grid"`: Local egocentric visual grid ($9 \times 15 \times 15$).
- `"memory"`: Global allocentric remembered building grid ($9 \times 40 \times 52$).
- `"agent"`: Normalized agent status scalars ($5$).
- `"previous_action"`: Integer action index from the prior step ($0..6$).

```
Raw Observation:
├── local grid  (9, 15, 15)  ──> [Local CNN]  ──> (1800,)  ──┐
├── memory grid (9, 40, 52)  ──> [Memory CNN] ──> (8320,)  ──┤ Concatenation
├── agent state (5,)         ───────────────────> (5,)     ──┼─> (10132,)
└── prev_action (1,)         ──> [One-Hot(7)] ──> (7,)     ──┘      │
                                                                    ▼
                                                          [Fusion Linear Layer]
                                                          Linear(10132 -> 128)
                                                          (0 non-linear depth!)
                                                                    │
                                                                    ▼
                                                            (128,) bottleneck
                                                                    │
                                                                    ▼
                                                               [GRU(128, 128)]
                                                                    │
                                                                    ▼
                                                           GRU Output (128,)
                                                                   / \
                                                                  /   \
                                                                 ▼     ▼
                                                          Action Head  Value Head
                                                          Linear(128->7) Linear(128->1)
                                                          (with masks)
```

---

### A. Local Vision CNN Branch

- **Code Location:** [`agents/cnn.py#L270-L278`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L270-L278)
- **Input Dimension:** $(B, 9, 15, 15)$ where $B$ is batch size, 9 channels represent physical passability (N, E, S, W), obstacles, visibility mask, victims, exits, and agent center.
- **Design Intent:** Unpooled spatial convolution preserving exact cell coordinates without pooling.

#### Transformation Pipeline & Dimensional Formulas:
1. **`local_cnn[0]`:** `nn.Conv2d(in_channels=9, out_channels=32, kernel_size=3, stride=1, padding=1)`
   $$H_1 = \left\lfloor \frac{15 + 2(1) - 3}{1} \right\rfloor + 1 = 15, \quad W_1 = 15$$
   - Output Tensor Shape: $(B, 32, 15, 15)$
   - Parameter Count:
     - Weights: $32 \times 9 \times 3 \times 3 = 2,592$
     - Biases: $32$
     - Total: $2,624$
2. **`local_cnn[1]`:** `nn.ReLU()`
   - Output Tensor Shape: $(B, 32, 15, 15)$
   - Parameters: $0$
3. **`local_cnn[2]`:** `nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)`
   $$H_2 = \left\lfloor \frac{15 + 2(1) - 3}{1} \right\rfloor + 1 = 15, \quad W_2 = 15$$
   - Output Tensor Shape: $(B, 64, 15, 15)$
   - Parameter Count:
     - Weights: $64 \times 32 \times 3 \times 3 = 18,432$
     - Biases: $64$
     - Total: $18,496$
4. **`local_cnn[3]`:** `nn.ReLU()`
   - Output Tensor Shape: $(B, 64, 15, 15)$
   - Parameters: $0$
5. **`local_cnn[4]`:** `nn.Conv2d(in_channels=64, out_channels=8, kernel_size=1, stride=1, padding=0)`
   $$H_3 = \left\lfloor \frac{15 + 2(0) - 1}{1} \right\rfloor + 1 = 15, \quad W_3 = 15$$
   - Output Tensor Shape: $(B, 8, 15, 15)$
   - Parameter Count:
     - Weights: $8 \times 64 \times 1 \times 1 = 512$
     - Biases: $8$
     - Total: $520$
6. **`local_cnn[5]`:** `nn.Flatten()`
   $$D_{\text{local}} = 8 \times 15 \times 15 = 1,800$$
   - Output Tensor Shape: $(B, 1800)$
   - Parameters: $0$

**Local CNN Parameter Subtotal:** $2,624 + 18,496 + 520 = \mathbf{21,640}$ (1.512% of total).

---

### B. Global Memory Map CNN Branch

- **Code Location:** [`agents/cnn.py#L281-L291`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L281-L291)
- **Input Dimension:** $(B, 9, 40, 52)$ representing the accumulated building exploration map padded to max building dimensions ($40 \times 52$).
- **Design Intent:** "Zero MaxPool2d" architecture to avoid erasing narrow 1-cell doorways; spatial downsampling is achieved via a single stride-2 convolution.

#### Transformation Pipeline & Dimensional Formulas:
1. **`memory_cnn[0]`:** `nn.Conv2d(in_channels=9, out_channels=32, kernel_size=3, stride=1, padding=1)`
   $$H_1 = \left\lfloor \frac{40 + 2(1) - 3}{1} \right\rfloor + 1 = 40, \quad W_1 = \left\lfloor \frac{52 + 2(1) - 3}{1} \right\rfloor + 1 = 52$$
   - Output Tensor Shape: $(B, 32, 40, 52)$
   - Parameter Count:
     - Weights: $32 \times 9 \times 3 \times 3 = 2,592$
     - Biases: $32$
     - Total: $2,624$
2. **`memory_cnn[1]`:** `nn.ReLU()`
   - Output Tensor Shape: $(B, 32, 40, 52)$
   - Parameters: $0$
3. **`memory_cnn[2]`:** `nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)`
   $$H_2 = \left\lfloor \frac{40 + 2(1) - 3}{2} \right\rfloor + 1 = \left\lfloor \frac{39}{2} \right\rfloor + 1 = 19 + 1 = 20$$
   $$W_2 = \left\lfloor \frac{52 + 2(1) - 3}{2} \right\rfloor + 1 = \left\lfloor \frac{51}{2} \right\rfloor + 1 = 25 + 1 = 26$$
   - Output Tensor Shape: $(B, 32, 20, 26)$
   - Parameter Count:
     - Weights: $32 \times 32 \times 3 \times 3 = 9,216$
     - Biases: $32$
     - Total: $9,248$
4. **`memory_cnn[3]`:** `nn.ReLU()`
   - Output Tensor Shape: $(B, 32, 20, 26)$
   - Parameters: $0$
5. **`memory_cnn[4]`:** `nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0)`
   $$H_3 = 20, \quad W_3 = 26$$
   - Output Tensor Shape: $(B, 16, 20, 26)$
   - Parameter Count:
     - Weights: $16 \times 32 \times 1 \times 1 = 512$
     - Biases: $16$
     - Total: $528$
6. **`memory_cnn[5]`:** `nn.Flatten()`
   $$D_{\text{memory}} = 16 \times 20 \times 26 = 8,320$$
   - Output Tensor Shape: $(B, 8320)$
   - Parameters: $0$

**Memory CNN Parameter Subtotal:** $2,624 + 9,248 + 528 = \mathbf{12,400}$ (0.866% of total).

---

### C. Agent State & Previous Action Embeddings

1. **Agent State Vector:**
   - Source: `obs["agent"]` ([`agents/cnn.py#L425`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L425))
   - Dimension: $D_{\text{agent}} = 5$ scalars (normalized agent row, normalized agent col, health status, sim step scalar, carrying flag).
   - Parameters: $0$.
2. **Previous Action One-Hot Encoding:**
   - Source: `obs["previous_action"]` processed by `_encode_previous_action` ([`agents/cnn.py#L344-L373`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L344-L373)).
   - Transformation: If $a_{t-1} \in \{0, 1, 2, 3, 4, 5, 6\}$, it is mapped via `torch.nn.functional.one_hot` into a 7-dimensional float vector:
     $$\mathbf{e}_{a} \in \mathbb{R}^7, \quad e_i = \mathbb{I}(a_{t-1} = i)$$
   - Dimension: $D_{\text{action}} = 7$.
   - Parameters: $0$.

**Combined Multi-Modal Representation:**
$$D_{\text{combined}} = D_{\text{local}} + D_{\text{memory}} + D_{\text{agent}} + D_{\text{action}} = 1,800 + 8,320 + 5 + 7 = \mathbf{10,132}$$

---

### D. Fusion Bottleneck Layer

- **Code Location:** [`agents/cnn.py#L303`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L303) and forward pass at [`agents/cnn.py#L432`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L432)
- **Module:** `self.fusion = nn.Linear(10132, 128)`
- **Mathematical Form:**
  $$\mathbf{z}_{\text{fused}} = \mathbf{W}_{\text{fusion}} \mathbf{x}_{\text{combined}} + \mathbf{b}_{\text{fusion}}$$
  where $\mathbf{W}_{\text{fusion}} \in \mathbb{R}^{128 \times 10,132}$ and $\mathbf{b}_{\text{fusion}} \in \mathbb{R}^{128}$.
- **Activation Function:** **NONE**. The forward pass directly executes:
  ```python
  combined = torch.cat([local_features, memory_features, agent_features, prev_action_features], dim=-1)
  fused = self.fusion(combined)
  ```
  There is **zero non-linear activation** applied to `fused` before it is shaped into `fused_sequence` and injected into the GRU.
- **Parameter Count:**
  - Weights: $128 \times 10,132 = 1,296,896$
  - Biases: $128$
  - Total: $\mathbf{1,297,024}$ (90.627% of total).

---

### E. Recurrent Core (GRU)

- **Code Location:** [`agents/cnn.py#L306-L310`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L306-L310)
- **Module:** `self.gru = nn.GRU(input_size=128, hidden_size=128, batch_first=True)`
- **Gates:** Reset gate $\mathbf{r}_t$, Update gate $\mathbf{z}_t$, Candidate state $\mathbf{n}_t$:
  $$\mathbf{r}_t = \sigma(\mathbf{W}_{ir} \mathbf{x}_t + \mathbf{b}_{ir} + \mathbf{W}_{hr} \mathbf{h}_{t-1} + \mathbf{b}_{hr})$$
  $$\mathbf{z}_t = \sigma(\mathbf{W}_{iz} \mathbf{x}_t + \mathbf{b}_{iz} + \mathbf{W}_{hz} \mathbf{h}_{t-1} + \mathbf{b}_{hz})$$
  $$\mathbf{n}_t = \tanh(\mathbf{W}_{in} \mathbf{x}_t + \mathbf{b}_{in} + \mathbf{r}_t \odot (\mathbf{W}_{hn} \mathbf{h}_{t-1} + \mathbf{b}_{hn}))$$
  $$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{n}_t + \mathbf{z}_t \odot \mathbf{h}_{t-1}$$
- **Parameter Breakdown:**
  - `weight_ih_l0`: Shape $[3 \times 128, 128] = [384, 128] = 49,152$
  - `weight_hh_l0`: Shape $[3 \times 128, 128] = [384, 128] = 49,152$
  - `bias_ih_l0`: Shape $[384] = 384$
  - `bias_hh_l0`: Shape $[384] = 384$
  - Total GRU Parameters: $49,152 + 49,152 + 384 + 384 = \mathbf{99,072}$ (6.922% of total).

---

### F. Actor and Critic Heads

- **Code Location:** [`agents/cnn.py#L322-L325`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L322-L325)
- **Shared Latent:** Both policy and value functions branch directly off the exact same 128-dimensional output from the GRU: $\mathbf{h}_t \in \mathbb{R}^{128}$.
- **Actor Head (`action_net`):**
  - Architecture: `nn.Linear(128, 7)`
  - Parameter Count: Weights: $7 \times 128 = 896$, Biases: $7$. Total: $\mathbf{903}$ (0.063%).
- **Critic Head (`value_net`):**
  - Architecture: `nn.Linear(128, 1)`
  - Parameter Count: Weights: $1 \times 128 = 128$, Biases: $1$. Total: $\mathbf{129}$ (0.009%).

Neither head contains a separate multi-layer perception (MLP) trunk; they are 1-layer projections of the recurrent state.

---

### G. Master Parameter Breakdown Table

The table below reports the exact parameter counts verified via execution on the runtime environment:

| Module Name | PyTorch Class | Input Dim / Shape | Output Dim / Shape | Weight Shape | Bias Shape | Trainable Params | % of Total |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`local_cnn.0`** | `Conv2d` | $(B, 9, 15, 15)$ | $(B, 32, 15, 15)$ | $[32, 9, 3, 3]$ | $[32]$ | 2,624 | 0.183% |
| **`local_cnn.2`** | `Conv2d` | $(B, 32, 15, 15)$ | $(B, 64, 15, 15)$ | $[64, 32, 3, 3]$ | $[64]$ | 18,496 | 1.292% |
| **`local_cnn.4`** | `Conv2d` | $(B, 64, 15, 15)$ | $(B, 8, 15, 15)$ | $[8, 64, 1, 1]$ | $[8]$ | 520 | 0.036% |
| *Subtotal: Local CNN* | `Sequential` | $(B, 9, 15, 15)$ | $(B, 1800)$ | — | — | **21,640** | **1.512%** |
| **`memory_cnn.0`** | `Conv2d` | $(B, 9, 40, 52)$ | $(B, 32, 40, 52)$ | $[32, 9, 3, 3]$ | $[32]$ | 2,624 | 0.183% |
| **`memory_cnn.2`** | `Conv2d` (s=2) | $(B, 32, 40, 52)$ | $(B, 32, 20, 26)$ | $[32, 32, 3, 3]$ | $[32]$ | 9,248 | 0.646% |
| **`memory_cnn.4`** | `Conv2d` | $(B, 32, 20, 26)$ | $(B, 16, 20, 26)$ | $[16, 32, 1, 1]$ | $[16]$ | 528 | 0.037% |
| *Subtotal: Memory CNN* | `Sequential` | $(B, 9, 40, 52)$ | $(B, 8320)$ | — | — | **12,400** | **0.866%** |
| **`fusion`** | `Linear` | $(B, 10132)$ | $(B, 128)$ | $[128, 10132]$ | $[128]$ | **1,297,024** | **90.627%** |
| **`gru`** | `GRU` | $(N, L, 128)$ | $(N, L, 128)$ | $[384, 128] \times 2$ | $[384] \times 2$ | **99,072** | **6.922%** |
| **`action_net`** | `Linear` | $(B, 128)$ | $(B, 7)$ | $[7, 128]$ | $[7]$ | **903** | **0.063%** |
| **`value_net`** | `Linear` | $(B, 128)$ | $(B, 1)$ | $[1, 128]$ | $[1]$ | **129** | **0.009%** |
| **TOTAL** | `RecurrentSpatialRescueCNN` | — | — | — | — | **1,431,168** | **100.000%** |

---

## 3. Dynamic State & Dataflow Mechanisms

### A. Recurrent State Management

Recurrent state propagation, memory persistence, and reset handling in `fire-rescue-project-2.0` operate through a synchronized contract between `MaskableRecurrentPPO` ([`agents/recurrent_ppo.py`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/recurrent_ppo.py)) and `RecurrentSpatialRescueCNN` ([`agents/cnn.py`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py)):

#### 1. Representation Structure
The policy uses a GRU, requiring only hidden state $\mathbf{h} \in \mathbb{R}^{1 \times N_{\text{envs}} \times 128}$. To maintain interface compatibility with Stable-Baselines3's recurrent abstractions, hidden states are wrapped in an `RNNStates` named tuple:
```python
RNNStates(pi=(h, zeros), vf=(h, zeros))
```
where `zeros` serves as an inert placeholder for the unused LSTM cell state $\mathbf{c}$.

#### 2. Rollout Collection State Tracking
In `collect_rollouts` ([`agents/recurrent_ppo.py#L205-L269`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/recurrent_ppo.py#L205-L269)):
- The persistent rollout state `self._last_lstm_states` is copied at rollout start.
- Step transition: `actions, values, log_probs, lstm_states = self.policy(obs_tensor, lstm_states, episode_starts, action_masks=action_masks)`.
- Inside `_forward_network` (single-step mode, $L=1$):
  $$h_{\text{eff}} = (1.0 - \text{starts}[:, 0].\text{view}(1, N_{\text{seq}}, 1)) \odot h$$
  If environment $i$ terminated on the previous step (`dones[i] == True`), `starts[i] == 1.0`, strictly zeroing out the recurrent state for that specific environment while leaving other asynchronous environments unperturbed.
- The rollout buffer stores the pre-step hidden state `self._last_lstm_states` via `rollout_buffer.add(..., lstm_states=self._last_lstm_states)`.
- At rollout end, `self._last_lstm_states` is retained across the rollout boundary, guaranteeing continuity into subsequent rollout iterations.

#### 3. PPO Update & BPTT Unrolling
In `_get_samples` and `train` ([`agents/recurrent_ppo.py#L97-L149`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/recurrent_ppo.py#L97-L149)):
- Trajectories are partitioned into sequence chunks via `create_sequencers`.
- The initial hidden state at the start of each sequence chunk is retrieved from `rollout_buffer.hidden_states_pi`.
- During `evaluate_actions`, $L > 1$:
  - If a chunk contains no internal episode boundaries (`torch.all(starts[:, 1:] == 0.0)`), the initial state is masked once and processed in a single C++ batched call: `gru_out, next_h = self.gru(fused_sequence, h)` ([`agents/cnn.py#L466`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L466)).
  - If mid-chunk resets occur, the network unrolls step-by-step applying per-timestep resets: `h = (1.0 - step_reset) * h`.
- Full temporal gradients $\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_{t-k}}$ propagate backward through time across the full sequence chunk length (verified in `tests/test_targeted_recurrent_verification.py::test_test1_temporal_gradient_propagation`).

---

### B. Previous Action Feedback Pipeline

The previous action is provided to the network to enable the recurrent policy to sense its own inertia and break cyclic oscillation:
1. **Environment Level:** `PreviousActionWrapper` ([`rescue_env/wrappers.py#L10-L41`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/rescue_env/wrappers.py#L10-L41)) augments the observation space with key `"previous_action": Box(low=0, high=6, shape=(), dtype=np.int64)`.
   - On `reset()`: `obs["previous_action"] = np.int64(0)`.
   - On `step(action)`: records executed action and returns `obs["previous_action"] = np.int64(action)`.
2. **Policy Ingestion:** In `_encode_previous_action` ([`agents/cnn.py#L344-L373`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L344-L373)):
   - Handles scalars, numpy arrays, or pre-batched tensors.
   - Converts valid action indices $a \in [0..6]$ to 7-dimensional one-hot vectors via `torch.nn.functional.one_hot`.
   - Concatenates the 7-dim vector into the 10,132-dim combined vector.

---

### C. Legal-Action Masking Pipeline

FlameBox enforces strict physical rules (agents cannot walk through solid concrete, pick up non-existent victims, or pick up victims when hands are full):
1. **Mask Generation:** `RescueEnv.action_masks()` ([`rescue_env/envs/single_agent.py#L163-L197`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/rescue_env/envs/single_agent.py#L163-L197)):
   - Action 0 (STAY): always `True`.
   - Actions 1..4 (MOVE N, E, S, W): `True` iff tile conductance $\ge 0.5$ and no debris/closed walls.
   - Action 5 (PICKUP): `True` iff not carrying and an unrescued victim is within Manhattan distance $\le 1$.
   - Action 6 (RESCUE): evaluated at exit portals.
2. **Wrapper Propagation:** `PreviousActionWrapper.action_masks()` delegates directly to `env.action_masks()`.
3. **Rollout Masking:** `MaskableRecurrentPPO.collect_rollouts` queries `get_action_masks(env)` and passes it to `policy.forward(...)`.
4. **Logit Masking:** `distribution = self.action_dist.proba_distribution(action_logits=logits); distribution.apply_masking(action_masks)`. Masked action logits are replaced with $-10^8$, forcing $\pi(a_{\text{illegal}} | s) \equiv 0.0$.
5. **Buffer Storage & Training:** `MaskableRecurrentDictRolloutBuffer` stores masks in `self.action_masks` ($[B, N_{\text{envs}}, 7]$) and enforces identical masking during `evaluate_actions` so that policy loss, clipped ratios, and action entropy are evaluated strictly over legal actions.

---

## 4. Capacity & Compression Analysis

### A. Spatial Compression: 10,132 to 128 (79.16x)

The spatial feature extraction pipeline exhibits an extreme dimensionality bottleneck:
- **Local Grid Input:** $9 \times 15 \times 15 = 2,025$ floats.
  - Convolved to $8 \times 15 \times 15 = 1,800$ features (compression: $1.125\times$).
- **Memory Grid Input:** $9 \times 40 \times 52 = 18,720$ floats.
  - Convolved to $16 \times 20 \times 26 = 8,320$ features (compression: $2.25\times$).
- **Concatenated Spatial Vector:** $1,800 + 8,320 = 10,120$ spatial cells.

When combined with agent and action scalars ($10,132$ total), the network maps directly to $128$ dimensions:
$$\text{Spatial Compression Ratio} = \frac{10,132}{128} = \mathbf{79.156\times}$$

Relative to raw input observation values ($2,025 + 18,720 + 5 + 7 = 20,757$):
$$\text{Raw Observation Compression Ratio} = \frac{20,757}{128} = \mathbf{162.16\times}$$

#### Implications of Flat Spatial Vectors:
Because pooling was completely removed to preserve single-cell doorways (width = 1 cell), the memory CNN produces 8,320 distinct spatial activations. However, flattening this tensor discards 2D inductive bias. The subsequent linear layer must learn $128 \times 8,320 = 1,064,960$ independent weights to correlate any arbitrary memory cell $(r, c)$ with the 128-dimensional output. Without localized receptive fields or hierarchical pooling, learning global spatial navigation requires enormous sample efficiency.

---

### B. Analysis of the 90.6% Weight Concentration in a 0-Depth Linear Fusion

The parameter distribution of `RecurrentSpatialRescueCNN` is exceptionally skewed:
- **Feature Extractors (Local + Memory CNNs):** $34,040$ params (**2.38%**)
- **Recurrence & Heads (GRU + Actor + Critic):** $100,104$ params (**7.00%**)
- **Fusion Layer (`self.fusion`):** $1,297,024$ params (**90.63%**)

#### Why Did This Occur?
1. **Unpooled Dimensionality:** Preserving high spatial resolution ($15 \times 15$ local, $20 \times 26$ memory) forces a 10,132-element input vector.
2. **Single-Stage Projection:** Instead of a multi-layer bottleneck (e.g. $10,132 \to 1,024 \to 256 \to 128$), the architecture connects 10,132 directly to 128 in one matrix.
3. **Absence of Non-Linearity:** In `agents/cnn.py`:
   ```python
   self.fusion = nn.Linear(total_fusion_dim, self.hidden_size)
   ```
   Notice that while `SpatialRescueFeaturesExtractor` ([`agents/cnn.py#L738-L741`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L738-L741)) defined `fusion` as `nn.Sequential(nn.Linear(...), nn.ReLU())`, the actual production policy `RecurrentSpatialRescueCNN` ([`agents/cnn.py#L303`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L303)) instantiated `self.fusion = nn.Linear(...)` without any activation function!

#### Representational Limitation:
A linear layer $z = Wx + b$ is an affine transformation. It **cannot compute cross-feature products or non-linear intersections**. For example, determining whether an unvisited room in memory channel 5 is accessible via an open door visible in local channel 0 cannot be resolved by a linear combination; it requires a logical conjunction (non-linear activation). By omitting non-linear depth in fusion, **all non-linear feature interaction is deferred entirely to the GRU**.

---

### C. Temporal Information Bottleneck: 128-dim GRU for 384 Steps

A single 128-dimensional continuous vector $\mathbf{h}_t \in \mathbb{R}^{128}$ must simultaneously encode:
1. **Exploration Frontier:** Which parts of the $40 \times 52$ grid have been visited vs. remain unsearched.
2. **Spatial Topology:** Path history, dead-ends, and backtrack decision points.
3. **Target Tracking:** Victim position when detected.
4. **Exit Vector:** Direction and path back to the exit portal.
5. **Short-Term Momentum:** Recent action history to suppress 2-step oscillation.
6. **Task Phase:** Searching vs. carrying.

#### Information Capacity Over Episode Horizons:
- **Short Horizons ($T \le 16$, Near $d=2$):** Information load is minimal. The victim is visible in the local grid ($15 \times 15$); the agent only needs to execute 3–4 steps to reach and rescue. 128 units easily hold this state.
- **Medium Horizons ($T \le 64$, Medium $d=6$):** The victim is in an adjacent room. The agent explores 1–2 doorways, locates the victim in ~10–20 steps, and returns. 128 units remain adequate.
- **Distant Horizons ($T = 384$, Distant $d=\text{None}$):** Searching an unmapped $40 \times 52$ building requires traversing 10+ rooms across 200–350 steps.
  - In a 128-dimensional vector, the continuous state undergoes 384 successive matrix multiplications with tanh non-linearities:
    $$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t)$$
  - Without external memory addressing or spatial topological readouts, early trajectory details are overwritten by subsequent observations (state drift / catastrophic forgetting).
  - When the recurrent state saturates or loses its directional gradient, the policy defaults to high-entropy or oscillatory local behaviors.

---

## 5. Correlation with Empirical Stage 2 Results

The empirical performance recorded across held-out seeds (50000..50019) in `models/curriculum_stage2/stage2_final_eval.json` and `stage2_best_eval.json` directly reflects these architectural characteristics:

### Empirical Evaluation Summary Table

| Metric | Near ($d=2$, Horizon=64) | Medium ($d=6$, Horizon=128) | Distant ($d=\text{None}$, Horizon=384) |
| :--- | :---: | :---: | :---: |
| **Deterministic Rescue Success** | **100.0%** | **80.0%** *(85.0% in checkpoint)* | **0.0%** |
| **Stochastic Rescue Success** | **90.0% – 95.0%** | **75.0%** | **0.0%** |
| **Deterministic Discovery Rate** | **100.0%** | **100.0%** | **0.0%** |
| **Stochastic Discovery Rate** | **100.0%** | **95.0% – 100.0%** | **5.0%** |
| **Deterministic Pickup Rate** | **100.0%** | **80.0%** | **0.0%** |
| **Stochastic Pickup Rate** | **95.0% – 100.0%** | **70.0% – 75.0%** | **0.0% – 5.0%** |
| **Mean Successful Episode Length**| **3.15 – 4.30 steps** | **7.31 – 20.73 steps** | **N/A (0 rescues)** |
| **Deterministic Oscillation %** | **0.0% – 22.1%** | **61.0% – 74.6%** | **85.48%** |
| **Stochastic Oscillation %** | **17.0% – 27.9%** | **31.8% – 32.2%** | **40.44% – 42.47%** |

---

### A. The Near ($100\%$) and Medium ($80-85\%$) Regimes

1. **Near ($d=2$):**
   - The victim is located within Manhattan distance 2 of an exit.
   - The agent requires only $3.15$ steps on average to complete the rescue.
   - At this distance, the victim is directly within the local CNN's $15 \times 15$ receptive field (`local_grid[6] > 0`).
   - The policy operates purely as a local reactive visual servomechanism. Memory and recurrence are barely exercised. Success is a robust **100%**.
2. **Medium ($d=6$):**
   - The victim is positioned 6 steps away (typically in the adjacent hallway or next room).
   - Discovery rate is **100%** deterministic.
   - Mean episode length is $7.3$ steps deterministic, $20.7$ steps stochastic.
   - The 128-dim GRU successfully retains the exit vector and navigates through 1 doorway, achieving **80.0% – 85.0%** rescue success.

---

### B. The Distant ($0-5\%$) Collapse & The Oscillation Trap

In the Distant regime ($d=\text{None}$):
- Victims are placed anywhere in the $40 \times 52$ building (distances often 25–60 steps through labyrinthine walls).
- Discovery rate collapses to **0.0%** (deterministic) and **5.0%** (stochastic).
- Rescues collapse to **0.0%**.
- Deterministic Oscillation Frequency explodes to **85.48%**.

#### Why Does the Policy Oscillate?
1. **Loss of Search Gradient:** In distant scenarios, neither the local field nor the memory map contains an observed victim at $t=0$. The potential reward gradient $\Phi(s)$ provides no directional bias towards an unobserved victim.
2. **Representation Collapse:** The 10,132-dim memory map is flattened and linearly compressed into 128 dimensions without non-linear feature crossing. The network cannot extract a clean "frontier direction vector" from the 8,320 memory cells.
3. **The Local Minimum of Action Masking:**
   - Legal action masks prevent walking into walls.
   - Moving back and forth between two open floor tiles (e.g., Action 1 North, Action 3 South) is always legal.
   - Because the 128-dim GRU saturates after 30–50 steps of random wandering without finding a victim, its state becomes invariant to time.
   - Under a stationary hidden state, the deterministic policy outputs the same action pair in a periodic cycle:
     $$s_t \to a_1 (\text{North}) \to s_{t+1} \to a_3 (\text{South}) \to s_t \to a_1 (\text{North})$$
   - This produces an **85.48% oscillation rate**, trapping the agent in a 2-tile loop until the 384-step timeout expires.

---

### C. Grounded Epistemic Breakdown: Facts vs. Hypotheses

To maintain strict scientific rigor, we differentiate verified empirical facts from causal hypotheses:

```
┌────────────────────────────────────────────────────────────────────────┐
│                          VERIFIED FACTS                                │
├────────────────────────────────────────────────────────────────────────┤
│ 1. fusion is 1,297,024 params (90.627% of total 1,431,168).           │
│ 2. fusion has 0 non-linear depth (single nn.Linear without activation).│
│ 3. Spatial compression ratio is 79.16x (10,132 -> 128).                │
│ 4. GRU hidden state is 128 units unrolled over 384 steps in Distant.   │
│ 5. Near rescues: 100.0% (mean length: 3.15 steps).                     │
│ 6. Medium rescues: 80.0% - 85.0% (mean length: 7.3 - 20.7 steps).      │
│ 7. Distant rescues: 0.0% (discovery: 0-5%, oscillation: 85.48%).      │
└────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        PLAUSIBLE HYPOTHESES                            │
├────────────────────────────────────────────────────────────────────────┤
│ Hypothesis A (Spatial Capacity Bottleneck):                            │
│ The 79x linear projection without non-linearity cannot synthesize      │
│ multi-channel spatial relationships between memory and agent position, │
│ preventing the policy from forming a coherent global exploration plan. │
│                                                                        │
│ Hypothesis B (Temporal Capacity Bottleneck):                           │
│ 128 continuous GRU units cannot maintain an active exploration map    │
│ of a 40x52 building across 384 steps, resulting in memory erasure.     │
│                                                                        │
│ Hypothesis C (Exploration / Credit Assignment Confound):               │
│ In Distant mode, with victim unobserved at start, unguided PPO lacks   │
│ exploration incentives; even an infinite-capacity network would        │
│ struggle without intrinsic curiosity, frontier bonuses, or curriculum. │
└────────────────────────────────────────────────────────────────────────┘
```

**Assessment:**  
While Hypothesis C (sparse exploration reward on hidden victims) is a known RL challenge, Hypotheses A and B are structural hardware limitations of the network. A network where 90.6% of parameters are locked in a zero-depth linear matrix multiplication is severely compromised in its ability to extract non-linear spatial navigation cues from an 8,320-element memory map.

---

## 6. Checkpoint & Model Metadata Audit

We inspected all serialized model weights in `models/` to verify architectural consistency and metadata tracking:

### Verified Checkpoints:
1. `models/curriculum_stage1/stage1/stage1_qualified.zip`:
   - Policy: `RecurrentSpatialRescueCNN`
   - Total Parameters: **1,431,168**
   - Training timesteps: 100,720
   - Qualification Gate: `PASSED` (100% near deterministic rescue)
   - Status: Immutable baseline checkpoint.
2. `models/controlled_pilot_stage2/stage2/stage2_final.zip`:
   - Policy: `RecurrentSpatialRescueCNN`
   - Total Parameters: **1,431,168**
   - Training timesteps: 117,104
   - Near retention: 100.0%, Medium rescue: 25.0%, Medium pickup: 45.0%.
3. `models/curriculum_stage2/stage2/stage2_final.zip` & `latest_model.zip`:
   - Policy: `RecurrentSpatialRescueCNN`
   - Total Parameters: **1,431,168**
   - Training timesteps: 601,456 (eval at 600,728)
   - Substage: `2B`
   - Near retention: 90.0% – 100.0%
   - Medium rescue: 80.0% – 85.0%
   - Distant rescue: 0.0% (pickup: 0.0% – 5.0%)

All checkpoints share the exact same parameter count (1,431,168) and tensor shapes.

---

## 7. Actionable Architectural Recommendations for Phase 2+

Based on the quantitative findings of this audit, we outline concrete architectural modifications for consideration in future phases:

### 1. Introduce Non-Linear Depth to the Fusion Trunk
- **Current Defect:** Single `nn.Linear(10132, 128)` with 0 activations.
- **Remedy:** Replace the single linear layer with a 2-stage non-linear projection or MLP:
  $$\mathbf{x} \in \mathbb{R}^{10,132} \xrightarrow{\text{Linear}} \mathbb{R}^{512} \xrightarrow{\text{LayerNorm / ReLU}} \mathbb{R}^{512} \xrightarrow{\text{Linear}} \mathbb{R}^{256} \xrightarrow{\text{ReLU}} \mathbb{R}^{128}$$
  This provides non-linear feature crossing between memory and local vision before bottlenecking into the recurrent core.

### 2. Spatial Spatial Pooling / Conv-Bottleneck on Memory Branch
- **Current Defect:** `memory_cnn` flattens $16 \times 20 \times 26 = 8,320$ values directly, discarding 2D locality.
- **Remedy:** Add an additional stride-2 convolution or spatial cross-attention layer to reduce the memory feature map from $20 \times 26$ to $10 \times 13$ or $5 \times 7$ before flattening, cutting memory features from 8,320 to ~800 while preserving topological structure.

### 3. Scale GRU Hidden Dimension (128 -> 256)
- **Current Defect:** 128 hidden units must remember visited rooms across 384 steps.
- **Remedy:** Increase `hidden_size` to 256. At `hidden_size = 256`, GRU parameters increase from 99,072 to 296,448, which is easily accommodated within compute budgets.

### 4. Separate Actor and Critic MLP Trunks
- **Current Defect:** `action_net` and `value_net` are single linear layers reading the identical 128-dim GRU output. Value estimation interfering with policy gradient updates is a known failure mode in shared recurrent backbones.
- **Remedy:** Add separate 64-dim MLP heads (`nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output))`) for the actor and critic.

---

## 8. Final Audit Verdict

- **Architectural Correctness:** **PASS**. The pipeline correctly implements recurrent masked PPO with multi-step BPTT, asynchronous environment reset isolation, previous action one-hot injection, and legal-action logit masking (15/15 unit tests pass).
- **Capacity & Bottleneck Assessment:** **STRUCTURALLY CONSTRAINED**. The 79x zero-depth linear fusion bottleneck (absorbing 90.6% of network weights) and 128-dimensional GRU provide sufficient capacity for near ($d=2$, 100%) and medium ($d=6$, 80–85%) tasks, but impose severe representational barriers on distant unguided search ($d=\text{None}$, 0%), manifesting as an 85.5% periodic oscillation failure mode.
