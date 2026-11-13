# Recurrent PPO Implementation Audit Report: `fire-rescue-project-2.0`

**Auditor:** Subagent 1 (Recurrent PPO Implementation Auditor)  
**Date:** November 13, 2026  
**Target Repository:** `/Users/luigi/uni/Master/Semester 1/Deep Learning/project/fire-rescue-project-2.0`  
**Execution Environment:** PyTorch 2.15.0, Stable-Baselines3 2.9.2, sb3_contrib 2.9.2 (Python 3.14)

---

## 1. Executive Summary & Audit Verdict

### Primary Question Answer
> **Is the 128-unit GRU actually being trained correctly through time using PPO with legal-action masking, or is the current implementation merely a feed-forward PPO policy containing a GRU?**

**Verdict: THE IMPLEMENTATION IS MERELY A FEED-FORWARD PPO POLICY CONTAINING A GRU.**  
The policy **does not undergo Backpropagation Through Time (BPTT)**. The GRU operates strictly over **isolated 1-step transitions (`seq_len = 1`)** during policy updates. Hidden states are **never stored in the rollout buffer**, minibatches are **randomly permuted individual timesteps**, and temporal gradients between consecutive environment steps are **identically zero ($\nabla_{s_t} \mathcal{L}_{t+1} \equiv 0$)**.

Furthermore, hidden state propagation during rollout collection is severely corrupted: hidden states **leak across episode terminations** within rollouts, yet are **wiped to zero at arbitrary rollout boundaries** due to batch-size shape mismatches. The policy also attempts to condition on a `previous_action` feature that is **never provided by the environment**, rendering that input permanently zero.

While legal-action masking is correctly applied in isolation (logits masked to $-10^8$ and entropy restricted to legal actions), the combination of masking with recurrence in `fire-rescue-project-2.0` is an illusion: `MaskablePPO` from `sb3_contrib` possesses no recurrent machinery.

---

### Audit Scorecard

| Category | Requirement / Property | Status | Primary Reason / Code Evidence |
| :--- | :--- | :---: | :--- |
| **A. Algorithm** | A1. PPO Algorithm Identity | **PASS** | `sb3_contrib.MaskablePPO` in [`training/train_ppo.py#L21`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/training/train_ppo.py#L21) |
| | A2. Recurrent PPO Class | **FAIL** | Uses standard `MaskablePPO`, NOT `RecurrentPPO` |
| | A3. GRU in Policy Module | **PASS** | `nn.GRU(128, 128)` defined in [`agents/cnn.py#L285`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L285) |
| | A4. Algorithmic Recurrence Support | **FAIL** | `MaskablePPO` and its buffer do not support RNN states or sequences |
| | A5. Optimizer Parameter Coverage | **PASS** | Optimizer contains all GRU parameters ([`agents/cnn.py#L310`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L310)) |
| **B. Rollout** | B1. Hidden State Rollout Tracking | **FAIL** | `self.hidden_state` tracked via side-effect, ignores `episode_starts` |
| | B2. Episode Boundary Resets | **FAIL** | `episode_starts` not passed in `collect_rollouts`; hidden state leaks across episodes |
| | B3. Rollout Boundary Persistence | **FAIL** | `train()` overwrites `self.hidden_state` with shape `(1, 64, 128)`; next rollout resets to 0 |
| | B4. Buffer State Storage | **FAIL** | `MaskableDictRolloutBuffer` does NOT store hidden states or sequence starts |
| | B5. Minibatch Construction | **FAIL** | Transitions randomly shuffled (`indices = np.random.permutation`); no sequences |
| | B6. Sequence Padding & Masks | **FAIL** | No sequence handling, no padding, no temporal masks |
| | B7. Value Bootstrapping | **PASS** | `TimeLimit.truncated` handled via `DummyVecEnv` terminal observations |
| | B8. Previous Action Feature | **FAIL** | `RescueEnv` never produces `previous_action`; feature vector is permanently all zeros |
| **C. BPTT** | C1. Multi-Step Gradient Flow | **FAIL** | GRU unrolled for exactly 1 step (`seq_len = 1`); 0 multi-step gradient |
| | C2. Evaluate Actions Hidden States | **FAIL** | `evaluate_actions` receives `lstm_states=None`; uses $h=0$ or arbitrary previous minibatch state |
| | C3. BPTT Existence Proof | **FAIL** | Mathematically and empirically proven: $\nabla_{s_t} \mathcal{L}_{t+1} \equiv 0$ |
| **D. Masking** | D1. Masking in Rollout | **PASS** | Invalid actions masked to $-10^8$ logits before action sampling |
| | D2. Masking in PPO Update | **PASS** | Buffer stores `action_masks`; identical masks applied during `evaluate_actions` |
| | D3. Legal-Action Entropy | **PASS** | `MaskableCategorical.entropy` sums $-p \log p$ strictly over legal actions |
| | D4. Inference Determinism/Sampling | **PASS** | Both `argmax` and `Categorical.sample` strictly respect action masks |
| | D5. Ratio & KL Numerical Validity | **FAIL** | Finite, but theoretically invalid: ratio $\neq 1.0$ at epoch 0 due to hidden state mismatch |

---

## 2. Complete Training Path Trace

The diagram below traces the true operational flow of data and gradients in `fire-rescue-project-2.0`:

```
[Observation from RescueEnv]
  │ (keys: 'grid', 'memory', 'agent'; MISSING 'previous_action')
  ▼
[RecurrentSpatialRescueCNN._forward_network]
  │ Local CNN (1,800) + Memory CNN (8,320) + Agent (5) + Zeros(7) -> Linear(10132, 128)
  │ fused: (1, 128)
  │ Input to GRU: fused.unsqueeze(1) -> shape (1, 1, 128)  [seq_len = 1]
  │ h: from self.hidden_state (NOT reset on episode resets due to missing episode_starts)
  ▼
[Action Sampling & Masking]
  │ action_net(latent) -> MaskableCategorical(logits) -> apply_masking(masks) -> sample/argmax
  ▼
[Rollout Storage in MaskableDictRolloutBuffer]
  │ STORED: observations, actions, rewards, episode_starts, values, log_probs, action_masks
  │ DROPPED / NOT STORED: GRU hidden states (h_t), sequence indices, previous actions
  ▼
[PPO Update: MaskablePPO.train()]
  │ Buffer Permutation: indices = np.random.permutation(n_steps * n_envs)
  │ Minibatch: 64 UNRELATED, SHUFFLED transitions from across time and episodes!
  ▼
[policy.evaluate_actions()]
  │ lstm_states is NONE; episode_starts is NONE
  │ batch_size = 64 != 1 -> self.hidden_state RESET TO ALL ZEROS for Minibatch 1
  │ For Minibatch 2+: self.hidden_state holds detached garbage from Minibatch 1
  │ GRU runs on fused.unsqueeze(1) of shape (64, 1, 128) -> seq_len = 1
  ▼
[Loss & Backward Pass]
  │ loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss
  │ loss.backward()
  │ Gradients flow ONLY within each single transition:
  │   loss -> heads -> GRU_Cell(x_i, 0) -> fusion -> CNNs
  │ Gradients across time steps (t -> t-1): STRICTLY ZERO! (BPTT = 0)
  ▼
[Post-Update Rollout Corruption]
  │ self.hidden_state now has shape (1, 64, 128)
  │ Rollout starts with batch_size = 1 -> shape mismatch -> RESET TO ZEROS!
  │ Any active episode spanning rollout boundary loses all recurrent memory!
```

---

## 3. Detailed Audit Findings with Code Citations

### A. Algorithm Identification

#### 1. PPO Algorithm Actually Used in `train_ppo.py`
In [`training/train_ppo.py#L21`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/training/train_ppo.py#L21) and [`#L398-L412`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/training/train_ppo.py#L398-L412):
```python
from sb3_contrib import MaskablePPO
...
model = MaskablePPO(
    policy=RecurrentSpatialRescueCNN,
    env=train_env,
    learning_rate=args.learning_rate,
    n_steps=args.n_steps,
    batch_size=args.batch_size,
    n_epochs=args.n_epochs,
    gamma=args.gamma,
    ent_coef=args.ent_coef,
    verbose=1,
    device=device,
    tensorboard_log="./logs/",
    seed=args.seed,
)
```
The algorithm is `sb3_contrib.ppo_mask.ppo_mask.MaskablePPO`.

#### 2. Comparison with `RecurrentPPO`
- `MaskablePPO` subclasses `stable_baselines3.common.on_policy_algorithm.OnPolicyAlgorithm`. It is a **feed-forward PPO algorithm** augmented with action masks.
- `sb3_contrib` maintains a completely separate algorithm for recurrent policies: `sb3_contrib.ppo_recurrent.ppo_recurrent.RecurrentPPO`.
- `sb3_contrib` **does not provide a merged `MaskableRecurrentPPO` class**.
- The author attempted to inject recurrence solely by creating a custom policy subclassing `MaskableActorCriticPolicy` ([`agents/cnn.py#L167`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L167)) and passing it to standard `MaskablePPO`.

#### 3 & 4. Trainable Policy vs. Algorithmic Recurrence
- In [`agents/cnn.py#L285-L289`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L285-L289), `self.gru` is instantiated:
  ```python
  self.gru = nn.GRU(
      input_size=self.hidden_size,
      hidden_size=self.hidden_size,
      batch_first=True,
  )
  ```
- However, the outer algorithm (`MaskablePPO`), its rollout buffer (`MaskableDictRolloutBuffer`), and its training loop (`MaskablePPO.train()`) have **zero awareness of recurrence**. Recurrence is purely a local layer in the PyTorch graph that is called once per transition.

#### 5. Optimizer Parameter Registration
In [`agents/cnn.py#L310-L314`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L310-L314):
```python
self.optimizer = self.optimizer_class(
    self.parameters(),
    lr=lr_schedule(1),
    **self.optimizer_kwargs,
)
```
All parameters of `self.gru` (`weight_ih_l0`, `weight_hh_l0`, `bias_ih_l0`, `bias_hh_l0`) are registered submodules and are present in `self.parameters()`. Verification via Python confirmed:
```python
set(model.policy.gru.parameters()).issubset({p for group in optimizer.param_groups for p in group['params']}) == True
```

---

### B. Recurrent Rollout Handling

#### 1. Hidden State Tracking in `collect_rollouts`
In `sb3_contrib/ppo_mask/ppo_mask.py#L228-L230`:
```python
if use_masking:
    action_masks = get_action_masks(env)

actions, values, log_probs = self.policy(obs_tensor, action_masks=action_masks)
```
Notice that `self.policy(...)` is called **without** `lstm_states` and **without** `episode_starts`!
In [`agents/cnn.py#L375-L387`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L375-L387):
```python
h = lstm_states if lstm_states is not None else self.hidden_state
if h is None or h.shape[1] != batch_size or h.device != device:
    h = torch.zeros((1, batch_size, self.hidden_size), dtype=torch.float32, device=device)

if episode_starts is not None:
    starts = torch.as_tensor(episode_starts, dtype=torch.float32, device=device).view(1, batch_size, 1)
    h = h * (1.0 - starts)

gru_out, next_hidden = self.gru(fused.unsqueeze(1), h)
gru_out = gru_out.squeeze(1)

self.hidden_state = next_hidden.detach()
```
Because `lstm_states` is `None`, the policy reads from `self.hidden_state`. It updates `self.hidden_state = next_hidden.detach()` as an internal side effect.

#### 2. Hidden State Leakage Across Episodes
Because `collect_rollouts` never passes `episode_starts` to `self.policy`, the reset block:
```python
if episode_starts is not None:
    starts = torch.as_tensor(episode_starts, dtype=torch.float32, device=device).view(1, batch_size, 1)
    h = h * (1.0 - starts)
```
**NEVER EXECUTES DURING ROLLOUTS**.  
When an environment terminates an episode and resets, `self.hidden_state` retains the final hidden state of the ended episode and passes it into step 0 of the new episode.

**Empirical Verification:**  
A test environment with `max_steps=2` was run through `model.learn(total_timesteps=8)`. The norm of `h_before` was monitored at each step:
- Step 0 (ep 0, step 0): `h_before = None` (init to 0)
- Step 1 (ep 0, step 1, terminates): `h_before norm = 0.873`
- Step 2 (ep 1, step 0, **NEW EPISODE**): `h_before norm = 0.989` (**FAILED TO RESET!**)
- Step 3 (ep 1, step 1, terminates): `h_before norm = 1.092`
- Step 4 (ep 2, step 0, **NEW EPISODE**): `h_before norm = 1.110` (**FAILED TO RESET!**)

#### 3. Hidden State Destruction at Rollout Boundaries
During training updates (`model.train()`), minibatches of size `batch_size = 64` are forwarded through `evaluate_actions()`.  
At the end of `train()`, `self.hidden_state` holds `next_hidden.detach()` with shape `(1, 64, 128)`.  
When the next rollout starts, the environment has `batch_size = 1` (or `num_envs`).  
Line 376 executes:
```python
if h is None or h.shape[1] != batch_size or h.device != device:
    h = torch.zeros((1, batch_size, self.hidden_size), dtype=torch.float32, device=device)
```
Since `h.shape[1] == 64 != 1`, **`h` is immediately reset to all zeros**!  
**Result:** Any episode that was active when the 128-step rollout ended has its hidden state completely erased upon entering step 129!

#### 4. Rollout Buffer State Storage
In `sb3_contrib/ppo_mask/ppo_mask.py#L156-L162`:
```python
if self.rollout_buffer_class is None:
    if isinstance(self.observation_space, spaces.Dict):
        self.rollout_buffer_class = MaskableDictRolloutBuffer
    else:
        self.rollout_buffer_class = MaskableRolloutBuffer
```
`MaskableDictRolloutBuffer` subclasses SB3's standard `DictRolloutBuffer`.  
Inspecting its sample fields (`data._fields`):
```python
('observations', 'actions', 'old_values', 'old_log_prob', 'advantages', 'returns', 'action_masks')
```
- Hidden states (`lstm_states` or `rnn_states`) are **not stored**.
- Initial hidden states per trajectory chunk are **not stored**.
- In contrast, `sb3_contrib.common.recurrent.buffers.RecurrentDictRolloutBuffer` specifically allocates:
  `hidden_state_buffer_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)`
  and yields `rollout_data.lstm_states`.

#### 5 & 6. Minibatch Sampling and Sequence Handling
In `sb3_contrib/common/maskable/buffers.py#L112-L135`:
```python
def get(self, batch_size: int | None = None) -> Generator[MaskableDictRolloutBufferSamples, None, None]:
    assert self.full, ""
    indices = np.random.permutation(self.buffer_size * self.n_envs)
    ...
    start_idx = 0
    while start_idx < self.buffer_size * self.n_envs:
        yield self._get_samples(indices[start_idx : start_idx + batch_size])
        start_idx += batch_size
```
- The buffer **randomly permutes all transitions across time and environments** (`np.random.permutation`).
- Minibatches are unordered collections of single-step transitions.
- There are no sequence chunks, no temporal unrolling, no sequence starts, and no padding masks.

#### 7. Value Bootstrapping for Truncated vs. Terminated Episodes
In `sb3_contrib/ppo_mask/ppo_mask.py#L251-L260`:
```python
for idx, done in enumerate(dones):
    if (
        done
        and infos[idx].get("terminal_observation") is not None
        and infos[idx].get("TimeLimit.truncated", False)
    ):
        terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
        with th.no_grad():
            terminal_value = self.policy.predict_values(terminal_obs)[0]
        rewards[idx] += self.gamma * terminal_value
```
- In `train_ppo.py`, `train_env = make_train_env(...)` wraps `CurriculumRescueEnv` in a `Monitor`.
- When initialized inside `MaskablePPO`, SB3 automatically wraps non-vectorized environments in `DummyVecEnv`.
- `DummyVecEnv` correctly detects `truncated=True` from `RescueEnv` and populates `infos[idx]["TimeLimit.truncated"] = True` and `infos[idx]["terminal_observation"]`.
- The terminal value is added to `rewards[idx]`, and `compute_returns_and_advantage` correctly calculates GAE advantages.
- *Caveat:* When `predict_values(terminal_obs)` is invoked, `lstm_states` is not passed; it uses whatever `self.hidden_state` happens to reside in the policy.

#### 8. Previous Action Input
In [`agents/cnn.py#L200-L206`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L200-L206):
```python
if isinstance(observation_space, spaces.Dict):
    space_dict = dict(observation_space.spaces)
    if "previous_action" not in space_dict:
        space_dict["previous_action"] = spaces.Box(
            low=0, high=6, shape=(), dtype=np.float32
        )
    observation_space = spaces.Dict(space_dict)
```
- The policy artificially injects `"previous_action"` into its internal copy of `observation_space`.
- But in `rescue_env/envs/single_agent.py#L64-L80`, `RescueEnv.observation_space` only defines `{"grid", "memory", "agent"}`.
- Neither `RescueEnv` nor `CurriculumRescueEnv` nor any wrapper ever generates `"previous_action"`.
- In [`agents/cnn.py#L359-L367`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L359-L367):
  ```python
  prev_action = obs.get("previous_action", None)
  ...
  prev_action_features = self._encode_previous_action(prev_action, batch_size, device)
  ```
- In [`agents/cnn.py#L322-L323`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L322-L323):
  ```python
  if prev_action is None:
      return torch.zeros((batch_size, self.action_dim), dtype=torch.float32, device=device)
  ```
- **Finding:** Across all rollouts, training updates, and evaluations, `previous_action_features` is **always `torch.zeros((batch_size, 7))`**. It is a dead feature branch.

---

### C. Backpropagation Through Time (BPTT)

#### 1. GRU Gradient Propagation During Updates
In `sb3_contrib/ppo_mask/ppo_mask.py#L339-L343`:
```python
values, log_prob, entropy = self.policy.evaluate_actions(
    rollout_data.observations,
    actions,
    action_masks=rollout_data.action_masks,
)
```
In [`agents/cnn.py#L516`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L516):
```python
latent, _ = self._forward_network(obs, lstm_states=lstm_states, episode_starts=episode_starts)
```
In [`agents/cnn.py#L383-L384`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L383-L384):
```python
gru_out, next_hidden = self.gru(fused.unsqueeze(1), h)
gru_out = gru_out.squeeze(1)
```
- In PyTorch, `nn.GRU(..., batch_first=True)` expects input of shape `(batch_size, seq_len, input_size)`.
- Here, `fused.unsqueeze(1)` has shape `(batch_size, 1, 128)`.
- The sequence length `seq_len` is **1**.
- The GRU unrolls for **exactly one step**.
- When `loss.backward()` executes, gradients backpropagate from `loss` through `gru_out[:, 0, :]` to `fused` and `h`.
- But `h` was either a freshly created zero tensor (`torch.zeros(...)`) or was detached (`next_hidden.detach()`).
- **Gradients can never flow into earlier timesteps.**

#### 2. Behavior of `evaluate_actions` on Shuffled Minibatches
When `MaskablePPO` passes a minibatch of 64 transitions:
1. For Minibatch 1: `self.hidden_state` (from rollout) has `shape[1] = 1`. Since $1 \neq 64$, `h` is initialized to **all zeros**.
2. For Minibatch 2: `self.hidden_state` has `shape[1] = 64` from Minibatch 1. Since $64 == 64$, `h` is **taken from Minibatch 1**.
3. But Minibatch 1 was a random collection of transitions! Transition 5 in Minibatch 2 receives the hidden state produced by transition 5 in Minibatch 1. These transitions have zero temporal relation.
4. In neither case does `evaluate_actions` receive the true hidden state $h_t$ that was present when observation $s_t$ was collected during the rollout.

#### 3. Mathematical and Code Proof of Zero BPTT

**Mathematical Proof:**  
Consider two consecutive environment transitions $(s_t, a_t)$ and $(s_{t+1}, a_{t+1})$ from the same episode.  
In a true recurrent RL algorithm with BPTT horizon $K \ge 2$:
$$\mathcal{L}_{t+1} = f(\pi_\theta(a_{t+1} \mid s_{t+1}, h_{t+1}))$$
$$h_{t+1} = \text{GRU}(e_\theta(s_{t+1}), h_t)$$
$$h_t = \text{GRU}(e_\theta(s_t), h_{t-1})$$
By the chain rule:
$$\frac{\partial \mathcal{L}_{t+1}}{\partial s_t} = \frac{\partial \mathcal{L}_{t+1}}{\partial h_{t+1}} \cdot \frac{\partial h_{t+1}}{\partial h_t} \cdot \frac{\partial h_t}{\partial e_\theta(s_t)} \cdot \frac{\partial e_\theta(s_t)}{\partial s_t} \neq 0$$
In `fire-rescue-project-2.0`:
During training, the buffer draws transitions $i$ and $j$.  
Minibatch inputs to `self.gru` are $X = [e_\theta(s^{(0)}), \dots, e_\theta(s^{(B-1)})]^T \in \mathbb{R}^{B \times 1 \times D}$.  
For any sample $i \in \{0, \dots, B-1\}$:
$$h^{(i)} = \text{GRU\_Cell}(e_\theta(s^{(i)}), h_{init}^{(i)})$$
The loss on sample $i$ is $\mathcal{L}^{(i)} = g(h^{(i)})$.  
For any sample $j \neq i$:
$$\frac{\partial h^{(i)}}{\partial s^{(j)}} = 0 \implies \frac{\partial \mathcal{L}^{(i)}}{\partial s^{(j)}} = 0$$
Even if $s^{(j)}$ was the chronologically preceding state $s_{t}$ and $s^{(i)}$ was $s_{t+1}$, because they are placed into separate batch slots:
$$\frac{\partial \mathcal{L}_{t+1}}{\partial s_t} \equiv 0$$
Furthermore, $h_{init}$ is not a function of $s_t$ in the PyTorch graph (it is a detached leaf or zero).

**Empirical Code Verification:**  
A test script created two consecutive observations $s_0$ and $s_1$ with `requires_grad=True`, passed them into `policy.evaluate_actions` as a batch (identical to `MaskablePPO.train()`), and computed $\nabla_{s_0} \mathcal{L}(s_1)$:
```python
values, log_prob, entropy = policy.evaluate_actions(batch_obs, actions, action_masks=masks)
loss_1 = values[1]
grad_s0 = torch.autograd.grad(loss_1, obs_0['grid'], retain_graph=True)[0]
assert torch.all(grad_s0 == 0.0)
```
**Result:** `grad_s0` is strictly all zeros (`torch.all(grad_s0 == 0.0)` is `True`).  
**Conclusion:** Backpropagation Through Time across environment steps is completely non-existent.

---

### D. Legal-Action Masking

#### 1 & 2. Masking Enforcement in Rollout and PPO Updates
- In `collect_rollouts` (`ppo_mask.py#L228`): `action_masks` are fetched from `env.action_masks()` and passed to `policy(obs_tensor, action_masks=action_masks)`.
- In `RecurrentSpatialRescueCNN.forward` ([`agents/cnn.py#L485`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L485)):
  ```python
  distribution = self.action_dist.proba_distribution(action_logits=logits)
  if action_masks is not None:
      distribution.apply_masking(action_masks)
  ```
- In `MaskableCategorical.apply_masking` (`sb3_contrib/common/maskable/distributions.py#L57-L61`):
  `logits = th.where(self.masks, self._original_logits, -1e8)`
- Illegal action logits are clamped to $-10^8$. Their softmax probability is strictly $0.0$.
- In `MaskableDictRolloutBuffer`, `action_masks` is recorded for every transition and re-applied identically during `train()`.

#### 3. Entropy Computation
In `sb3_contrib/common/maskable/distributions.py#L74-L82`:
```python
def entropy(self) -> th.Tensor:
    if self.masks is None:
        return super().entropy()

    device = self.logits.device
    p_log_p = self.logits * self.probs
    p_log_p = th.where(self.masks, p_log_p, th.tensor(0.0, device=device))
    return -p_log_p.sum(-1)
```
Illegal action terms are explicitly zeroed out in `p_log_p`. Entropy is calculated strictly over the support of valid actions.

#### 4. Deterministic and Stochastic Inference
- Stochastic action selection calls `distribution.sample()`. Since masked actions have $p = 0.0$, illegal actions are sampled with probability 0.
- Deterministic action selection calls `distribution.get_actions(deterministic=True)`, which performs `argmax` on the masked logits. Since illegal actions have logit $-10^8$, they cannot be selected unless all actions are masked.
- Unit tests in `test_recurrent_spatial_cnn.py#L63-L109` verify this with 100 repeated samples per state.

#### 5. Numerical Validity of PPO Ratios and KL
In `sb3_contrib/ppo_mask/ppo_mask.py#L352`:
`ratio = th.exp(log_prob - rollout_data.old_log_prob)`
In standard PPO, on epoch 0 before optimizer updates:
$$\pi_\theta(a_t \mid s_t) \equiv \pi_{\theta_{\text{old}}}(a_t \mid s_t) \implies \text{ratio} \equiv 1.0, \quad \text{approx\_kl} \equiv 0.0$$
In `fire-rescue-project-2.0`:
Because `evaluate_actions` evaluates $s_t$ with $h = 0$ (or a detached state from an unrelated transition) instead of $h_t^{\text{rollout}}$:
$$\pi_\theta(a_t \mid s_t, h_{\text{eval}}) \neq \pi_{\theta_{\text{old}}}(a_t \mid s_t, h_t^{\text{rollout}})$$
**Empirical Test:**  
Even at step 0 of epoch 0 before updating parameters, the importance sampling ratio was measured on a 64-step rollout:
- $\min(\text{ratio}) = 0.99596$
- $\max(\text{ratio}) = 1.00782$
- Mean absolute deviation in $\log \pi = 0.0019$
This mismatch violates the foundational PPO trust region assumption at the start of every update epoch.

---

## 4. Critical Bugs Catalog

| Bug ID | Severity | File & Location | Description & Impact |
| :--- | :---: | :--- | :--- |
| **BUG-01** | **BLOCKER** | [`training/train_ppo.py#L398`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/training/train_ppo.py#L398) | **Fake Recurrent PPO Architecture**: `MaskablePPO` is paired with `RecurrentSpatialRescueCNN`. `MaskablePPO` uses `MaskableDictRolloutBuffer`, which randomly shuffles all transitions and has zero RNN sequence support. |
| **BUG-02** | **BLOCKER** | [`agents/cnn.py#L383`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L383) | **Zero BPTT (Sequence Length 1)**: GRU receives `fused.unsqueeze(1)` (`seq_len = 1`). Gradients never backpropagate across time steps. BPTT horizon is 0. |
| **BUG-03** | **CRITICAL** | [`agents/cnn.py#L376-L377`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L376-L377) | **Rollout Boundary Hidden State Wipeout**: `train()` leaves `self.hidden_state` with shape `(1, 64, 128)`. The next rollout step has `batch_size = 1`. The shape mismatch triggers a reset to zeros, destroying ongoing episode memory every 128 steps. |
| **BUG-04** | **CRITICAL** | `sb3_contrib/ppo_mask/ppo_mask.py#L230` | **Episode Hidden State Leakage**: `collect_rollouts` does not pass `episode_starts` to `self.policy(...)`. Hidden states do not reset when episodes end; memory leaks into the next episode. |
| **BUG-05** | **MAJOR** | [`rescue_env/envs/single_agent.py#L64`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/rescue_env/envs/single_agent.py#L64), [`agents/cnn.py#L322`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L322) | **Permanently Dead `previous_action` Feature**: Policy allocates a 7-dim one-hot input for previous action, but `RescueEnv` never outputs it. It defaults to all zeros 100% of the time. |
| **BUG-06** | **MAJOR** | [`training/evaluate.py#L233`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/training/evaluate.py#L233) vs [`agents/cnn.py#L516`](file:///Users/luigi/uni/Master/Semester%201/Deep%20Learning/project/fire-rescue-project-2.0/agents/cnn.py#L516) | **Train-vs-Eval Distribution Shift**: During evaluation, `evaluate.py` propagates GRU hidden states sequentially over 64–384 steps. But the model was trained with $h \approx 0$ on single-step inputs. The hidden state representation at test time is completely out-of-distribution. |
| **BUG-07** | **MAJOR** | `sb3_contrib/ppo_mask/ppo_mask.py#L352` | **Corrupted PPO Policy Ratio**: Re-evaluating actions in `train()` with zero/detached hidden states means $\pi_\theta(a \mid s, h_{\text{eval}}) \neq \pi_{\text{old}}(a \mid s, h_{\text{rollout}})$. Ratios deviate from 1.0 before optimization begins. |

---

## 5. Why Existing Tests Passed Despite Fatal Bugs

A critical question for the audit is: *How did `fire-rescue-project-2.0` pass test suites (such as `test_recurrent_spatial_cnn.py` and `test_verification_benchmarks.py`) if the algorithm is fundamentally broken?*

Reviewing the tests reveals the root cause:
1. **Isolated Unit Calls vs. System Integration**:  
   In `test_verification_benchmarks.py#L188-L240` and `test_recurrent_spatial_cnn.py#L111-L160`, the tests explicitly invoke:
   ```python
   policy(obs, lstm_states=h_prev, episode_starts=starts, return_hidden_state=True)
   ```
   The unit test manually supplied `lstm_states` and `episode_starts` directly to the policy method! The test verified that *if* someone passes these arguments, the GRU branch updates correctly. It never tested whether `MaskablePPO` actually passes them (it does not).
2. **Superficial End-to-End Test**:  
   In `test_recurrent_spatial_cnn.py#L161-L194` (`test_sb3_maskable_ppo_end_to_end`):
   ```python
   model = MaskablePPO(RecurrentSpatialRescueCNN, env, n_steps=8, batch_size=8, ...)
   model.learn(total_timesteps=16)
   action, state = model.predict(obs, action_masks=env.action_masks())
   ```
   This test only checked that `model.learn(16)` executed without throwing an exception, and that `model.predict()` returned an action and a state tensor. It did not check whether gradients flowed across time steps or whether hidden states were stored in the buffer.
3. **Synthetic Backward Test with Sequence Length 1**:  
   In `test_recurrent_spatial_cnn.py#L214-L246` (`test_gradients_flow_through_all_branches`):  
   The test computed:
   ```python
   values, log_prob, entropy = policy.evaluate_actions(obs, actions, action_masks=masks)
   loss = values.mean() - log_prob.mean() - 0.01 * entropy.mean()
   loss.backward()
   assert policy.gru.weight_ih_l0.grad is not None and policy.gru.weight_ih_l0.grad.abs().sum() > 0
   ```
   This proved that the GRU parameters receive gradients for a *single-step* forward pass (`seq_len = 1`). It did not verify BPTT across consecutive timesteps.

---

## 6. Recommended Minimal Fixes for Phase 2

To achieve genuine recurrent RL with legal-action masking in `fire-rescue-project-2.0`, the following minimal fixes must be implemented:

### 1. Build a Unified `MaskableRecurrentPPO` Algorithm
Instead of forcing a recurrent policy into feed-forward `MaskablePPO`, construct a proper `MaskableRecurrentPPO` class by combining `sb3_contrib.RecurrentPPO` and `sb3_contrib.MaskablePPO`:
- **Subclass `RecurrentPPO`**: This provides the native recurrent rollout buffer (`RecurrentDictRolloutBuffer`), trajectory chunking, sequence padding, temporal masks, and hidden state storage (`RNNStates`).
- **Integrate Masking**:
  - Extend `RecurrentDictRolloutBuffer` to store `action_masks` alongside observations and hidden states.
  - Update `collect_rollouts` to query `env.action_masks()` and pass them to `self.policy(..., action_masks=action_masks)`.
  - Update `train()` to pass `action_masks=rollout_data.action_masks` to `policy.evaluate_actions(...)`.

### 2. Update Policy Interface to `RecurrentActorCriticPolicy`
- Subclass `sb3_contrib.common.recurrent.policies.RecurrentActorCriticPolicy` rather than `MaskableActorCriticPolicy`.
- Ensure `forward()`, `evaluate_actions()`, and `predict_values()` adhere to the standard recurrent signature:
  `evaluate_actions(obs, actions, lstm_states, episode_starts, action_masks=None)`
- Unroll the GRU across the sequence dimension `seq_len` for each padded trajectory chunk during `evaluate_actions()`, using the true initial hidden state stored at the chunk start.

### 3. Supply `previous_action` via an Environment Wrapper
- Create a lightweight `PreviousActionWrapper(gymnasium.Wrapper)` that tracks the previous action executed in `step()` (initialized to 0 on `reset()`) and adds `"previous_action"` to the observation dictionary.
- Wrap `train_env` and `eval_env` so that the policy's 7-dim action feature receives genuine, dynamic action history rather than permanent zeros.

### 4. Remove Policy Instance Variable State Leaks
- Eliminate `self.hidden_state` as an internal mutable state attribute in `_forward_network`.
- Enforce that all hidden state flow is explicit: passed into the function as an argument and returned as an output, managed entirely by the recurrent rollout buffer and the training loop.
