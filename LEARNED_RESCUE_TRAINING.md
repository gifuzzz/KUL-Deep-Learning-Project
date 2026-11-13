# Learning a reliable rescue policy

Branch: `experiment/spatial-observations`. Active goal: train a policy that completes
rescues reliably on separate scenarios, with learned action selection and no inference-time
navigator, waypoint or programmed route hints. Existing checkpoints are preserved.

## First milestones

1. Fit varied distance-3 rescues across all four exit directions. Repeated training
   scenarios are a learning sanity check, not a generalization result.
2. Reach at least 95% completed rescues on the fixed near-rescue validation set, then
   confirm deterministic and sampled performance on at least 200 fresh scenarios.
3. Increase distance and separately measure initially hidden victims, pickups and returns.
   Near-exit success alone will not be described as successful distant exploration.

## Corrected navigation reward

The previous `state_potential` returned zero for *any* position outside the building,
even empty-handed. For seed 20002, leaving could earn +0.199 shaped reward while a
useful approach move earned about +0.100. Correct potential shaping can preserve
optimal discounted behavior under its assumptions; these observations nevertheless
show misleading immediate guidance at a common failure boundary, not proof that the
optimal global policy was changed.

The potential now uses the native world graph, including grass and legal exit crossings.
Pickup targets include native-eligible adjacent grass. Carrying targets are the first
outside grass cells where automatic rescue happens. Empty-handed regression seeds
20000 and 20002 now receive negative shaping for the incorrect outward move and
positive shaping for returning toward pickup. The final carrying move has distance
one to rescue grass, then zero on rescue. The observation/action contract and native
physics have not changed.

The native route function is used **only to calculate a training reward distance**.
No path, direction or subtask enters the policy. This continues to use privileged
simulator state for reward guidance, including true victim positions. Evaluation
has all shaping disabled.

## Training and architecture

The runner supports eight environments with 128 steps each, keeping 1024 total
rollout samples per PPO update. Learning rate 3e-4 is being tested rather than the
previous 1e-4. Batch size 64, four epochs, gamma .99 and entropy coefficient .01
remain unchanged. MPS is available and used.

A separate `SpatialRescueCNN` retains the original two local convolutions, projects
64 channels to 8 with a 1x1 convolution, then flattens all 15x15 positions. It avoids
local 3x3 average pooling while retaining the remembered-map branch and 128-feature
fusion. The original `RescueCNN` class is unchanged for old checkpoint compatibility.
A save/reload test verifies the new extractor reproduces masked policy probabilities.
These architecture and optimizer changes are experiments, not individually proven causes.

## Learning sanity check

Training roots: 0, 4 (left exit); 5, 13 (top); 9, 15 (right); 14, 31 (bottom).
Every selected victim is placed at distance 3 under the existing placement rule.
The original CNN, corrected reward, higher learning rate and vectorized collection
completed **8/8 deterministic training scenarios** at saved timestep 76,800.
Training sampled success was about 97% at that point. Training on the repeated set
was then stopped intentionally and its best checkpoint preserved:

`models/learned_rescue_20261112/sanity_world_reward_continue/best_model.zip`

On separate unshaped roots 10000–10099 at maximum distance 3, it rescued **42/100**,
with **61/100 pickups** and **19 carrying timeouts**. This confirms that fitting the
training cases is not sufficient; generalization and return behavior remain incomplete.

## Broad training underway

`models/learned_rescue_20261112/spatial_d4/` trains a fresh spatial CNN on varied
maps with maximum victim placement distance 4, minimum 1, 64-step episodes, seed 46,
a 262,144-step initial budget, and fixed validation roots 10000–10049. Both shaping
options are enabled in training and disabled in validation. The global marker is off.
This is not a controlled attribution study: the goal is a reliable policy; any successful
combination must be measured before claiming which individual change caused it.

Source snapshots are in `models/learned_rescue_20261112/source_initial_fix/`.
The stopped learning-check continuation has explicit stop-reason metadata. Logs and
checkpoints from earlier experiments are preserved.

## Near-rescue model passed fresh evaluation

Frozen checkpoint:
`models/learned_rescue_20261112/candidate_d4_122880.zip`

It was selected on fixed validation, where it reached 50/50 rescues, then frozen
before evaluation on roots **50000–50199**. All shaping was disabled, maximum
placement distance was 4 (minimum 1), the episode limit was 64, and inference used MPS.

| Selection | Rescues /200 | Pickups /200 | Return after pickup | Carrying timeouts |
|---|---:|---:|---:|---:|
| Deterministic | 195 | 198 | 195/198 | 3 |
| Sampled, action seed 123 | 198 | 198 | 198/198 | 0 |

There were no drops. Mean successful lengths were 5.08 deterministic and 6.08 sampled.
199 of these 200 victims were initially visible. This is a reliable **near-rescue**
result; it does not establish distant search. Original data, selection metadata and
checkpoint SHA-256 are saved beside the candidate. The broad distance-4 run was
stopped intentionally after the candidate passed; its earlier best is preserved.
The result combines corrected reward, spatial encoder, optimizer and collection changes;
it does not identify their individual causal contributions.

Reproduce near-rescue inference from the project directory:

```sh
.venv/bin/python -m training.evaluate \
  --model models/learned_rescue_20261112/candidate_d4_122880.zip \
  --victim-distance 4 --episodes 200 --scenario-start 50000 --device mps
```

Add `--stochastic` for sampled actions. These roots are now consumed and will not be
used to select further changes.

## Farther hidden-victim continuation

Before continuation, the near model scored 7/50 deterministic and 9/50 sampled rescues
on validation roots 10000–10049 with minimum placement distance 5, maximum 8 and a
128-step limit. Of the 36 initially hidden victims, it rescued **zero** in either mode.

`models/learned_rescue_20261112/spatial_d8/` starts from the frozen near checkpoint,
uses the same observation/action contract, eight environments and 128 rollout steps
each, seed 47, learning rate 1e-4, unchanged .01 entropy coefficient, both training
shaping options, and a 262,144-step initial budget. Validation remains unshaped.

With minimum distance above 1, placement uses distances from all open exit portal
cells. Thus these victims are 5–8 steps from their nearest exit, but can be far from
the first exit where the firefighter starts. The 50 validation scenarios have initial
firefighter-to-victim distances of 5–76 steps (median 9), with 36 initially hidden.
Do not compare this task directly with the first-exit near-placement rule.

The near checkpoint remains unchanged regardless of the farther continuation outcome.
Source snapshots for the expanded runner/evaluator are in
`models/learned_rescue_20261112/source_far_stage/`.

## Imitation-assisted search experiments

Farther pure PPO plateaued at 24/50 validation rescues; relative-coordinate and
exploration-bonus variants did not resolve the failures. These experimental models
were stopped with their best checkpoints preserved.

`training/train_imitation.py` uses an observed-map teacher **only for training labels**.
The teacher reads the local observation, known own position and static grass rules;
it receives no hidden victim coordinates. Saved policies contain only the CNN and
policy/value heads. Validation uses the neural policy alone, with normal movement
and interaction masks. There are no inference-time waypoints or subtask features.
This method is imitation-assisted CNN training following PPO, not pure PPO and not
the earlier navigator-hint hybrid.

The observed teacher solved all 50 search validation scenarios within 753 steps;
only 36 finished within 256 and 45 within 512. These are teacher results, not neural
policy results. The search experiments therefore allow 1024 steps and retain the
separate 64-step near-rescue benchmark.

Additional optional extractors preserve fine global memory features and read a
15x15 remembered crop centered on the agent. They retain the same external policy
inputs: 11x15x15 grid, 13x64x64 memory, five agent features. Added residual branches
were initialized to preserve the parent policy exactly; initialization metadata and
source snapshots are saved beside the experiments. The current-camera view remains
visibility-limited; the crop reads only previously observed memory.

Initial FIFO-replay imitation runs had low training losses but plateaued around
20–26/50 deterministic search rescues. A sampled validation check of the remembered
local variant achieved 35/50, still below reliability. A 20-episode consistency probe
found 4826 identical repeated observations with zero conflicting teacher labels.
A separate 133-state float16 replay precision probe changed no argmax actions
(maximum probability difference below .00047). Neither probe proves general absence
of aliasing, but neither supported blaming those mechanisms for the observed plateau.

The current run is `models/learned_rescue_20261112/imitation_d8_aggregate/`.
It starts from `deterministic_dagger_start_bc50000.zip`, executes the learner
**deterministically** with zero teacher-action execution probability, and labels
learner-visited states. Its replay uses an 8192-entry reservoir over unique states,
plus recent corrections in each batch, to retain diversity across houses instead
of filling a FIFO with repeated loop states. Position features remain float32;
image replay uses float16. Training roots remain 1000–3047, with validation roots
10000–10049. Fresh search test roots 60000–60199 remain unused.

At 80000 new label steps, deterministic validation reached **28/50**. Over 46000
unique states had been observed, with zero conflicting labels so far. This is
validation progress, not a verified reliable search model. Do not promote it over
the independently verified near-rescue model yet. The active goal remains incomplete
until a frozen search policy passes independent completed-rescue evaluation, including
initially hidden victims and near-task retention.

Current source snapshots: `models/learned_rescue_20261112/source_imitation_aggregate/`.
Per-run evaluation histories and stopped-run metadata distinguish completed,
plateaued and currently running experiments. All earlier checkpoints are retained.
