# CNN + PPO rescue inspection — 12 November 2026

This document describes the active `fire-rescue-project`, not the backup or the older
`Rescue-Environment`. The existing README contains historical navigator commands and
results that do not describe the current policy. Its pending edits were preserved.

## Confirmed implementation

The handoff matches the inspected environment, memory, CNN, and training scripts.
The policy observes an 11×15×15 local grid, a 13×64×64 remembered map, and five agent
features. Memory records observed contents plus explored/building masks; it computes
no routes. The two CNN branches feed a 128-feature fusion layer. There is no recurrent
network, waypoint, or subtask input.

Actions map to native commands `(0, 1, 2, 3, 4, 10, 5)`: wait, four moves, pickup,
water. Pickup/drop is masked while carrying. Native C++ automatically rescues the
single category-A victim when the carrier moves onto grass. Other victim categories
could require treatment, which this action set does not provide.

The current training scripts match the handoff, including privileged shortest-distance
potential shaping in `train_ppo`, followed by curricula without it. Evaluation has no
shaping. Potential shaping supplies a reward, not an inference-time route.

Distance-4 training has completed: its saved 20-episode evaluations peak and finish at
35% success, with final mean episode length 42.45 steps. Distance-3 remains 40%, and
`no_drop` peaks at 60% and finishes at 45%. These use changing evaluation scenarios;
20 episodes also give noisy estimates. They cannot establish improvement between runs.

MPS is built into this Python's Torch but unavailable in this tool session. All new
measurements and training below use CPU with one Torch thread. Source checkpoints
were loaded with an environment to check observation/action spaces. Matching spaces
cannot prove matching action meanings; only current no-drop-era checkpoints were used.

## Fresh diagnosis

Seeds 20000–20099 now serve as a development/diagnostic set, not an untouched final
test. Every policy below sees identical unshaped 64-step scenarios with victim placement
maximum 3 and minimum 1. The distance is from the selected exit in this configuration,
not a guaranteed distance from the nearest exit.

| Policy | Action selection | Rescues / 100 | Pickups / 100 |
|---|---|---:|---:|
| `no_drop/best_model.zip` | deterministic | 25 | 28 |
| `curriculum_d3/last_model.zip` | deterministic | 29 | 32 |
| `curriculum_d4/last_model.zip` | deterministic | 25 | 28 |
| `curriculum_d3/last_model.zip` | sampled, seed 123 | 73 | 83 |
| uniformly random valid actions | sampled, seed 123 | 20 | 49 |

No drops occurred in the sampled-policy evaluation. Its rescue-after-pickup rate was
73/83 (88%); 17 episodes timed out before pickup and 10 while carrying.

For all three deterministic policies, failed episodes visited a median of just two
positions. Seed 20000 repeatedly alternates right and left for all 64 steps.
Thus the main observed failure is getting to pickup, rather than dropping victims.
Sampling can escape these loops; it does not demonstrate robust learned navigation.

All 100 victims were visible initially. For the distance-3 checkpoint, deterministic
success split by actual placement distance was 23/24 at distance 1, 6/34 at distance 2,
and 0/42 at distance 3. These are short-range rescue tests, not exploration evidence.

Raw episode diagnostics and traces are in `logs/handoff_audit_20261112/`.

## Reproducible tools added

`training.evaluate` now accepts `--device`, `--scenario-start`, and `--output`.
It records per-episode seeds, rescue/pickup/timeout/length/drop results, current action
mapping, and sampling configuration. Existing result files are protected. Random
baselines no longer require a checkpoint. Pass an explicit model path: the legacy
default is still `models/baseline/best_model.zip`.

For example, reproduce the sampled diagnostic result without overwriting its JSON:

```sh
.venv/bin/python -m training.evaluate \
  --model models/curriculum_d3/last_model.zip \
  --victim-distance 3 --episodes 100 --scenario-start 20000 --stochastic
```

`training.train_rescue_experiment` continues an explicit current checkpoint into a
new directory, preserving the source. It retains the CNN and PPO parameters, uses
training RNG seed 43, and validates deterministically every 5120 steps on the fixed
50 scenario roots 10000–10049. Validation has no shaping. The output contains config,
evaluations, best and final checkpoints. Its default budget is 30,720 additional
steps. `--navigation-shaping` enables the existing potential reward during training;
pickup/step shaping remains enabled in both conditions.

```sh
.venv/bin/python -m training.train_rescue_experiment \
  --output models/my_new_d3_potential --navigation-shaping
```

The existing viewer is incompatible with checkpoint loading: it refers to the removed
`env.navigation` attribute/constructor parameter. Its action labels also call water
“rescue outside.” It was inspected, not used or repaired in this experiment.

## How to interpret and continue

1. First learn reliable approach → pickup → return on visible victims. The deterministic
   two-cell loops and distance-3 failures are a more immediate problem than distant search.
2. Use sampled evaluation as well as deterministic evaluation. For example, a policy can
   assign left 40% and right 35% in one state; choosing the largest probability every time
   can repeat a loop, while sampling sometimes escapes. Sampling adds no navigator.
3. Compare training changes on the same fixed validation maps. Keep a separate test range
   for final comparisons, and never select the next change based on that final range.
4. Only increase distance after this stage works. Measure actual distance, initially hidden
   victims, pickups, and completed returns. Maximum placement distance alone does not
   guarantee an exploration task.

The experiment is a short pilot with one training seed per condition. Even a higher
held-out success rate does not establish a repeatable advantage for potential shaping.
No network architecture or observation/action semantics were changed. Existing checkpoints,
the original training scripts, the user's README edits, and viewer files were preserved.

Verification: 9 existing focused tests passed (memory, memory integration, reward shaping,
action masks; CNN import assertions also passed). Two new integration tests passed separately:
random evaluation/output preservation and an 8-step train/save/reload continuation with source
checkpoint hash preservation. This was not a full legacy-suite or browser-viewer validation.


## Completed training pilot and final test

Both continuations started from `no_drop/best_model.zip` at 95,000 saved timesteps
and trained 30,720 additional steps on CPU. Both finished at 125,720 timesteps.
Only the existing navigation-potential reward differed between conditions; both
retained pickup/step shaping and the inherited CNN/PPO settings. The shaping run
started before the runner's TensorBoard destination was moved into the output
directory; its TensorBoard events are in `logs/MaskablePPO_0`. Both console logs
are saved beside the experiment checkpoints.

| Additional steps | Control validation rescues / 50 | Potential validation rescues / 50 |
|---:|---:|---:|
| 5120 | 17 | 17 |
| 10240 | 19 | 20 |
| 15360 | 17 | 21 |
| 20480 | 17 | 20 |
| 25600 | 19 | 17 |
| 30720 | 21 | 19 |

The starting checkpoint rescued 19/50 on this fixed validation set. Both
continuations peaked at 21/50. Selected checkpoints were the control's 125,720-step
model and the potential run's 110,360-step model. Selection used only deterministic
validation performance, before running the final test.

Final test: roots **30000–30099**, maximum victim distance 3, minimum 1, unshaped
64-step episodes, CPU, action seed 123 for sampling. All four policies saw exactly
the same environment roots. This range is now consumed; further development should
use validation/development ranges and reserve a new final-test range.

| Checkpoint | Deterministic rescues / 100 | Sampled rescues / 100 | Sampled pickups / 100 |
|---|---:|---:|---:|
| `models/no_drop/best_model.zip` | 34 | 67 | 73 |
| `models/curriculum_d3/last_model.zip` | 28 | 66 | 80 |
| `models/experiment_d3_control_20261112/best_model.zip` | 32 | 66 | 70 |
| `models/experiment_d3_potential_20261112/best_model.zip` | 34 | 60 | 67 |

**No measured improvement from this training pilot.** Retaining potential shaping
alone did not resolve the behavior. Do not promote these experimental checkpoints
as better models, and do not infer that potential shaping can never help from this
single-seed, short continuation. Sampling helps these particular checkpoints avoid
deterministic loops, but near-exit success remains unreliable.

For the next training investigation, keep placement distances small and isolate
approach/pickup learning before adding distant search. A useful next diagnosis is
whether the local CNN features distinguish nearby victim directions reliably;
its 3×3 average pooling compresses spatial information, but this pilot did not test
whether that is the cause. No architecture change is justified as a proven fix yet.

Per-episode final results and a compact `summary.json` are in
`logs/handoff_audit_20261112/`. Existing training commands and default model choices
were left unchanged.
