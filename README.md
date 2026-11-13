# Single-agent rescue training

The distant-rescue controller now passes **200/200 held-out scenarios** with
victims 20–40 walking cells from the nearest exit, initially outside view.
It combines observed-map planning with a CNN action policy trained by
demonstrations, DAgger, and PPO. Search/route planning is programmed; evaluation
actions are selected by the neural policy. See
[DISTANT_RESCUE_RESULTS.md](DISTANT_RESCUE_RESULTS.md) for evidence and limits.

```sh
.venv/bin/python -m training.evaluate --far
.venv/bin/python -m training.web_viewer --checkpoint models/distant_rescue/best_model.zip --seed 90073
```

Earlier near-exit experiments remain available below.

Run commands from this directory using the existing virtual environment.
Dependencies are listed in `requirements.txt`; install with
`uv pip install --python .venv/bin/python -r requirements.txt`.

```sh
.venv/bin/python -m training.train_ppo
.venv/bin/python -m training.evaluate
.venv/bin/python -m training.visualize
```

The current stage cycles through four scenarios with reset seeds 42, 1, 5, and
6, covering left, bottom, top, and right exits. Each map and victim placement
repeats exactly when its turn comes around. Episodes have a 64-step limit.
Training continues from `models/fixed_scenario/last_model.zip`, preserving its
weights, optimizer, and PPO settings, with a new 61,440-step training budget.
The checkpoint must exist before starting this stage.
Training evaluates the deterministic policy every 2048 steps on 20 validation
scenarios (seeds 10000–10019), with native rescue rewards.
TensorBoard records `eval/success_rate` and
`eval/mean_reward`. The best evaluated checkpoint is saved to
`models/four_scenarios/best_model.zip`; the final policy is saved to
`models/four_scenarios/last_model.zip`.
These paths are replaced by subsequent training runs. The original
`ppo_rescue.zip` and previous unmasked checkpoints are preserved.

Training uses SB3-Contrib's `MaskablePPO` and a mask-aware evaluation callback.
`env.action_masks()` returns seven booleans in policy action order: wait, up,
right, down, left, pick up/drop, rescue. It disables blocked moves, pickup when
no eligible victim is adjacent, and rescue without an eligible carried victim
outside. Dropping a carried victim remains valid. Wait is always available.
Masks do not choose a route or reward progress; PPO still learns its strategy.
Evaluation and visualization explicitly pass the current mask to prediction.
The environment continues to accept unmasked actions for manual interaction.
Exactly one firefighter is present after every reset, and building size remains
4. The CNN, visible exits, `ent_coef=0.01`, shaped rewards, optimizer seed, and
other PPO settings are unchanged. Training
still uses MPS; launch from a terminal where MPS is available.
Training logs use `logs/MaskablePPO_four_*` and
`logs/four_scenarios_evaluation/`. Earlier runs and their logs are preserved.
Evaluation and visualization default to the new run's best checkpoint.
Success on the four training scenarios does not establish generalization.
Validation checks all 20 scenarios once per checkpoint and determines which
checkpoint is saved as best. Standalone evaluation defaults to another 50
scenarios (seeds 20000–20049), separate from training and checkpoint selection.

A temporary CPU check with the same CNN and PPO settings first solved this
scenario at the 6144-step evaluation. After 10,240 steps, deterministic actions
succeeded in 20/20 repeated attempts at 4 steps each; sampled actions succeeded
in 20/20 at a mean of 7.2 steps. These are same-scenario results from one training
seed, not a generalization test. The temporary check did not save a project
checkpoint; the normal training command saves the configured MPS run.

Both training and evaluation log `mean_successful_ep_length` and `timeout_rate`
under their `rollout/` and `eval/` prefixes. Training metrics use the latest 100
completed episodes; evaluation metrics use the 20 validation episodes.
Successful length excludes failures; it is NaN when there are no successes.
Timeout rate is the fraction of episodes ended by the step limit (0 to 1).
A rescue on the last permitted step is a success, not a timeout. Standalone
evaluation prints the same metrics, showing timeout rate as a percentage and
`n/a` when there are no successful episodes.

The observation contains an `(11, 15, 15)` local grid at the default visibility
radius and five agent features: normalized row, normalized column, health,
normalized role, and `carrying_victim` (0 or 1). This last feature distinguishes
a carried victim from one dropped on the firefighter's cell.

The last grid channel marks visible interior exit cells with 1, and all other
cells with 0. It respects the simulator's visibility mask and does not reveal
unseen exits. It identifies exits regardless of open/blocked status; existing
geometry and movement masks determine whether passage is possible. The first
channels retain their meanings. Outward passage channels at visible exits now
match actual open/unblocked movement instead of incorrectly indicating walls.
The CNN adapts to the channel count automatically. Navigation-enabled far-rescue
models have eight agent inputs, adding the observed-map waypoint and subtask to
the original five. Use the matching far-rescue profile when loading them.

The grid input changed from 10 to 11 channels; the earlier carrying feature also
changed the agent input from four to five values. Previous checkpoints cannot
be used directly with this observation space. Train a fresh model before
running evaluation or visualization. Earlier checkpoints require their original
observation code and, for standard PPO, their original evaluation code.

Standalone evaluation defaults to the 50 test scenarios and a 64-step limit;
`--random-scenarios` explicitly selects that same mode. Use
`--training-scenarios` to check the four training scenarios separately. Both
evaluation and visualization accept an optional checkpoint path, for example
`.venv/bin/python -m training.evaluate models/four_scenarios/last_model`.
Visualization accepts `--seed`, defaulting to the familiar seed 42:

```sh
.venv/bin/python -m training.evaluate --training-scenarios
.venv/bin/python -m training.evaluate
.venv/bin/python -m training.visualize --seed 6
```

The environment varies scenarios freely by default. `fixed_seed=42` repeats one
scenario; `scenario_seeds=(42, 1, 5, 6)` cycles through a set. An explicit seeded
reset restarts the cycle; ordinary episode resets advance it. These two modes
cannot be enabled together.

Training rewards are +10 per rescue, +1 for the first pickup of each victim,
and -0.001 per step. Dropping and picking up the same victim does not earn
another pickup bonus. Evaluation uses only the native +10 rescue reward.
Compare success rates rather than shaped training reward between experiments.
Checkpoint selection detects and preserves better policies; it does not itself
fix learning instability.

```sh
.venv/bin/python -m unittest tests.test_action_masks tests.test_evaluation tests.test_exit_observation tests.test_fixed_scenario tests.test_scenario_set -v
```

Compare CPU and MPS training speed from your normal terminal:

```sh
.venv/bin/python -m training.benchmark
```

This uses the training CNN, masks, and shaped rewards, with 10 epochs and batch
size 64. Each device runs three fresh seeded models, each with 1024 untimed
warmup steps and 2048 timed steps. It prints individual times, median throughput,
and the faster device. Timing includes simulation and PPO updates; it excludes
evaluation, model saving, and initialization. No checkpoints or logs are written.
Unavailable MPS is skipped; unsupported MPS operations are reported.

For a quicker test or to compare a different update configuration:

```sh
.venv/bin/python -m training.benchmark --repeats 1 --steps 1024
.venv/bin/python -m training.benchmark --epochs 4 --batch-size 256
```

Stop other training jobs before benchmarking. CPU and MPS can sample different
actions despite sharing a seed; this measures throughput, not learning quality.
