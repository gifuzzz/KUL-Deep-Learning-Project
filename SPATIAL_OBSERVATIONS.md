# Spatial observation screening

Branch: `experiment/spatial-observations`.

## Change

`RescueEnv(memory_agent_marker=True)` appends channel 13 to the remembered map:
1 at the firefighter's current interior cell, 0 everywhere else. The original 13
channels are unchanged, including explored mask 11 and building mask 12. The marker
is regenerated, not remembered. On outside grass it is empty; the existing agent
coordinates still report the outside position. It reveals no unseen geometry or
victims and computes no routes.

The option defaults to false. Existing observation shapes, action meanings, source
checkpoints and training entry points are preserved. Marker checkpoints require the
14-channel environment; SB3's space checks reject 13/14-channel mismatches.

`RescueCNN` already derives its channel count from the observation space. No change
to its convolutions, pooling, fusion, policy heads or PPO algorithm was needed.
Baseline policy: 274,888 parameters. Marker policy: 275,176 (+288, approximately 0.1%).
Adding a channel also changes random initialization after that layer; the paired
runs share random seeds, not identical weights throughout the network.

## Protocol

The new `--fresh` option in `training.train_rescue_experiment` permits a matched
comparison without loading an old model into the changed observation space.
`--memory-agent-marker` is supported by both training and evaluation.

- Fresh baseline and marker policies, paired training seeds 43, 44 and 45.
- 30,720 additional environment steps per run (six runs, 184,320 steps total).
- MPS training/validation; one CPU Torch thread. MPS was available in this session.
- Same PPO parameters as `training.train_ppo`: learning rate 1e-4, n_steps 1024,
  batch size 64, four epochs, gamma .99, entropy coefficient .01.
- Placement distance maximum 3, minimum 1; 64-step episodes.
- Pickup/step and navigation-potential shaping enabled in both training conditions.
  The potential uses privileged simulator state as reward guidance, not policy input.
- Unshaped deterministic validation every 5120 steps on fixed roots 10000–10049.
- Checkpoints selected only on validation. New final-test roots 40000–40099 are
  reserved until all training/selection completes. Previous roots 30000–30099 were
  already consumed by the earlier experiment.
- Report deterministic and sampled rescues separately, plus pickup and carrying
  timeout diagnostics, parameters and training wall time.

These are short screening runs, not an asymptotic comparison. The 100 final scenario
roots will be shared by all policies; three policies on those roots are not 300
independent scenario draws. A weak result does not establish that a representation
cannot learn with a larger budget.

Artifacts are preserved in `models/marker_ablation_20261112/`: the trial plan, source
snapshots, training logs, per-run configurations, evaluation archives and best/final
checkpoints. Previous experiment outputs are untouched.

Example commands, from the project directory:

```sh
.venv/bin/python -m training.train_rescue_experiment --fresh \
  --memory-agent-marker --navigation-shaping --seed 43 --steps 30720 \
  --output models/my_new_marker_trial

.venv/bin/python -m training.evaluate \
  --model models/my_new_marker_trial/best_model.zip \
  --memory-agent-marker --victim-distance 3 --stochastic
```

The training runner requires a new output directory. Evaluation's `--output` similarly
refuses to overwrite an existing JSON result. Pass `--device mps` or `--device cpu` to
choose explicitly; `auto` selects MPS when available, otherwise CPU.

## Verification

Marker tests check original-channel equality against the baseline, unchanged actions
and masks, movement/reset without a marker trail, an empty marker on outside rescue,
CNN feature shape/finiteness, and rejection of an old 13-channel checkpoint. CLI tests
exercise random evaluation and protected result output with the marker both on and off.
The existing source-checkpoint-preservation integration test still passes.

No physical wall geometry, hidden-information visibility, local window size, or
checkpoint action semantics changed. Those comparisons remain separate experiments.

## Additional diagnosis: leaving the building before pickup

Rechecking the previous distance-3 deterministic diagnostic traces (roots 20000–20099),
59 of the 71 failed episodes entered outside grass; failed episodes spent 2264 steps
there in total. Native `observe()` returns no local visible cells when the firefighter
is outside the native building (`sim.cpp`, the `valid(ag.r, ag.c)` visibility guard).
Memory and scalar position remain available. This can make recovery after an incorrect
exit harder; it does not prove why the policy initially chose the wrong move. It is
also a consequence of behavior, not evidence that exterior blindness alone causes the
failures. Exterior visibility was not changed during the marker comparison.

Final focused verification: **15 passed** in the combined memory, marker, action-mask,
reward-shaping, evaluation-CLI and experiment-runner checks. The CNN import-time
assertions passed as well. `git diff --check` passed. A read-only subagent review
found no correctness issues in the marker implementation or matching CLI flags.
The legacy viewer and full historical test suite were not repaired or validated.

## Completed screening results

All six runs completed their preset budget: **184,320 training steps total**,
with training and final evaluation on MPS. Checkpoints were frozen using validation
before the final roots 40000–40099 were used. All 100 final victims were visible
initially. This is an approach/pickup/return test, not distant exploration.

| Training seed | Baseline validation /50 | Marker validation /50 | Baseline deterministic /100 | Marker deterministic /100 | Baseline sampled /100 | Marker sampled /100 |
|---:|---:|---:|---:|---:|---:|---:|
| 43 | 16 | 22 | 16 | 32 | 41 | 51 |
| 44 | 20 | 12 | 24 | 12 | 49 | 38 |
| 45 | 13 | 18 | 22 | 24 | 40 | 48 |

Mean final success over the three training seeds:

- Deterministic: baseline **20.7%**, marker **22.7%**.
- Sampled: baseline **43.3%**, marker **45.7%**.

The averages describe three training seeds evaluated on the same 100 scenario roots,
not 300 independently generated maps. Sampled evaluation uses action seed 123
(reset per scenario as 123 + scenario root). No architecture was selected or tuned
from these final results. Roots 40000–40099 are now consumed.

The table below reports the median number of distinct post-action positions in
failed episodes, and failed episodes that spent at least one step outside:

| Policy | Deterministic median positions in failures | Deterministic failures ever outside | Sampled pickups /100 | Sampled carrying timeouts /100 |
|---|---:|---:|---:|---:|
| baseline_s43 | 29.0 | 44 | 64 | 23 |
| marker_s43 | 31.0 | 64 | 68 | 17 |
| baseline_s44 | 25.0 | 71 | 64 | 15 |
| marker_s44 | 30.0 | 32 | 65 | 27 |
| baseline_s45 | 29.0 | 53 | 69 | 29 |
| marker_s45 | 30.5 | 65 | 63 | 15 |

Training wall time per run includes validation; it is not a controlled speed benchmark:

| Run | Wall seconds | Selected checkpoint step |
|---|---:|---:|
| baseline_s43 | 176.1 | 20480 |
| marker_s43 | 170.8 | 30720 |
| baseline_s44 | 168.1 | 25600 |
| marker_s44 | 181.6 | 30720 |
| baseline_s45 | 177.0 | 30720 |
| marker_s45 | 175.8 | 30720 |

Full per-episode results, scenario visibility/distance metadata, loop/outside diagnostics,
selected-checkpoint hashes and evaluator source are in
`models/marker_ablation_20261112/final_test/`. The default observation remains
13 channels; no experimental checkpoint was promoted to the default model.

## Interpretation

The marker won two seed pairs but lost one, with only +2.0 percentage points in
mean deterministic success and +2.3 points in mean sampled success. That is not a
clear or repeatable improvement from this short screen. Mean sampled pickup rates
were almost identical: baseline 65.7%, marker 65.3%. Do not present the marker as a
solution to approaching victims.

These fresh 30,720-step policies fail differently from the older distance-3 checkpoint:
failed deterministic episodes visited median counts of 25–31 distinct positions,
rather than the older checkpoint's median of two. Wandering through outside grass
is not evidence of useful exploration. Return after pickup also varies substantially
by training seed. Replayed carrying failures also end in two-cell oscillations:
baseline seed-44 policy on scenario 40004 alternates (29,31)/(29,30), and its
marker counterpart on scenario 40002 alternates (35,0)/(34,0). Thus wider wandering
does not imply loops are eliminated. Full traces are saved in
`final_test/representative_carrying_failures.json` beneath the experiment directory.
No network capacity limit, pooling failure, or need for recurrence
has been established by this trial.

The marker remains opt-in on the experiment branch. Existing production defaults and
checkpoints are preserved. A useful next representation comparison would retain finer
local spatial information, or test lossless wall rasterization while keeping physical
geometry and information visibility fixed. Neither wall encoding nor full revelation
was changed in this marker screen.
