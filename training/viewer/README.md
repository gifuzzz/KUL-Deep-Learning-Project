# Rescue observation and action inspector

Run from `fire-rescue-project` with the project's existing virtual environment:

```sh
.venv/bin/python -m training.web_viewer --checkpoint models/fixed_scenario/last_model.zip
```

Open **http://127.0.0.1:8765**. The inspector runs a separate simulator and uses
CPU inference. It does not modify checkpoints, training code, or a running
training process. Omit `--checkpoint` to start with manual exploration, then
choose and load a checkpoint in the page. Checkpoints must be inside `models/`
and match the current observation and action spaces. Loading or reloading a
checkpoint starts a fresh episode with the same seed.

## What you can inspect

- Physical building geometry, victims, exits, and the firefighter, with a follow
  view or whole-building view. This is global observer information, clearly
  separate from what the policy receives.
- All policy grid channels and all 15 native simulator channels, as small
  multiples and an enlarged selected layer. The exact inputs come from the
  current `RescueEnv`; unseen cells are marked without altering the tensor.
- Exact cell values before/after an action, the five agent features, and the
  combined grid/agent feature vector. Local grids move with the firefighter; before/after
  values at the same local index can refer to different world cells.
- Raw actor logits, probabilities before masking, probabilities after masking,
  masked log probabilities, the chosen action, and the critic's value estimate.
- Native FlameBox command and return values, number of commands applied,
  native rescue reward, reward shaping adjustment, returned reward, episode
  reward, `terminated`, `truncated`, `info`, and the next observation.
- A scrubber and action history, plus a full JSON export containing each recorded
  frame and transition. Complete raw tensors are available under **Complete
  action exchange**. These are inference outputs; the viewer performs no
  learning updates or gradient inspection.

Manual buttons and arrow keys move; **P** picks up/drops, **R** rescues, and **W**
waits. **Space** steps the policy when the page has focus. In the local grid,
arrow keys instead move the inspected cell. **Test masked actions** explicitly
allows invalid manual attempts so you can inspect their actual simulator result.
Policy actions always use the current mask.

The default scenario is seed 42 with a 64-step limit and victims near an exit.
Change the seed and reset to inspect another scenario. **Training rewards**
enables the environment's shaping; turning it off and resetting uses native
rescue rewards only. The current environment starts without fire, so zero fire,
smoke, and water layers are expected. Playback speed changes display cadence,
not physics ticks per action. Historical steps are read-only: return to the
latest state before taking a new live action.

## Standalone HTML replay

Record a real policy episode and inspect it without a server or Python:

```sh
.venv/bin/python -m training.web_viewer \
  --checkpoint models/fixed_scenario/last_model.zip \
  --export outputs/rescue-inspection.html
```

Open the resulting HTML file in your browser. Layer selection, before/after
inspection, values, policy outputs, playback, history, and JSON export work
offline. New manual actions, different seeds, and model loading require the
live server. Add `--sampled` to record sampled instead of highest-probability
actions. Use `--seed 0`, `--max-steps 128`, or `--port 8766` as needed.

The server binds only to `127.0.0.1`, serves only its three frontend assets,
and accepts only same-origin API requests. Use the printed numeric host rather
than `localhost`. One server is one shared inspection session; use one tab per
server to avoid concurrent controls.

```sh
.venv/bin/python -m unittest tests.test_web_viewer -v
```
