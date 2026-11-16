# FlameBox Visual Agent Debugger — User Guide & Inspection Manual

## 1. Quickstart: Launching the Debugger

The FlameBox Visual Agent Debugger runs as a local web application on your Mac.

### Launch Command
From `fire-rescue-project-2.0`:
```bash
.venv/bin/python -m training.visual_debugger --port 8000
```
*(Alternatively, `.venv/bin/python -m training.web_viewer` also launches the visual debugger).*

Open your browser to:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 2. Interface Walkthrough

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🔥 FlameBox Visual Agent Debugger                  [READY] [DISTANT RESCUE]     │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Trajectories: [Category: All ▾] [Trajectory: Distant Seed 50068 (Det) ▾] [Load] │
│ Live Sim    : [Checkpoint: final_model.zip ▾] [Tier: Distant ▾] [Seed: 50068]   │
│               [Mode: Deterministic ▾]  [Run & Inspect]                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│ [⏮] [◀] [▶ Play] [▶] [⏭]  Speed: [1x ▾]  Step 14 of 72                          │
│ [━━━━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━] │
│        ▲ first_discovery (step 8)  ▲ first_pickup (step 14)  ▲ rescue (step 72) │
├─────────────────────────┬─────────────────────────┬─────────────────────────────┤
│ PANEL 1: Ground Truth   │ PANEL 2: Agent View     │ PANEL 3: Decisions          │
│ (Privileged Simulation) │ (Exact Policy Inputs)   │ (Neural Probabilities)      │
│                         │                         │                             │
│ • Full building layout  │ • [Local] [Memory]      │ • Wait          [ 0.1%] -4.2│
│ • Real walls & openings │ • Channel: [Combined ▾] │ • Move Up       [ 0.2%] -3.1│
│ • Firefighter position  │ • Exact float on hover  │ • Move Right ▶  [88.4%] +2.9│
│ • Hidden victim marker  │ • 5 Agent Scalars       │ • Move Down     [ 9.2%] +0.6│
│ • Visited trail history │ • Previous Action       │ • Move Left     [ 0.1%] -5.0│
│ • Exit boundaries       │ • Legal action masks    │ • Pick/Drop     [ 0.0%] MSK │
│                         │                         │ • Rescue Outside[ 2.0%] -1.2│
│                         │                         │                             │
│                         │                         │ V(s): +9.84 | ||h||₂: 10.38 │
│                         │                         │ Step: +0.00 | Cum: +1.20    │
├─────────────────────────┴─────────────────────────┴─────────────────────────────┤
│ Outcome: SUCCESS (+10 Rescue) | Steps: 72 | Coverage: 48.2% | Oscillations: 0   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Navigating the Synchronized Panels

### Panel 1 — Ground-Truth Simulation View (Left)
- **What it shows:** The complete simulation world, including hidden walls, unobserved rooms, hidden victims, and exterior exits.
- **Visual Legend:**
  - 🟠 **Orange Circle:** Firefighter agent.
  - ⭐ **Gold Star / Ring:** Indicates the firefighter is actively carrying a victim.
  - 🟣 **Magenta Circle with Cross:** Victim location.
  - 🟢 **Glowing Green Cells:** Exterior exits where victims can be evacuated.
  - 🟫 **Brown Blocks:** Debris obstacles blocking movement.
  - ⬛ **Dark Borders:** Solid impassable walls.
  - 🔹 **Dotted Blue Line:** Movement trail showing the firefighter's path from step 0 to the current frame.
- **Interactive Tooltip:** Hovering over any grid cell displays its exact coordinate `[Row, Col]` and zone compartment ID.

### Panel 2 — Agent's Actual View (Center)
- **What it shows:** The exact input tensors received by the neural network policy at inference time.
- **View Toggles:**
  - **Local Observation (15×15):** The 9-channel square local field-of-view centered around the firefighter.
  - **Global Remembered Map (40×52):** The 9-channel memory map cropped to the active building footprint.
- **Channel Selector:**
  - **Combined Semantic View (Default):** Composite rendering overlaying passability walls (cyan), debris (brown), explored cells (light blue), known victim (magenta), known exit (green), and firefighter position (orange).
  - **Ch 0 – 3:** Directional passability (North, East, South, West).
  - **Ch 4:** Debris / rubble obstacle mask.
  - **Ch 5:** Explored / line-of-sight visitation mask.
  - **Ch 6:** Known victim position.
  - **Ch 7:** Known exterior exit locations.
  - **Ch 8:** Firefighter position.
- **Interactive Cell Tooltip:** Hovering over any cell in Panel 2 displays the exact floating-point tensor value stored in that cell (e.g. `1.0000` or `0.0000`).
- **Auxiliary Inputs Card:**
  - **Agent Scalars:** Normalized `[Row, Col, Health, Role, Carrying]`.
  - **Previous Action:** Name and index of the primitive action taken on the preceding step.
  - **Action Masks:** Visual indicators showing which actions are legally permitted (`✓`) or masked (`✗`).

### Panel 3 — Neural Decisions & Recurrent Diagnostics (Right)
- **What it shows:** The policy's inner decision-making process at the exact current timestep.
- **Action Probabilities Bar Chart:**
  - Displays all 7 primitive actions: `Wait`, `Move Up`, `Move Right`, `Move Down`, `Move Left`, `Pick Up / Drop`, `Rescue Outside`.
  - **Solid Blue Fill:** Probability percentage ($P(a)$) from the masked softmax distribution.
  - **Gold Highlight:** The selected action executed by the agent.
  - **Strikethrough / Red Border:** Masked (illegal) actions strictly blocked by the environment.
  - **Raw Logit Value:** Unnormalized pre-softmax logit displayed beside each bar.
- **Recurrent KPI Cards:**
  - **Critic Value $V(s)$:** Expected cumulative discounted return estimated by the value head.
  - **GRU Hidden Norm $||h_t||_2$:** L2 norm of the 128-unit recurrent memory vector, reflecting memory activation strength.
  - **Step Reward & Cumulative Reward:** Reward received on the transition.
  - **Carrying Status & Victim Visibility:** Real-time state flags.

---

## 4. Key Event Markers on the Timeline

Along the timeline scrubber, colored marker dots pinpoint critical milestones and behavioral failures:
- 🔵 **Blue Marker (`first_discovery`):** Timestep when a victim first enters the agent's line-of-sight.
- 🟡 **Yellow Marker (`first_pickup`):** Timestep when the agent successfully executes "Pick Up" and starts carrying.
- 🟢 **Green Marker (`rescue_completed`):** Timestep when the agent steps onto an exit cell and evacuates the victim.
- 🔴 **Red Marker (`two_cell_oscillation`):** Timestep where the agent cycles between two adjacent cells.
- 🟣 **Purple Marker (`room_transition`):** Timestep where the agent passes through a doorway into a new room.
- 🟠 **Orange Marker (`timeout`):** Timestep where the episode exhausts its maximum step budget.

> **Tip:** Click directly on any event marker dot along the slider (or in the "Episode Key Events" list) to instantly jump the viewer to that exact moment!

---

## 5. Keyboard Shortcuts

| Key | Action |
|---|---|
| <kbd>Space</kbd> | Toggle Play / Pause |
| <kbd>←</kbd> (Left Arrow) | Step backward one frame |
| <kbd>→</kbd> (Right Arrow) | Step forward one frame |
| <kbd>Home</kbd> | Jump to start (Step 0) |
| <kbd>End</kbd> | Jump to final step |

---

## 6. Inspecting Specific Failure Modes

### Example 1: Doorway Oscillation
1. Open the trajectory dropdown.
2. Select `[Distant] Seed 50027 · FAIL · 384s (Det) - Doorway Oscillation`.
3. Scrub to the red marker around step 310.
4. Observe Panel 1: The firefighter is carrying a victim and reaches the door connecting Zone 0 and Zone 1.
5. Observe Panel 3: Action probabilities alternate between `Move Left` and `Move Right` with near equal logits, causing the agent to repeatedly flip across the doorway threshold until timeout.

### Example 2: Complete Distant Multi-Room Rescue
1. Select `[Distant] Seed 50068 · 🏆 RESCUED · 72s (Det) - Successful Distant Rescue`.
2. Press <kbd>Space</kbd> to watch playback at 1x or 2x speed.
3. Notice:
   - At Step 8 (blue marker): Victim is discovered deep in an interior room.
   - At Step 14 (yellow marker): Firefighter enters room and picks up victim.
   - Steps 15–71: Firefighter navigates out of the room, through the central corridor, directly to the south exit.
   - At Step 72 (green marker): Firefighter steps onto the exterior exit; reward leaps to `+10.00`!
