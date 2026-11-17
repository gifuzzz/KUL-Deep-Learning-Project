/**
 * FlameBox Visual Agent Debugger Frontend Controller.
 * Synchronizes Panel 1 (Ground Truth), Panel 2 (Policy Inputs), and Panel 3 (Decisions).
 */

const ACTION_NAMES = [
  "Wait",
  "Move Up",
  "Move Right",
  "Move Down",
  "Move Left",
  "Pick Up / Drop",
  "Water Hose",
];

const MEMORY_CHANNEL_NAMES = [
  "Ch 0: Passability North",
  "Ch 1: Passability East",
  "Ch 2: Passability South",
  "Ch 3: Passability West",
  "Ch 4: Debris / Obstacle",
  "Ch 5: Explored / Visited",
  "Ch 6: Known Victim",
  "Ch 7: Known Exit",
  "Ch 8: Agent Position",
  "Ch 9: Spatial Visit Memory",
];

// App State
let trajectory = null;
let currentStepIdx = 0;
let isPlaying = false;
let playInterval = null;
let playbackSpeed = 1.0;
let activeViewMode = "local"; // "local" (15x15) or "memory" (40x52)
let activeChannel = "composite"; // "composite" or 0..8
let allSavedTrajectories = [];

// DOM Elements
const statusBadge = document.getElementById("status-badge");
const episodeTag = document.getElementById("episode-tag");
const stepBadge = document.getElementById("step-badge");
const errorBanner = document.getElementById("error-banner");

const categoryFilter = document.getElementById("category-filter");
const trajectorySelect = document.getElementById("trajectory-select");
const btnLoadTraj = document.getElementById("btn-load-traj");

const checkpointSelect = document.getElementById("checkpoint-select");
const tierSelect = document.getElementById("tier-select");
const seedInput = document.getElementById("seed-input");
const policyMode = document.getElementById("policy-mode");
const btnRunLive = document.getElementById("btn-run-live");

const btnFirst = document.getElementById("btn-first");
const btnPrev = document.getElementById("btn-prev");
const btnPlay = document.getElementById("btn-play");
const btnNext = document.getElementById("btn-next");
const btnLast = document.getElementById("btn-last");
const playbackSpeedSelect = document.getElementById("playback-speed");

const timelineSlider = document.getElementById("timeline-slider");
const timelineMarkers = document.getElementById("timeline-markers");
const scrubberStepInfo = document.getElementById("scrubber-step-info");

const worldCanvas = document.getElementById("world-canvas");
const worldCtx = worldCanvas.getContext("2d");
const chkVisitedTrail = document.getElementById("chk-visited-trail");
const worldHoverInfo = document.getElementById("world-hover-info");

const agentCanvas = document.getElementById("agent-canvas");
const agentCtx = agentCanvas.getContext("2d");
const btnToggleLocal = document.getElementById("btn-toggle-local");
const btnToggleMemory = document.getElementById("btn-toggle-memory");
const channelSelect = document.getElementById("channel-select");
const channelDimsBadge = document.getElementById("channel-dims-badge");
const agentHoverInfo = document.getElementById("agent-hover-info");

const scalarRow = document.getElementById("scalar-row");
const scalarCol = document.getElementById("scalar-col");
const scalarHealth = document.getElementById("scalar-health");
const scalarRole = document.getElementById("scalar-role");
const scalarCarrying = document.getElementById("scalar-carrying");
const prevActionName = document.getElementById("prev-action-name");
const actionMasksList = document.getElementById("action-masks-list");

const actionBarsList = document.getElementById("action-bars-list");
const kpiValue = document.getElementById("kpi-value");
const kpiGru = document.getElementById("kpi-gru");
const kpiStepReward = document.getElementById("kpi-step-reward");
const kpiCumReward = document.getElementById("kpi-cum-reward");
const summaryAction = document.getElementById("summary-action");
const summaryCarrying = document.getElementById("summary-carrying");
const summaryDiscovered = document.getElementById("summary-discovered");
const eventsLogList = document.getElementById("events-log-list");

const metaSeed = document.getElementById("meta-seed");
const metaTier = document.getElementById("meta-tier");
const metaMode = document.getElementById("meta-mode");
const metaModel = document.getElementById("meta-model");
const metaHash = document.getElementById("meta-hash");

const perfOutcome = document.getElementById("perf-outcome");
const perfSteps = document.getElementById("perf-steps");
const perfCoverage = document.getElementById("perf-coverage");
const perfOscillations = document.getElementById("perf-oscillations");
const perfDiscovery = document.getElementById("perf-discovery");
const perfPickup = document.getElementById("perf-pickup");
const perfRescue = document.getElementById("perf-rescue");


// Initialize on page load
window.addEventListener("DOMContentLoaded", async () => {
  setupEventListeners();
  await loadInitialData();
});

function setupEventListeners() {
  // Playback buttons
  btnPlay.addEventListener("click", togglePlay);
  btnFirst.addEventListener("click", () => jumpToStep(0));
  btnPrev.addEventListener("click", () => jumpToStep(currentStepIdx - 1));
  btnNext.addEventListener("click", () => jumpToStep(currentStepIdx + 1));
  btnLast.addEventListener("click", () => {
    if (trajectory && trajectory.steps) jumpToStep(trajectory.steps.length - 1);
  });

  playbackSpeedSelect.addEventListener("change", (e) => {
    playbackSpeed = parseFloat(e.target.value);
    if (isPlaying) {
      clearInterval(playInterval);
      startPlaybackTimer();
    }
  });

  timelineSlider.addEventListener("input", (e) => {
    jumpToStep(parseInt(e.target.value, 10));
  });

  // Toggles & Channels
  btnToggleLocal.addEventListener("click", () => {
    activeViewMode = "local";
    btnToggleLocal.classList.add("active");
    btnToggleMemory.classList.remove("active");
    channelDimsBadge.textContent = "9 × 15 × 15";
    renderAgentView();
  });

  btnToggleMemory.addEventListener("click", () => {
    activeViewMode = "memory";
    btnToggleMemory.classList.add("active");
    btnToggleLocal.classList.remove("active");
    channelDimsBadge.textContent = "9 × 40 × 52";
    renderAgentView();
  });

  channelSelect.addEventListener("change", (e) => {
    activeChannel = e.target.value;
    renderAgentView();
  });

  chkVisitedTrail.addEventListener("change", () => {
    renderWorldView();
  });

  // Trajectory browser
  categoryFilter.addEventListener("change", populateTrajectoryDropdown);
  btnLoadTraj.addEventListener("click", loadSelectedTrajectory);

  // Live simulation
  btnRunLive.addEventListener("click", runLiveSimulation);

  // Keyboard shortcuts
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.code === "Space") {
      e.preventDefault();
      togglePlay();
    } else if (e.code === "ArrowLeft") {
      e.preventDefault();
      jumpToStep(currentStepIdx - 1);
    } else if (e.code === "ArrowRight") {
      e.preventDefault();
      jumpToStep(currentStepIdx + 1);
    } else if (e.code === "Home") {
      e.preventDefault();
      jumpToStep(0);
    } else if (e.code === "End") {
      e.preventDefault();
      if (trajectory && trajectory.steps) jumpToStep(trajectory.steps.length - 1);
    }
  });

  // Canvas tooltips
  worldCanvas.addEventListener("mousemove", onWorldCanvasHover);
  agentCanvas.addEventListener("mousemove", onAgentCanvasHover);
}

async function loadInitialData() {
  try {
    showError(null);
    statusBadge.textContent = "Loading...";

    // 1. Fetch checkpoints
    const cpRes = await fetch("/api/checkpoints");
    const cpData = await cpRes.json();
    checkpointSelect.innerHTML = "";
    cpData.checkpoints.forEach((cp) => {
      const opt = document.createElement("option");
      opt.value = cp.path;
      opt.textContent = `${cp.filename} (${cp.size_kb} KB)`;
      checkpointSelect.appendChild(opt);
    });

    // 2. Fetch saved trajectories
    const trajRes = await fetch("/api/trajectories");
    const trajData = await trajRes.json();
    allSavedTrajectories = trajData.trajectories || [];
    populateTrajectoryDropdown();

    // Auto-load first trajectory if available
    if (allSavedTrajectories.length > 0) {
      await loadTrajectoryByFilename(allSavedTrajectories[0].filename);
    } else {
      statusBadge.textContent = "Ready (No Trajectories)";
    }
  } catch (err) {
    showError(`Failed to load initial metadata: ${err.message}`);
    statusBadge.textContent = "Error";
  }
}

function populateTrajectoryDropdown() {
  const selectedCat = categoryFilter.value;
  trajectorySelect.innerHTML = "";

  const filtered = allSavedTrajectories.filter((t) => {
    if (selectedCat === "all") return true;
    return t.category === selectedCat;
  });

  if (filtered.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "(No matching trajectories)";
    trajectorySelect.appendChild(opt);
    return;
  }

  filtered.forEach((t) => {
    const opt = document.createElement("option");
    opt.value = t.filename;
    const mode = t.deterministic ? "Det" : "Stoch";
    const status = t.rescue_completed ? "🏆 RESCUED" : (t.timeout ? "⏰ TIMEOUT" : "FAIL");
    opt.textContent = `[${t.tier}] Seed ${t.seed} · ${status} · ${t.total_steps}s (${mode}) - ${t.category_readable}`;
    trajectorySelect.appendChild(opt);
  });
}

async function loadSelectedTrajectory() {
  const filename = trajectorySelect.value;
  if (!filename) return;
  await loadTrajectoryByFilename(filename);
}

async function loadTrajectoryByFilename(filename) {
  try {
    pause();
    showError(null);
    statusBadge.textContent = "Loading Trajectory...";

    const res = await fetch(`/api/trajectory?file=${encodeURIComponent(filename)}`);
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const data = await res.json();

    setTrajectoryData(data);
    statusBadge.textContent = "Replaying";
    statusBadge.className = "badge ready";
  } catch (err) {
    showError(`Error loading trajectory: ${err.message}`);
    statusBadge.textContent = "Error";
    statusBadge.className = "badge error";
  }
}

async function runLiveSimulation() {
  const modelPath = checkpointSelect.value;
  if (!modelPath) {
    showError("Please select a checkpoint first");
    return;
  }

  const seed = parseInt(seedInput.value, 10);
  const tier = tierSelect.value;
  const deterministic = policyMode.value === "deterministic";

  pause();
  showError(null);
  btnRunLive.disabled = true;
  btnRunLive.textContent = "Executing...";
  statusBadge.textContent = "Running Simulation...";
  statusBadge.className = "badge ready";

  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_path: modelPath,
        seed: seed,
        tier: tier,
        deterministic: deterministic,
        save_trajectory: true,
      }),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Execution error: ${errText}`);
    }

    const data = await res.json();
    setTrajectoryData(data);

    // Refresh trajectories list in background
    const trajRes = await fetch("/api/trajectories");
    const trajData = await trajRes.json();
    allSavedTrajectories = trajData.trajectories || [];
    populateTrajectoryDropdown();

    statusBadge.textContent = "Simulation Ready";
  } catch (err) {
    showError(`Failed to run live episode: ${err.message}`);
    statusBadge.textContent = "Execution Failed";
  } finally {
    btnRunLive.disabled = false;
    btnRunLive.textContent = "Run & Inspect";
  }
}

function setTrajectoryData(data) {
  trajectory = data;
  currentStepIdx = 0;

  const totalSteps = trajectory.steps ? trajectory.steps.length : 0;
  timelineSlider.max = Math.max(0, totalSteps - 1);
  timelineSlider.value = 0;

  // Metadata cards
  const meta = trajectory.metadata || {};
  const summary = trajectory.summary || {};

  metaSeed.textContent = meta.seed !== undefined ? meta.seed : "—";
  metaTier.textContent = meta.tier || "—";
  metaMode.textContent = meta.deterministic ? "Deterministic (Argmax)" : "Sampled (Stochastic)";
  metaModel.textContent = meta.model_path ? meta.model_path.split("/").pop() : "—";
  metaHash.textContent = meta.model_hash || "—";

  episodeTag.textContent = summary.category_readable || "Active Run";
  if (summary.rescue_completed) {
    episodeTag.style.borderColor = "var(--accent-green)";
    episodeTag.style.color = "var(--accent-green)";
  } else if (summary.category === "doorway_oscillation") {
    episodeTag.style.borderColor = "var(--accent-red)";
    episodeTag.style.color = "var(--accent-red)";
  } else {
    episodeTag.style.borderColor = "var(--accent-blue)";
    episodeTag.style.color = "var(--accent-blue)";
  }

  perfOutcome.textContent = summary.rescue_completed ? "SUCCESS (+10 Rescue)" : (summary.timeout ? "TIMEOUT" : "FAILED");
  perfOutcome.style.color = summary.rescue_completed ? "var(--accent-green)" : "var(--accent-red)";
  perfSteps.textContent = summary.total_steps || totalSteps;
  perfCoverage.textContent = summary.explored_coverage ? `${(summary.explored_coverage * 100).toFixed(1)}%` : "—";
  perfOscillations.textContent = summary.oscillation_count !== undefined ? summary.oscillation_count : 0;
  perfDiscovery.textContent = summary.discovery_step !== null && summary.discovery_step !== undefined ? `Step ${summary.discovery_step}` : "None";
  perfPickup.textContent = summary.pickup_step !== null && summary.pickup_step !== undefined ? `Step ${summary.pickup_step}` : "None";
  perfRescue.textContent = summary.rescue_step !== null && summary.rescue_step !== undefined ? `Step ${summary.rescue_step}` : "None";

  // Timeline Event Markers
  renderTimelineMarkers();

  // Events Log list
  renderEventsLog();

  // Render initial step
  jumpToStep(0);
}

function renderTimelineMarkers() {
  timelineMarkers.innerHTML = "";
  if (!trajectory || !trajectory.events || !trajectory.steps || trajectory.steps.length <= 1) return;

  const total = trajectory.steps.length - 1;
  trajectory.events.forEach((ev) => {
    const dot = document.createElement("div");
    dot.className = `event-marker-dot ${ev.type}`;
    const pct = (ev.step / total) * 100;
    dot.style.left = `${pct}%`;
    dot.title = `Step ${ev.step}: ${ev.description}`;
    dot.addEventListener("click", () => jumpToStep(ev.step));
    timelineMarkers.appendChild(dot);
  });
}

function renderEventsLog() {
  eventsLogList.innerHTML = "";
  if (!trajectory || !trajectory.events || trajectory.events.length === 0) {
    eventsLogList.innerHTML = '<p class="empty-hint">No significant failure or milestone events detected.</p>';
    return;
  }

  trajectory.events.forEach((ev) => {
    const item = document.createElement("div");
    item.className = `event-log-item ${ev.type}`;
    item.innerHTML = `<span><strong>Step ${ev.step}:</strong> ${ev.description}</span><span class="badge step">Go</span>`;
    item.addEventListener("click", () => jumpToStep(ev.step));
    eventsLogList.appendChild(item);
  });
}

function jumpToStep(step) {
  if (!trajectory || !trajectory.steps || trajectory.steps.length === 0) return;
  const clamped = Math.max(0, Math.min(step, trajectory.steps.length - 1));
  currentStepIdx = clamped;
  timelineSlider.value = clamped;

  scrubberStepInfo.textContent = `Step ${clamped} of ${trajectory.steps.length - 1}`;
  stepBadge.textContent = `Step ${clamped} / ${trajectory.steps.length - 1}`;

  renderFrame(trajectory.steps[clamped]);
}

function togglePlay() {
  if (isPlaying) {
    pause();
  } else {
    play();
  }
}

function play() {
  if (!trajectory || !trajectory.steps || trajectory.steps.length === 0) return;
  isPlaying = true;
  btnPlay.textContent = "⏸ Pause";
  btnPlay.style.backgroundColor = "var(--accent-yellow)";

  if (currentStepIdx >= trajectory.steps.length - 1) {
    currentStepIdx = 0;
  }
  startPlaybackTimer();
}

function pause() {
  isPlaying = false;
  btnPlay.textContent = "▶ Play";
  btnPlay.style.backgroundColor = "var(--accent-green)";
  if (playInterval) {
    clearInterval(playInterval);
    playInterval = null;
  }
}

function startPlaybackTimer() {
  const intervalMs = Math.max(20, Math.round(1000 / (4 * playbackSpeed)));
  playInterval = setInterval(() => {
    if (currentStepIdx < trajectory.steps.length - 1) {
      jumpToStep(currentStepIdx + 1);
    } else {
      pause();
    }
  }, intervalMs);
}


/* -------------------------------------------------------------
 * RENDERING: Panel 1 (Ground Truth World Canvas)
 * ------------------------------------------------------------- */
function renderWorldView() {
  if (!trajectory || !trajectory.steps || trajectory.steps.length === 0) return;
  const step = trajectory.steps[currentStepIdx];
  const staticGT = trajectory.static_ground_truth || {};
  const H = staticGT.height || 15;
  const W = staticGT.width || 18;

  const canvasW = worldCanvas.width;
  const canvasH = worldCanvas.height;
  worldCtx.clearRect(0, 0, canvasW, canvasH);

  const cellMargin = 20;
  const cellW = (canvasW - 2 * cellMargin) / W;
  const cellH = (canvasH - 2 * cellMargin) / H;
  const cellSize = Math.min(cellW, cellH);

  const offsetX = (canvasW - W * cellSize) / 2;
  const offsetY = (canvasH - H * cellSize) / 2;

  // 1. Draw zones / rooms
  const isEdgeMode = (staticGT.geometry_mode === "edge");
  const zones = staticGT.zones || [];
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const z = zones[r] ? zones[r][c] : -1;
      let fill = "#131b26"; // Exterior
      if (z >= 0) {
        // Distinct room tint
        const hue = (z * 65 + 180) % 360;
        fill = `hsl(${hue}, 28%, 14%)`;
      } else if (!isEdgeMode) {
        // Legacy cell wall block
        fill = "#1e293b";
      }
      worldCtx.fillStyle = fill;
      worldCtx.fillRect(offsetX + c * cellSize, offsetY + r * cellSize, cellSize, cellSize);

      // Subtle cell border
      worldCtx.strokeStyle = "#1a2433";
      worldCtx.lineWidth = 0.5;
      worldCtx.strokeRect(offsetX + c * cellSize, offsetY + r * cellSize, cellSize, cellSize);
    }
  }

  // 2. Draw physical walls (where south == 0 or east == 0 between interior cells)
  const south = staticGT.south || [];
  const east = staticGT.east || [];
  worldCtx.strokeStyle = "#475569";
  worldCtx.lineWidth = 2.5;

  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const x = offsetX + c * cellSize;
      const y = offsetY + r * cellSize;
      const zCur = zones[r] ? zones[r][c] : -1;

      // South edge
      if (r < H - 1) {
        const isSouthPassable = south[r] && south[r][c] === 1;
        if (!isSouthPassable && (zCur >= 0 || isEdgeMode)) {
          worldCtx.beginPath();
          worldCtx.moveTo(x, y + cellSize);
          worldCtx.lineTo(x + cellSize, y + cellSize);
          worldCtx.stroke();
        }
      }
      // East edge
      if (c < W - 1) {
        const isEastPassable = east[r] && east[r][c] === 1;
        if (!isEastPassable && (zCur >= 0 || isEdgeMode)) {
          worldCtx.beginPath();
          worldCtx.moveTo(x + cellSize, y);
          worldCtx.lineTo(x + cellSize, y + cellSize);
          worldCtx.stroke();
        }
      }
    }
  }

  // Perimeter walls (along outer building boundaries, leaving openings at exits)
  const exitSet = new Set((staticGT.exits || []).map(([er, ec]) => `${er},${ec}`));
  for (let c = 0; c < W; c++) {
    const x = offsetX + c * cellSize;
    // Top perimeter (r = 0, North boundary)
    if (!exitSet.has(`0,${c}`)) {
      worldCtx.beginPath();
      worldCtx.moveTo(x, offsetY);
      worldCtx.lineTo(x + cellSize, offsetY);
      worldCtx.stroke();
    }
    // Bottom perimeter (r = H - 1, South boundary)
    if (!exitSet.has(`${H - 1},${c}`)) {
      worldCtx.beginPath();
      worldCtx.moveTo(x, offsetY + H * cellSize);
      worldCtx.lineTo(x + cellSize, offsetY + H * cellSize);
      worldCtx.stroke();
    }
  }
  for (let r = 0; r < H; r++) {
    const y = offsetY + r * cellSize;
    // Left perimeter (c = 0, West boundary)
    if (!exitSet.has(`${r},0`)) {
      worldCtx.beginPath();
      worldCtx.moveTo(offsetX, y);
      worldCtx.lineTo(offsetX, y + cellSize);
      worldCtx.stroke();
    }
    // Right perimeter (c = W - 1, East boundary)
    if (!exitSet.has(`${r},${W - 1}`)) {
      worldCtx.beginPath();
      worldCtx.moveTo(offsetX + W * cellSize, y);
      worldCtx.lineTo(offsetX + W * cellSize, y + cellSize);
      worldCtx.stroke();
    }
  }

  // 3. Highlight exterior exits
  const exits = staticGT.exits || [];
  exits.forEach(([er, ec]) => {
    worldCtx.fillStyle = "rgba(52, 211, 153, 0.4)";
    worldCtx.fillRect(offsetX + ec * cellSize, offsetY + er * cellSize, cellSize, cellSize);
    worldCtx.strokeStyle = "var(--accent-green)";
    worldCtx.lineWidth = 2;
    worldCtx.strokeRect(offsetX + ec * cellSize, offsetY + er * cellSize, cellSize, cellSize);
  });

  // 4. Draw debris
  const gt = step.ground_truth || {};
  const debris = gt.debris || [];
  debris.forEach(([dr, dc]) => {
    worldCtx.fillStyle = "#b08763";
    worldCtx.fillRect(offsetX + dc * cellSize + 2, offsetY + dr * cellSize + 2, cellSize - 4, cellSize - 4);
  });

  // 5. Draw visited trail
  if (chkVisitedTrail.checked && trajectory.steps) {
    worldCtx.strokeStyle = "rgba(56, 189, 248, 0.4)";
    worldCtx.lineWidth = 2;
    worldCtx.setLineDash([3, 3]);
    worldCtx.beginPath();
    for (let i = 0; i <= currentStepIdx; i++) {
      const pos = trajectory.steps[i].ground_truth.agent_pos;
      const cx = offsetX + pos[1] * cellSize + cellSize / 2;
      const cy = offsetY + pos[0] * cellSize + cellSize / 2;
      if (i === 0) worldCtx.moveTo(cx, cy);
      else worldCtx.lineTo(cx, cy);
    }
    worldCtx.stroke();
    worldCtx.setLineDash([]);
  }

  // 6. Draw victims
  const victims = gt.victims || [];
  victims.forEach((v) => {
    if (v.rescued) return; // Already outside
    if (v.carried_by === 0) return; // Carried by firefighter

    const vx = offsetX + v.c * cellSize + cellSize / 2;
    const vy = offsetY + v.r * cellSize + cellSize / 2;

    worldCtx.fillStyle = "#ec4899";
    worldCtx.beginPath();
    worldCtx.arc(vx, vy, cellSize * 0.35, 0, 2 * Math.PI);
    worldCtx.fill();
    worldCtx.strokeStyle = "#fff";
    worldCtx.lineWidth = 1.5;
    worldCtx.stroke();

    // Cross inside
    worldCtx.strokeStyle = "#fff";
    worldCtx.beginPath();
    worldCtx.moveTo(vx - cellSize * 0.15, vy);
    worldCtx.lineTo(vx + cellSize * 0.15, vy);
    worldCtx.moveTo(vx, vy - cellSize * 0.15);
    worldCtx.lineTo(vx, vy + cellSize * 0.15);
    worldCtx.stroke();
  });

  // 7. Draw Firefighter Agent
  const agentPos = gt.agent_pos || [0, 0];
  const ax = offsetX + agentPos[1] * cellSize + cellSize / 2;
  const ay = offsetY + agentPos[0] * cellSize + cellSize / 2;

  worldCtx.fillStyle = "var(--accent-orange)";
  worldCtx.beginPath();
  worldCtx.arc(ax, ay, cellSize * 0.4, 0, 2 * Math.PI);
  worldCtx.fill();
  worldCtx.strokeStyle = "#fff";
  worldCtx.lineWidth = 2;
  worldCtx.stroke();

  // Carrying victim badge / glow
  if (gt.carrying) {
    worldCtx.strokeStyle = "var(--accent-yellow)";
    worldCtx.lineWidth = 2.5;
    worldCtx.beginPath();
    worldCtx.arc(ax, ay, cellSize * 0.55, 0, 2 * Math.PI);
    worldCtx.stroke();

    // Sparkle symbol
    worldCtx.fillStyle = "var(--accent-yellow)";
    worldCtx.font = `${Math.round(cellSize * 0.4)}px sans-serif`;
    worldCtx.textAlign = "center";
    worldCtx.textBaseline = "middle";
    worldCtx.fillText("★", ax, ay);
  }
}


/* -------------------------------------------------------------
 * RENDERING: Panel 2 (Agent Actual View Canvas)
 * ------------------------------------------------------------- */
function renderAgentView() {
  if (!trajectory || !trajectory.steps || trajectory.steps.length === 0) return;
  const step = trajectory.steps[currentStepIdx];
  const inputs = step.policy_inputs || {};

  const canvasW = agentCanvas.width;
  const canvasH = agentCanvas.height;
  agentCtx.clearRect(0, 0, canvasW, canvasH);

  // Extract tensor: local (9, 15, 15) or memory (9, 40, 52)
  let tensor = activeViewMode === "local" ? inputs.grid : inputs.memory;
  if (!tensor || tensor.length === 0) return;

  const C = tensor.length;
  const H = tensor[0].length;
  const W = tensor[0][0].length;

  channelDimsBadge.textContent = `${C} × ${H} × ${W}`;

  const cellW = canvasW / W;
  const cellH = canvasH / H;
  const cellSize = Math.min(cellW, cellH);

  const offsetX = (canvasW - W * cellSize) / 2;
  const offsetY = (canvasH - H * cellSize) / 2;

  if (activeChannel === "composite") {
    // Composite Semantic Rendering:
    // Ch 0-3: Passability North, East, South, West
    // Ch 4: Debris
    // Ch 5: Explored / Visited
    // Ch 6: Victim
    // Ch 7: Exit
    // Ch 8: Agent Position
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const x = offsetX + c * cellSize;
        const y = offsetY + r * cellSize;

        const explored = tensor[5] ? tensor[5][r][c] : 0;
        const debris = tensor[4] ? tensor[4][r][c] : 0;
        const victim = tensor[6] ? tensor[6][r][c] : 0;
        const exit = tensor[7] ? tensor[7][r][c] : 0;
        const agent = tensor[8] ? tensor[8][r][c] : 0;

        let bg = "#0b0f17"; // Unknown / unobserved
        if (explored > 0.5) {
          bg = "#182333"; // Observed interior
        }
        if (exit > 0.5) {
          bg = "rgba(52, 211, 153, 0.35)"; // Exit
        }
        if (debris > 0.5) {
          bg = "#b08763"; // Debris block
        }

        agentCtx.fillStyle = bg;
        agentCtx.fillRect(x, y, cellSize, cellSize);

        // Visited memory tint (Ch 9, if present in memory view)
        const visitMem = tensor[9] ? tensor[9][r][c] : 0;
        if (visitMem > 0 && activeViewMode === "memory") {
          agentCtx.fillStyle = `rgba(245, 158, 11, ${Math.min(0.35, 0.12 + visitMem * 0.23)})`;
          agentCtx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
        }

        // Grid lines
        agentCtx.strokeStyle = "#1a2536";
        agentCtx.lineWidth = 0.5;
        agentCtx.strokeRect(x, y, cellSize, cellSize);

        // Passability edges (Cyan lines)
        if (explored > 0.5) {
          agentCtx.strokeStyle = "#38bdf8";
          agentCtx.lineWidth = 1.5;

          // Ch 0: North
          if (tensor[0] && tensor[0][r][c] > 0.5) {
            agentCtx.beginPath();
            agentCtx.moveTo(x + 2, y);
            agentCtx.lineTo(x + cellSize - 2, y);
            agentCtx.stroke();
          }
          // Ch 1: East
          if (tensor[1] && tensor[1][r][c] > 0.5) {
            agentCtx.beginPath();
            agentCtx.moveTo(x + cellSize, y + 2);
            agentCtx.lineTo(x + cellSize, y + cellSize - 2);
            agentCtx.stroke();
          }
          // Ch 2: South
          if (tensor[2] && tensor[2][r][c] > 0.5) {
            agentCtx.beginPath();
            agentCtx.moveTo(x + 2, y + cellSize);
            agentCtx.lineTo(x + cellSize - 2, y + cellSize);
            agentCtx.stroke();
          }
          // Ch 3: West
          if (tensor[3] && tensor[3][r][c] > 0.5) {
            agentCtx.beginPath();
            agentCtx.moveTo(x, y + 2);
            agentCtx.lineTo(x, y + cellSize - 2);
            agentCtx.stroke();
          }
        }

        // Victim (Ch 6)
        if (victim > 0) {
          agentCtx.fillStyle = "#ec4899";
          agentCtx.beginPath();
          agentCtx.arc(x + cellSize / 2, y + cellSize / 2, cellSize * 0.35, 0, 2 * Math.PI);
          agentCtx.fill();
        }

        // Agent (Ch 8)
        if (agent > 0.5) {
          agentCtx.fillStyle = "var(--accent-orange)";
          agentCtx.beginPath();
          agentCtx.arc(x + cellSize / 2, y + cellSize / 2, cellSize * 0.38, 0, 2 * Math.PI);
          agentCtx.fill();
          agentCtx.strokeStyle = "#fff";
          agentCtx.lineWidth = 1.5;
          agentCtx.stroke();
        }
      }
    }
  } else {
    // Single Channel Heatmap
    const chIdx = parseInt(activeChannel, 10);
    const channelMatrix = tensor[chIdx] || [];

    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const val = channelMatrix[r] ? channelMatrix[r][c] : 0;
        const x = offsetX + c * cellSize;
        const y = offsetY + r * cellSize;

        // Color based on value 0.0 -> 1.0
        if (chIdx === 9) {
          if (val === 0) {
            agentCtx.fillStyle = "#0c121c";
          } else {
            // Warm amber/gold heatmap for visit memory
            const rC = Math.round(40 + val * 215);
            const gC = Math.round(val * 190);
            const bC = Math.round((1.0 - val) * 140 + val * 40);
            agentCtx.fillStyle = `rgb(${rC}, ${gC}, ${bC})`;
          }
        } else {
          const brightness = Math.round(val * 220);
          agentCtx.fillStyle = `rgb(${brightness}, ${Math.round(brightness * 0.7)}, ${Math.round(brightness * 0.9)})`;
          if (val === 0) agentCtx.fillStyle = "#0c121c";
        }
        agentCtx.fillRect(x, y, cellSize, cellSize);

        agentCtx.strokeStyle = "#1b2738";
        agentCtx.lineWidth = 0.5;
        agentCtx.strokeRect(x, y, cellSize, cellSize);
      }
    }
  }

  // Update Scalar Badges
  const agentScalars = inputs.agent || [0, 0, 1, 0.33, 0];
  scalarRow.textContent = agentScalars[0] !== undefined ? agentScalars[0].toFixed(2) : "0.00";
  scalarCol.textContent = agentScalars[1] !== undefined ? agentScalars[1].toFixed(2) : "0.00";
  scalarHealth.textContent = agentScalars[2] !== undefined ? agentScalars[2].toFixed(2) : "1.00";
  scalarRole.textContent = agentScalars[3] !== undefined ? agentScalars[3].toFixed(2) : "0.33";

  const isCarrying = agentScalars[4] > 0.5;
  scalarCarrying.textContent = isCarrying ? "YES" : "NO";
  scalarCarrying.style.color = isCarrying ? "var(--accent-yellow)" : "var(--text-main)";

  // Update Previous Action
  const prevAct = inputs.previous_action !== undefined ? inputs.previous_action : 0;
  prevActionName.textContent = `${ACTION_NAMES[prevAct] || prevAct} (Action ${prevAct})`;

  // Update Action Masks
  const masks = inputs.action_mask || [true, true, true, true, true, true, true];
  actionMasksList.innerHTML = "";
  masks.forEach((legal, idx) => {
    const pill = document.createElement("span");
    pill.className = `mask-pill ${legal ? "legal" : "masked"}`;
    pill.textContent = `${ACTION_NAMES[idx]}: ${legal ? "✓" : "✗"}`;
    actionMasksList.appendChild(pill);
  });
}


/* -------------------------------------------------------------
 * RENDERING: Panel 3 (Neural Decisions & Diagnostics)
 * ------------------------------------------------------------- */
function renderDecisionsView() {
  if (!trajectory || !trajectory.steps || trajectory.steps.length === 0) return;
  const step = trajectory.steps[currentStepIdx];
  const outputs = step.policy_outputs || {};
  const trans = step.transition || {};

  const probs = outputs.action_probs || [0, 0, 0, 0, 0, 0, 0];
  const rawLogits = outputs.raw_logits || [0, 0, 0, 0, 0, 0, 0];
  const selected = outputs.selected_action !== undefined ? outputs.selected_action : 0;
  const masks = (step.policy_inputs && step.policy_inputs.action_mask) || [true, true, true, true, true, true, true];

  // Action bars chart
  actionBarsList.innerHTML = "";
  probs.forEach((p, idx) => {
    const row = document.createElement("div");
    const isSelected = idx === selected;
    const isMasked = !masks[idx];
    row.className = `action-bar-row ${isSelected ? "selected" : ""} ${isMasked ? "masked" : ""}`;

    const pct = (p * 100).toFixed(1);
    const logit = rawLogits[idx] !== undefined ? rawLogits[idx].toFixed(2) : "—";

    row.innerHTML = `
      <span class="action-name">${isSelected ? "▶ " : ""}${ACTION_NAMES[idx]}</span>
      <div class="action-bar-track">
        <div class="action-bar-fill" style="width: ${pct}%"></div>
      </div>
      <span class="action-prob-val">${pct}%</span>
      <span class="action-logit-val">${logit}</span>
    `;
    actionBarsList.appendChild(row);
  });

  // KPI Metrics
  kpiValue.textContent = outputs.value_estimate !== undefined ? (outputs.value_estimate >= 0 ? `+${outputs.value_estimate.toFixed(3)}` : outputs.value_estimate.toFixed(3)) : "—";
  kpiGru.textContent = outputs.gru_hidden_norm !== undefined ? outputs.gru_hidden_norm.toFixed(2) : "—";
  kpiStepReward.textContent = trans.reward !== undefined ? (trans.reward >= 0 ? `+${trans.reward.toFixed(3)}` : trans.reward.toFixed(3)) : "0.00";
  kpiCumReward.textContent = trans.cumulative_reward !== undefined ? (trans.cumulative_reward >= 0 ? `+${trans.cumulative_reward.toFixed(3)}` : trans.cumulative_reward.toFixed(3)) : "0.00";

  // Decision Summary Pills
  summaryAction.textContent = outputs.action_name ? `${outputs.action_name} (${selected})` : ACTION_NAMES[selected];
  summaryCarrying.textContent = step.ground_truth.carrying ? "YES (Carrying Victim)" : "NO";
  summaryCarrying.style.color = step.ground_truth.carrying ? "var(--accent-yellow)" : "var(--text-sub)";

  summaryDiscovered.textContent = trans.victim_seen ? "YES (Observed)" : "NO";
  summaryDiscovered.style.color = trans.victim_seen ? "var(--accent-blue)" : "var(--text-sub)";
}

function renderFrame(step) {
  renderWorldView();
  renderAgentView();
  renderDecisionsView();
}


/* -------------------------------------------------------------
 * Canvas Hover Tooltips
 * ------------------------------------------------------------- */
function onWorldCanvasHover(e) {
  if (!trajectory || !trajectory.static_ground_truth) return;
  const rect = worldCanvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  const staticGT = trajectory.static_ground_truth;
  const H = staticGT.height || 15;
  const W = staticGT.width || 18;
  const cellSize = Math.min((worldCanvas.width - 40) / W, (worldCanvas.height - 40) / H);
  const offsetX = (worldCanvas.width - W * cellSize) / 2;
  const offsetY = (worldCanvas.height - H * cellSize) / 2;

  const col = Math.floor((mouseX - offsetX) / cellSize);
  const row = Math.floor((mouseY - offsetY) / cellSize);

  if (row >= 0 && row < H && col >= 0 && col < W) {
    const zone = staticGT.zones && staticGT.zones[row] ? staticGT.zones[row][col] : -1;
    worldHoverInfo.textContent = `[Row ${row}, Col ${col}] Zone: ${zone >= 0 ? zone : "Ext"} (Sim Ground Truth)`;
  } else {
    worldHoverInfo.textContent = "Hover cell for coordinates";
  }
}

function onAgentCanvasHover(e) {
  if (!trajectory || !trajectory.steps) return;
  const rect = agentCanvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  const step = trajectory.steps[currentStepIdx];
  const tensor = activeViewMode === "local" ? step.policy_inputs.grid : step.policy_inputs.memory;
  if (!tensor || tensor.length === 0) return;

  const H = tensor[0].length;
  const W = tensor[0][0].length;
  const cellSize = Math.min(agentCanvas.width / W, agentCanvas.height / H);
  const offsetX = (agentCanvas.width - W * cellSize) / 2;
  const offsetY = (agentCanvas.height - H * cellSize) / 2;

  const col = Math.floor((mouseX - offsetX) / cellSize);
  const row = Math.floor((mouseY - offsetY) / cellSize);

  if (row >= 0 && row < H && col >= 0 && col < W) {
    if (activeChannel === "composite") {
      const exp = tensor[5] ? tensor[5][row][col] : 0;
      const deb = tensor[4] ? tensor[4][row][col] : 0;
      const vic = tensor[6] ? tensor[6][row][col] : 0;
      const agt = tensor[8] ? tensor[8][row][col] : 0;
      agentHoverInfo.textContent = `[${row}, ${col}] Exp:${exp.toFixed(1)} Deb:${deb.toFixed(1)} Vic:${vic.toFixed(1)} Agt:${agt.toFixed(1)}`;
    } else {
      const chIdx = parseInt(activeChannel, 10);
      const val = tensor[chIdx] && tensor[chIdx][row] ? tensor[chIdx][row][col] : 0;
      agentHoverInfo.textContent = `[${row}, ${col}] ${MEMORY_CHANNEL_NAMES[chIdx]}: ${val.toFixed(4)}`;
    }
  } else {
    agentHoverInfo.textContent = "Hover cell for exact tensor value";
  }
}

function showError(msg) {
  if (!msg) {
    errorBanner.hidden = true;
    errorBanner.textContent = "";
  } else {
    errorBanner.hidden = false;
    errorBanner.textContent = msg;
  }
}
