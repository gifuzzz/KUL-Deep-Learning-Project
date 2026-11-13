/* Plain browser UI; all observations and decisions come from the Python inspector. */
'use strict';
const $ = id => document.getElementById(id);
const replay = window.INSPECTION_REPLAY || null;
let frames = [], selected = 0, layer = 3, cell = [7, 7], busy = false, playing = false;
let layerKey = '', modelKey = '';
const fmt = (value, digits = 4) => value == null ? '—' : Number(value).toLocaleString('en', {maximumFractionDigits: digits});
const signed = value => (value > 0 ? '+' : '') + fmt(value);
const current = () => frames[selected];
const latest = () => replay ? current() : frames[frames.length - 1];
const isLive = () => !!replay || selected === frames.length - 1;
function element(tag, text, className) {
  const el = document.createElement(tag);
  if (text != null) el.textContent = text;
  if (className) el.className = className;
  return el;
}
function pause() { playing = false; $('play').textContent = '▶ Play'; }
function error(message) { $('error').textContent = message; $('error').hidden = !message; }
async function request(path, body, reset = false) {
  if (busy) return false;
  busy = true; error(''); updateControls();
  try {
    if (replay) {
      if (path === '/api/state' || path === '/api/reset') { frames = replay; selected = 0; }
      else if (path === '/api/step') selected = Math.min(selected + 1, replay.length - 1);
      else throw new Error('This is a recorded replay. Start the local server for live inspection.');
      if (current().finished) pause();
      return true;
    }
    const response = await fetch(path, body === undefined ? {} : {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'The request failed.');
    if (reset || !frames.length) frames = [data]; else frames.push(data);
    selected = frames.length - 1;
    if (data.finished) pause();
    return true;
  } catch (err) { pause(); error(err.message); return false; }
  finally { busy = false; if (frames.length) render(); else updateControls(); }
}
function inputs(frame = current()) {
  const before = $('phase').value === 'before' && frame.transition;
  const native = $('source').value === 'native';
  return {
    grid: before ? (native ? frame.transition.native_input.grid : frame.transition.input.grid)
                 : (native ? frame.native_observation.grid : frame.observation.grid),
    visibility: before ? frame.transition.native_input.grid[5] : frame.native_observation.grid[5],
    features: before ? frame.transition.input.agent : frame.observation.agent,
    policy: before ? frame.transition.policy_output : frame.policy,
    mask: before ? frame.transition.action_masks : frame.action_masks,
    person: before ? frame.transition.agent_before : frame.scene.agents[0],
    metadata: frame.layers[native ? 'native' : 'policy'], before
  };
}
function canvasContext(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, rect.width), height = Math.max(1, rect.height);
  canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {ctx, width, height};
}
function mix(color, fraction) {
  const rgb = color.slice(1).match(/../g).map(x => parseInt(x, 16));
  return `rgb(${rgb.map(x => Math.round(249 + (x - 249) * fraction)).join(',')})`;
}
function paintLayer(canvas, grid, metadata, visibility, large = false) {
  const {ctx, width, height} = canvasContext(canvas), rows = grid.length, cols = grid[0].length;
  const cw = width / cols, ch = height / rows;
  const flat = grid.flat(), max = Math.max(1, ...flat);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const value = grid[r][c], unknown = !visibility[r][c] && metadata.native_index !== 5;
    ctx.fillStyle = unknown ? ((r + c) % 2 ? '#dce2e9' : '#e8edf3') : mix(metadata.color, Math.max(0, value / max));
    ctx.fillRect(c * cw, r * ch, cw + .3, ch + .3);
    if (large) {
      ctx.strokeStyle = unknown ? '#d6dee7' : '#dce3ec'; ctx.lineWidth = .45;
      ctx.strokeRect(c * cw, r * ch, cw, ch);
      if ($('cell-values').checked) {
        ctx.font = `${Math.max(6, Math.min(10, cw * .34))}px "Avenir Next",sans-serif`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillStyle = !unknown && value / max > .6 ? '#fff' : '#4c6078';
        ctx.fillText(fmt(value, 2), (c + .5) * cw, (r + .5) * ch);
      }
    }
  }
  if (large) {
    const center = Math.floor(rows / 2);
    ctx.strokeStyle = '#244f9f'; ctx.lineWidth = 1.7;
    ctx.beginPath(); ctx.arc((center + .5) * cw, (center + .5) * ch, Math.min(cw, ch) * .25, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo((center + .5) * cw, (center + .12) * ch); ctx.lineTo((center + .5) * cw, (center + .88) * ch);
    ctx.moveTo((center + .12) * cw, (center + .5) * ch); ctx.lineTo((center + .88) * cw, (center + .5) * ch); ctx.stroke();
    ctx.strokeStyle = '#142c4b'; ctx.lineWidth = 2;
    ctx.strokeRect(cell[1] * cw + 1, cell[0] * ch + 1, cw - 2, ch - 2);
  }
}
function paintWorld(frame) {
  const scene = frame.scene, person = scene.agents[0];
  const {ctx, width, height} = canvasContext($('world'));
  const follow = $('world-view').value === 'follow';
  let minC, minR, spanC, spanR;
  if (follow) { spanR = 18; spanC = spanR * width / height; minC = person.c + .5 - spanC / 2; minR = person.r + .5 - spanR / 2; }
  else { minC = -scene.margin; minR = -scene.margin; spanC = scene.w + 2 * scene.margin; spanR = scene.h + 2 * scene.margin; }
  const scale = Math.min(width / spanC, height / spanR);
  const ox = (width - spanC * scale) / 2 - minC * scale, oy = (height - spanR * scale) / 2 - minR * scale;
  const x = c => ox + c * scale, y = r => oy + r * scale;
  ctx.fillStyle = '#dce7e2'; ctx.fillRect(0, 0, width, height);
  const tones = ['#e8edf4', '#fbfcfe', '#f2f5fa', '#f8fafc', '#eef3f9'];
  for (let r = 0; r < scene.h; r++) for (let c = 0; c < scene.w; c++) {
    if (x(c + 1) < 0 || x(c) > width || y(r + 1) < 0 || y(r) > height) continue;
    ctx.fillStyle = scene.zone[r][c] < 0 ? '#34404a' : tones[scene.zone[r][c] % tones.length];
    ctx.fillRect(x(c), y(r), scale + .2, scale + .2);
    for (const [field, color] of [['smoke', '117,110,133'], ['water', '66,148,184'], ['fire', '234,105,47'], ['debris', '127,99,69']]) {
      const value = scene[field][r][c];
      if (value > .001) { ctx.fillStyle = `rgba(${color},${Math.min(.9, Math.sqrt(value))})`; ctx.fillRect(x(c), y(r), scale + .2, scale + .2); }
    }
    if (scale > 8) { ctx.strokeStyle = '#d5dfe855'; ctx.lineWidth = .5; ctx.strokeRect(x(c), y(r), scale, scale); }
  }
  const side = frame.observation.grid[0].length, radius = Math.floor(side / 2);
  for (let r = 0; r < side; r++) for (let c = 0; c < side; c++) {
    if (frame.native_observation.grid[5][r][c]) {
      ctx.fillStyle = '#527bca12'; ctx.fillRect(x(person.c + c - radius), y(person.r + r - radius), scale, scale);
    }
  }
  ctx.strokeStyle = '#5579af88'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
  ctx.strokeRect(x(person.c - radius), y(person.r - radius), side * scale, side * scale); ctx.setLineDash([]);
  function line(points, color, thickness) {
    ctx.strokeStyle = color; ctx.lineWidth = thickness; ctx.beginPath();
    ctx.moveTo(x(points[0]), y(points[1])); ctx.lineTo(x(points[2]), y(points[3])); ctx.stroke();
  }
  ctx.strokeStyle = '#6f7e92'; ctx.lineWidth = Math.max(1.4, scale * .09); ctx.strokeRect(x(0), y(0), scene.w * scale, scene.h * scale);
  for (const wall of scene.walls) line(wall, '#6f7e92', Math.max(1.25, scale * .08));
  for (const opening of scene.openings) {
    line(opening.line, '#edf2f6', Math.max(3, scale * .25));
    line(opening.line, !opening.open || opening.blocked ? '#bd7859' : opening.kind === 'exit' ? '#27966a' : '#b8c7d7', Math.max(2, scale * .15));
  }
  for (const victim of scene.victims) {
    if (victim.rescued || victim.dead || victim.carried_by >= 0) continue;
    const size = Math.max(4, scale * .3);
    ctx.fillStyle = '#bc6090'; ctx.fillRect(x(victim.c + .5) - size, y(victim.r + .5) - size, size * 2, size * 2);
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.strokeRect(x(victim.c + .5) - size, y(victim.r + .5) - size, size * 2, size * 2);
    if (scale >= 10) { ctx.font = `600 ${Math.max(9, scale * .45)}px "Avenir Next",sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = '#fff'; ctx.fillText(victim.category, x(victim.c + .5), y(victim.r + .5)); }
  }
  const carrying = scene.victims.some(v => v.carried_by === person.id && !v.dead && !v.rescued);
  ctx.fillStyle = '#335fba'; ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(x(person.c + .5), y(person.r + .5), Math.max(5, scale * .34), 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  if (carrying) { ctx.fillStyle = '#c76494'; ctx.beginPath(); ctx.arc(x(person.c + .83), y(person.r + .18), Math.max(3, scale * .16), 0, Math.PI * 2); ctx.fill(); }
  $('world-position').textContent = `Row ${person.r}, col ${person.c}${carrying ? ' · carrying' : ''}`;
}
function renderLayers() {
  const view = inputs(), source = $('source').value;
  if (layer >= view.metadata.length) layer = 0;
  const key = `${source}:${view.metadata.length}`;
  if (layerKey !== key) {
    $('layers').replaceChildren();
    view.metadata.forEach((meta, i) => {
      const button = element('button', null, 'layer'); button.dataset.layer = i;
      const title = element('span', null, 'layer-title'); title.append(element('span', meta.name), element('small', String(i)));
      const canvas = element('canvas'); canvas.width = 120; canvas.height = 120; canvas.setAttribute('aria-hidden', 'true');
      const stat = element('span', '', 'layer-stat'); button.append(title, canvas, stat);
      button.addEventListener('click', () => { layer = i; render(); }); $('layers').append(button);
    });
    layerKey = key;
  }
  [...$('layers').children].forEach((button, i) => {
    button.classList.toggle('selected', layer === i); button.setAttribute('aria-pressed', String(layer === i));
    const values = view.grid[i].flat();
    button.querySelector('.layer-stat').textContent = `${values.filter(v => v !== 0).length} nonzero`;
    button.title = `${view.metadata[i].description}; range ${fmt(Math.min(...values))} to ${fmt(Math.max(...values))}`;
    paintLayer(button.querySelector('canvas'), view.grid[i], view.metadata[i], view.visibility);
  });
  const meta = view.metadata[layer], values = view.grid[layer].flat();
  $('layer-title').textContent = meta.name; $('layer-description').textContent = meta.description;
  $('shape').textContent = `${view.grid.length} × ${view.grid[0].length} × ${view.grid[0][0].length}`;
  $('layer-range').textContent = `Range ${fmt(Math.min(...values))}–${fmt(Math.max(...values))}${meta.native_index == null ? '' : ` · native ${meta.native_index}`}`;
  $('phase-note').textContent = `${view.before ? 'Before' : 'After'} action · local coordinates`;
  $('layers-caption').textContent = source === 'policy' ? 'The actual spatial inputs to the CNN. Select any layer for a closer look.' : 'All 15 local simulator channels, including channels excluded from the policy.';
  paintLayer($('local'), view.grid[layer], meta, view.visibility, true);
}
function renderCell() {
  const frame = current(), native = $('source').value === 'native', view = inputs();
  const before = frame.transition ? (native ? frame.transition.native_input.grid : frame.transition.input.grid) : null;
  const after = native ? frame.native_observation.grid : frame.observation.grid;
  const [r, c] = cell, radius = Math.floor(after[0].length / 2);
  $('cell-position').textContent = `Local (${r}, ${c}) · world (${view.person.r + r - radius}, ${view.person.c + c - radius}) · ${view.visibility[r][c] ? 'visible' : 'unseen'}`;
  $('cell-values-table').replaceChildren();
  view.metadata.forEach((meta, i) => {
    const row = element('tr'), old = before?.[i][r][c], value = after[i][r][c];
    row.append(element('td', meta.name), element('td', fmt(old)), element('td', fmt(value)));
    row.append(element('td', old == null ? '—' : signed(value - old), old != null && Math.abs(value - old) > 1e-7 ? 'changed' : ''));
    $('cell-values-table').append(row);
  });
}
function renderFeatures() {
  const frame = current(), view = inputs(), before = frame.transition?.input.agent;
  $('features').replaceChildren();
  frame.feature_names.forEach((name, i) => {
    const row = element('tr');
    row.append(element('td', name), element('td', fmt(before?.[i])), element('td', fmt(frame.observation.agent[i]), before && before[i] !== frame.observation.agent[i] ? 'changed' : ''));
    $('features').append(row);
  });
  const features = view.policy?.features || [];
  $('feature-count').textContent = features.length ? `${features.length} features` : 'Load a checkpoint';
  $('feature-vector').replaceChildren();
  const max = Math.max(.001, ...features.map(Math.abs));
  features.forEach((value, i) => {
    const item = element('i'); item.title = `Feature ${i}: ${value}`;
    item.style.backgroundColor = mix(value >= 0 ? '#426eb8' : '#b87385', Math.abs(value) / max);
    $('feature-vector').append(item);
  });
}
function renderDecision() {
  const frame = current(), view = inputs(), p = view.policy;
  $('decision-title').textContent = view.before ? `Decision for step ${frame.step}` : frame.finished ? 'Terminal model output' : 'Next decision';
  $('value-note').textContent = frame.finished && !view.before ? 'Inspection only; this episode has ended' : 'Expected discounted training reward';
  $('value').textContent = fmt(p?.value, 3);
  $('action-bars').replaceChildren();
  frame.action_names.forEach((name, i) => {
    const row = element('div', null, `action-row${!view.mask[i] ? ' masked' : ''}${p?.suggested_action === i ? ' suggested' : ''}`);
    const label = element('span', null, 'name'); label.append(element('span', String(i), 'action-id'), element('span', name));
    const track = element('span', null, 'prob-track');
    track.style.setProperty('--p', `${(p?.probabilities[i] || 0) * 100}%`);
    track.style.setProperty('--u', `${(p?.unmasked_probabilities[i] || 0) * 100}%`);
    track.append(element('span', null, 'prob-fill'), element('span', null, 'prob-unmasked'));
    row.append(label, track, element('span', p ? `${fmt(p.probabilities[i] * 100, 1)}%` : '—', 'prob-number'));
    row.title = `${view.mask[i] ? 'Available' : 'Masked'} · native command ${frame.native_commands[i]}${p ? ` · logit ${p.logits[i]} · unmasked probability ${p.unmasked_probabilities[i]} · masked log probability ${p.log_probabilities[i]}` : ''}`;
    $('action-bars').append(row);
  });
}
function renderResult() {
  const frame = current(), t = frame.transition, target = $('result'); target.replaceChildren();
  if (!t) { target.append(element('p', 'Take a manual action or step the policy to inspect the exact transition.', 'empty')); $('result-caption').textContent = 'No action taken in this episode'; return; }
  $('result-caption').textContent = `Step ${t.step} · ${t.source} selection · ${t.was_valid ? 'valid action' : 'masked action tested'}`;
  const title = element('div', null, 'result-action'); title.append(element('span', t.action_name), element('small', `${t.action} → command ${t.native_command}`)); target.append(title);
  function line(label, value, className = '') { const row = element('div', null, `result-line ${className}`); row.append(element('span', label), element('span', value)); target.append(row); }
  line('Native rescue reward', signed(t.native_reward)); line('Reward shaping adjustment', signed(t.shaping_adjustment)); line('Returned reward', signed(t.reward), 'total');
  line('Episode reward', fmt(frame.total_reward));
  const a = t.agent_before, b = frame.scene.agents[0]; line('Firefighter position', `(${a.r}, ${a.c}) → (${b.r}, ${b.c})`);
  const flags = element('div', null, 'result-flags');
  flags.append(element('span', t.action === 0 ? 'Wait / tick advanced' : t.native_output.actions_succeeded ? 'Command applied' : 'No effect', `badge ${t.native_output.actions_succeeded ? 'good' : ''}`));
  flags.append(element('span', `terminated: ${t.terminated}`, `badge ${t.terminated ? 'good' : ''}`), element('span', `truncated: ${t.truncated}`, `badge ${t.truncated ? 'warn' : ''}`)); target.append(flags);
}
function exchange(frame) {
  const t = frame.transition;
  return {
    seed: frame.seed, step: frame.step, checkpoint: frame.checkpoint,
    input: {observation: t ? t.input : frame.observation, native_observation: t ? t.native_input : frame.native_observation,
      action_masks: t ? t.action_masks : frame.action_masks},
    policy_output: t ? t.policy_output : frame.policy,
    action: t ? {index: t.action, native_command: t.native_command, name: t.action_name, source: t.source, was_valid: t.was_valid} : null,
    output: t ? {observation: frame.observation, action_masks: frame.action_masks, reward: t.reward,
      native_reward: t.native_reward, shaping_adjustment: t.shaping_adjustment, terminated: t.terminated,
      truncated: t.truncated, info: t.info, native_output: t.native_output, next_policy_output: frame.policy} : null
  };
}
function renderHistory() {
  $('seek').max = frames.length - 1; $('seek').value = selected;
  $('seek-label').textContent = `Step ${current().step}`;
  $('history').replaceChildren();
  frames.forEach((frame, i) => {
    const t = frame.transition, row = element('tr', null, i === selected ? 'selected' : '');
    row.tabIndex = 0; row.setAttribute('aria-label', `Inspect step ${frame.step}`);
    const outcome = !t ? 'Ready' : t.info.success ? 'Rescued' : t.truncated ? 'Timeout' : t.terminated ? 'Finished' : 'Running';
    const nativeResult = !t ? '—' : t.action === 0 ? 'Wait' : t.native_output.actions_succeeded ? 'Applied' : 'No effect';
    for (const value of [frame.step, t?.action_name || 'Episode reset', t?.source || '—', nativeResult, t ? signed(t.reward) : '—', fmt(frame.total_reward), outcome]) row.append(element('td', String(value)));
    row.addEventListener('click', () => selectHistory(i)); row.addEventListener('keydown', e => { if (e.key === 'Enter') selectHistory(i); });
    $('history').append(row);
  });
}
function updateControls() {
  const frame = latest(), unavailable = busy || !frame || !isLive() || frame.finished;
  $('step').disabled = unavailable || !frame?.policy; $('play').disabled = !playing && (unavailable || !frame?.policy);
  $('reset').disabled = busy; $('load').disabled = busy || !!replay;
  $('back').disabled = busy || selected <= 0; $('forward').disabled = busy || selected >= frames.length - 1;
  $('live').disabled = busy || selected === frames.length - 1; $('export').disabled = !frames.length;
  [...$('manual-actions').children].forEach((button, i) => { button.disabled = !!replay || unavailable || (!frame.action_masks[i] && !$('allow-invalid').checked); });
}
function render() {
  const frame = current();
  if (!frame) return;
  const size = frame.observation.grid[0].length;
  cell = [Math.max(0, Math.min(size - 1, cell[0])), Math.max(0, Math.min(size - 1, cell[1]))];
  const status = frame.info.success ? 'Victim rescued' : frame.finished ? frame.info.timeout ? 'Episode timed out' : 'Episode finished' : isLive() ? 'Episode running' : 'Recorded state';
  $('episode-status').textContent = status; $('step-count').textContent = `Step ${frame.step} / ${frame.max_steps}`;
  $('status-dot').className = `status-dot ${frame.info.success ? 'success' : frame.finished ? 'timeout' : ''}`;
  const key = JSON.stringify(frame.checkpoints);
  if (modelKey !== key) {
    const previous = $('checkpoint').value; $('checkpoint').replaceChildren(new Option('Manual exploration', ''));
    for (const path of frame.checkpoints) $('checkpoint').append(new Option(path, path));
    $('checkpoint').value = frame.checkpoint || previous; modelKey = key;
  }
  $('model-note').textContent = frame.checkpoint ? `${frame.checkpoint} · ${fmt(frame.trained_steps, 0)} training steps · ${replay ? 'recorded replay' : 'CPU inference'}` : 'Manual exploration. Load a checkpoint to inspect policy probabilities, features, and value estimates.';
  $('phase').disabled = !frame.transition;
  paintWorld(frame); renderLayers(); renderCell(); renderFeatures(); renderDecision(); renderResult(); renderHistory();
  if ($('raw-details').open) $('raw').textContent = JSON.stringify(exchange(frame), null, 2);
  updateControls();
}
function selectHistory(index) { if (busy) return; pause(); selected = Math.max(0, Math.min(frames.length - 1, index)); render(); }
async function takeAction(action) {
  if (busy || !isLive() || latest()?.finished) return;
  const ok = await request('/api/step', {action, mode: $('mode').value, allow_invalid: $('allow-invalid').checked});
  if (ok) $('history').parentElement.scrollTop = $('history').parentElement.scrollHeight;
}
$('reset').addEventListener('click', async () => { pause(); await request('/api/reset', {seed: Number($('seed').value), shaping: $('shaping').checked}, true); });
$('load').addEventListener('click', async () => { pause(); await request('/api/model', {checkpoint: $('checkpoint').value || null}, true); });
$('step').addEventListener('click', () => takeAction(null));
$('play').addEventListener('click', async () => {
  if (playing) { pause(); return; }
  if (busy || !isLive() || latest()?.finished) return;
  playing = true; $('play').textContent = 'Ⅱ Pause';
  while (playing) {
    await takeAction(null);
    if (!playing) break;
    await new Promise(resolve => setTimeout(resolve, 1000 / Number($('speed').value)));
  }
});
const shortActions = ['Wait', '↑ Up', '→ Right', '↓ Down', '← Left', 'Pick/drop', 'Rescue'];
shortActions.forEach((name, i) => {
  const button = element('button', name); button.title = `Action ${i}`;
  button.addEventListener('click', () => { pause(); takeAction(i); }); $('manual-actions').append(button);
});
$('back').addEventListener('click', () => selectHistory(selected - 1));
$('forward').addEventListener('click', () => selectHistory(selected + 1));
$('live').addEventListener('click', () => selectHistory(frames.length - 1));
$('seek').addEventListener('input', () => selectHistory(Number($('seek').value)));
for (const id of ['source', 'phase', 'world-view', 'cell-values']) $(id).addEventListener('change', () => { if (frames.length) render(); });
$('allow-invalid').addEventListener('change', updateControls);
$('raw-details').addEventListener('toggle', () => { if (current() && $('raw-details').open) $('raw').textContent = JSON.stringify(exchange(current()), null, 2); });
$('local').addEventListener('click', event => {
  if (!current()) return;
  const rect = $('local').getBoundingClientRect(), size = inputs().grid[0].length;
  cell = [Math.min(size - 1, Math.floor((event.clientY - rect.top) / rect.height * size)), Math.min(size - 1, Math.floor((event.clientX - rect.left) / rect.width * size))];
  renderLayers(); renderCell();
});
$('local').addEventListener('mousemove', event => {
  if (!current()) return;
  const view = inputs(), rect = $('local').getBoundingClientRect(), size = view.grid[0].length;
  const r = Math.max(0, Math.min(size - 1, Math.floor((event.clientY - rect.top) / rect.height * size)));
  const c = Math.max(0, Math.min(size - 1, Math.floor((event.clientX - rect.left) / rect.width * size)));
  $('local').title = `${view.metadata[layer].name} [${r}, ${c}] = ${view.grid[layer][r][c]}${view.visibility[r][c] ? '' : ' (unseen)'}`;
});
$('local').addEventListener('keydown', event => {
  const delta = {ArrowUp: [-1, 0], ArrowRight: [0, 1], ArrowDown: [1, 0], ArrowLeft: [0, -1]}[event.key];
  if (!delta || !current()) return;
  event.preventDefault(); event.stopPropagation(); const size = inputs().grid[0].length;
  cell = cell.map((value, i) => Math.max(0, Math.min(size - 1, value + delta[i]))); renderLayers(); renderCell();
});
document.addEventListener('keydown', event => {
  if (event.target.closest('input,select,textarea,button,summary,canvas') || event.ctrlKey || event.metaKey || event.altKey || event.repeat || !latest()) return;
  const action = {ArrowUp: 1, ArrowRight: 2, ArrowDown: 3, ArrowLeft: 4, p: 5, r: 6, w: 0}[event.key];
  if (action != null && !replay) { event.preventDefault(); pause(); takeAction(action); }
  else if (event.code === 'Space' && !$('step').disabled) { event.preventDefault(); pause(); takeAction(null); }
});
$('export').addEventListener('click', () => {
  const payload = {format: 'rescue-inspector-v1', note: 'Simulation state is observer information, not additional policy input.', frames};
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], {type: 'application/json'}));
  const anchor = element('a'); anchor.href = url; anchor.download = `rescue-seed-${latest().seed}-trace.json`;
  document.body.append(anchor); anchor.click(); anchor.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
let resizeFrame;
window.addEventListener('resize', () => { cancelAnimationFrame(resizeFrame); resizeFrame = requestAnimationFrame(() => { if (current()) render(); }); });
if (replay) {
  for (const id of ['seed', 'checkpoint', 'mode', 'shaping', 'allow-invalid']) $(id).disabled = true;
  $('seed').value = replay[0].seed;
  $('shaping').checked = replay[0].reward_shaping;
  $('step').replaceChildren(document.createTextNode('Next recorded step'));
  $('live').textContent = 'Last recorded step';
  document.querySelector('.decision-panel .canvas-hint').textContent = 'Read-only replay. Use playback controls or the history to inspect recorded actions.';
  document.querySelector('footer').textContent = 'Standalone recording. Tensors and policy outputs were captured from the real simulator. Start the local server to choose new actions or scenarios.';
}
request('/api/state', undefined, true);
