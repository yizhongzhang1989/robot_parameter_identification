/* Panel logic: poll state, drive the forms, keep the 3D view and the charts
 * looking at the same snapshot.
 */

import { SERIES, drawCondition, drawErrors, drawFriction, drawResidual, drawTrace }
  from '/charts.js';
import { LANGUAGES, applyStatic, getLang, onLangChange, setLang, t }
  from '/i18n.js';

const POLL_MS = 400;
// A directory scan does not belong on the fast path; saved runs change once
// per campaign, not four times a second.
const RUNS_EVERY = 25;
const PHASES = ['A_gravity', 'B_friction', 'C_inertia', 'D_validation'];

const $ = (id) => document.getElementById(id);
const state = { snapshot: null, selected: null, frames: [], unit: 'A',
                ticks: 0, runs: [], profile: null,
                // The live-signal overlay: which quantity the plot shows,
                // which joints are in it, and the frames behind it.
                signal: 'position_deg', hidden: new Set(), trace: [],
                jointNames: [], cursor: 0, rate: 0, rows: [], rowKey: '',
                lastFrameAt: 0, plot: 'normal' };

/* ---------------- language ---------------- */

const langSelect = $('lang');
langSelect.innerHTML = LANGUAGES.map(
  (entry) => `<option value="${entry.code}">${entry.label}</option>`).join('');
langSelect.value = getLang();
langSelect.addEventListener('change', () => setLang(langSelect.value));
onLangChange(() => {
  if (state.snapshot) renderAll(state.snapshot);
  renderSignals();
});
document.documentElement.lang = getLang() === 'zh' ? 'zh-CN' : 'en';
applyStatic(document);

/* ---------------- transport ---------------- */

async function post(path, body) {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  const data = await response.json().catch(() => ({ ok: false, message: 'bad reply' }));
  if (!data.ok) toast(data.message || 'failed', 'err');
  return data;
}

let toastTimer = null;
function toast(message, kind) {
  const node = $('toast');
  node.textContent = message;
  node.className = kind || '';
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.add('hidden'), 4000);
}

/* ---------------- obstacles ---------------- */

window.__dash = {
  updateObstacle: async (id, changes) => {
    await post('/api/obstacles', { action: 'update', id, changes });
  },
  onViewer: (data) => {
    if (data.frames && data.frames.length !== state.frames.length) {
      state.frames = data.frames;
      fillFrames();
    }
  },
};

function fillFrames() {
  const select = $('obst-frame');
  const keep = select.value;
  select.innerHTML = '';
  for (const name of state.frames) {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = name;
    select.appendChild(option);
  }
  select.value = keep || defaultFrame();
}

/** A bench belongs on the robot's base, not on the kinematic root. */
function defaultFrame() {
  const driven = (state.snapshot?.driven_joints || [])[0] || '';
  const prefix = driven.replace(/joint\d+$/, '');
  const preferred = state.frames.find(
    (name) => prefix && name.startsWith(prefix) && /base/i.test(name));
  return preferred || state.frames.find((n) => /base_link$/i.test(n))
    || state.frames[0] || '';
}

$('obst-add').addEventListener('click', async () => {
  const frame = $('obst-frame').value || defaultFrame();
  if (!frame) return toast(t('obst.nomodel'), 'err');
  const result = await post('/api/obstacles', {
    action: 'add',
    obstacle: { parent_frame: frame, size_m: [0.2, 0.2, 0.2],
                xyz_m: [0.3, 0.0, 0.1] },
  });
  if (result.ok && result.obstacle) {
    setTimeout(() => window.__viewer.select(result.obstacle.id), 150);
  }
});

$('obst-del').addEventListener('click', async () => {
  if (!state.selected) return;
  await post('/api/obstacles', { action: 'remove', id: state.selected });
  window.__viewer.select(null);
});

$('obst-clear').addEventListener('click', async () => {
  await post('/api/obstacles', { action: 'clear' });
  window.__viewer.select(null);
});

$('obst-save').addEventListener('click', async () => {
  const data = await post('/api/obstacles',
                          { action: 'save', name: $('obst-file').value.trim() });
  if (data.ok) toast(t('obst.saved', { v: data.path }), 'ok');
});

window.__viewer.onChange = (id) => {
  state.selected = id;
  $('obst-del').disabled = !id;
  $('obst-form').classList.toggle('hidden', !id);
  renderObstacleList();
  fillForm();
};

function currentObstacle() {
  return (state.snapshot?.obstacles || []).find((o) => o.id === state.selected);
}

function fillForm() {
  const box = currentObstacle();
  if (!box) return;
  $('obst-frame').value = box.parent_frame;
  const set = (id, value) => { const el = $(id); if (document.activeElement !== el) el.value = value; };
  set('sz-x', box.size_m[0]); set('sz-y', box.size_m[1]); set('sz-z', box.size_m[2]);
  set('px', box.xyz_m[0]); set('py', box.xyz_m[1]); set('pz', box.xyz_m[2]);
  set('rr', box.rpy_deg[0]); set('rp', box.rpy_deg[1]); set('ry', box.rpy_deg[2]);
}

const FIELD_IDS = ['sz-x', 'sz-y', 'sz-z', 'px', 'py', 'pz', 'rr', 'rp', 'ry'];
for (const id of FIELD_IDS.concat(['obst-frame'])) {
  $(id).addEventListener('change', async () => {
    if (!state.selected) return;
    const num = (fid) => parseFloat($(fid).value) || 0;
    await post('/api/obstacles', {
      action: 'update',
      id: state.selected,
      changes: {
        parent_frame: $('obst-frame').value,
        size_m: [num('sz-x'), num('sz-y'), num('sz-z')],
        xyz_m: [num('px'), num('py'), num('pz')],
        rpy_deg: [num('rr'), num('rp'), num('ry')],
      },
    });
  });
}

function renderObstacleList() {
  const list = $('obst-list');
  const boxes = state.snapshot?.obstacles || [];
  $('obst-count').textContent = String(boxes.length);
  const where = state.snapshot?.obstacle_file || '';
  $('obst-file').placeholder = where.split('/').pop() || 'obstacles.json';
  // Saying where it already goes is the point: the name is what gets carried
  // to the next arm, and the launch argument is where it is picked up again.
  $('obst-where').textContent = where ? t('obst.where', { v: where }) : '';
  list.innerHTML = '';
  for (const box of boxes) {
    const item = document.createElement('li');
    item.className = box.id === state.selected ? 'sel' : '';
    item.innerHTML = `<span>${box.name}</span>`
      + `<span class="frame">${box.parent_frame}</span>`;
    item.addEventListener('click', () => window.__viewer.select(box.id));
    list.appendChild(item);
  }
}

/* ---------------- run control ---------------- */

$('btn-rehearse').addEventListener('click', () => post('/api/campaign', { mode: 'rehearsal' }));
$('btn-hardware').addEventListener('click', () => post('/api/campaign', { mode: 'hardware' }));
$('btn-optimal').addEventListener('click', () => post('/api/campaign', {
  mode: 'optimal_excitation',
  options: {
    optimal_training_trajectories: parseInt($('optimal-training').value, 10),
    optimal_validation_trajectories: parseInt($('optimal-validation').value, 10),
    optimal_friction_postures: parseInt($('optimal-postures').value, 10),
    optimal_friction_repeats: parseInt($('optimal-repeats').value, 10),
    fourier_base_frequency_hz: parseFloat($('optimal-frequency').value),
    fourier_duration_s: parseFloat($('optimal-duration').value),
    reuse_friction: $('optimal-reuse-friction').checked,
  },
}));
$('btn-home').addEventListener('click', () => post('/api/home', {}));
$('btn-stop').addEventListener('click', () => post('/api/stop', {}));
$('btn-sweep').addEventListener('click', () => post('/api/campaign', {
  mode: 'load_sweep',
  options: { resume: $('sweep-resume').checked },
}));

/* ---------------- render ---------------- */

function pill(id, ok, label) {
  const node = $(id);
  node.className = 'pill ' + (ok === null ? '' : ok ? 'good' : 'bad');
  node.textContent = label;
}

function renderConnection(snapshot) {
  const connection = snapshot.connection || {};
  pill('pill-desc', !!connection.description_ok, t('pill.model'));
  pill('pill-tel', !!connection.telemetry_ok, t('pill.telemetry'));
  pill('pill-act', !!connection.action_ok, t('pill.action'));

  const collision = snapshot.collision || {};
  if (!collision.available) pill('pill-collide', null, t('pill.clearance'));
  else pill('pill-collide', collision.clear,
            t(collision.clear ? 'pill.clear' : 'pill.contact'));

  const quantity = connection.effort_source === 'torque'
    ? `${t('live.torque')} (N·m)` : `${t('live.current')} (A)`;
  const rows = [
    ['conn.transport', connection.transport],
    ['conn.topic', connection.topic],
    ['conn.action', connection.action],
    ['conn.effort_unit', quantity],
    ['conn.sample_age', connection.sample_age_s == null
      ? '—' : `${connection.sample_age_s}s`],
    ['conn.driven', (snapshot.driven_joints || []).length || '—'],
    ['conn.profile', snapshot.profile_edited
      ? t('profile.from_panel') : (snapshot.profile_source || 'none')],
    ['conn.shapes', collision.robot_shapes ?? '—'],
    ['conn.obstacles', collision.enabled_obstacles ?? 0],
  ];
  const extra = connection.extra_topics || [];
  if (extra.length) rows.splice(2, 0, ['conn.extra_topics', extra.join(', ')]);
  $('conn-table').innerHTML = rows.map(
    ([k, v]) => `<tr><td>${t(k)}</td><td class="num">${v ?? '—'}</td></tr>`)
    .join('');

  const missing = (connection.missing_guards || []).slice();
  if (snapshot.profile_source === 'derived' && !snapshot.current_guard) {
    missing.push('current ceiling (derived profile)');
  }
  if (snapshot.profile_source === 'none') {
    $('guards').textContent = t('conn.noprofile');
    $('guards').style.color = 'var(--bad)';
    return;
  }
  $('guards').textContent = missing.length
    ? t('conn.guards_off', { v: missing.join(', ') })
    : t('conn.guards_all');
  $('guards').style.color = missing.length ? 'var(--warn)' : 'var(--muted)';
}

/** Phase names double as progress states ("finished", "failed"), which have
 * no translation entry; t() hands back the key, so fall through to it. */
function phaseLabel(name) {
  const key = 'phase.' + name;
  const text = t(key);
  return text === key ? name : text;
}

/** The sweep has its own progress shape: it reports a joint and a level rather
 * than one of the campaign's phases, so it gets its own line instead of being
 * forced through a phase bar that has no box for it. */
function renderSweep(snapshot) {
  const progress = snapshot.progress || {};
  const line = $('sweep-line');
  if (progress.mode !== 'load_sweep') { line.textContent = ''; return; }
  if (progress.phase === 'designing') {
    line.textContent = t('sweep.designing', { j: progress.joint || '—' });
    return;
  }
  if (progress.phase === 'complete' || progress.phase === 'failed') {
    line.textContent = progress.phase === 'failed'
      ? t('sweep.failed', { v: progress.error || '' })
      : t('sweep.done', { n: progress.driven, s: progress.skipped });
    return;
  }
  const share = progress.total
    ? Math.round(100 * (progress.passes || 0) / progress.total) : 0;
  line.textContent = t('sweep.running', {
    j: progress.joint_name || progress.joint,
    l: progress.level, L: progress.levels,
    v: progress.speed_deg_s,
    n: progress.driven || 0, t: progress.total || 0, p: share,
    m: Math.round((progress.elapsed_s || 0) / 60),
  });
}

function renderRun(snapshot) {
  const running = snapshot.state === 'running';
  // Jogging holds the trajectory action open, so a campaign cannot have it.
  const busy = running || !!snapshot.jogging;
  $('run-state').textContent = running
    ? (snapshot.activity || t('state.running')) : t('state.idle');
  $('run-state').className = 'state' + (running ? ' running' : '');
  $('btn-rehearse').disabled = busy || !snapshot.have_model;
  $('btn-hardware').disabled = busy || !snapshot.rehearsal_passed;
  $('btn-optimal').disabled = busy || !snapshot.rehearsal_passed;
  // Homing runs no identification, so the rehearsal gate does not apply; it
  // would only block recovering an arm the plant already refuses to arm.
  $('btn-home').disabled = busy || !snapshot.have_model;
  $('btn-stop').disabled = !running && !snapshot.jogging;
  // The sweep fits nothing, so the rehearsal gate does not apply to it either.
  // What it does need is a collision screen with the arm actually in it, which
  // it proves for itself before planning any motion.
  $('btn-sweep').disabled = busy || !snapshot.have_model;
  renderSweep(snapshot);
  $('ack-hint').textContent = snapshot.rehearsal_passed
    ? t('run.armed') : t('run.locked');

  const progress = snapshot.progress || {};
  const current = progress.phase || '';
  const done = current === 'finished';
  $('phasebar').innerHTML = PHASES.map((name) => {
    const at = PHASES.indexOf(current);
    const index = PHASES.indexOf(name);
    const cls = current === name ? 'now' : (at > index || done) ? 'done' : '';
    return `<div class="ph ${cls}">${phaseLabel(name)}</div>`;
  }).join('');

  const bits = [];
  if (progress.phase) bits.push(phaseLabel(progress.phase));
  if (progress.elapsed_s != null) bits.push(`${Math.round(progress.elapsed_s)}s`);
  if (progress.observations != null) {
    bits.push(`${progress.observations} ${t('run.samples')}`);
  }
  if (progress.pose != null) {
    bits.push(`${t('run.pose')} ${progress.pose}/${progress.poses ?? '?'}`);
  }
  if (progress.trajectory != null) {
    bits.push(`${t('optimal.trajectory')} ${progress.trajectory}/${progress.trajectories ?? '?'}`);
  }
  if (progress.worst_deg != null) {
    bits.push(t('run.worst', { v: progress.worst_deg }));
  }
  $('progress-line').textContent = bits.join(' · ') || t('run.notstarted');
  const failed = progress.phase === 'failed';
  if (progress.error) {
    $('progress-line').textContent =
      `${failed ? t('run.failed') : t('run.stopped')}: ${progress.error}`;
    $('progress-line').style.color = failed ? 'var(--bad)' : 'var(--warn)';
  } else {
    $('progress-line').style.color = '';
  }
}

/* ---------------- live signals over the 3D view ----------------
 * Every quantity at once, because they only mean anything together: a current
 * is a reading, a current at that speed and that temperature is evidence.
 *
 * Fed from /api/telemetry rather than the state poll. The state poll carries a
 * collision report and a directory listing and cannot run fast; the robot
 * publishes at a hundred hertz or more, and sampling that four times a second
 * does not give a slower signal, it gives a different one with the peaks
 * removed.
 */

const SIGNALS = [
  { key: 'position_deg', label: 'live.position', unit: '°', digits: 1 },
  { key: 'speed_deg_s', label: 'live.speed', unit: '°/s', digits: 1 },
  { key: 'drive_current_a', label: 'live.current', unit: 'A', digits: 3 },
  { key: 'joint_torque_nm', label: 'live.torque', unit: 'N·m', digits: 3 },
  { key: 'temperature_c', label: 'live.temperature', unit: '°C', digits: 1 },
  { key: 'voltage_v', label: 'live.voltage', unit: 'V', digits: 1 },
];
const TELEMETRY_MS = 100;
// A frame older than this is not a slow reading, it is a stopped feed. Leaving
// the last one on screen is worse than showing nothing: a frozen number looks
// exactly like a live one.
const TELEMETRY_STALE_MS = 1500;
// Deep enough to hold a few seconds at the rate a real arm publishes.
const TRACE_DEPTH = 1600;
// Plot heights per mode; 'big' is a ceiling, not a promise.
const PLOT_HEIGHT = { normal: 200, big: 460 };

/** The signals this robot actually publishes.
 *
 * One it does not publish is left out rather than drawn flat at zero, which
 * would be a reading. */
function availableSignals(frame) {
  return SIGNALS.filter(
    (signal) => Array.isArray(frame[signal.key]) && frame[signal.key].length);
}

function buildSignalRows(names, offered) {
  $('signal-head').innerHTML = '<span class="dot"></span><span class="name"></span>'
    + '<span class="track"></span>'
    + offered.map((signal) => `<span class="val">${signal.unit}</span>`).join('');
  $('joint-bars').innerHTML = names.map((name, index) =>
    `<div class="jb" data-joint="${index}" title="${name}">`
    + `<span class="dot" style="background:${SERIES[index % SERIES.length]}"></span>`
    + `<span class="name">${name}</span>`
    + `<span class="track"><span class="fill" style="background:`
    + `${SERIES[index % SERIES.length]}"></span></span>`
    + offered.map(() => '<span class="val">—</span>').join('')
    + '</div>').join('');
  $('signal-tabs').innerHTML = offered.map((signal) =>
    `<button class="small" data-signal="${signal.key}">${t(signal.label)}</button>`)
    .join('');
  state.rows = [...$('joint-bars').children].map((row) => ({
    row,
    fill: row.querySelector('.fill'),
    values: [...row.querySelectorAll('.val')],
  }));
  state.rowKey = `${names.join()}|${offered.map((s) => s.key).join()}|${getLang()}`;
}

function renderSignals() {
  const frame = state.trace[state.trace.length - 1];
  const names = state.jointNames;
  const panel = $('jointbars');
  const fresh = Date.now() - state.lastFrameAt < TELEMETRY_STALE_MS;
  if (!frame || !names.length || !fresh) {
    panel.classList.add('waiting');
    $('signal-rate').textContent = '';
    $('signal-waiting').textContent =
      t(state.lastFrameAt ? 'signals.lost' : 'signals.waiting');
    return;
  }
  panel.classList.remove('waiting');

  const offered = availableSignals(frame);
  if (!offered.length) return;
  const key = `${names.join()}|${offered.map((s) => s.key).join()}|${getLang()}`;
  if (key !== state.rowKey) buildSignalRows(names, offered);
  if (!offered.some((signal) => signal.key === state.signal)) {
    state.signal = offered[0].key;
  }
  for (const button of $('signal-tabs').children) {
    button.classList.toggle('sel', button.dataset.signal === state.signal);
  }

  // Only position gets a bar, and its length is the joint's own travel: a bar
  // that ends where the joint does says how much room is left, which is the
  // one thing a number on its own cannot.
  const reach = state.snapshot?.reach_deg || [];
  const positions = frame.position_deg || [];
  const sample = state.snapshot?.sample || {};
  state.rows.forEach((entry, index) => {
    const limit = reach[index] > 0 ? reach[index] : 180;
    const value = positions[index];
    const half = Number.isFinite(value)
      ? Math.min(50, Math.abs(value) / limit * 50) : 0;
    entry.fill.style.left = `${value >= 0 ? 50 : 50 - half}%`;
    entry.fill.style.width = `${half}%`;
    entry.row.classList.toggle('off', state.hidden.has(index));
    // A drive that is faulted or not enabled is still publishing numbers, so
    // the numbers cannot say so; the joint's own name is where it shows.
    const off = sample.enabled ? sample.enabled[index] === false : false;
    const fault = sample.fault_code ? sample.fault_code[index] : 0;
    entry.row.classList.toggle('faulted', off || !!fault);
    offered.forEach((signal, column) => {
      const reading = (frame[signal.key] || [])[index];
      entry.values[column].textContent =
        Number.isFinite(reading) ? reading.toFixed(signal.digits) : '—';
      entry.values[column].classList.toggle(
        'bad', signal.key === 'temperature_c' && reading > 45);
    });
  });

  drawTraceChart(names, offered.find((entry) => entry.key === state.signal));
}

function drawTraceChart(names, signal) {
  if (!signal || state.plot === 'off') return;
  const canvas = $('chart-trace');
  // Thinning to the canvas rather than to a constant: expanding the plot is
  // asking for more of the signal, not the same picture drawn larger.
  const points = Math.max(120, Math.round(canvas.clientWidth || 300));
  const step = Math.max(1, Math.ceil(state.trace.length / points));
  const window = state.trace.filter((_frame, index) => index % step === 0);
  const series = names.map((name, index) => ({
    colour: SERIES[index % SERIES.length],
    values: state.hidden.has(index)
      ? [] : window.map((frame) => (frame[signal.key] || [])[index]),
  }));
  const span = state.trace.length > 1
    ? (state.trace[state.trace.length - 1].at - state.trace[0].at) / 1000 : 0;
  drawTrace(canvas, series, {
    unit: signal.unit,
    window: span > 0.5 ? `${span.toFixed(1)}s` : '',
    empty: t('signals.nodata'),
  });
}

async function pollTelemetry() {
  let data;
  try {
    data = await (await fetch(`/api/telemetry?since=${state.cursor}`,
                              { cache: 'no-store' })).json();
  } catch (error) {
    // The panel has to say the feed is gone; silence looks like a slow robot.
    return renderSignals();
  }
  if (data.joint_names && data.joint_names.join() !== state.jointNames.join()) {
    state.jointNames = data.joint_names;
    state.trace = [];
  }
  const now = Date.now();
  for (const frame of data.frames || []) {
    // The bridge stamps how old each frame was when it left, so the plot is on
    // the robot's clock rather than on however the poll happened to land.
    state.trace.push({ ...frame, at: now - (frame.age_s || 0) * 1000 });
  }
  if (data.frames?.length) state.lastFrameAt = now;
  if (state.trace.length > TRACE_DEPTH) {
    state.trace.splice(0, state.trace.length - TRACE_DEPTH);
  }
  state.cursor = data.cursor ?? state.cursor;
  state.rate = rateOf(state.trace);
  $('signal-rate').textContent = state.rate ? `${state.rate} Hz` : '';
  renderSignals();
}

function rateOf(trace) {
  if (trace.length < 8) return 0;
  const span = (trace[trace.length - 1].at - trace[0].at) / 1000;
  return span > 0.2 ? Math.round((trace.length - 1) / span) : 0;
}

$('signal-tabs').addEventListener('click', (event) => {
  const key = event.target.dataset?.signal;
  if (!key) return;
  state.signal = key;
  renderSignals();
});

$('joint-bars').addEventListener('click', (event) => {
  const row = event.target.closest('.jb');
  if (!row) return;
  const index = Number(row.dataset.joint);
  if (state.hidden.has(index)) state.hidden.delete(index);
  else state.hidden.add(index);
  renderSignals();
});

$('plot-modes').addEventListener('click', (event) => {
  const mode = event.target.dataset?.plot;
  if (!mode) return;
  state.plot = mode;
  applyPlotMode();
  renderSignals();
});

function applyPlotMode() {
  const panel = $('jointbars');
  panel.classList.toggle('expanded', state.plot === 'big');
  panel.classList.toggle('noplot', state.plot === 'off');
  for (const button of $('plot-modes').children) {
    button.classList.toggle('sel', button.dataset.plot === state.plot);
  }
  if (state.plot === 'off') return;
  // Expanded takes what the view can spare rather than a fixed number, or the
  // panel outgrows the window it floats over.
  const room = ($('stage').clientHeight || 600) - 300;
  // charts.js caches the logical height on first draw, so the cache is what
  // has to change; the attribute alone would be ignored.
  $('chart-trace').dataset.logicalHeight = String(
    state.plot === 'big'
      ? Math.max(PLOT_HEIGHT.normal, Math.min(PLOT_HEIGHT.big, room))
      : PLOT_HEIGHT.normal);
}
applyPlotMode();

/* ---------------- jogging ----------------
 * Drives the same FollowJointTrajectory action the campaign uses, so there is
 * no second command path to keep safe. The sliders follow the arm while nobody
 * is touching them, which is what makes them readable as a position display as
 * well as a control.
 */

const jog = { rows: [], key: '', held: false };

function renderJog(snapshot) {
  const names = (snapshot.driven_joints || []).length
    ? snapshot.driven_joints : (snapshot.joint_names || []);
  const limits = snapshot.reach_deg || [];
  const jogging = !!snapshot.jogging;
  const ready = names.length > 0 && limits.length === names.length;

  const button = $('btn-jog');
  button.textContent = t(jogging ? 'jog.disable' : 'jog.enable');
  button.classList.toggle('danger', !jogging);
  button.classList.toggle('stop', jogging);
  button.disabled = snapshot.state === 'running' || !ready;
  $('btn-jog-zero').disabled = !jogging;

  const key = `${names.join()}|${limits.join()}|${getLang()}`;
  if (key !== jog.key) {
    if (!ready) {
      $('jog-sliders').innerHTML = `<p class="hint">${t('jog.nomodel')}</p>`;
      jog.rows = [];
      jog.key = key;
      return;
    }
    buildJogRows(names, limits);
    jog.key = key;
  }
  for (const row of jog.rows) row.slider.disabled = !jogging;
  if (!jog.held) showJogPose(snapshot.sample?.position_deg || []);
}

function buildJogRows(names, limits) {
  $('jog-sliders').innerHTML = names.map((name, index) =>
    `<div class="jogrow"><span class="name" title="${name}">${name}</span>`
    + `<input type="range" data-joint="${index}" disabled`
    + ` min="${-limits[index]}" max="${limits[index]}" step="0.1" value="0"/>`
    + `<span class="val" data-readout="${index}">0.0\u00b0</span></div>`).join('');
  jog.rows = [...$('jog-sliders').children].map((row) => ({
    slider: row.querySelector('input'),
    readout: row.querySelector('.val'),
  }));
}

/** Show a pose on the sliders without commanding anything. */
function showJogPose(pose) {
  jog.rows.forEach((row, index) => {
    const value = pose[index];
    if (!Number.isFinite(value)) return;
    row.slider.value = String(value);
    row.readout.textContent = `${value.toFixed(1)}\u00b0`;
  });
}

function jogPose() {
  return jog.rows.map((row) => parseFloat(row.slider.value) || 0);
}

$('jog-sliders').addEventListener('input', (event) => {
  const index = event.target.dataset?.joint;
  if (index == null) return;
  jog.held = true;
  jog.rows[index].readout.textContent =
    `${(parseFloat(event.target.value) || 0).toFixed(1)}\u00b0`;
});

// On release, not on every pixel: one goal per gesture, so the controller is
// not handed a new trajectory sixty times a second.
$('jog-sliders').addEventListener('change', async (event) => {
  if (event.target.dataset?.joint == null) return;
  jog.held = false;
  await post('/api/jog', { action: 'move', position_deg: jogPose() });
});

$('btn-jog').addEventListener('click', async () => {
  const jogging = !!state.snapshot?.jogging;
  await post('/api/jog', { action: jogging ? 'stop' : 'start' });
});

$('btn-jog-zero').addEventListener('click', async () => {
  const zeros = jog.rows.map(() => 0);
  showJogPose(zeros);
  await post('/api/jog', { action: 'move', position_deg: zeros });
});

function renderResult(snapshot) {
  const result = snapshot.result;
  state.unit = snapshot.effort_unit === 'newton_metre' ? 'N·m' : 'A';
  drawErrors($('chart-error'), result, state.unit);
  drawCondition($('chart-condition'), result, 1000);

  const joints = result?.joints || [];
  // The result carries the names the run was recorded against; the live model
  // may already describe a different arm.
  const names = (result?.joint_names || []).length
    ? result.joint_names : (snapshot.joint_names || []);
  const label = (index) => names[index] || `${index + 1}`;
  const select = $('friction-joint');
  if (select.options.length !== joints.length) {
    select.innerHTML = joints.map(
      (_, i) => `<option value="${i}">${label(i)}</option>`).join('');
  }
  const index = Math.min(parseInt(select.value || '0', 10), Math.max(0, joints.length - 1));
  drawFriction($('chart-friction'), joints[index], result?.friction_samples?.[index],
               state.unit);
  drawResidual($('chart-residual'), result?.residual_samples?.[index]);

  const verdict = result?.verdict?.joints || [];
  $('verdict-table').innerHTML = verdict.map((entry, i) =>
    `<tr><td>${label(i)}</td><td class="num ${entry.state === 'pass' ? 'ok' : 'bad'}">`
    + `${t('verdict.' + entry.state)}</td></tr>`).join('');

  const recovery = result?.rehearsal_check;
  const note = $('recovery');
  if (!recovery || !recovery.available) {
    note.textContent = '';
  } else if (recovery.passed) {
    note.textContent = t('recovery.ok', { v: recovery.worst_coulomb_error,
                                          t: recovery.tolerance });
    note.style.color = 'var(--ok)';
  } else {
    note.textContent = t('recovery.bad', { v: recovery.worst_coulomb_error,
                                           t: recovery.tolerance });
    note.style.color = 'var(--bad)';
  }

  $('params').innerHTML = joints.map((entry, i) => {
    const friction = entry.friction || {};
    const components = entry.components || {};
    const physical = (friction.coulomb ?? 0) >= 0 && (friction.viscous ?? 0) >= 0;
    return `<div class="param-row"><span title="${label(i)}">${label(i)}</span>`
      + `<span class="muted">c ${(friction.coulomb ?? 0).toFixed(3)} ${state.unit}`
      + `${components.load_friction ? ` \u00b7 \u03bc ${(friction.load_friction ?? 0).toFixed(3)}` : ''}`
      + ` \u00b7 v ${(friction.viscous ?? 0).toFixed(4)} ${state.unit}/(\u00b0/s)</span>`
      + `<span class="badge ${physical ? 'ok' : 'bad'}">`
      + `${t(physical ? 'params.physical' : 'params.unphysical')}</span></div>`;
  }).join('') || `<span class="muted">${t('params.none')}</span>`;
  renderComparison(result, names);
}

function renderComparison(result, names) {
  const card = $('compare-card');
  const comparison = result?.comparison;
  card.classList.toggle('hidden', !comparison || !Object.keys(comparison).length);
  if (!comparison || !Object.keys(comparison).length) return;
  if (!comparison.available) {
    $('compare-summary').innerHTML = `<p class="hint">${t('compare.unavailable')}: `
      + `${comparison.reason || ''}</p>`;
    $('compare-table').innerHTML = '';
    return;
  }
  const signed = (value) => Number.isFinite(value)
    ? `${value >= 0 ? '+' : ''}${value.toFixed(1)}%` : '—';
  const statusClass = comparison.target_met ? 'ok' : 'bad';
  $('compare-summary').innerHTML = `<p><span class="badge ${statusClass}">`
    + `${t(comparison.target_met ? 'compare.met' : 'compare.missed')}</span></p>`
    + `<p class="hint">${t('compare.mean')}: `
    + `${comparison.optimal_mean_validation_rms_a.toFixed(4)} / `
    + `${comparison.sweep_mean_validation_rms_a.toFixed(4)} ${state.unit || 'A'} `
    + `(${signed(comparison.mean_improvement_percent)}); ${t('compare.worst')}: `
    + `${comparison.optimal_worst_validation_rms_a.toFixed(4)} / `
    + `${comparison.sweep_worst_validation_rms_a.toFixed(4)} ${state.unit || 'A'} `
    + `(${signed(comparison.worst_improvement_percent)})</p>`;
  const rows = comparison.joints || [];
  $('compare-table').innerHTML = `<tr><th>${t('live.joint')}</th>`
    + `<th class="num">${t('compare.optimal')}</th>`
    + `<th class="num">${t('compare.sweep')}</th>`
    + `<th class="num">${t('compare.improvement')}</th></tr>`
    + rows.map((entry, index) => `<tr><td>${names[index] || entry.name || index + 1}</td>`
      + `<td class="num">${entry.optimal_validation_rms_a.toFixed(4)}</td>`
      + `<td class="num">${entry.sweep_validation_rms_a.toFixed(4)}</td>`
      + `<td class="num ${entry.optimal_better ? 'ok' : 'bad'}">`
      + `${signed(entry.improvement_percent)}</td></tr>`).join('');
}

function renderRuns(runs) {
  const list = $('runs');
  if (!runs || !runs.length) {
    list.innerHTML = `<li class="muted">${t('out.none')}</li>`;
    return;
  }
  list.innerHTML = runs.map((run) =>
    `<li><span>${run.name}</span>`
    + `<a href="${run.report}" target="_blank" rel="noopener">`
    + `${t('out.report')}</a></li>`).join('');
}

$('friction-joint').addEventListener('change', () => renderResult(state.snapshot || {}));

/* ---------------- robot envelope ----------------
 * The profile is editable before any file exists: with none the module derives
 * one from the URDF, and that derivation is what this form shows. Saving is a
 * way to keep the answer, not a step needed before running.
 */

const JOINT_COLUMNS = [
  { key: 'position_deg', label: 'profile.position', unit: '°' },
  { key: 'workspace_deg', label: 'profile.workspace', unit: '°' },
  { key: 'continuous_current_a', label: 'profile.continuous', unit: 'A' },
  { key: 'peak_current_a', label: 'profile.peak', unit: 'A' },
];
// Blank means "nobody measured this", which is a live distinction only for the
// current ceilings; every other number always has a value.
const BLANKABLE = new Set(['continuous_current_a', 'peak_current_a']);
const ENVELOPE_KEYS = [
  ['temperature_c', '°C'], ['sustained_speed_deg_s', '°/s'],
  ['peak_speed_deg_s', '°/s'], ['position_margin_deg', '°'],
  ['minimum_voltage_v', 'V'], ['maximum_voltage_v', 'V'],
  ['sustained_current_window_s', 's'], ['sustained_speed_window_s', 's'],
  ['current_slew_a_s', 'A/s'], ['sender_gap_s', 's'],
  ['telemetry_stale_s', 's'], ['probe_current_fraction', ''],
];

function openProfile() {
  $('profile-modal').classList.remove('hidden');
  loadProfile();
}

async function loadProfile() {
  const data = await (await fetch('/api/profile', { cache: 'no-store' })).json();
  renderProfile(data);
}

function renderProfile(data) {
  const profile = data.profile;
  $('profile-origin').textContent = data.origin || '';
  $('profile-file').placeholder = data.save_target || 'profile.yaml';
  if (!profile) {
    $('profile-state').textContent = t('profile.nomodel');
    $('profile-joints').innerHTML = '';
    $('profile-envelope').innerHTML = '';
    return;
  }
  $('profile-state').textContent = data.editable
    ? t(data.current_guard ? 'profile.guard_on' : 'profile.guard_off')
    : t('profile.busy');
  $('profile-name').value = profile.name || '';

  const names = profile.joints.names || [];
  const head = JOINT_COLUMNS.map(
    (column) => `<th>${t(column.label)}<span class="unit"> ${column.unit}</span></th>`);
  const rows = names.map((name, index) => `<tr><td>${name}</td>` + JOINT_COLUMNS.map(
    (column) => {
      const value = (profile.limits[column.key] || [])[index];
      return `<td><input type="number" step="any" data-limit="${column.key}"`
        + ` data-index="${index}" value="${value == null ? '' : value}"`
        + `${BLANKABLE.has(column.key) ? ' placeholder="∞"' : ''}/></td>`;
    }).join('') + '</tr>');
  $('profile-joints').innerHTML =
    `<tr><th>${t('profile.joint')}</th>${head.join('')}</tr>${rows.join('')}`;

  $('profile-envelope').innerHTML = ENVELOPE_KEYS.map(([key, unit]) =>
    `<label class="field"><span>${t(`profile.${key}`)}`
    + `${unit ? ` (${unit})` : ''}</span>`
    + `<input type="number" step="any" data-envelope="${key}"`
    + ` value="${profile.envelope[key] ?? ''}"/></label>`).join('');

  for (const node of $('profile-modal').querySelectorAll('input')) {
    node.disabled = !data.editable;
  }
  $('profile-file').disabled = false;
  state.profile = profile;
}

function collectProfile() {
  const base = state.profile;
  if (!base) return null;
  const limits = {};
  for (const column of JOINT_COLUMNS) {
    const values = (base.limits[column.key] || []).slice();
    if (!values.length && column.key === 'workspace_deg') continue;
    limits[column.key] = values;
  }
  for (const node of $('profile-joints').querySelectorAll('input[data-limit]')) {
    const key = node.dataset.limit;
    const index = Number(node.dataset.index);
    if (!limits[key]) limits[key] = [];
    const text = node.value.trim();
    limits[key][index] = text === '' ? null : Number(text);
  }
  // An all-blank workspace row is no cap at all, and the schema says that by
  // omitting the key rather than by carrying a row of nulls.
  if ((limits.workspace_deg || []).every((entry) => entry == null)) {
    delete limits.workspace_deg;
  }
  const envelope = {};
  for (const node of $('profile-envelope').querySelectorAll('input[data-envelope]')) {
    const text = node.value.trim();
    if (text !== '') envelope[node.dataset.envelope] = Number(text);
  }
  return {
    schema_version: base.schema_version,
    name: $('profile-name').value.trim() || base.name,
    joints: { prefix: base.joints.prefix, names: base.joints.names },
    limits,
    envelope,
    notes: base.notes || {},
  };
}

$('btn-profile').addEventListener('click', openProfile);
$('profile-close').addEventListener('click',
  () => $('profile-modal').classList.add('hidden'));
$('profile-modal').addEventListener('click', (event) => {
  if (event.target === $('profile-modal')) $('profile-modal').classList.add('hidden');
});
$('profile-apply').addEventListener('click', async () => {
  const profile = collectProfile();
  if (!profile) return;
  const data = await post('/api/profile/apply', { profile });
  if (data.ok) {
    toast(t('profile.applied'), 'ok');
    renderProfile(data.profile);
  }
});
$('profile-reset').addEventListener('click', async () => {
  const data = await post('/api/profile/reset', {});
  if (data.ok) {
    toast(t('profile.was_reset'), 'ok');
    renderProfile(data.profile);
  }
});
$('profile-save').addEventListener('click', async () => {
  const data = await post('/api/profile/save', { name: $('profile-file').value.trim() });
  if (data.ok) toast(t('profile.saved', { v: data.path }), 'ok');
});

/* ---------------- poll ---------------- */

function renderAll(snapshot) {
  renderConnection(snapshot);
  renderRun(snapshot);
  renderJog(snapshot);
  renderObstacleList();
  if (state.selected) fillForm();
  renderResult(snapshot);
  // Redrawn from the cache: the listing is polled rarely, and a language
  // change must not leave it in the old one until the next scan.
  renderRuns(state.runs);
  $('notes').textContent = (snapshot.notes || []).join('\n');
}

async function pollRuns() {
  try {
    const data = await (await fetch('/api/runs', { cache: 'no-store' })).json();
    state.runs = data.runs || [];
    renderRuns(state.runs);
  } catch (error) { /* the panel is still usable without the listing */ }
}

async function poll() {
  try {
    const snapshot = await (await fetch('/api/state', { cache: 'no-store' })).json();
    state.snapshot = snapshot;
    renderAll(snapshot);
  } catch (error) {
    pill('pill-desc', false, t('pill.offline'));
  }
  if (state.ticks % RUNS_EVERY === 0) pollRuns();
  state.ticks += 1;
}
setInterval(poll, POLL_MS);
poll();
setInterval(pollTelemetry, TELEMETRY_MS);
pollTelemetry();
