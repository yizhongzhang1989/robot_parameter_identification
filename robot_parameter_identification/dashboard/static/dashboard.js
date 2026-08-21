/* Panel logic: poll state, drive the forms, keep the 3D view and the charts
 * looking at the same snapshot.
 */

import { drawCondition, drawErrors, drawFriction, drawResidual } from '/charts.js';
import { LANGUAGES, applyStatic, getLang, onLangChange, setLang, t }
  from '/i18n.js';

const POLL_MS = 400;
// A directory scan does not belong on the fast path; saved runs change once
// per campaign, not four times a second.
const RUNS_EVERY = 25;
const PHASES = ['A_gravity', 'B_friction', 'C_inertia', 'D_validation'];

const $ = (id) => document.getElementById(id);
const state = { snapshot: null, selected: null, frames: [], unit: 'A',
                ticks: 0, runs: [] };

/* ---------------- language ---------------- */

const langSelect = $('lang');
langSelect.innerHTML = LANGUAGES.map(
  (entry) => `<option value="${entry.code}">${entry.label}</option>`).join('');
langSelect.value = getLang();
langSelect.addEventListener('change', () => setLang(langSelect.value));
onLangChange(() => { if (state.snapshot) renderAll(state.snapshot); });
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
    ['conn.profile', snapshot.profile_source || 'none'],
    ['conn.shapes', collision.robot_shapes ?? '—'],
    ['conn.obstacles', collision.enabled_obstacles ?? 0],
  ];
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
  $('run-state').textContent = running
    ? (snapshot.activity || t('state.running')) : t('state.idle');
  $('run-state').className = 'state' + (running ? ' running' : '');
  $('btn-rehearse').disabled = running || !snapshot.have_model;
  $('btn-hardware').disabled = running || !snapshot.rehearsal_passed;
  // Homing runs no identification, so the rehearsal gate does not apply; it
  // would only block recovering an arm the plant already refuses to arm.
  $('btn-home').disabled = running || !snapshot.have_model;
  $('btn-stop').disabled = !running;
  // The sweep fits nothing, so the rehearsal gate does not apply to it either.
  // What it does need is a collision screen with the arm actually in it, which
  // it proves for itself before planning any motion.
  $('btn-sweep').disabled = running || !snapshot.have_model;
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

function renderJointBars(snapshot) {
  const sample = snapshot.sample;
  const names = snapshot.joint_names || [];
  if (!sample || !names.length) { $('joint-bars').innerHTML = ''; return; }
  const positions = sample.position_deg || [];
  $('joint-bars').innerHTML = names.map((name, index) => {
    const value = positions[index] ?? 0;
    const pct = Math.min(100, Math.abs(value) / 180 * 100);
    const left = value >= 0 ? 50 : 50 - pct / 2;
    return `<div class="jb"><span class="name" title="${name}">${name}</span>`
      + `<span class="track"><span class="fill" style="left:${left}%;width:${pct / 2}%"></span></span>`
      + `<span class="val">${value.toFixed(1)}°</span></div>`;
  }).join('');
}

/* Columns appear only when the robot actually publishes them, so an empty
 * column never gets mistaken for a reading of zero. */
const LIVE_COLUMNS = [
  { key: 'position_deg', label: 'live.position', unit: '°', digits: 2 },
  { key: 'speed_deg_s', label: 'live.speed', unit: '°/s', digits: 2 },
  { key: 'drive_current_a', label: 'live.current', unit: 'A', digits: 3 },
  { key: 'joint_torque_nm', label: 'live.torque', unit: 'N·m', digits: 3 },
  { key: 'temperature_c', label: 'live.temperature', unit: '°C', digits: 1 },
  { key: 'voltage_v', label: 'live.voltage', unit: 'V', digits: 1 },
];

function renderLive(snapshot) {
  const table = $('live-table');
  const sample = snapshot.sample;
  const names = snapshot.joint_names || [];
  if (!sample || !names.length) {
    table.innerHTML = `<tr><td class="muted">${t('live.waiting')}</td></tr>`;
    return;
  }
  const fitted = snapshot.connection?.effort_source === 'torque'
    ? 'joint_torque_nm' : 'drive_current_a';
  const columns = LIVE_COLUMNS.filter(
    (column) => Array.isArray(sample[column.key]) && sample[column.key].length);
  const status = sample.enabled || sample.fault_code;

  const head = `<tr><th>${t('live.joint')}</th>`
    + columns.map((column) => {
      const mark = column.key === fitted
        ? ` <span class="tag">${t('live.fitted')}</span>` : '';
      return `<th class="num">${t(column.label)} <span class="unit">`
        + `${column.unit}</span>${mark}</th>`;
    }).join('')
    + (status ? `<th class="num">${t('live.fault')}</th>` : '')
    + '</tr>';

  const body = names.map((name, index) => {
    const cells = columns.map((column) => {
      const value = sample[column.key][index];
      const text = Number.isFinite(value) ? value.toFixed(column.digits) : '—';
      const hot = column.key === 'temperature_c' && value > 45;
      return `<td class="num${hot ? ' bad' : ''}">${text}</td>`;
    }).join('');
    let flag = '';
    if (status) {
      const code = sample.fault_code ? sample.fault_code[index] : 0;
      const off = sample.enabled ? sample.enabled[index] === false : false;
      const bad = off || (code !== undefined && code !== 0);
      const label = off ? t('live.disabled') : (code ? String(code) : 'ok');
      flag = `<td class="num${bad ? ' bad' : ' ok'}">${label}</td>`;
    }
    return `<tr><td title="${name}">${name}</td>${cells}${flag}</tr>`;
  }).join('');

  table.innerHTML = head + body;
}

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
    const physical = (friction.coulomb ?? 0) >= 0 && (friction.viscous ?? 0) >= 0;
    return `<div class="param-row"><span title="${label(i)}">${label(i)}</span>`
      + `<span class="muted">c ${(friction.coulomb ?? 0).toFixed(3)} ${state.unit}`
      + ` \u00b7 v ${(friction.viscous ?? 0).toFixed(4)} ${state.unit}/(\u00b0/s)</span>`
      + `<span class="badge ${physical ? 'ok' : 'bad'}">`
      + `${t(physical ? 'params.physical' : 'params.unphysical')}</span></div>`;
  }).join('') || `<span class="muted">${t('params.none')}</span>`;
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

/* ---------------- poll ---------------- */

function renderAll(snapshot) {
  renderConnection(snapshot);
  renderRun(snapshot);
  renderLive(snapshot);
  renderJointBars(snapshot);
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
