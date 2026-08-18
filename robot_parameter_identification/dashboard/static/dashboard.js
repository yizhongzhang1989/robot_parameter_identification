/* Panel logic: poll state, drive the forms, keep the 3D view and the charts
 * looking at the same snapshot.
 */

import { drawCondition, drawErrors, drawFriction, drawResidual } from '/charts.js';

const POLL_MS = 400;
const PHASES = ['A_gravity', 'B_friction', 'C_inertia', 'D_validation'];

const $ = (id) => document.getElementById(id);
const state = { snapshot: null, selected: null, frames: [], unit: 'A' };

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
  if (!frame) return toast('no model yet', 'err');
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

/* ---------------- render ---------------- */

function pill(id, ok, label) {
  const node = $(id);
  node.className = 'pill ' + (ok === null ? '' : ok ? 'good' : 'bad');
  node.textContent = label;
}

function renderConnection(snapshot) {
  const connection = snapshot.connection || {};
  pill('pill-desc', !!connection.description_ok, 'model');
  pill('pill-tel', !!connection.telemetry_ok, 'telemetry');
  pill('pill-act', !!connection.action_ok, 'action');

  const collision = snapshot.collision || {};
  if (!collision.available) pill('pill-collide', null, 'clearance');
  else pill('pill-collide', collision.clear, collision.clear ? 'clear' : 'contact');

  const rows = [
    ['transport', connection.transport],
    ['topic', connection.topic],
    ['action', connection.action],
    ['effort unit', connection.effort_unit],
    ['sample age', connection.sample_age_s == null ? '—' : `${connection.sample_age_s}s`],
    ['joints driven', (snapshot.driven_joints || []).length || '—'],
    ['profile', snapshot.profile_source || 'none'],
    ['robot shapes', collision.robot_shapes ?? '—'],
    ['obstacles', collision.enabled_obstacles ?? 0],
  ];
  $('conn-table').innerHTML = rows.map(
    ([k, v]) => `<tr><td>${k}</td><td class="num">${v ?? '—'}</td></tr>`).join('');

  const missing = (connection.missing_guards || []).slice();
  if (snapshot.profile_source === 'derived' && !snapshot.current_guard) {
    missing.push('current ceiling (derived profile)');
  }
  if (snapshot.profile_source === 'none') {
    $('guards').textContent = 'No profile: waiting for the controller to name '
      + 'the joints it drives. Nothing can run until then.';
    $('guards').style.color = 'var(--bad)';
    return;
  }
  $('guards').textContent = missing.length
    ? `Guards off because the value is not available: ${missing.join(', ')}.`
    : 'All guards active.';
  $('guards').style.color = missing.length ? 'var(--warn)' : 'var(--muted)';
}

function renderRun(snapshot) {
  const running = snapshot.state === 'running';
  $('run-state').textContent = running ? (snapshot.activity || 'running') : 'idle';
  $('run-state').className = 'state' + (running ? ' running' : '');
  $('btn-rehearse').disabled = running || !snapshot.have_model;
  $('btn-hardware').disabled = running || !snapshot.rehearsal_passed;
  // Homing runs no identification, so the rehearsal gate does not apply; it
  // would only block recovering an arm the plant already refuses to arm.
  $('btn-home').disabled = running || !snapshot.have_model;
  $('btn-stop').disabled = !running;
  $('ack-hint').textContent = snapshot.rehearsal_passed
    ? 'Rehearsal passed: the hardware run is armed.'
    : 'A rehearsal must pass before a campaign may move the arm. '
      + 'Homing is always available.';

  const progress = snapshot.progress || {};
  const current = progress.phase || '';
  const done = current === 'finished';
  $('phasebar').innerHTML = PHASES.map((name) => {
    const at = PHASES.indexOf(current);
    const index = PHASES.indexOf(name);
    const cls = current === name ? 'now' : (at > index || done) ? 'done' : '';
    return `<div class="ph ${cls}">${name.split('_')[1]}</div>`;
  }).join('');

  const bits = [];
  if (progress.phase) bits.push(progress.phase);
  if (progress.elapsed_s != null) bits.push(`${Math.round(progress.elapsed_s)}s`);
  if (progress.observations != null) bits.push(`${progress.observations} samples`);
  if (progress.pose != null) bits.push(`pose ${progress.pose}/${progress.poses ?? '?'}`);
  if (progress.worst_deg != null) bits.push(`worst ${progress.worst_deg}\u00b0 from zero`);
  $('progress-line').textContent = bits.join(' · ') || 'not started';
  const failed = progress.phase === 'failed';
  if (progress.error) {
    $('progress-line').textContent =
      `${failed ? 'failed' : 'stopped'}: ${progress.error}`;
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
    return `<div class="jb"><span class="name">${name.slice(-7)}</span>`
      + `<span class="track"><span class="fill" style="left:${left}%;width:${pct / 2}%"></span></span>`
      + `<span class="val">${value.toFixed(1)}°</span></div>`;
  }).join('');
}

function renderResult(snapshot) {
  const result = snapshot.result;
  state.unit = snapshot.effort_unit === 'newton_metre' ? 'N·m' : 'A';
  drawErrors($('chart-error'), result, state.unit);
  drawCondition($('chart-condition'), result, 1000);

  const joints = result?.joints || [];
  const select = $('friction-joint');
  if (select.options.length !== joints.length) {
    select.innerHTML = joints.map((_, i) => `<option value="${i}">J${i + 1}</option>`).join('');
  }
  const index = Math.min(parseInt(select.value || '0', 10), Math.max(0, joints.length - 1));
  drawFriction($('chart-friction'), joints[index], result?.friction_samples?.[index],
               state.unit);
  drawResidual($('chart-residual'), result?.residual_samples?.[index]);

  const verdict = result?.verdict?.joints || [];
  $('verdict-table').innerHTML = verdict.map((entry, i) =>
    `<tr><td>J${i + 1}</td><td class="num ${entry.state === 'pass' ? 'ok' : 'bad'}">`
    + `${entry.state}</td></tr>`).join('');

  const recovery = result?.rehearsal_check;
  const note = $('recovery');
  if (!recovery || !recovery.available) {
    note.textContent = '';
  } else if (recovery.passed) {
    note.textContent = `Rehearsal recovered the planted friction to within `
      + `${recovery.worst_coulomb_error} (tolerance ${recovery.tolerance}).`;
    note.style.color = 'var(--ok)';
  } else {
    note.textContent = `Rehearsal ran but did NOT recover the planted friction: `
      + `worst error ${recovery.worst_coulomb_error} exceeds `
      + `${recovery.tolerance}. The hardware button stays locked.`;
    note.style.color = 'var(--bad)';
  }

  $('params').innerHTML = joints.map((entry, i) => {
    const friction = entry.friction || {};
    const physical = (friction.coulomb ?? 0) >= 0 && (friction.viscous ?? 0) >= 0;
    return `<div class="param-row"><span>J${i + 1}</span>`
      + `<span class="muted">c ${(friction.coulomb ?? 0).toFixed(3)} ${state.unit}`
      + ` \u00b7 v ${(friction.viscous ?? 0).toFixed(4)} ${state.unit}/(\u00b0/s)</span>`
      + `<span class="badge ${physical ? 'ok' : 'bad'}">`
      + `${physical ? 'physical' : 'unphysical'}</span></div>`;
  }).join('') || '<span class="muted">no result yet</span>';
}

$('friction-joint').addEventListener('change', () => renderResult(state.snapshot || {}));

/* ---------------- poll ---------------- */

async function poll() {
  try {
    const snapshot = await (await fetch('/api/state', { cache: 'no-store' })).json();
    state.snapshot = snapshot;
    renderConnection(snapshot);
    renderRun(snapshot);
    renderJointBars(snapshot);
    renderObstacleList();
    if (state.selected) fillForm();
    renderResult(snapshot);
    $('notes').textContent = (snapshot.notes || []).join('\n');
  } catch (error) {
    pill('pill-desc', false, 'offline');
  }
}
setInterval(poll, POLL_MS);
poll();
