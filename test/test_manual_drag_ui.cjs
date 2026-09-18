const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { systemConfigFixture } = require('./system_config_fixture.cjs');

const source = fs.readFileSync(path.join(__dirname,
  '../robot_parameter_identification/dashboard/static/dashboard.js'), 'utf8');
const configSource = fs.readFileSync(path.join(__dirname,
  '../robot_parameter_identification/dashboard/static/system_config.js'), 'utf8');
const i18nSource = fs.readFileSync(path.join(__dirname,
  '../robot_parameter_identification/dashboard/static/i18n.js'), 'utf8');
const html = fs.readFileSync(path.join(__dirname,
  '../robot_parameter_identification/dashboard/static/index.html'), 'utf8');
const systemConfig = systemConfigFixture({
  dashboard: { drag_test: { maximum_speed_deg_s: 92 } },
});
const elements = new Map();
const requests = [];
const confirmations = [];
const events = [];
const toasts = [];
const element = (id) => {
  if (!elements.has(id)) {
    const classes = new Set();
    elements.set(id, {
      id, value: '', disabled: true, handlers: {},
      type: typeof systemConfig.controls[id]?.value === 'boolean' ? 'checkbox' : 'number',
      classList: {
        toggle(name, enabled) { if (enabled) classes.add(name); else classes.delete(name); },
        contains(name) { return classes.has(name); },
      },
      addEventListener(event, handler) { this.handlers[event] = handler; },
    });
  }
  return elements.get(id);
};
let confirmed = true;
let confirmEffect = () => {};
let postReply = async () => ({ ok: true });
let snapshotReply = async () => { throw new Error('Offline fixture'); };
let badReply = false;
const context = vm.createContext({
  $: element,
  systemConfig,
  document: { querySelectorAll: () => Object.keys(systemConfig.controls).map(element) },
  state: { ticks: 1 },
  window: { confirm: (message) => {
    confirmations.push(message);
    confirmEffect();
    return confirmed;
  } },
  t: (key, values = {}) => key === 'gravtest.selected_arm' ? `Arm: ${values.arm}` : key,
  fetch: async (route, options) => {
    if (route === '/api/state') return snapshotReply();
    assert.ok(['/api/gravity-test', '/api/stop', '/api/recover-position'].includes(route));
    assert.equal(options.method, 'POST');
    assert.equal(options.headers['Content-Type'], 'application/json');
    const body = JSON.parse(options.body);
    requests.push(JSON.parse(JSON.stringify({ route, body })));
    const answer = await postReply(route, body);
    return { json: async () => {
      if (badReply) throw new Error('Invalid JSON fixture');
      return answer;
    } };
  },
  publishLocalEvent: (message, level, source) => events.push({ message, level, source }),
  toast: (message, kind) => toasts.push({ message, kind }),
  gravityArmed: () => false,
  gravityStatus: () => ({ level: 'info', message: '', passive: true }),
  gravityOptions: () => ({
    static_poses: 5, gravity_validation_poses: 2,
    gravity_probe_speeds_deg_s: [1, 3],
  }),
  renderGravityTestResult() {},
  activityLabel: (activity) => activity,
  renderSweep() {},
  seedGravityFields() {},
  renderWorkspace() {},
  refreshPlannedPoses() {},
  renderAll: (snapshot) => context.renderRun(snapshot),
  renderControllers() {},
  pill() {},
  pollRuns() {},
  RUNS_EVERY: 40,
});

vm.runInContext(`
  ${configSource.replace(/^export /gm, '')}
  ${source.slice(source.indexOf('async function post('), source.indexOf('let toastTimer'))}
  const GRAVITY_HOLD_TEST = 'gravity_hold_test';
  const GRAVITY_DRAG_TEST = 'gravity_drag_test';
  const GRAVITY_TEST_ACKNOWLEDGEMENT = 'I_AM_HOLDING_ARM_AND_ESTOP_READY';
  ${source.slice(source.indexOf('function holdPoseCount('),
    source.indexOf("$('btn-rescreen').addEventListener"))}
  ${source.slice(source.indexOf('function renderGravity(snapshot)'),
    source.indexOf('function renderGravityTestResult(snapshot)'))}
  ${source.slice(source.indexOf('function renderRun(snapshot)'),
    source.indexOf('/* ---------------- live signals'))}
  ${source.slice(source.indexOf('async function poll()'),
    source.indexOf('setInterval(poll, POLL_MS)'))}
`, context);

assert.equal(requests.length, 0, 'Registering handlers must not execute a command');

async function verifyRecovery() {
  const translations = vm.createContext({
    document: { documentElement: {}, querySelectorAll: () => [] },
    localStorage: { getItem: () => null, setItem() {} },
  });
  vm.runInContext(i18nSource.replace(/^export /gm, ''), translations);
  context.t = translations.t;
  const recover = element('btn-gravtest-recover');
  const recoveryMarkup = html.match(/<button id="btn-gravtest-recover"[^>]*>[\s\S]*?<\/button>/g);
  assert.equal(recoveryMarkup?.length, 1);
  assert.match(recoveryMarkup[0], /class="danger hidden" disabled/);
  assert.match(recoveryMarkup[0], /data-i18n="gravtest.recover"/);
  assert.match(recoveryMarkup[0], /data-i18n-title="gravtest.recover_hint"/);
  assert.match(html, /<div class="gravity-test-run-actions">\s*<button id="btn-gravtest-stop"/);
  assert.match(html, /id="btn-gravtest-stop"[^>]*>[\s\S]*?<\/button>\s*<button id="btn-gravtest-recover"/);
  recover.getAttribute = (name) => ({
    'data-i18n': 'gravtest.recover', 'data-i18n-title': 'gravtest.recover_hint',
  })[name];

  const pending = (arm = 'left') => ({
    state: 'paused', activity: 'gravity_drag_test', have_model: true,
    rehearsal_passed: true,
    connection: { description_ok: true, telemetry_ok: true, action_ok: false },
    gravity_test: { available: true, arm },
    hold_plan: { available: true, id: 'recovery-plan', poses: Number(element('gravtest-poses').value) },
    current_recovery: {
      required: true, available: true, running: false,
      reason: 'Position restoration was not verified.', arm,
    },
  });
  const render = (snapshot) => {
    context.state.snapshot = snapshot;
    context.renderRun(snapshot);
  };
  const locked = () => {
    for (const id of ['btn-rehearse', 'btn-hardware', 'btn-optimal', 'btn-home',
      'btn-sweep', 'btn-grav-rehearse', 'btn-grav-plan', 'btn-grav-run',
      'btn-grav-pause', 'btn-grav-resume', 'btn-gravtest-plan', 'btn-gravtest-hold',
      'btn-gravtest-drag', 'btn-rescreen', 'home-speed', 'jog-speed',
      'grav-transit-speed', 'gravtest-transit-speed']) {
      assert.equal(element(id).disabled, true, `${id} stays locked pending recovery`);
    }
  };
  let blockedChecks = 0;
  const blocked = async (snapshot, label) => {
    render(snapshot);
    assert.equal(recover.disabled, true, label);
    const requestCount = requests.length;
    const confirmationCount = confirmations.length;
    await recover.handlers.click();
    assert.equal(requests.length, requestCount, `${label}: no request`);
    assert.equal(confirmations.length, confirmationCount, `${label}: no confirmation`);
    blockedChecks += 1;
  };

  requests.length = 0;
  confirmations.length = 0;
  context.state.snapshotOnline = false;
  await blocked(pending(), 'No live state yet');
  context.state.snapshotOnline = true;
  const validRecovery = pending().current_recovery;
  for (const payload of [undefined, null, {}, [], true, 'unknown',
    { ...validRecovery, required: undefined }, { ...validRecovery, required: 'true' },
    { ...validRecovery, required: false }, { ...validRecovery, available: undefined },
    { ...validRecovery, available: 'true' }, { ...validRecovery, available: false },
    { ...validRecovery, running: undefined }, { ...validRecovery, running: 'false' },
    { ...validRecovery, running: true }, { ...validRecovery, reason: undefined },
    { ...validRecovery, reason: {} }, { ...validRecovery, arm: null },
    { ...validRecovery, arm: {} }, { ...validRecovery, arm: ' ' }]) {
    await blocked({ ...pending(), current_recovery: payload }, `Recovery ${JSON.stringify(payload)}`);
  }
  for (const runState of [undefined, null, '', 'unknown', 'running']) {
    await blocked({ ...pending(), state: runState }, `State ${runState}`);
  }
  for (const connection of [undefined, {},
    { description_ok: true, telemetry_ok: false },
    { description_ok: false, telemetry_ok: true },
    { description_ok: true, telemetry_ok: 'true' }]) {
    await blocked({ ...pending(), connection }, `Connection ${JSON.stringify(connection)}`);
  }
  await blocked({ ...pending(), jogging: true }, 'Jogging');
  await blocked({ ...pending(), planning: true }, 'Planning');
  context.state.planPending = true;
  await blocked(pending(), 'Plan request pending');
  context.state.planPending = false;
  await blocked({ ...pending(), state: 'running', current_recovery: {
    required: false, available: false, running: false, reason: '',
  } }, 'Manual drag without pending recovery');
  assert.equal(recover.classList.contains('hidden'), true);
  context.state.snapshot = null;
  context.renderPositionRecovery(null);
  assert.equal(recover.disabled, true);
  await recover.handlers.click();
  assert.equal(requests.length, 0);

  for (const language of ['en', 'zh']) {
    translations.setLang(language);
    translations.applyStatic({ querySelectorAll: () => [recover] });
    assert.equal(recover.textContent, translations.t('gravtest.recover'));
    assert.notEqual(recover.textContent, 'gravtest.recover');
    assert.equal(recover.title, translations.t('gravtest.recover_hint'));
    for (const arm of ['left', 'right', 'station_3']) {
      const snapshot = pending(arm);
      render(snapshot);
      assert.equal(recover.disabled, false);
      assert.equal(recover.classList.contains('hidden'), false);
      assert.ok(recover.title.includes(translations.t('gravtest.selected_arm', { arm })));
      assert.ok(recover.title.includes(snapshot.current_recovery.reason));
      locked();
      const requestCount = requests.length;
      confirmed = false;
      await recover.handlers.click();
      assert.equal(requests.length, requestCount, 'Cancel sends no request');
      assert.equal(confirmations.at(-1), translations.t('gravtest.selected_arm', { arm })
        + '\n\n' + translations.t('gravtest.recover_confirm'));
      confirmed = true;
      const before = JSON.stringify(snapshot);
      await recover.handlers.click();
      assert.deepEqual(requests.at(-1), { route: '/api/recover-position', body: {
        acknowledgement: 'I_AM_HOLDING_ARM_AND_ESTOP_READY',
      } });
      assert.equal(requests.length, requestCount + 1);
      assert.equal(context.state.snapshot, snapshot);
      assert.equal(JSON.stringify(snapshot), before, 'POST acceptance must not mutate recovery state');
      assert.equal(events.at(-1).message, translations.t('gravtest.recover_requested'));
      assert.equal(recover.disabled, true, 'Wait for a newer snapshot after POST acceptance');
      await recover.handlers.click();
      assert.equal(requests.length, requestCount + 1);
      locked();
    }
  }

  translations.setLang('en');
  const unnamed = pending();
  delete unnamed.current_recovery.arm;
  delete unnamed.gravity_test.arm;
  render(unnamed);
  assert.equal(recover.disabled, false, 'Arm label is optional, not a right-only gate');
  confirmed = false;
  await recover.handlers.click();
  assert.equal(confirmations.at(-1), translations.t('gravtest.recover_confirm'));
  confirmed = true;
  confirmEffect = () => { context.state.snapshotOnline = false; };
  const beforeRecheck = requests.length;
  await recover.handlers.click();
  assert.equal(requests.length, beforeRecheck, 'Recheck connectivity after confirmation');
  confirmEffect = () => {};
  context.state.snapshotOnline = true;

  let finishRecovery;
  const deferred = new Promise((resolve) => { finishRecovery = resolve; });
  postReply = async (route) => route === '/api/recover-position' ? deferred : { ok: true };
  render(pending());
  const beforeBusy = requests.length;
  const inFlight = recover.handlers.click();
  assert.equal(requests.length, beforeBusy + 1);
  assert.equal(recover.disabled, true);
  render(pending());
  await recover.handlers.click();
  assert.equal(requests.length, beforeBusy + 1, 'Polling cannot clear the in-flight lock');
  for (const activity of ['gravity_hold_test', 'gravity_drag_test']) {
    const snapshot = pending();
    snapshot.state = 'running';
    snapshot.activity = activity;
    snapshot.current_recovery.running = true;
    snapshot.current_recovery.available = false;
    render(snapshot);
    assert.equal(recover.disabled, true);
    assert.equal(element('btn-gravtest-stop').disabled, false);
    assert.equal(element('btn-stop').disabled, false);
    locked();
    await element('btn-gravtest-stop').handlers.click();
    assert.deepEqual(requests.at(-1), { route: '/api/stop', body: {} });
  }
  finishRecovery({ ok: true });
  await inFlight;
  assert.equal(context.state.snapshot.current_recovery.required, true);
  assert.equal(recover.disabled, true);
  locked();
  postReply = async () => ({ ok: true });
  const verified = pending();
  verified.state = 'idle';
  verified.current_recovery = { required: false, available: false, running: false, reason: '' };
  render(verified);
  assert.equal(recover.disabled, true);
  assert.equal(recover.classList.contains('hidden'), true);
  for (const id of ['btn-home', 'btn-hardware', 'btn-gravtest-plan', 'btn-gravtest-hold',
    'btn-gravtest-drag']) assert.equal(element(id).disabled, false, `${id}: verified snapshot`);

  for (const failure of ['refused', 'network', 'bad_json']) {
    render(pending());
    events.length = 0;
    toasts.length = 0;
    badReply = failure === 'bad_json';
    postReply = async () => {
      if (failure === 'network') throw new Error('Lost response');
      return { ok: false, message: 'Recovery refused by fixture' };
    };
    await recover.handlers.click();
    assert.deepEqual(toasts.at(-1), { kind: 'err', message: failure === 'network'
      ? translations.t('gravtest.recover_unconfirmed')
      : failure === 'bad_json' ? 'bad reply' : 'Recovery refused by fixture' });
    assert.equal(events.length, 0, 'Failed POST must not report acceptance');
    assert.equal(recover.disabled, true);
    assert.equal(context.state.snapshot.current_recovery.required, true);
    locked();
  }
  badReply = false;
  postReply = async () => ({ ok: true });
  const beforePolling = requests.length;
  render(pending());
  assert.equal(recover.disabled, false, 'A fresh available snapshot permits an explicit retry');
  for (const reply of [
    async () => { throw new Error('Disconnected'); },
    async () => ({ ok: false, status: 503 }),
    async () => ({ ok: true, json: async () => { throw new Error('Invalid state JSON'); } }),
  ]) {
    snapshotReply = reply;
    await context.poll();
    assert.equal(context.state.snapshotOnline, false);
    assert.equal(recover.disabled, true);
    await recover.handlers.click();
    assert.equal(requests.length, beforePolling);
  }
  snapshotReply = async () => ({ ok: true, json: async () => pending('right') });
  await context.poll();
  assert.equal(recover.disabled, false);
  assert.equal(requests.length, beforePolling, 'Reconnection and rendering never auto-recover');
  requests.length = 0;
  console.log(`PASS: recovery handler, ${blockedChecks} fail-closed states, EN/ZH labels and confirms, `
    + 'both arms, cancel, duplicate lock, Stop while recovering, response errors, snapshot-only release, offline polling');
}

async function main() {
  vm.runInContext('initializeSystemControls(systemConfig)', context);
  await element('btn-gravtest-drag').handlers.click();
  assert.deepEqual(requests.pop(), {
    route: '/api/gravity-test',
    body: {
      mode: 'gravity_drag_test',
      options: {
        maximum_speed_deg_s: 92,
        acknowledgement: 'I_AM_HOLDING_ARM_AND_ESTOP_READY',
      },
    },
  });
  element('gravtest-speed-stop').value = '47';
  vm.runInContext('initializeSystemControls(systemConfig)', context);
  await element('btn-gravtest-drag').handlers.click();
  assert.equal(requests.pop().body.options.maximum_speed_deg_s, 47);
  assert.equal(elements.has('gravtest-drag-seconds'), false);
  confirmed = false;
  await element('btn-gravtest-drag').handlers.click();
  assert.equal(requests.length, 0);
  confirmed = true;
  element('gravtest-poses').value = '4';
  element('gravtest-hold-seconds').value = '2';
  element('gravtest-transit-speed').value = '5';
  context.state.snapshot = {
    state: 'idle', have_model: true, gravity_test: { available: true },
    hold_plan: { available: true, id: 'hold-plan-4', poses: 4 },
  };
  await element('btn-gravtest-hold').handlers.click();
  assert.deepEqual(requests.pop().body.options, {
    poses: 4, seconds: 2, transit_speed_deg_s: 5, plan_id: 'hold-plan-4',
    acknowledgement: 'I_AM_HOLDING_ARM_AND_ESTOP_READY',
  });

  for (const arm of ['left', 'station_3']) {
    context.snapshot = {
      state: 'idle', have_model: true,
      gravity_test: { available: true, arm, source: `/calibrations/${arm}` },
    };
    vm.runInContext('renderGravity(snapshot)', context);
    assert.equal(element('gravtest-capability').textContent, `Arm: ${arm}`);
    assert.equal(element('btn-gravtest-drag').disabled, false);
    context.snapshot.gravity_test = {
      available: false, arm, reason_code: 'invalid_source',
      reason: 'source must describe the selected arm',
    };
    vm.runInContext('renderGravity(snapshot)', context);
    assert.equal(element('btn-gravtest-drag').disabled, true);
    assert.equal(element('btn-gravtest-plan').disabled, true);
    assert.equal(element('btn-gravtest-hold').disabled, true);
    assert.equal(element('gravtest-capability').textContent,
      `Arm: ${arm}: gravtest.invalid_source: source must describe the selected arm`);
  }

  await verifyRecovery();

  for (const state of ['idle', 'running', 'paused']) {
    for (const activity of ['', 'gravity', 'gravity_hold_test', 'gravity_drag_test']) {
      context.snapshot = { state, activity, have_model: false, planning: true };
      vm.runInContext('renderGravity(snapshot)', context);
      assert.equal(element('btn-gravtest-stop').disabled,
        !(state === 'running'
          && ['gravity_hold_test', 'gravity_drag_test'].includes(activity)),
        `${state}/${activity}`);
    }
  }
  await element('btn-gravtest-stop').handlers.click();
  assert.deepEqual(requests.pop(), { route: '/api/stop', body: {} });
  assert.equal(requests.length, 0);
  console.log('PASS: drag/hold payloads, selected-arm labels and refusal gates, cancel, 12 Stop states, and Stop route');
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
