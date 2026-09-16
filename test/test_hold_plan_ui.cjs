const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const staticPath = path.join(__dirname,
  '../robot_parameter_identification/dashboard/static');
const source = fs.readFileSync(path.join(staticPath, 'dashboard.js'), 'utf8');
const elements = new Map();
const requests = [];
const events = [];
const highlights = [];
let confirmations = 0;
let previewFetches = 0;
let flying = false;
let flightStarts = 0;
let flightStops = 0;
let reply = { ok: true, hold_plan: { available: true, id: 'hold-4', poses: 4 } };
let preview = { groups: [{ phase: 'hold_set', poses: [
  { index: 1, pose_deg: [10, 20] }, { index: 2, pose_deg: [30, 40] },
] }] };
const element = (id) => {
  if (!elements.has(id)) elements.set(id, {
    value: '', disabled: true, handlers: {}, style: {},
    classList: { toggle() {} },
    addEventListener(event, handler) { this.handlers[event] = handler; },
  });
  return elements.get(id);
};
const ready = () => ({
  state: 'idle', have_model: true, preview_token: 7,
  gravity_test: { available: true },
  hold_plan: { available: true, id: 'hold-4', poses: 4 },
});
const context = vm.createContext({
  $: element,
  state: { snapshot: ready(), plannedPoses: [], previewToken: 7, poseAt: 0 },
  window: {
    confirm: () => { confirmations += 1; return true; },
    __viewer: {
      highlight: (...args) => highlights.push(args),
      flying: () => flying,
      fly: () => { flying = true; flightStarts += 1; },
      stopFlying: () => { flying = false; flightStops += 1; },
    },
  },
  t: (key) => key,
  post: async (route, body) => {
    requests.push(JSON.parse(JSON.stringify({ route, body })));
    return reply;
  },
  fetch: async (route, options) => {
    assert.equal(route, '/api/preview');
    assert.equal(options.cache, 'no-store');
    previewFetches += 1;
    return { json: async () => preview };
  },
  publishLocalEvent: (...args) => events.push(args),
  gravityArmed: () => false,
  gravityStatus: () => ({ level: 'info', message: 'grav.locked', passive: false }),
  gravityOptions: () => ({
    static_poses: 5, gravity_validation_poses: 2,
    gravity_probe_speeds_deg_s: [1, 3],
  }),
  renderGravityTestResult() {},
  activityLabel: (name) => name,
  progressText: () => '',
});
const slice = (start, end) => source.slice(source.indexOf(start), source.indexOf(end));
vm.runInContext(`
  const GRAVITY_HOLD_TEST = 'gravity_hold_test';
  const GRAVITY_DRAG_TEST = 'gravity_drag_test';
  const GRAVITY_TEST_ACKNOWLEDGEMENT = 'I_AM_HOLDING_ARM_AND_ESTOP_READY';
  ${slice('function holdPoseCount(', "$('btn-rescreen').addEventListener")}
  ${slice('function renderGravity(snapshot)', 'function renderGravityTestResult(snapshot)')}
  ${slice('async function requestPlan(', '/* ---------------- collapsible panel groups')}
  ${slice('function phaseLabel(', 'function activityLabel(')}
  ${slice('function renderSceneActivity(', '/** The sweep has its own progress shape')}
`, context);
const render = () => vm.runInContext('renderGravity(state.snapshot)', context);
const click = (id) => element(id).handlers.click();
const editCount = (value) => {
  element('gravtest-poses').value = value;
  element('gravtest-poses').handlers.input();
};

async function main() {
  element('gravtest-poses').value = '4';
  element('gravtest-hold-seconds').value = '2';
  element('gravtest-transit-speed').value = '5';
  for (const plan of [undefined, { available: false },
    { available: true, id: '', poses: 4 },
    { available: true, id: 123, poses: 4 },
    { available: true, id: 'other-count', poses: 5 }]) {
    context.state.snapshot = { ...ready(), hold_plan: plan };
    render();
    assert.equal(element('btn-gravtest-hold').disabled, true);
    assert.equal(element('btn-gravtest-plan').disabled, false);
    assert.equal(events.at(-1)[0], 'gravtest.plan_missing');
    await click('btn-gravtest-hold');
  }
  assert.equal(requests.length, 0);
  assert.equal(confirmations, 0);
  assert.ok(events.some(([message, level]) =>
    message === 'gravtest.plan_missing' && level === 'warning'));

  for (const phase of ['complete', 'failed', 'stopped']) {
    context.state.holdPlanMissing = false;
    context.state.snapshot = {
      ...ready(), hold_plan: { available: false },
      progress: { mode: 'gravity_hold_test', phase, error: 'original hold result' },
    };
    const previousEvents = events.length;
    render();
    assert.equal(events.length, previousEvents, phase);
    assert.equal(element('btn-gravtest-hold').disabled, true);
    assert.equal(element('btn-gravtest-plan').disabled, false);
  }

  context.state.snapshot = ready();
  render();
  assert.equal(element('btn-gravtest-hold').disabled, false);
  for (const count of ['5', '', '4.5', '0', '21', 'abc']) {
    editCount(count);
    assert.equal(element('btn-gravtest-hold').disabled, true, count);
    await click('btn-gravtest-hold');
  }
  editCount('4');
  element('gravtest-hold-seconds').value = '7';
  render();
  assert.equal(element('btn-gravtest-hold').disabled, false);
  await click('btn-gravtest-hold');
  assert.deepEqual(requests.pop(), {
    route: '/api/gravity-test', body: { mode: 'gravity_hold_test', options: {
      poses: 4, seconds: 7, transit_speed_deg_s: 5, plan_id: 'hold-4',
      acknowledgement: 'I_AM_HOLDING_ARM_AND_ESTOP_READY',
    } },
  });
  assert.equal(confirmations, 1);

  for (const overrides of [{ state: 'running' }, { state: 'paused' },
    { jogging: true }, { planning: true }, { have_model: false },
    { gravity_test: { available: false, reason_code: 'unsupported_arm' } }]) {
    context.state.snapshot = { ...ready(), ...overrides };
    render();
    assert.equal(element('btn-gravtest-hold').disabled, true);
    assert.equal(element('btn-gravtest-plan').disabled, true);
    await click('btn-gravtest-plan');
    await click('btn-gravtest-hold');
  }
  assert.equal(requests.length, 0);
  assert.equal(confirmations, 1);

  context.state.snapshot = ready();
  let resolvePlan;
  reply = new Promise((resolve) => { resolvePlan = resolve; });
  const planning = click('btn-gravtest-plan');
  assert.equal(element('btn-gravtest-hold').disabled, true);
  assert.equal(element('btn-gravtest-plan').disabled, true);
  assert.equal(element('btn-grav-plan').disabled, true);
  await click('btn-gravtest-plan');
  await click('btn-gravtest-hold');
  assert.equal(requests.length, 1);
  assert.deepEqual(requests.pop(), {
    route: '/api/plan', body: { mode: 'gravity_hold_test', options: { poses: 4 } },
  });
  assert.equal(confirmations, 1);
  assert.equal(previewFetches, 0);
  editCount('5');
  resolvePlan({ ok: true, hold_plan: ready().hold_plan });
  await planning;
  assert.equal(previewFetches, 1);
  assert.equal(element('btn-gravtest-hold').disabled, true);
  assert.equal(element('btn-gravtest-plan').disabled, false);
  assert.equal(context.state.plannedPoses.length, 2);
  assert.deepEqual(highlights.pop(), ['hold_set', 1]);
  editCount('4');
  assert.equal(element('btn-gravtest-hold').disabled, false);

  const reviewControls = ['pose-prev', 'pose-next', 'pose-worst', 'pose-fly'];
  const renderScene = (activity) => {
    context.activity = activity;
    vm.runInContext('renderSceneActivity(activity)', context);
  };
  const assertControls = (disabled) => {
    for (const id of reviewControls) assert.equal(element(id).disabled, disabled, id);
  };
  for (const mode of ['gravity_hold_test', 'gravity_drag_test', 'gravity', 'hardware']) {
    renderScene({});
    assertControls(false);
    click('pose-fly');
    assert.equal(flying, true);
    assert.equal(context.state.flyAuto, false);
    const starts = flightStarts;
    const stops = flightStops;
    const activity = { id: `${mode}-live`, mode,
      focus: { phase: 'hold_set', index: 2 } };
    renderScene(activity);
    assert.equal(flightStops, stops + 1);
    assert.equal(flying, false);
    assert.equal(flightStarts, starts);
    assertControls(true);
    assert.equal(context.state.poseAt, 1);
    assert.deepEqual(highlights.pop(), ['hold_set', 2]);
    assert.equal(element('pose-where').textContent, '2 / 2');
    const highlightCount = highlights.length;
    for (const id of reviewControls) click(id);
    assert.equal(highlights.length, highlightCount);
    assert.equal(context.state.poseAt, 1);
    assert.equal(flightStarts, starts);

    preview.groups[0].poses.reverse();
    await vm.runInContext('refreshPlannedPoses(state.previewToken + 1)', context);
    assertControls(true);
    assert.equal(context.state.poseAt, 0);
    assert.deepEqual(highlights.pop(), ['hold_set', 2]);
    assert.equal(flightStarts, starts);
    renderScene(activity);
    assert.equal(flightStops, stops + 1);
    renderScene({});
    assertControls(false);
    preview.groups[0].poses.reverse();
    await vm.runInContext('refreshPlannedPoses(state.previewToken + 1)', context);
  }
  const scenePreviewFetches = previewFetches;
  renderScene({ id: 'rehearsal', mode: 'gravity_rehearsal',
    tour: { autoplay: true, available: true } });
  assertControls(false);
  assert.equal(flying, true);
  assert.equal(context.state.flyAuto, true);
  click('pose-fly');
  assert.equal(flying, false);
  renderScene({});
  assertControls(false);
  click('pose-fly');
  renderScene({});
  assert.equal(flying, true);
  click('pose-fly');
  click('pose-next');
  assert.equal(context.state.poseAt, 1);
  click('pose-prev');
  assert.equal(context.state.poseAt, 0);
  click('pose-worst');
  assert.equal(context.state.poseAt, 0);

  reply = { ok: true };
  preview = { groups: [{ phase: 'A_gravity', poses: [{ index: 1, pose_deg: [50] }] }] };
  await click('btn-grav-plan');
  assert.equal(previewFetches, scenePreviewFetches + 1);
  assert.deepEqual(requests.pop(), { route: '/api/plan', body: {
    mode: 'gravity', options: { static_poses: 5, gravity_validation_poses: 2,
      gravity_probe_speeds_deg_s: [1, 3] },
  } });
  assert.equal(context.state.plannedPoses.length, 1);
  assert.deepEqual(highlights.pop(), ['A_gravity', 1]);
  assert.equal(element('btn-gravtest-hold').disabled, true);
  assert.equal(confirmations, 1);

  reply = { ok: false };
  await click('btn-gravtest-plan');
  assert.equal(previewFetches, scenePreviewFetches + 1);
  assert.equal(element('btn-gravtest-hold').disabled, true);
  assert.equal(element('btn-gravtest-plan').disabled, false);
  context.post = async () => { throw new Error('offline'); };
  await click('btn-gravtest-plan');
  assert.equal(element('btn-gravtest-plan').disabled, false);
  assert.ok(events.some(([message, level]) => message.includes('offline') && level === 'error'));

  const html = fs.readFileSync(path.join(staticPath, 'index.html'), 'utf8');
  assert.match(html, /id="btn-gravtest-plan" disabled\s+data-i18n="gravtest.plan"/);
  const translations = fs.readFileSync(path.join(staticPath, 'i18n.js'), 'utf8');
  for (const key of ['gravtest.plan', 'gravtest.plan_missing', 'phase.hold_set']) {
    assert.match(translations, new RegExp(`'${key.replaceAll('.', '\\.')}': \\{\\s*en: [\\s\\S]*?zh:`));
  }
  console.log('PASS: hold plan gates, count edits, duration independence, plan_id payload, request lock, shared previews, hardware focus locks, manual flight cancellation, idle/rehearsal review, failures, and labels');
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});