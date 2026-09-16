const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { systemConfigFixture } = require('./system_config_fixture.cjs');

const staticPath = path.join(__dirname,
  '../robot_parameter_identification/dashboard/static');
const source = fs.readFileSync(path.join(staticPath, 'dashboard.js'), 'utf8');
const html = fs.readFileSync(path.join(staticPath, 'index.html'), 'utf8');
const translations = fs.readFileSync(path.join(staticPath, 'i18n.js'), 'utf8');
const configSource = fs.readFileSync(path.join(staticPath, 'system_config.js'), 'utf8');
const elements = new Map();
const requests = [];
const speedIds = ['home-speed', 'jog-speed', 'grav-transit-speed', 'gravtest-transit-speed'];
const systemConfig = systemConfigFixture({
  dashboard: {
    home: { transit_speed_deg_s: 7.5 }, jog: { transit_speed_deg_s: 8.5 },
    hold_test: { poses: 7, seconds: 2.5, transit_speed_deg_s: 4.1 },
    drag_test: { maximum_speed_deg_s: 83 },
    gravity: { gravity_probe_speeds_deg_s: [1.2, 3.4] },
    optimal: { reuse_friction: true },
  },
  campaign: { static_poses: 28, gravity_validation_poses: 9, gravity_probe_deg: 6,
    transit_speed_deg_s: 6.7 },
});
speedIds.forEach((id, index) => { systemConfig.controls[id].max = [48, 36, 50, 42][index]; });
const element = (id) => {
  assert.ok(elements.has(id), `Missing HTML element: ${id}`);
  return elements.get(id);
};
for (const match of html.matchAll(/<[^>]+\bid="([^"]+)"[^>]*>/g)) {
  const attributes = Object.fromEntries([...match[0].matchAll(/([\w-]+)="([^"]*)"/g)]
    .map((attribute) => [attribute[1], attribute[2]]));
  elements.set(match[1], {
    ...attributes, value: attributes.value || '', handlers: {},
    checked: /\schecked(?:\s|\/?>)/.test(match[0]),
    disabled: /\sdisabled(?:\s|\/?>)/.test(match[0]),
    classList: { toggle() {} },
    addEventListener(event, handler) { this.handlers[event] = handler; },
  });
}
const ready = () => ({
  state: 'idle', have_model: true, gravity_armed: true,
  gravity_test: { available: true },
  hold_plan: { available: true, id: 'hold-config', poses: systemConfig.values.dashboard.hold_test.poses },
});
let reply = { ok: true };
const context = vm.createContext({
  $: element, systemConfig, state: { snapshot: ready() },
  document: {
    activeElement: null,
    querySelectorAll: () => [...elements.values()].filter((input) =>
      input.type === 'number' || input.type === 'checkbox'),
  },
  window: { confirm: () => true }, t: (key) => key,
  post: async (route, body) => {
    requests.push(JSON.parse(JSON.stringify({ route, body })));
    return reply;
  },
  publishLocalEvent() {}, renderGravityTestResult() {},
  refreshPlannedPoses: async () => {},
});
const slice = (start, end) => {
  const first = source.indexOf(start);
  const last = source.indexOf(end, first);
  assert.ok(first >= 0 && last > first, `Missing source boundary: ${start}`);
  return source.slice(first, last);
};
vm.runInContext(`
  ${configSource.replace(/^export /gm, '')}
  ${slice("$('btn-home').addEventListener", 'function renderGravityTestResult(snapshot)')}
  ${slice('function seedGravityFields(snapshot)', 'function renderRun(snapshot)')}
  ${slice("$('btn-jog').addEventListener", "$('btn-jog-zero').addEventListener")}
  ${slice('async function requestPlan(', 'function poseReviewLocked()')}
`, context);
const render = (overrides = {}) => {
  context.state.snapshot = { ...ready(), ...overrides };
  vm.runInContext('renderGravity(state.snapshot)', context);
};
const click = async (id) => {
  if (!element(id).disabled) await element(id).handlers.click();
};
const invoke = (id) => element(id).handlers.click();
const take = (route) => {
  assert.equal(requests.length, 1);
  const request = requests.pop();
  assert.equal(request.route, route);
  return request.body;
};

async function main() {
  vm.runInContext('initializeSystemControls(systemConfig)', context);
  for (const id of speedIds) {
    const input = element(id);
    const control = systemConfig.controls[id];
    assert.deepEqual([input.type, input.min, input.max, input.step, input.value],
      ['number', ...['min', 'max', 'step', 'value'].map((key) => String(control[key]))], id);
  }
  assert.equal(element('optimal-reuse-friction').checked, true);
  for (const key of ['run.home_speed', 'jog.speed', 'grav.transit_speed',
    'gravtest.transit_speed']) {
    assert.ok(html.includes(`data-i18n="${key}"`));
    assert.match(translations, new RegExp(`'${key.replaceAll('.', '\\.')}': \\{`
      + ` en: '[^']*deg/s[^']*', zh: '[^']*[\\u4e00-\\u9fff][^']*deg/s[^']*'`));
  }
  for (const custom of [false, true]) {
    const speeds = custom ? [12.3, 8.4, 6.7, 4.2]
      : speedIds.map((id) => systemConfig.controls[id].value);
    if (custom) speedIds.forEach((id, index) => { element(id).value = String(speeds[index]); });
    vm.runInContext('initializeSystemControls(systemConfig)', context);
    render();
    await invoke('btn-home');
    assert.deepEqual(take('/api/home'), { transit_speed_deg_s: speeds[0] });
    await invoke('btn-jog');
    assert.deepEqual(take('/api/jog'), {
      action: 'start', transit_speed_deg_s: speeds[1],
    });
    const gravity = {
      static_poses: 28, gravity_validation_poses: 9, gravity_probe_deg: 6,
      gravity_probe_speeds_deg_s: [1.2, 3.4], transit_speed_deg_s: speeds[2],
    };
    await click('btn-grav-rehearse');
    assert.deepEqual(take('/api/campaign'), {
      mode: 'gravity_rehearsal', options: gravity,
    });
    render();
    await click('btn-grav-run');
    assert.deepEqual(take('/api/campaign'), { mode: 'gravity', options: gravity });
    await click('btn-gravtest-hold');
    assert.deepEqual(take('/api/gravity-test'), {
      mode: 'gravity_hold_test', options: {
        poses: 7, seconds: 2.5, transit_speed_deg_s: speeds[3], plan_id: 'hold-config',
        acknowledgement: 'I_AM_HOLDING_ARM_AND_ESTOP_READY',
      },
    });
    if (custom) element('gravtest-speed-stop').value = '45';
    vm.runInContext('initializeSystemControls(systemConfig)', context);
    await click('btn-gravtest-drag');
    assert.deepEqual(take('/api/gravity-test'), {
      mode: 'gravity_drag_test', options: {
        maximum_speed_deg_s: custom ? 45 : 83,
        acknowledgement: 'I_AM_HOLDING_ARM_AND_ESTOP_READY',
      },
    });
    await click('btn-grav-plan');
    assert.deepEqual(take('/api/plan'), { mode: 'gravity', options: gravity });
    render({ jogging: true });
    assert.equal(element('jog-speed').disabled, true);
    await invoke('btn-jog');
    assert.deepEqual(take('/api/jog'), { action: 'stop' });
  }

  render();
  assert.equal(element('btn-grav-run').disabled, false);
  element('grav-transit-speed').value = '9.5';
  element('grav-transit-speed').handlers.input();
  assert.equal(element('btn-grav-run').disabled, true);
  assert.equal(element('btn-gravtest-hold').disabled, false);
  await click('btn-grav-rehearse');
  assert.equal(take('/api/campaign').options.transit_speed_deg_s, 9.5);
  render();
  assert.equal(element('btn-grav-run').disabled, false);

  for (const overrides of [{ state: 'running' }, { state: 'paused' },
    { jogging: true }, { planning: true }]) {
    render(overrides);
    for (const id of speedIds) assert.equal(element(id).disabled, true, id);
    for (const id of ['btn-grav-plan', 'btn-grav-rehearse', 'btn-grav-run',
      'btn-gravtest-hold', 'btn-gravtest-drag']) await click(id);
    assert.equal(requests.length, 0);
  }
  render();
  let finish;
  reply = new Promise((resolve) => { finish = resolve; });
  const planning = click('btn-gravtest-plan');
  assert.deepEqual(take('/api/plan'), {
    mode: 'gravity_hold_test', options: { poses: 7 },
  });
  for (const id of speedIds) assert.equal(element(id).disabled, true, id);
  finish({ ok: true, hold_plan: ready().hold_plan });
  await planning;
  for (const id of speedIds) assert.equal(element(id).disabled, false, id);

  for (const ceiling of [12.5, 0.1, 0.05, 90, undefined, null, 'invalid']) {
    for (const value of ['42.3', '', '0', '80']) {
      for (const id of speedIds) element(id).value = value;
      render({ motion_speed_max_deg_s: ceiling });
      for (const id of speedIds) {
        const uiMax = systemConfig.controls[id].max;
        const expected = String(Number.isFinite(ceiling) ? Math.min(uiMax, ceiling) : uiMax);
        assert.equal(element(id).max, expected, id);
        assert.equal(element(id).value, value, `No silent clamping: ${id}`);
      }
    }
  }
  context.state.snapshot.have_model = false;
  context.state.snapshot.gravity_defaults = {
    transit_speed_deg_s: 7.8, static_poses: 31, gravity_validation_poses: 10,
    gravity_probe_deg: 7, gravity_probe_speeds_deg_s: [0.7, 2.1],
  };
  vm.runInContext('seedGravityFields(state.snapshot)', context);
  assert.equal(element('grav-transit-speed').value, '80');
  assert.notEqual(context.state.gravitySeeded, true);
  context.state.snapshot.have_model = true;
  vm.runInContext('seedGravityFields(state.snapshot)', context);
  assert.equal(Number(element('grav-transit-speed').value), 7.8);
  assert.deepEqual(['grav-poses', 'grav-check', 'grav-arc', 'grav-slow', 'grav-fast']
    .map((id) => Number(element(id).value)), [31, 10, 7, 0.7, 2.1]);
  element('grav-transit-speed').value = '11.2';
  vm.runInContext('initializeSystemControls(systemConfig); seedGravityFields(state.snapshot)', context);
  assert.equal(element('grav-transit-speed').value, '11.2');
  context.state.gravitySeeded = false;
  context.document.activeElement = element('grav-transit-speed');
  vm.runInContext('seedGravityFields(state.snapshot)', context);
  assert.equal(element('grav-transit-speed').value, '11.2');
  assert.equal(element('gravtest-speed-stop').value, '45');
  assert.equal(requests.length, 0);
  console.log('PASS: custom config defaults, bounds, bilingual labels, actual handler payloads, '
    + 'arming, busy/session/planning locks, config/server ceilings without clamping, '
    + 'and saved gravity overrides after model readiness');
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});