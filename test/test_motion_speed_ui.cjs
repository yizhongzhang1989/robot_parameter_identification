const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const staticPath = path.join(__dirname,
  '../robot_parameter_identification/dashboard/static');
const source = fs.readFileSync(path.join(staticPath, 'dashboard.js'), 'utf8');
const html = fs.readFileSync(path.join(staticPath, 'index.html'), 'utf8');
const translations = fs.readFileSync(path.join(staticPath, 'i18n.js'), 'utf8');
const elements = new Map();
const requests = [];
const speedDefaults = {
  'home-speed': '10', 'jog-speed': '10',
  'grav-transit-speed': '10', 'gravtest-transit-speed': '5',
};
const element = (id) => {
  assert.ok(elements.has(id), `Missing HTML element: ${id}`);
  return elements.get(id);
};
for (const match of html.matchAll(/<[^>]+\bid="([^"]+)"[^>]*>/g)) {
  const attributes = Object.fromEntries([...match[0].matchAll(/([\w-]+)="([^"]*)"/g)]
    .map((attribute) => [attribute[1], attribute[2]]));
  elements.set(match[1], {
    ...attributes, value: attributes.value || '', handlers: {},
    disabled: /\sdisabled(?:\s|\/?>)/.test(match[0]),
    classList: { toggle() {} },
    addEventListener(event, handler) { this.handlers[event] = handler; },
  });
}
const ready = () => ({
  state: 'idle', have_model: true, gravity_armed: true,
  gravity_test: { available: true },
  hold_plan: { available: true, id: 'hold-5', poses: 5 },
});
let reply = { ok: true };
const context = vm.createContext({
  $: element, state: { snapshot: ready() }, document: { activeElement: null },
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
  for (const [id, value] of Object.entries(speedDefaults)) {
    const input = element(id);
    assert.deepEqual([input.type, input.min, input.max, input.step, input.value],
      ['number', '0.1', '60', '0.1', value], id);
  }
  for (const key of ['run.home_speed', 'jog.speed', 'grav.transit_speed',
    'gravtest.transit_speed']) {
    assert.ok(html.includes(`data-i18n="${key}"`));
    assert.match(translations, new RegExp(`'${key.replaceAll('.', '\\.')}': \\{`
      + ` en: '[^']*deg/s[^']*', zh: '[^']*[\\u4e00-\\u9fff][^']*deg/s[^']*'`));
  }
  for (const custom of [false, true]) {
    const speeds = custom ? [12.3, 8.4, 6.7, 4.2] : [10, 10, 10, 5];
    Object.keys(speedDefaults).forEach((id, index) => {
      element(id).value = String(speeds[index]);
    });
    render();
    await invoke('btn-home');
    assert.deepEqual(take('/api/home'), { transit_speed_deg_s: speeds[0] });
    await invoke('btn-jog');
    assert.deepEqual(take('/api/jog'), {
      action: 'start', transit_speed_deg_s: speeds[1],
    });
    const gravity = {
      static_poses: 24, gravity_validation_poses: 8, gravity_probe_deg: 5,
      gravity_probe_speeds_deg_s: [1, 3], transit_speed_deg_s: speeds[2],
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
        poses: 5, seconds: 3, transit_speed_deg_s: speeds[3], plan_id: 'hold-5',
        acknowledgement: 'I_AM_HOLDING_ARM_AND_ESTOP_READY',
      },
    });
    element('gravtest-speed-stop').value = custom ? '45' : '120';
    await click('btn-gravtest-drag');
    assert.deepEqual(take('/api/gravity-test'), {
      mode: 'gravity_drag_test', options: {
        maximum_speed_deg_s: custom ? 45 : 120,
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
    for (const id of Object.keys(speedDefaults)) assert.equal(element(id).disabled, true, id);
    for (const id of ['btn-grav-plan', 'btn-grav-rehearse', 'btn-grav-run',
      'btn-gravtest-hold', 'btn-gravtest-drag']) await click(id);
    assert.equal(requests.length, 0);
  }
  render();
  let finish;
  reply = new Promise((resolve) => { finish = resolve; });
  const planning = click('btn-gravtest-plan');
  assert.deepEqual(take('/api/plan'), {
    mode: 'gravity_hold_test', options: { poses: 5 },
  });
  for (const id of Object.keys(speedDefaults)) assert.equal(element(id).disabled, true, id);
  finish({ ok: true, hold_plan: ready().hold_plan });
  await planning;
  for (const id of Object.keys(speedDefaults)) assert.equal(element(id).disabled, false, id);

  for (const [ceiling, expected] of [[12.5, '12.5'], [0.1, '0.1'], [90, '60'],
    [undefined, '60'], [null, '60'], ['invalid', '60']]) {
    for (const value of ['42.3', '', '0', '80']) {
      for (const id of Object.keys(speedDefaults)) element(id).value = value;
      render({ motion_speed_max_deg_s: ceiling });
      for (const id of Object.keys(speedDefaults)) {
        assert.equal(element(id).max, expected, id);
        assert.equal(element(id).value, value, `No silent clamping: ${id}`);
      }
    }
  }
  context.state.snapshot.gravity_defaults = { transit_speed_deg_s: 7.8 };
  vm.runInContext('seedGravityFields(state.snapshot)', context);
  assert.equal(Number(element('grav-transit-speed').value), 7.8);
  element('grav-transit-speed').value = '11.2';
  vm.runInContext('seedGravityFields(state.snapshot)', context);
  assert.equal(element('grav-transit-speed').value, '11.2');
  context.state.gravitySeeded = false;
  context.document.activeElement = element('grav-transit-speed');
  vm.runInContext('seedGravityFields(state.snapshot)', context);
  assert.equal(element('grav-transit-speed').value, '11.2');
  assert.equal(element('gravtest-speed-stop').value, '45');
  assert.equal(requests.length, 0);
  console.log('PASS: speed defaults, bounds, bilingual labels, actual handler payloads, '
    + 'arming, busy/session/planning locks, server ceilings without clamping, and seeding');
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});