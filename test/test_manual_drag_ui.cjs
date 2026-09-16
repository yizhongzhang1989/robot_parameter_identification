const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { systemConfigFixture } = require('./system_config_fixture.cjs');

const source = fs.readFileSync(path.join(__dirname,
  '../robot_parameter_identification/dashboard/static/dashboard.js'), 'utf8');
const configSource = fs.readFileSync(path.join(__dirname,
  '../robot_parameter_identification/dashboard/static/system_config.js'), 'utf8');
const systemConfig = systemConfigFixture({
  dashboard: { drag_test: { maximum_speed_deg_s: 92 } },
});
const elements = new Map();
const requests = [];
const element = (id) => {
  if (!elements.has(id)) {
    elements.set(id, {
      id, value: '', disabled: true, handlers: {},
      type: typeof systemConfig.controls[id]?.value === 'boolean' ? 'checkbox' : 'number',
      classList: { toggle() {} },
      addEventListener(event, handler) { this.handlers[event] = handler; },
    });
  }
  return elements.get(id);
};
let confirmed = true;
const context = vm.createContext({
  $: element,
  systemConfig,
  document: { querySelectorAll: () => Object.keys(systemConfig.controls).map(element) },
  state: {},
  window: { confirm: () => confirmed },
  t: (key) => key,
  post: async (route, body) => {
    requests.push(JSON.parse(JSON.stringify({ route, body })));
    return { ok: true };
  },
  publishLocalEvent() {},
  gravityArmed: () => false,
  gravityStatus: () => ({ level: 'info', message: '', passive: true }),
  gravityOptions: () => ({
    static_poses: 5, gravity_validation_poses: 2,
    gravity_probe_speeds_deg_s: [1, 3],
  }),
  renderGravityTestResult() {},
});

vm.runInContext(`
  ${configSource.replace(/^export /gm, '')}
  const GRAVITY_HOLD_TEST = 'gravity_hold_test';
  const GRAVITY_DRAG_TEST = 'gravity_drag_test';
  const GRAVITY_TEST_ACKNOWLEDGEMENT = 'I_AM_HOLDING_ARM_AND_ESTOP_READY';
  ${source.slice(source.indexOf('function holdPoseCount('),
    source.indexOf("$('btn-rescreen').addEventListener"))}
  ${source.slice(source.indexOf('function renderGravity(snapshot)'),
    source.indexOf('function renderGravityTestResult(snapshot)'))}
`, context);

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
  console.log('PASS: config-initialized drag/hold payloads, later edits, cancel, 12 Stop states, and Stop route');
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});