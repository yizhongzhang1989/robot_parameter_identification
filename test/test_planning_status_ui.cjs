const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { systemConfigFixture } = require('./system_config_fixture.cjs');

const source = fs.readFileSync(path.join(__dirname,
  '../robot_parameter_identification/dashboard/static/dashboard.js'), 'utf8');
const elements = new Map();
const events = [];
const element = (id) => {
  if (!elements.has(id)) elements.set(id, {
    style: {}, classList: { toggle() {} }, textContent: '',
  });
  return elements.get(id);
};
const state = { localEvents: [] };
const context = vm.createContext({
  state, window: {}, $: element, t: (key) => key,
  systemConfig: systemConfigFixture(),
  gravityArmed: () => false,
  gravityOptions: () => ({ static_poses: 5, gravity_validation_poses: 2,
    gravity_probe_speeds_deg_s: [1, 3] }),
  matchingHoldPlan: () => true,
  planBlocked: () => false,
  renderGravityTestResult() {},
  activityLabel: (mode) => mode,
  progressText: (progress) => `${progress.target || progress.mode}:${progress.phase}`,
  publishLocalEvent: (message, level, source) => {
    events.push({ message, level, source, stamp_s: 20 });
    state.localEvents.push(events.at(-1));
  },
  GRAVITY_HOLD_TEST: 'gravity_hold_test',
  GRAVITY_DRAG_TEST: 'gravity_drag_test',
});
const slice = (start, end) => source.slice(source.indexOf(start), source.indexOf(end));
vm.runInContext(`
  ${slice('function gravityStatus(snapshot)', 'function renderGravityTestResult(snapshot)')}
  ${slice('function renderActivity(snapshot)', '/** Apply the mode-neutral activity contract')}
`, context);

function render(snapshot) {
  context.snapshot = { state: 'idle', have_model: true,
    gravity_test: { available: true }, ...snapshot };
  vm.runInContext('renderGravity(snapshot); renderActivity(snapshot)', context);
}

for (const target of ['gravity_hold_test', 'future_planner']) {
  state.gravityStatusKey = '';
  state.localEvents = [];
  events.length = 0;
  render({ planning: true, progress: { mode: 'planning', phase: 'designing', target } });
  assert.equal(element('progress-line').textContent, `${target}:designing`);
  assert.equal(events.length, 0);
  render({ planning: false, progress: { phase: 'idle' }, activity_feed: {
    state: 'idle', progress: { phase: 'idle' },
    events: [{ source: 'planner', message: 'plan ready', stamp_s: 10 }],
  } });
  assert.equal(element('progress-line').textContent, 'plan ready');
  assert.equal(events.length, 0);
}

render({ planning: true, progress: {
  mode: 'planning', phase: 'designing', target: 'gravity',
} });
assert.equal(element('progress-line').textContent, 'grav.planning');
assert.equal(events.at(-1).source, 'gravity');
state.localEvents = [];
events.length = 0;
render({ planning: false, progress: { mode: 'gravity', phase: 'complete' },
  activity_feed: { state: 'idle', progress: { mode: 'gravity', phase: 'complete' },
    events: [{ source: 'planner', message: 'gravity plan ready', stamp_s: 10 }] } });
assert.equal(element('progress-line').textContent, 'gravity plan ready');
assert.equal(events.length, 0);
assert.equal(state.gravityStatus.passive, true);
render({ state: 'paused', activity: 'gravity_hold_test',
  progress: { mode: 'gravity_hold_test', phase: 'failed', error: 'recovery required' },
  result: { mode: 'gravity_hold_test', result: 'FAIL', reason: 'UDP stale' } });
assert.equal(element('btn-gravtest-hold').disabled, true);
assert.equal(element('btn-gravtest-drag').disabled, true);
render({ state: 'idle', activity: '',
  progress: { mode: 'gravity_hold_test', phase: 'failed', error: 'UDP stale',
    recovery_verified: true },
  result: { mode: 'gravity_hold_test', result: 'FAIL', reason: 'UDP stale' } });
assert.equal(element('btn-gravtest-hold').disabled, false);
assert.equal(element('btn-gravtest-drag').disabled, false);
assert.equal(element('btn-grav-rehearse').disabled, false);
assert.equal(element('btn-grav-run').disabled, true);
console.log('PASS: planner status isolation and passive gravity rehearsal prerequisite');