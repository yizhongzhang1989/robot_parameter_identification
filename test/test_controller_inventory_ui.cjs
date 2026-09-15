const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const staticDir = path.join(__dirname,
  '../robot_parameter_identification/dashboard/static');
const source = fs.readFileSync(path.join(staticDir, 'dashboard.js'), 'utf8');
const translations = fs.readFileSync(path.join(staticDir, 'i18n.js'), 'utf8');
const elements = new Map();
function makeElement(tagName = 'span') {
  return {
    tagName, children: [], style: {}, textContent: '',
    set innerHTML(value) { throw new Error(`Unsafe HTML: ${value}`); },
    appendChild(child) { this.children.push(child); },
    replaceChildren(...children) { this.children = children; },
  };
}
function element(id) {
  if (!elements.has(id)) elements.set(id, makeElement());
  return elements.get(id);
}
Object.defineProperty(element('conn-table'), 'innerHTML', {
  set(value) { this.connectionMarkup = value; },
});
const context = vm.createContext({
  $: element,
  document: { createElement: makeElement },
  state: { snapshot: null, polling: false, ticks: 1 },
  RUNS_EVERY: 25,
  pollRuns() {},
  pill() {},
  describeEnvelope: () => '-',
});
vm.runInContext(`
  ${translations.slice(translations.indexOf('const DICT ='),
    translations.indexOf("let current = 'en'"))}
  let current = 'en';
  ${translations.slice(translations.indexOf('export function t('),
    translations.indexOf('/** Fill every')).replace('export ', '')}
  ${source.slice(source.indexOf('function renderControllers('),
    source.indexOf('function phaseLabel('))}
  ${source.slice(source.indexOf('async function poll()'),
    source.indexOf('setInterval(poll,'))}
  function renderAll(snapshot) { renderConnection(snapshot); }
`, context);

const inventory = (items, extra = {}) => ({
  manager: '/controller_manager', available: true, age_s: 0.2, error: '', items, ...extra,
});
const controller = (name, state, claimed_interfaces = ['joint/effort']) => ({
  name, type: `package/${name}`, state, claimed_interfaces,
});
const items = [
  controller('arm_trajectory_controller', 'active'),
  controller('joint_state_broadcaster', 'active', []),
  controller('direct_current_controller', 'inactive'),
  controller('unconfigured_controller', 'unconfigured'),
];
const rows = () => element('controllers-body').children.map(
  (row) => row.children.map((cell) => cell.textContent));
const activeText = () => element('active-controllers').textContent;
function render(value) {
  context.controllers = value;
  vm.runInContext('renderControllers(controllers)', context);
}

async function main() {
  render(inventory(items));
  assert.deepEqual(rows(), items.map((item) => [item.name, item.type, item.state]));
  assert.equal(activeText(), 'arm_trajectory_controller');
  assert.equal(element('controllers-status').textContent, '');
  render(inventory(items, { age_s: 999, error: 'stale' }));
  assert.equal(activeText(), 'arm_trajectory_controller');

  const renamedBroadcaster = { ...controller('sensor_states', 'active', []),
    type: 'force_torque_sensor_broadcaster/ForceTorqueSensorBroadcaster' };
  render(inventory([renamedBroadcaster, controller('idle_controller', 'active', [])]));
  assert.equal(activeText(), 'idle_controller');
  assert.equal(rows().length, 2);
  render(inventory([items[1], renamedBroadcaster]));
  assert.equal(activeText(), 'No active controllers');
  assert.equal(rows().length, 2);

  render(inventory(items.map((item) => ({ ...item,
    state: item.name === 'direct_current_controller' ? 'active' : 'inactive' }))));
  assert.equal(activeText(), 'direct_current_controller');
  assert.equal(rows().length, 4);
  render(inventory([controller('uppercase', 'ACTIVE'), items[2], items[3]]));
  assert.equal(activeText(), 'No active controllers');
  assert.equal(rows().length, 3);
  render(inventory([]));
  assert.equal(activeText(), 'No active controllers');
  assert.equal(element('controllers-status').textContent, 'No controllers');
  assert.deepEqual(rows(), []);

  for (const error of ['waiting', 'unavailable', 'timeout', 'query_failed', 'stale', '']) {
    render(inventory(items));
    render(inventory(items, { available: false, age_s: null, error }));
    assert.deepEqual(rows(), []);
    assert.notEqual(activeText(), 'No active controllers');
    assert.equal(activeText(), element('controllers-status').textContent);
    assert.equal(activeText(), vm.runInContext(
      `t('controllers.${error || 'unavailable'}')`, context));
  }
  for (const value of [undefined, null, {}, inventory(items, { available: 'true' })]) {
    render(value);
    assert.equal(activeText(), 'Controllers unavailable');
    assert.deepEqual(rows(), []);
  }

  const payload = '<img src=x onerror="alert(1)"> & <script>bad()</script>';
  render(inventory([{ name: payload, type: payload, state: 'active', claimed_interfaces: [] },
    { name: 'other', type: 'other', state: payload, claimed_interfaces: [] }]));
  assert.equal(activeText(), payload);
  assert.deepEqual(rows()[0], [payload, payload, 'active']);
  assert.equal(rows()[1][2], payload);
  const longName = 'full_controller_name_'.repeat(30);
  render(inventory([controller(longName, 'active')]));
  assert.equal(activeText(), longName);
  assert.equal(rows()[0][0], longName);

  vm.runInContext("current = 'zh'", context);
  render(inventory([]));
  assert.equal(activeText(), '\u65e0\u6d3b\u52a8\u63a7\u5236\u5668');
  assert.equal(element('controllers-status').textContent, '\u65e0\u63a7\u5236\u5668');
  vm.runInContext("current = 'en'", context);

  for (const failure of [
    async () => { throw new Error('network disconnected'); },
    async () => ({ ok: false, status: 503, json: async () => ({ connection: { controllers: inventory(items) } }) }),
    async () => ({ ok: true, json: async () => { throw new Error('invalid JSON'); } }),
  ]) {
    context.fetch = async () => ({ ok: true, json: async () => ({ connection: { controllers: inventory(items) } }) });
    await vm.runInContext('poll()', context);
    assert.equal(activeText(), 'arm_trajectory_controller');
    context.fetch = failure;
    await vm.runInContext('poll()', context);
    assert.equal(activeText(), 'Controllers unavailable');
    assert.deepEqual(rows(), []);
    assert.equal(context.state.polling, false);
    vm.runInContext("current = 'zh'; renderAll(state.snapshot)", context);
    assert.equal(activeText(), '\u63a7\u5236\u5668\u6570\u636e\u4e0d\u53ef\u7528');
    vm.runInContext("current = 'en'", context);
  }
  const connectionSource = source.slice(source.indexOf('function renderConnection('),
    source.indexOf('function phaseLabel('));
  assert.match(connectionSource, /renderControllers\(connection\.controllers\)/);
  const html = fs.readFileSync(path.join(staticDir, 'index.html'), 'utf8');
  const keys = [...html.matchAll(/data-i18n="(controllers\.[^"]+)"/g)].map((match) => match[1]);
  for (const key of keys) {
    for (const language of ['en', 'zh']) {
      assert.ok(vm.runInContext(`DICT['${key}']['${language}']`, context), `${key}/${language}`);
    }
  }
  assert.match(html.slice(html.indexOf('<header'), html.indexOf('</header>')), /id="active-controllers"/);
  assert.match(html.slice(html.indexOf('id="group-control"'), html.indexOf('id="group-space"')), /id="controllers-table"/);
  console.log('PASS: inventory, exact active filter, broadcaster, empty/unavailable, escaping, updates, EN/ZH, and poll disconnect/recovery');
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});