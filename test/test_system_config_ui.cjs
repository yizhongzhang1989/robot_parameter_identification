const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { systemConfigFixture } = require('./system_config_fixture.cjs');

const source = fs.readFileSync(path.join(__dirname,
  '../robot_parameter_identification/dashboard/static/system_config.js'), 'utf8');
const html = fs.readFileSync(path.join(__dirname,
  '../robot_parameter_identification/dashboard/static/index.html'), 'utf8');
const dashboard = fs.readFileSync(path.join(__dirname,
  '../robot_parameter_identification/dashboard/static/dashboard.js'), 'utf8');
const viewer = fs.readFileSync(path.join(__dirname,
  '../robot_parameter_identification/dashboard/static/viewer.js'), 'utf8');
let moduleId = 0;
const loadModule = () => import(`data:text/javascript;base64,${Buffer.from(source)
  .toString('base64')}#${moduleId++}`);

function makeDocument() {
  const inputs = [
    { id: 'home-speed', type: 'number', value: '', disabled: false },
    { id: 'gravtest-speed-stop', type: 'number', value: '', disabled: false },
    { id: 'show-mesh', type: 'checkbox', checked: false, disabled: false },
  ];
  const scene = { textContent: '', classList: { remove() {} } };
  const status = { textContent: '' };
  return {
    inputs, scene, status, body: { inert: true },
    querySelectorAll: () => inputs,
    getElementById: (id) => ({ 'scene-status': scene, 'run-state': status })[id],
  };
}

function fixture() {
  const config = systemConfigFixture({
    ui: { polling: { state_ms: 750 },
      obstacle: { size_m: [0.12, 0.34, 0.56], xyz_m: [0.4, -0.2, 0.3] },
      view: { ghost: false, frames: true } },
    dashboard: {
      home: { transit_speed_deg_s: 7.5 },
      drag_test: { maximum_speed_deg_s: 83 },
      optimal: { reuse_friction: true },
    },
    campaign: { optimal_training_trajectories: 17, optimal_validation_trajectories: 4,
      fourier_duration_s: 45, optimal_friction_postures: 4, optimal_friction_repeats: 3,
      fourier_base_frequency_hz: 0.12 },
  });
  Object.assign(config.controls['home-speed'], { min: 0.2, max: 45, step: 0.2 });
  Object.assign(config.controls['gravtest-speed-stop'], { min: 2, max: 100, step: 2 });
  return config;
}

async function main() {
  const originalFetch = global.fetch;
  const originalDocument = global.document;
  try {
    const root = global.document = makeDocument();
    const config = fixture();
    const requests = [];
    let resolveResponse;
    global.fetch = (route, options) => {
      requests.push({ route, options });
      return new Promise((resolve) => { resolveResponse = resolve; });
    };
    const module = await loadModule();
    const first = module.loadSystemConfig();
    const second = module.loadSystemConfig();
    assert.equal(first, second);
    assert.equal(root.body.inert, true);
    assert.equal(root.inputs[0].value, '');
    resolveResponse({ ok: true, json: async () => config });
    assert.equal(await first, config);
    assert.deepEqual(requests, [{ route: '/api/system-config', options: { cache: 'no-store' } }]);
    assert.deepEqual(root.inputs.map((input) => input.value ?? input.checked), ['7.5', '83', true]);
    assert.deepEqual(['min', 'max', 'step'].map((key) => root.inputs[0][key]), ['0.2', '45', '0.2']);
    root.inputs[0].value = '9.1';
    root.inputs[1].value = '72';
    root.inputs[2].checked = false;
    module.initializeSystemControls(config);
    await module.loadSystemConfig();
    assert.deepEqual(root.inputs.map((input) => input.value ?? input.checked), ['9.1', '72', false]);
    assert.equal(requests.length, 1);

    const htmlInputs = [...html.matchAll(/<input\b[^>]*>/g)].filter(([tag]) =>
      /type="(?:number|checkbox)"/.test(tag));
    assert.match(html, /<body inert>/);
    for (const [tag] of htmlInputs) {
      const id = tag.match(/\bid="([^"]+)"/)[1];
      assert.match(tag, /\bdata-system-config\b/, id);
      assert.doesNotMatch(tag, /\b(?:value|min|max|step)="|\bchecked\b/, id);
      assert.ok(config.controls[id], `Missing fixture for ${id}`);
    }
    const inputs = htmlInputs.map(([tag]) => ({
      id: tag.match(/\bid="([^"]+)"/)[1], type: tag.match(/\btype="([^"]+)"/)[1],
      value: '', checked: false,
    }));
    module.initializeSystemControls(config, { querySelectorAll: () => inputs });
    for (const input of inputs) {
      const control = config.controls[input.id];
      assert.equal(input.type === 'checkbox' ? input.checked : input.value,
        input.type === 'checkbox' ? control.value : String(control.value), input.id);
      for (const key of ['min', 'max', 'step']) {
        assert.equal(input[key], control[key] === undefined ? undefined : String(control[key]), input.id);
      }
    }
    const handlers = new Map();
    const payloads = [];
    const context = vm.createContext({
      systemConfig: config,
      $: (id) => inputs.find((input) => input.id === id)
        || (id === 'obst-frame' ? { value: 'fixture_base' }
          : { addEventListener: (_event, handler) => handlers.set(id, handler) }),
      post: async (route, body) => { payloads.push(JSON.parse(JSON.stringify({ route, body }))); return { ok: false }; },
    });
    for (const [start, end] of [["$('btn-optimal').addEventListener", "$('btn-home').addEventListener"],
      ["$('obst-add').addEventListener", "$('obst-del').addEventListener"]]) {
      vm.runInContext(dashboard.slice(dashboard.indexOf(start), dashboard.indexOf(end)), context);
    }
    await handlers.get('btn-optimal')();
    assert.deepEqual(payloads.pop(), { route: '/api/campaign', body: { mode: 'optimal_excitation', options: {
      optimal_training_trajectories: 17, optimal_validation_trajectories: 4,
      optimal_friction_postures: 4, optimal_friction_repeats: 3,
      fourier_base_frequency_hz: 0.12, fourier_duration_s: 45, reuse_friction: true,
    } } });
    await handlers.get('obst-add')();
    assert.deepEqual(payloads.pop(), { route: '/api/obstacles', body: { action: 'add', obstacle: {
      parent_frame: 'fixture_base', size_m: [0.12, 0.34, 0.56], xyz_m: [0.4, -0.2, 0.3],
    } } });

    const defaultRoot = makeDocument();
    module.initializeSystemControls(systemConfigFixture(), defaultRoot);
    assert.equal(defaultRoot.inputs[1].value, '120');
    for (const script of [dashboard, viewer]) {
      const stripped = script.replace(/^import[\s\S]*?;\n/gm, '');
      new vm.Script(`(async () => { ${stripped} })()`);
      const gate = script.indexOf('await loadSystemConfig()');
      assert.ok(gate >= 0 && gate < script.indexOf('setInterval('));
    }
    assert.match(dashboard, /import '\/viewer\.js';/);
    assert.match(dashboard, /document\.body\.inert = false;/);
    for (const name of ['mesh', 'frames', 'labels', 'ghost', 'gravity']) {
      assert.ok(viewer.includes(`document.getElementById('show-${name}').checked`));
    }

    for (const response of [
      { ok: false, status: 503 },
      { ok: true, json: async () => ({ values: {}, controls: {} }) },
      { ok: true, json: async () => ({ ...fixture(), controls: {} }) },
      { ok: true, json: async () => systemConfigFixture({ ui: { polling: { state_ms: 0 } } }) },
      { ok: true, json: async () => systemConfigFixture({ ui: { scene: { camera_position_m: [1] } } }) },
    ]) {
      global.document = makeDocument();
      let calls = 0;
      global.fetch = async (route) => {
        assert.equal(route, '/api/system-config');
        calls += 1;
        return response;
      };
      const failedModule = await loadModule();
      await assert.rejects(failedModule.loadSystemConfig());
      await assert.rejects(failedModule.loadSystemConfig());
      assert.equal(calls, 1);
      assert.equal(global.document.status.textContent, 'configuration error');
      assert.match(global.document.scene.textContent, /System configuration unavailable:/);
      assert.ok(global.document.inputs.every((input) => input.disabled && !input.value));
      assert.equal(global.document.body.inert, false);
    }
    console.log('PASS: shared config GET, all HTML control defaults and bounds, '
      + 'optimal/obstacle payloads, one-time initialization, user edits, '
      + 'and visible fail-closed initialization');
  } finally {
    global.fetch = originalFetch;
    global.document = originalDocument;
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});