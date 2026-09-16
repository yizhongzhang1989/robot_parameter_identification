let configPromise;
const initializedControls = new WeakSet();

function validateUI(ui) {
  const positive = [
    'polling.state_ms', 'polling.runs_every', 'polling.viewer_ms', 'polling.telemetry_ms',
    'telemetry.stale_ms', 'telemetry.trace_depth', 'plot.height.normal', 'plot.height.big',
    'editor.workspace_step_deg', 'editor.jog_step_deg', 'scene.camera_fov_deg',
    'scene.pixel_ratio_max', 'scene.grid_size_m', 'scene.grid_divisions', 'scene.axes_size_m',
    'scene.tour.duration_ms', 'scene.tour.min_segment_ms', 'scene.tour.max_segment_ms',
    'scene.tour.frame_ms',
  ];
  const valueAt = (path) => path.split('.').reduce((value, key) => value?.[key], ui);
  for (const path of positive) {
    const value = valueAt(path);
    if (!Number.isFinite(value) || value <= 0) {
      throw new Error(`Invalid system configuration: ui.${path}`);
    }
  }
  for (const path of ['polling.runs_every', 'telemetry.trace_depth', 'scene.grid_divisions']) {
    if (!Number.isInteger(valueAt(path))) {
      throw new Error(`Expected an integer for ui.${path}`);
    }
  }
  for (const path of ['scene.camera_position_m', 'scene.camera_target_m',
    'obstacle.size_m', 'obstacle.xyz_m']) {
    const value = valueAt(path);
    if (!Array.isArray(value) || value.length !== 3 || !value.every(Number.isFinite)) {
      throw new Error(`Expected a three-component vector for ui.${path}`);
    }
  }
  if (!['off', 'normal', 'big'].includes(ui.plot.mode)
    || typeof ui.plot.signal !== 'string' || !ui.plot.signal
    || typeof ui.scene.orbit_damping !== 'boolean'
    || !Number.isFinite(ui.scene.smoothing_tau_s) || ui.scene.smoothing_tau_s < 0
    || ui.scene.camera_fov_deg >= 180 || ui.obstacle.size_m.some((size) => size <= 0)
    || ui.plot.height.big < ui.plot.height.normal
    || ui.scene.tour.max_segment_ms < ui.scene.tour.min_segment_ms) {
    throw new Error('Invalid system configuration plot or scene settings');
  }
}

export function initializeSystemControls(systemConfig, root = document) {
  const settings = [...root.querySelectorAll('[data-system-config]')].map((input) => {
    const control = systemConfig.controls?.[input.id];
    if (!control || !['number', 'string', 'boolean'].includes(typeof control.value)) {
      throw new Error(`Missing system configuration control: ${input.id}`);
    }
    if (input.type === 'checkbox' && typeof control.value !== 'boolean') {
      throw new Error(`Expected a checkbox value for ${input.id}`);
    }
    if (input.type === 'number' && (typeof control.value === 'boolean'
      || (control.value !== '' && !Number.isFinite(Number(control.value))))) {
      throw new Error(`Expected a numeric value for ${input.id}`);
    }
    for (const attribute of ['min', 'max', 'step']) {
      if (control[attribute] !== undefined && !Number.isFinite(control[attribute])) {
        throw new Error(`Invalid ${attribute} for ${input.id}`);
      }
    }
    if (control.min > control.max || (control.step !== undefined && control.step <= 0)) {
      throw new Error(`Invalid bounds for ${input.id}`);
    }
    return { input, control };
  });
  for (const { input, control } of settings) {
    if (initializedControls.has(input)) continue;
    for (const attribute of ['min', 'max', 'step']) {
      if (control[attribute] !== undefined) input[attribute] = String(control[attribute]);
    }
    if (input.type === 'checkbox') input.checked = control.value;
    else input.value = String(control.value);
    initializedControls.add(input);
  }
}

export function showSystemConfigError(error, root = document) {
  const message = `System configuration unavailable: ${error.message || error}`;
  for (const input of root.querySelectorAll('button, input, select')) input.disabled = true;
  const status = root.getElementById('run-state');
  if (status) status.textContent = 'configuration error';
  const scene = root.getElementById('scene-status');
  if (scene) {
    scene.textContent = message;
    scene.classList.remove('hidden');
  }
  const activity = root.getElementById('progress-line');
  if (activity) {
    activity.textContent = message;
    activity.removeAttribute('data-i18n');
  }
  root.getElementById('loading')?.classList.add('hidden');
  if (root.body) root.body.inert = false;
}

export function loadSystemConfig() {
  if (!configPromise) {
    configPromise = (async () => {
      try {
        const response = await fetch('/api/system-config', { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const config = await response.json();
        if (!config || typeof config.path !== 'string' || !config.values?.ui
          || typeof config.controls !== 'object' || !config.controls
          || Array.isArray(config.controls)) {
          throw new Error('Invalid system configuration response');
        }
        validateUI(config.values.ui);
        initializeSystemControls(config);
        return config;
      } catch (error) {
        showSystemConfigError(error);
        throw error;
      }
    })();
  }
  return configPromise;
}