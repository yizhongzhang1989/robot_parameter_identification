function merge(base, overrides) {
  for (const [key, value] of Object.entries(overrides)) {
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      base[key] = merge(base[key] || {}, value);
    } else base[key] = value;
  }
  return base;
}

function systemConfigFixture(overrides = {}) {
  const values = merge({
    ui: {
      polling: { state_ms: 400, runs_every: 25, viewer_ms: 100, telemetry_ms: 100 },
      telemetry: { stale_ms: 1500, trace_depth: 1600 },
      plot: { signal: 'position_deg', mode: 'normal', height: { normal: 200, big: 460 } },
      editor: { workspace_step_deg: 1, jog_step_deg: 0.1 },
      obstacle: { size_m: [0.2, 0.2, 0.2], xyz_m: [0.3, 0, 0.1] },
      scene: {
        smoothing_tau_s: 0.06, camera_fov_deg: 45,
        camera_position_m: [1.1, -1.1, 0.9], camera_target_m: [0, 0, 0.35],
        pixel_ratio_max: 2, orbit_damping: true,
        grid_size_m: 3, grid_divisions: 30, axes_size_m: 0.25,
        tour: { duration_ms: 8000, min_segment_ms: 110, max_segment_ms: 420, frame_ms: 16 },
      },
      view: { mesh: true, frames: false, labels: false, ghost: true, gravity: false },
      sweep_resume: false,
    },
    dashboard: {
      home: { transit_speed_deg_s: 10 },
      jog: { transit_speed_deg_s: 10 },
      gravity: { gravity_probe_speeds_deg_s: [1, 3] },
      hold_test: { poses: 5, seconds: 3, transit_speed_deg_s: 5 },
      drag_test: { maximum_speed_deg_s: 120 },
      optimal: { reuse_friction: false },
    },
    campaign: {
      static_poses: 24, gravity_validation_poses: 8, gravity_probe_deg: 5,
      transit_speed_deg_s: 10, optimal_training_trajectories: 12,
      optimal_validation_trajectories: 3, fourier_duration_s: 30,
      optimal_friction_postures: 3, optimal_friction_repeats: 2,
      fourier_base_frequency_hz: 0.08,
    },
  }, overrides);
  const { ui, dashboard, campaign } = values;
  const number = (value, min, max, step) => ({ value, min, max, step });
  const controls = {
    'home-speed': number(dashboard.home.transit_speed_deg_s, 0.1, 60, 0.1),
    'jog-speed': number(dashboard.jog.transit_speed_deg_s, 0.1, 60, 0.1),
    'grav-poses': number(campaign.static_poses, 4, 60, 1),
    'grav-check': number(campaign.gravity_validation_poses, 2, 30, 1),
    'grav-arc': number(campaign.gravity_probe_deg, 1, 20, 0.5),
    'grav-slow': number(dashboard.gravity.gravity_probe_speeds_deg_s[0], 0.05, 20, 0.1),
    'grav-fast': number(dashboard.gravity.gravity_probe_speeds_deg_s.at(-1), 0.05, 20, 0.1),
    'grav-transit-speed': number(campaign.transit_speed_deg_s, 0.1, 60, 0.1),
    'gravtest-poses': number(dashboard.hold_test.poses, 1, 20, 1),
    'gravtest-hold-seconds': number(dashboard.hold_test.seconds, 0.5, 10, 0.5),
    'gravtest-transit-speed': number(dashboard.hold_test.transit_speed_deg_s, 0.1, 60, 0.1),
    'gravtest-speed-stop': number(dashboard.drag_test.maximum_speed_deg_s, 1, 120, 1),
    'optimal-training': number(campaign.optimal_training_trajectories, 2, 32, 1),
    'optimal-validation': number(campaign.optimal_validation_trajectories, 1, 8, 1),
    'optimal-duration': number(campaign.fourier_duration_s, 5, 120, 5),
    'optimal-postures': number(campaign.optimal_friction_postures, 1, 5, 1),
    'optimal-repeats': number(campaign.optimal_friction_repeats, 1, 5, 1),
    'optimal-frequency': number(campaign.fourier_base_frequency_hz, 0.02, 0.3, 0.01),
    'optimal-reuse-friction': { value: dashboard.optimal.reuse_friction },
    'sweep-resume': { value: ui.sweep_resume },
    'space-all-low': { value: '', step: 5 },
    'space-all-high': { value: '', step: 5 },
  };
  for (const key of ['mesh', 'frames', 'labels', 'ghost', 'gravity']) {
    controls[`show-${key}`] = { value: ui.view[key] };
  }
  ['sz-x', 'sz-y', 'sz-z'].forEach((id, index) => {
    controls[id] = { value: ui.obstacle.size_m[index], step: 0.01 };
  });
  ['px', 'py', 'pz'].forEach((id, index) => {
    controls[id] = { value: ui.obstacle.xyz_m[index], step: 0.01 };
  });
  for (const id of ['rr', 'rp', 'ry']) controls[id] = { value: 0, step: 1 };
  return { path: '/tmp/frontend-system-config.yaml', values, controls };
}

module.exports = { systemConfigFixture };