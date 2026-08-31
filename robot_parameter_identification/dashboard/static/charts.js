/* Charts for reading a fit rather than staring at a table of numbers.
 *
 * Plain canvas on purpose: four small plots do not justify a charting library,
 * and the ones that matter here are unusual enough (a fitted curve laid over
 * the cloud it was fitted to) that a library would mostly get in the way.
 */

const INK = '#e6e9ef';
const MUTED = '#8b93a5';
const LINE = '#262c38';
const SERIES = ['#4da3ff', '#e0b341', '#35c08a', '#e2565a',
                '#b07de0', '#4fd0d0', '#e08a4d'];

function setup(canvas) {
  const ratio = Math.min(devicePixelRatio || 1, 2);
  // The logical height is stashed on first use. Reading it back from the
  // height attribute would be self-feeding: assigning canvas.height writes
  // that same attribute, so every redraw would scale the panel again.
  if (!canvas.dataset.logicalHeight) {
    canvas.dataset.logicalHeight = String(+canvas.getAttribute('height') || 150);
  }
  const height = +canvas.dataset.logicalHeight;
  const width = Math.max(120, Math.round(canvas.clientWidth || 360));
  canvas.style.height = `${height}px`;
  const backingWidth = Math.round(width * ratio);
  const backingHeight = Math.round(height * ratio);
  if (canvas.width !== backingWidth) canvas.width = backingWidth;
  if (canvas.height !== backingHeight) canvas.height = backingHeight;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.font = '10px ui-monospace, monospace';
  return { ctx, width, height };
}

function empty(ctx, width, height, message) {
  ctx.fillStyle = MUTED;
  ctx.textAlign = 'center';
  ctx.fillText(message, width / 2, height / 2);
  ctx.textAlign = 'left';
}

function axes(ctx, box) {
  ctx.strokeStyle = LINE;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(box.left, box.top);
  ctx.lineTo(box.left, box.bottom);
  ctx.lineTo(box.right, box.bottom);
  ctx.stroke();
}

/* ---------------- grouped bars: error per joint ---------------- */

export function drawErrors(canvas, result, unit) {
  const { ctx, width, height } = setup(canvas);
  const joints = result?.joints || [];
  if (!joints.length) return empty(ctx, width, height, 'no result yet');

  const box = { left: 34, right: width - 6, top: 14, bottom: height - 26 };
  const groups = [
    ['residual_rms_a', 'training', '#3a4356'],
    ['holdout_rms_a', 'holdout', '#4da3ff'],
    ['validation_rms_a', 'validation', '#35c08a'],
  ];
  let peak = 0;
  for (const entry of joints) {
    for (const [key] of groups) peak = Math.max(peak, entry[key] || 0);
  }
  peak = peak || 1;

  axes(ctx, box);
  ctx.fillStyle = MUTED;
  ctx.fillText(peak.toFixed(3), 2, box.top + 4);
  ctx.fillText('0', 2, box.bottom + 3);

  const slot = (box.right - box.left) / joints.length;
  const barWidth = Math.max(2, (slot - 6) / groups.length);
  joints.forEach((entry, index) => {
    const x0 = box.left + index * slot + 3;
    groups.forEach(([key, , colour], slotIndex) => {
      const value = entry[key] || 0;
      const h = (value / peak) * (box.bottom - box.top);
      ctx.fillStyle = colour;
      ctx.fillRect(x0 + slotIndex * barWidth, box.bottom - h, barWidth - 1, h);
    });
    ctx.fillStyle = MUTED;
    ctx.textAlign = 'center';
    ctx.fillText(`J${index + 1}`, x0 + slot / 2 - 3, box.bottom + 12);
    ctx.textAlign = 'left';
  });

  let legendX = box.left;
  groups.forEach(([, label, colour]) => {
    ctx.fillStyle = colour;
    ctx.fillRect(legendX, height - 9, 7, 7);
    ctx.fillStyle = MUTED;
    ctx.fillText(label, legendX + 10, height - 3);
    legendX += ctx.measureText(label).width + 24;
  });
  ctx.fillStyle = MUTED;
  ctx.textAlign = 'right';
  ctx.fillText(unit || 'A', box.right, box.top + 4);
  ctx.textAlign = 'left';
}

/* ---------------- friction curve over its own samples ---------------- */

export function frictionValue(entry, velocity, load) {
  const friction = entry?.friction || {};
  const components = entry?.components || {};
  const transition = components.coulomb_transition_deg_s ?? 0;
  const stribeckSpeed = components.stribeck_speed_deg_s || 1;
  const sign = Math.sign(velocity);
  const reversal = transition > 0 ? Math.tanh(velocity / transition) : sign;
  const decay = sign * Math.exp(-Math.abs(velocity) / stribeckSpeed);
  return (friction.coulomb || 0) * reversal
    + (friction.load_friction || 0) * Math.abs(load || 0) * sign
    + (friction.stribeck || 0) * decay
    + (friction.load_stribeck || 0) * Math.abs(load || 0) * decay
    + (friction.viscous || 0) * velocity + (friction.offset || 0);
}

function frictionLoads(entry, samples) {
  if (!(entry?.components?.load_friction
        || entry?.components?.load_stribeck)) return [0];
  const loads = (samples || []).map((sample) => Math.abs(sample.load || 0))
    .sort((a, b) => a - b);
  if (!loads.length) return [0];
  const low = loads[Math.floor(loads.length * 0.05)];
  const high = loads[Math.floor(loads.length * 0.95)];
  return high - low > 1e-6 ? [low, high] : [high];
}

function loadShade(load, loads) {
  if (loads.length < 2 || loads[1] - loads[0] <= 1e-9) {
    return 'rgba(77,163,255,.65)';
  }
  const share = Math.max(0, Math.min(1,
    (Math.abs(load || 0) - loads[0]) / (loads[1] - loads[0])));
  return `rgba(77,${Math.round(120 + 80 * share)},255,${(0.3 + 0.6 * share).toFixed(2)})`;
}

export function drawFriction(canvas, entry, samples, unit) {
  const { ctx, width, height } = setup(canvas);
  if (!entry) return empty(ctx, width, height, 'no result yet');

  const speeds = (samples || []).map((s) => s.speed);
  const efforts = (samples || []).map((s) => s.effort);
  const maxSpeed = Math.max(10, ...speeds.map(Math.abs));
  const loads = frictionLoads(entry, samples);
  const curves = loads.map((load) => {
    const curve = [];
    for (let i = 0; i <= 120; i += 1) {
      const velocity = -maxSpeed + (2 * maxSpeed * i) / 120;
      curve.push([velocity, frictionValue(entry, velocity, load)]);
    }
    return curve;
  });
  const values = curves.flat().map((point) => point[1]).concat(efforts);
  const low = Math.min(...values);
  const high = Math.max(...values);
  const span = (high - low) || 1;

  const box = { left: 38, right: width - 6, top: 10, bottom: height - 22 };
  const sx = (v) => box.left + ((v + maxSpeed) / (2 * maxSpeed)) * (box.right - box.left);
  const sy = (e) => box.bottom - ((e - low) / span) * (box.bottom - box.top);

  axes(ctx, box);
  ctx.strokeStyle = LINE;
  ctx.beginPath();
  ctx.moveTo(sx(0), box.top);
  ctx.lineTo(sx(0), box.bottom);
  ctx.stroke();

  (samples || []).forEach((sample) => {
    if (sample.sweep) return;
    ctx.fillStyle = loadShade(sample.load, loads);
    ctx.fillRect(sx(sample.speed) - 1, sy(sample.effort) - 1, 2, 2);
  });

  // Sweep points are the pose-controlled ones: a single joint moving fast
  // about one pose. Everything else is low speed across many poses, so a
  // trend read across both groups is part pose, not all speed.
  ctx.fillStyle = 'rgba(120,220,150,.85)';
  (samples || []).forEach((sample) => {
    if (!sample.sweep) return;
    ctx.fillRect(sx(sample.speed) - 1.5, sy(sample.effort) - 1.5, 3, 3);
  });

  if (curves.length > 1) {
    ctx.fillStyle = 'rgba(224,179,65,.12)';
    ctx.beginPath();
    curves[0].forEach(([v, e], index) => {
      const x = sx(v), y = sy(e);
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    [...curves[1]].reverse().forEach(([v, e]) => ctx.lineTo(sx(v), sy(e)));
    ctx.closePath();
    ctx.fill();
  }
  ctx.strokeStyle = '#e0b341';
  ctx.lineWidth = 1.6;
  curves.forEach((curve, curveIndex) => {
    ctx.setLineDash(curveIndex === 0 && curves.length > 1 ? [4, 3] : []);
    ctx.beginPath();
    curve.forEach(([v, e], index) => {
      const x = sx(v), y = sy(e);
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  ctx.setLineDash([]);

  const friction = entry.friction || {};
  ctx.fillStyle = MUTED;
  ctx.textAlign = 'left';
  ctx.fillText(`c ${(friction.coulomb || 0).toFixed(3)}`
    + `${entry.components?.load_friction ? `  μ ${(friction.load_friction || 0).toFixed(3)}` : ''}`
    + `${entry.components?.load_stribeck ? `  FsL ${(friction.load_stribeck || 0).toFixed(3)}` : ''}`
    + `  v ${(friction.viscous || 0).toFixed(4)}`,
  box.left + 4, box.top + 10);

  ctx.fillStyle = MUTED;
  ctx.fillText(high.toFixed(2), 2, box.top + 4);
  ctx.fillText(low.toFixed(2), 2, box.bottom);
  ctx.textAlign = 'center';
  ctx.fillText(`speed deg/s  (${unit || 'A'} vertical)`,
               (box.left + box.right) / 2, height - 4);
  ctx.textAlign = 'right';
  ctx.fillStyle = 'rgba(120,220,150,.95)';
  ctx.fillText('sweep', box.right, box.top + 4);
  ctx.textAlign = 'left';
}

/* ---------------- residual against speed ---------------- */

export function drawResidual(canvas, points) {
  const { ctx, width, height } = setup(canvas);
  if (!points || !points.length) {
    return empty(ctx, width, height, 'run a campaign to populate');
  }
  const box = { left: 38, right: width - 6, top: 10, bottom: height - 20 };
  const speeds = points.map((p) => p.speed);
  const residuals = points.map((p) => p.residual);
  const maxSpeed = Math.max(1, ...speeds.map(Math.abs));
  const maxResidual = Math.max(1e-6, ...residuals.map(Math.abs));

  const sx = (v) => box.left + ((v + maxSpeed) / (2 * maxSpeed)) * (box.right - box.left);
  const sy = (r) => (box.top + box.bottom) / 2
    - (r / maxResidual) * ((box.bottom - box.top) / 2);

  axes(ctx, box);
  ctx.strokeStyle = LINE;
  ctx.beginPath();
  ctx.moveTo(box.left, sy(0));
  ctx.lineTo(box.right, sy(0));
  ctx.stroke();

  ctx.fillStyle = 'rgba(226,86,90,.5)';
  points.forEach((point) => {
    if (point.sweep) return;
    ctx.fillRect(sx(point.speed) - 1, sy(point.residual) - 1, 2, 2);
  });

  ctx.fillStyle = 'rgba(120,220,150,.85)';
  points.forEach((point) => {
    if (!point.sweep) return;
    ctx.fillRect(sx(point.speed) - 1.5, sy(point.residual) - 1.5, 3, 3);
  });

  ctx.fillStyle = MUTED;
  ctx.fillText(`+${maxResidual.toFixed(3)}`, 2, box.top + 4);
  ctx.fillText(`-${maxResidual.toFixed(3)}`, 2, box.bottom);
  ctx.textAlign = 'center';
  ctx.fillText('speed deg/s', (box.left + box.right) / 2, height - 3);
  ctx.textAlign = 'left';
}

/* ---------------- condition number per joint ---------------- */

export function drawCondition(canvas, result, ceiling) {
  const { ctx, width, height } = setup(canvas);
  const joints = result?.joints || [];
  if (!joints.length) return empty(ctx, width, height, 'no result yet');

  const box = { left: 40, right: width - 6, top: 12, bottom: height - 20 };
  const values = joints.map((entry) => entry.condition_number || 0);
  const cap = ceiling || 1000;
  const peak = Math.max(cap, ...values);
  axes(ctx, box);

  const limitY = box.bottom - (cap / peak) * (box.bottom - box.top);
  ctx.strokeStyle = 'rgba(226,86,90,.6)';
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(box.left, limitY);
  ctx.lineTo(box.right, limitY);
  ctx.stroke();
  ctx.setLineDash([]);

  const slot = (box.right - box.left) / joints.length;
  values.forEach((value, index) => {
    const h = (value / peak) * (box.bottom - box.top);
    ctx.fillStyle = value > cap ? '#e2565a' : SERIES[index % SERIES.length];
    ctx.fillRect(box.left + index * slot + 2, box.bottom - h, slot - 5, h);
    ctx.fillStyle = MUTED;
    ctx.textAlign = 'center';
    ctx.fillText(`J${index + 1}`, box.left + index * slot + slot / 2, box.bottom + 11);
    ctx.textAlign = 'left';
  });
  ctx.fillStyle = MUTED;
  ctx.fillText(String(Math.round(peak)), 2, box.top + 4);
  ctx.fillText(`cap ${cap}`, box.right - 46, limitY - 3);
}
