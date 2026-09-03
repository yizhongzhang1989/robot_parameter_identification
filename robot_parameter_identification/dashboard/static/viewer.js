/* 3D view: the arm from forward kinematics, plus an editable obstacle scene.
 *
 * Link poses come from the server as 4x4 matrices computed by the same model
 * the collision check and the regression use, so the picture cannot drift from
 * the maths.
 *
 * Dragging happens in world space but an obstacle is stored relative to the
 * frame it is bolted to, so every release converts back through the inverse of
 * that frame's world pose. Get this wrong and a box silently teleports the
 * moment the arm moves.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { ColladaLoader } from 'three/addons/loaders/ColladaLoader.js';
import { t } from '/i18n.js';

const POLL_MS = 100;
const SMOOTH_TAU = 0.06;

const host = document.getElementById('viewer');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d11);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(1.1, -1.1, 0.9);
camera.up.set(0, 0, 1);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
host.appendChild(renderer.domElement);

const orbit = new OrbitControls(camera, renderer.domElement);
orbit.target.set(0, 0, 0.35);
orbit.enableDamping = true;

scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x202430, 2.0));
const key = new THREE.DirectionalLight(0xffffff, 1.6);
key.position.set(2, -2, 3);
scene.add(key);
const grid = new THREE.GridHelper(3, 30, 0x2a3140, 0x1a1f28);
grid.rotation.x = Math.PI / 2;
scene.add(grid);
scene.add(new THREE.AxesHelper(0.25));

const linkGroup = new THREE.Group();
const frameGroup = new THREE.Group();
const obstacleGroup = new THREE.Group();
const ghostGroup = new THREE.Group();
const flierGroup = new THREE.Group();
const gravityGroup = new THREE.Group();
scene.add(linkGroup, frameGroup, obstacleGroup, ghostGroup, flierGroup,
         gravityGroup);

const gizmo = new TransformControls(camera, renderer.domElement);
gizmo.setSpace('local');
gizmo.addEventListener('dragging-changed', (event) => {
  orbit.enabled = !event.value;
  if (!event.value) commitSelection();
});
scene.add(gizmo);

const state = {
  frames: {},           // frame name -> THREE.Matrix4 (world)
  linkNodes: new Map(),  // link name -> Object3D
  boxNodes: new Map(),   // obstacle id -> Mesh
  massNodes: new Map(),  // link name -> {node, local, mass, label, world}
  massKey: '',
  ghostNodes: [],        // one skeleton per planned pose
  flierNodes: new Map(), // link name -> the second arm's Object3D
  flying: false,
  flight: 0,             // which flight is current; older loops stand down
  flyFrom: 0,
  previewToken: -1,
  obstacles: [],
  selected: null,
  showMesh: true,
  showFrames: false,
  showLabels: false,
  showGhost: true,
  showGravity: false,
  meshCache: new Map(),
};

window.__viewer = {
  select: selectObstacle,
  selected: () => state.selected,
  setMode: (mode) => gizmo.setMode(mode),
  frames: () => Object.keys(state.frames),
  highlight: (phase, index) => highlightPose(phase, index),
  plannedPoses: () => state.ghostNodes.length,
  fly: () => { if (!state.flying) flyTour(); },
  stopFlying,
  flying: () => state.flying,
  flierLinks: () => [...state.flierNodes.values()].filter((n) => n.visible).length,
  onChange: null,       // dashboard.js hooks this to refresh the form
};

/* ---------------- sizing ---------------- */

function resize() {
  const w = host.clientWidth || 1;
  const h = host.clientHeight || 1;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(host);
resize();

/* ---------------- mesh loading ---------------- */

const stl = new STLLoader();
const dae = new ColladaLoader();

function loadMesh(url) {
  if (state.meshCache.has(url)) return state.meshCache.get(url);
  const promise = new Promise((resolve) => {
    const isDae = /\.dae(\?|$)/i.test(url) || url.includes('.dae');
    const done = (obj) => resolve(obj);
    const fail = () => resolve(null);
    if (isDae) {
      dae.load(url, (result) => done(result.scene), undefined, fail);
    } else {
      stl.load(url, (geometry) => {
        geometry.computeVertexNormals();
        done(new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
          color: 0x9aa7bd, metalness: 0.15, roughness: 0.65,
        })));
      }, undefined, fail);
    }
  });
  state.meshCache.set(url, promise);
  return promise;
}

async function buildLinks(visuals) {
  linkGroup.clear();
  state.linkNodes.clear();
  for (const visual of visuals) {
    const node = new THREE.Group();
    node.name = visual.link;
    const proto = await loadMesh(visual.url);
    if (proto) {
      const clone = proto.clone(true);
      clone.position.set(...(visual.xyz || [0, 0, 0]));
      const rpy = visual.rpy || [0, 0, 0];
      clone.rotation.set(rpy[0], rpy[1], rpy[2], 'ZYX');
      const s = visual.scale || [1, 1, 1];
      clone.scale.set(s[0], s[1], s[2]);
      node.add(clone);
    }
    node.matrixAutoUpdate = false;
    linkGroup.add(node);
    state.linkNodes.set(visual.link, node);
  }
  document.getElementById('loading')?.classList.add('hidden');
}

/* ---------------- obstacles ---------------- */

const BOX_IDLE = 0xe0b341;
const BOX_SEL = 0x4da3ff;

function syncObstacles(list) {
  const seen = new Set();
  for (const item of list) {
    seen.add(item.id);
    let node = state.boxNodes.get(item.id);
    if (!node) {
      node = new THREE.Mesh(
        new THREE.BoxGeometry(1, 1, 1),
        new THREE.MeshStandardMaterial({
          color: BOX_IDLE, transparent: true, opacity: 0.45,
          metalness: 0.0, roughness: 0.9,
        }));
      node.add(new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
        new THREE.LineBasicMaterial({ color: 0xffffff, opacity: 0.35,
                                      transparent: true })));
      node.userData.obstacleId = item.id;
      obstacleGroup.add(node);
      state.boxNodes.set(item.id, node);
    }
    // While the user is dragging this box, the server's copy is stale.
    if (gizmo.dragging && state.selected === item.id) continue;
    node.matrixAutoUpdate = false;
    node.matrix.fromArray(item.matrix).transpose();
    node.matrix.decompose(node.position, node.quaternion, node.scale);
    node.matrixAutoUpdate = true;
    node.scale.set(item.size_m[0], item.size_m[1], item.size_m[2]);
    node.userData.size = item.size_m.slice();
    node.userData.parentFrame = item.parent_frame;
  }
  for (const [id, node] of [...state.boxNodes]) {
    if (seen.has(id)) continue;
    obstacleGroup.remove(node);
    state.boxNodes.delete(id);
    if (state.selected === id) selectObstacle(null);
  }
}

function selectObstacle(id) {
  state.selected = id;
  for (const [key2, node] of state.boxNodes) {
    node.material.color.setHex(key2 === id ? BOX_SEL : BOX_IDLE);
    node.material.opacity = key2 === id ? 0.6 : 0.45;
  }
  const node = id ? state.boxNodes.get(id) : null;
  if (node) gizmo.attach(node); else gizmo.detach();
  if (window.__viewer.onChange) window.__viewer.onChange(id);
}

/** Convert the dragged world pose back into the parent frame and publish it. */
async function commitSelection() {
  const id = state.selected;
  const node = id ? state.boxNodes.get(id) : null;
  if (!node) return;
  const parentName = node.userData.parentFrame;
  const parentWorld = state.frames[parentName];
  if (!parentWorld) return;

  const world = new THREE.Matrix4().compose(
    node.position, node.quaternion, new THREE.Vector3(1, 1, 1));
  const local = new THREE.Matrix4()
    .copy(parentWorld).invert().multiply(world);

  const position = new THREE.Vector3();
  const rotation = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  local.decompose(position, rotation, scale);
  const euler = new THREE.Euler().setFromQuaternion(rotation, 'ZYX');
  const deg = (r) => +(r * 180 / Math.PI).toFixed(3);

  const changes = {
    xyz_m: [+position.x.toFixed(4), +position.y.toFixed(4), +position.z.toFixed(4)],
    rpy_deg: [deg(euler.x), deg(euler.y), deg(euler.z)],
  };
  if (gizmo.getMode() === 'scale') {
    changes.size_m = [
      Math.max(0.001, +node.scale.x.toFixed(4)),
      Math.max(0.001, +node.scale.y.toFixed(4)),
      Math.max(0.001, +node.scale.z.toFixed(4)),
    ];
  }
  window.__dash?.updateObstacle(id, changes);
}

/* ---------------- picking ---------------- */

const ray = new THREE.Raycaster();
const pointer = new THREE.Vector2();

renderer.domElement.addEventListener('pointerdown', (event) => {
  if (gizmo.dragging) return;
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  ray.setFromCamera(pointer, camera);
  const hits = ray.intersectObjects(obstacleGroup.children, false);
  if (hits.length) selectObstacle(hits[0].object.userData.obstacleId);
  else if (event.target === renderer.domElement) selectObstacle(null);
});

window.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') selectObstacle(null);
});

/* ---------------- frames ---------------- */

function syncFrames(linkTf) {
  state.frames = {};
  for (const [name, flat] of Object.entries(linkTf || {})) {
    state.frames[name] = new THREE.Matrix4().fromArray(flat).transpose();
  }
  for (const [name, node] of state.linkNodes) {
    const world = state.frames[name];
    if (world) node.matrix.copy(world);
    node.visible = state.showMesh && !!world;
  }
  frameGroup.clear();
  if (state.showFrames) {
    for (const world of Object.values(state.frames)) {
      const axes = new THREE.AxesHelper(0.05);
      axes.matrixAutoUpdate = false;
      axes.matrix.copy(world);
      frameGroup.add(axes);
    }
  }
}

/* ---------------- gravity terms ---------------- */
/* What the URDF says gravity has to work with: a mass, and how far from the
 * joint it hangs. Drawn where forward kinematics puts it, so a lever entered
 * on the wrong axis is visible as a ball sitting outside its own link.
 */

const MASS_COLOR = 0xff6b6b;
const TOTAL_COLOR = 0x6ee7ff;
const MASS_MIN_R = 0.012;
const MASS_MAX_R = 0.042;

const ballGeometry = new THREE.SphereGeometry(1, 16, 12);
const labelLayer = document.getElementById('gravity-labels');

const totalNode = new THREE.Group();
const totalBall = new THREE.Mesh(ballGeometry, new THREE.MeshBasicMaterial({
  color: TOTAL_COLOR, wireframe: true, transparent: true, opacity: 0.8 }));
totalBall.scale.setScalar(0.05);
const weightArrow = new THREE.ArrowHelper(
  new THREE.Vector3(0, 0, -1), new THREE.Vector3(), 0.3, TOTAL_COLOR,
  0.06, 0.035);
totalNode.add(totalBall, weightArrow);
totalNode.visible = false;
gravityGroup.add(totalNode);
const totalLabel = makeLabel('total');

function makeLabel(extra) {
  if (!labelLayer) return null;
  const element = document.createElement('div');
  element.className = extra ? `mass-label ${extra}` : 'mass-label';
  labelLayer.appendChild(element);
  return element;
}

function buildGravity(links) {
  for (const entry of state.massNodes.values()) {
    gravityGroup.remove(entry.node);
    entry.label?.remove();
  }
  state.massNodes.clear();
  const heaviest = Math.max(...links.map((item) => item.mass_kg), 0) || 1;
  for (const item of links) {
    const local = new THREE.Vector3(...item.com_m);
    const node = new THREE.Group();
    node.matrixAutoUpdate = false;
    const ball = new THREE.Mesh(ballGeometry, new THREE.MeshStandardMaterial({
      color: MASS_COLOR, transparent: true, opacity: 0.85,
      metalness: 0.0, roughness: 0.5 }));
    ball.position.copy(local);
    // Radius by cube root, so it is the volume that tracks the mass: scaling
    // the radius directly gives a link twice as heavy eight times the ink.
    ball.scale.setScalar(MASS_MIN_R + (MASS_MAX_R - MASS_MIN_R)
                         * Math.cbrt(item.mass_kg / heaviest));
    const stem = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), local]),
      new THREE.LineBasicMaterial({ color: MASS_COLOR, transparent: true,
                                    opacity: 0.55 }));
    node.add(ball, stem);
    gravityGroup.add(node);
    const label = makeLabel();
    if (label) {
      label.innerHTML = `<b>${item.mass_kg.toFixed(3)} kg</b>`
        + `<span>${item.link}</span>`;
    }
    state.massNodes.set(item.link, { node, local, mass: item.mass_kg, label,
                                     world: null });
  }
}

/** Place every mass where this frame's kinematics puts it, and total them. */
function syncGravity() {
  const centre = new THREE.Vector3();
  let total = 0;
  for (const [name, entry] of state.massNodes) {
    const world = state.frames[name];
    entry.node.visible = !!world;
    entry.world = null;
    if (!world) continue;
    entry.node.matrix.copy(world);
    entry.world = entry.local.clone().applyMatrix4(world);
    centre.addScaledVector(entry.world, entry.mass);
    total += entry.mass;
  }
  totalNode.visible = total > 0;
  if (total <= 0) return;
  centre.divideScalar(total);
  totalNode.position.copy(centre);
  // The arrow reaches the floor, so it shows both which way weight pulls and
  // the point on the bench the whole robot's weight passes through.
  weightArrow.setLength(Math.max(0.08, centre.z), 0.06, 0.035);
  if (totalLabel) {
    totalLabel.innerHTML = `<b>${total.toFixed(3)} kg</b>`
      + `<span>${t('view.gravity_total')}</span>`;
  }
}

/** Labels follow the camera, so they are placed per rendered frame. */
const projected = new THREE.Vector3();
function placeLabels() {
  const width = renderer.domElement.clientWidth;
  const height = renderer.domElement.clientHeight;
  const place = (label, world) => {
    if (!label) return;
    if (!world) { label.style.display = 'none'; return; }
    projected.copy(world).project(camera);
    const onScreen = projected.z < 1 && Math.abs(projected.x) <= 1
      && Math.abs(projected.y) <= 1;
    label.style.display = onScreen ? 'block' : 'none';
    if (!onScreen) return;
    label.style.left = `${(projected.x * 0.5 + 0.5) * width}px`;
    label.style.top = `${(-projected.y * 0.5 + 0.5) * height}px`;
  };
  for (const entry of state.massNodes.values()) place(entry.label, entry.world);
  place(totalLabel, totalNode.visible ? totalNode.position : null);
}

function setGravityVisible(on) {
  gravityGroup.visible = on;
  labelLayer?.classList.toggle('hidden', !on);
}

/* ---------------- planned poses ---------------- */
/* Where the run is going, as one skeleton per designed pose. Fetched only when
 * the server says the plan changed: the poses are fixed once designed, and
 * re-sending them ten times a second would dwarf everything else on the wire.
 */

const GHOST_COLORS = {
  A_gravity: 0x4da3ff,
  D_validation: 0x39d98a,
  B_friction: 0xe0b341,
  C_inertia: 0xb388ff,
};
const GHOST_FALLBACK = 0x8b93a5;
const GHOST_DIM = 0.22;

async function loadPreview() {
  const response = await fetch('/api/preview', { cache: 'no-store' });
  const data = await response.json();
  // A run publishes each phase's poses as it designs them, so this arrives
  // mid-flight. The old tour is gone, but the intent to watch one is not.
  const wasFlying = state.flying;
  stopFlying();
  ghostGroup.clear();
  state.ghostNodes = [];
  for (const group of data.groups || []) {
    const color = GHOST_COLORS[group.phase] ?? GHOST_FALLBACK;
    for (const pose of group.poses || []) {
      const points = (pose.points || []).map((p) => new THREE.Vector3(...p));
      if (points.length < 2) continue;
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(points),
        new THREE.LineBasicMaterial({ color, transparent: true,
                                      opacity: GHOST_DIM }));
      const tip = new THREE.Mesh(ballGeometry, new THREE.MeshBasicMaterial({
        color, transparent: true, opacity: GHOST_DIM + 0.2 }));
      tip.position.copy(points[points.length - 1]);
      tip.scale.setScalar(0.012);
      ghostGroup.add(line, tip);
      state.ghostNodes.push({ phase: group.phase, index: pose.index,
                              pose: pose.pose_deg, line, tip });
    }
  }
  state.flyFrom = 0;
  // The previous loop is unwound by now; starting here rather than leaving it
  // to the next poll keeps the gap to one frame instead of four hundred ms.
  if (wasFlying && state.ghostNodes.length > 1) flyTour();
}

/** Draw the pose being executed solid, so the plan and the arm can be compared. */
function highlightPose(phase, index) {
  for (const node of state.ghostNodes) {
    const on = node.phase === phase && node.index === index;
    node.line.material.opacity = on ? 1.0 : GHOST_DIM;
    node.tip.material.opacity = on ? 1.0 : GHOST_DIM + 0.2;
    node.tip.scale.setScalar(on ? 0.022 : 0.012);
  }
}

/* ---------------- the tour, flown ---------------- */
/* A second arm in its own colour, walked along the planned tour. A rehearsal
 * moves nothing on the bench, so watching the run's own progress counter is
 * not watching the run. This flies the poses the plan contains.
 *
 * Deliberately far faster than the arm: the point is to see the shape of the
 * whole tour in a few seconds, not to sit through it. The pace is set by the
 * tour rather than by the transit, so twice as many poses does not mean twice
 * as long a wait, with a floor so a short tour is still followable.
 *
 * Every intermediate configuration comes from the server's forward
 * kinematics, along the straight joint-space line the arm is commanded to
 * take, so the animation cannot show a path the screen never cleared.
 */

const FLY_COLOR = 0xff9f43;
const FLY_TOUR_MS = 8000;      // about how long one lap takes to watch
const FLY_MIN_MS = 110;        // floor, so a long tour is quick but not a blur
const FLY_MAX_MS = 420;        // ceiling, so a short tour is not a crawl
const FLY_FRAME_MS = 16;       // one sample per display frame, no more

function buildFlier() {
  if (state.flierNodes.size || !state.linkNodes.size) return;
  const paint = new THREE.MeshStandardMaterial({
    color: FLY_COLOR, transparent: true, opacity: 0.55,
    metalness: 0.1, roughness: 0.6, depthWrite: false,
  });
  for (const [name, source] of state.linkNodes) {
    const node = source.clone(true);
    node.traverse((child) => { if (child.isMesh) child.material = paint; });
    node.matrixAutoUpdate = false;
    node.visible = false;
    flierGroup.add(node);
    state.flierNodes.set(name, node);
  }
}

function placeFlier(frames) {
  buildFlier();
  for (const [name, node] of state.flierNodes) {
    const flat = frames?.[name];
    if (flat) node.matrix.fromArray(flat).transpose();
    // Only the arm being flown: the rest of the robot is already drawn where
    // it is, and a second copy of it in orange is just clutter.
    node.visible = !!flat;
  }
}

async function transit(from, to, steps) {
  const answer = await fetch('/api/kinematics', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ from_deg: from, to_deg: to, steps }),
  }).then((r) => r.json()).catch(() => ({ ok: false }));
  return answer.ok ? answer.frames : null;
}

/** Play one transit over `ms` of wall clock, however often frames arrive.
 *
 * Which sample is shown comes from the clock, not from a count of ticks. A
 * browser that has stopped compositing the tab throttles both timers and
 * animation frames hard, and a tour paced by counting ticks does not then run
 * slowly, it crawls -- measured here at a sixteenth speed. Paced by the clock,
 * throttling costs smoothness and never duration.
 */
function playTransit(frames, ms, mine) {
  return new Promise((done) => {
    const started = performance.now();
    const last = frames.length - 1;
    const tick = () => {
      if (!state.flying || state.flight !== mine) return done();
      const share = Math.min(1, (performance.now() - started) / ms);
      placeFlier(frames[Math.round(share * last)]);
      if (share >= 1) return done();
      return setTimeout(tick, FLY_FRAME_MS);
    };
    tick();
  });
}

/** Walk the planned poses in order until told to stop or the list runs out.
 *
 * Numbered, because a plan republished mid-flight starts a new tour while the
 * old loop is still unwinding, and whichever finished last would otherwise
 * decide whether anything was flying at all.
 */
async function flyTour() {
  const poses = state.ghostNodes.map((node) => node.pose);
  if (poses.length < 2) return;
  const segment = Math.min(FLY_MAX_MS,
                           Math.max(FLY_MIN_MS, FLY_TOUR_MS / poses.length));
  const steps = Math.max(3, Math.round(segment / FLY_FRAME_MS));
  const mine = ++state.flight;
  state.flying = true;
  let at = state.flyFrom;
  // Held here so the next transit is fetched while this one is being watched.
  let ahead = transit(poses[at], poses[(at + 1) % poses.length], steps);
  while (state.flying && state.flight === mine) {
    const frames = await ahead;
    const next = (at + 1) % poses.length;
    ahead = transit(poses[next], poses[(next + 1) % poses.length], steps);
    if (!frames || !frames.length) break;
    window.__dash?.onFlying?.(next + 1, poses.length);
    await playTransit(frames, segment, mine);
    at = next;
    state.flyFrom = at;
  }
  if (state.flight !== mine) return;
  state.flying = false;
  placeFlier(null);
  window.__dash?.onFlying?.(0, poses.length);
}

function stopFlying() {
  // Puts the arm away here rather than leaving it to the loop's own tail:
  // several loops can be unwinding at once once a plan has been republished,
  // and "stopped" has to mean "gone from the canvas" whichever one gets there
  // first. Deliberately not bumping the flight number -- only a replacement
  // flight does that, and that is what makes the one it replaced stand down.
  state.flying = false;
  placeFlier(null);
  window.__dash?.onFlying?.(0, state.ghostNodes.length);
}

/* ---------------- poll ---------------- */

let inFlight = false;
async function poll() {
  if (inFlight) return;
  inFlight = true;
  try {
    const response = await fetch('/api/viewer', { cache: 'no-store' });
    const data = await response.json();
    if (!data.have_model) return;
    if (data.visuals && data.visuals.length !== state.linkNodes.size) {
      await buildLinks(data.visuals);
    }
    if (data.visuals && data.visuals.length === 0) {
      document.getElementById('loading')?.classList.add('hidden');
    }
    syncFrames(data.link_tf);
    const links = data.gravity?.links || [];
    // Rebuilt only when the masses themselves change, not every poll: the
    // markers are static geometry, it is the kinematics under them that moves.
    const key = links.map((item) => `${item.link}:${item.mass_kg}`).join('|');
    if (key !== state.massKey) {
      state.massKey = key;
      buildGravity(links);
    }
    syncGravity();
    if (data.preview_token !== state.previewToken) {
      state.previewToken = data.preview_token;
      await loadPreview();
    }
    state.obstacles = data.obstacles || [];
    syncObstacles(state.obstacles);
    if (data.moving) highlightPose(data.moving.phase, data.moving.index);
    window.__dash?.onViewer?.(data);
  } catch (error) {
    /* the panel already reports connection health */
  } finally {
    inFlight = false;
  }
}
setInterval(poll, POLL_MS);
poll();

/* ---------------- options ---------------- */

const bind = (id, key2, after) => {
  const element = document.getElementById(id);
  if (!element) return;
  element.addEventListener('change', () => {
    state[key2] = element.checked;
    if (after) after();
  });
};
bind('show-mesh', 'showMesh');
bind('show-frames', 'showFrames');
bind('show-labels', 'showLabels');
bind('show-ghost', 'showGhost', () => { ghostGroup.visible = state.showGhost; });
ghostGroup.visible = state.showGhost;
bind('show-gravity', 'showGravity', () => setGravityVisible(state.showGravity));
setGravityVisible(state.showGravity);

for (const [id, mode] of [['gizmo-move', 'translate'],
                          ['gizmo-rotate', 'rotate'],
                          ['gizmo-scale', 'scale']]) {
  document.getElementById(id)?.addEventListener('click', () => {
    gizmo.setMode(mode);
    for (const other of ['gizmo-move', 'gizmo-rotate', 'gizmo-scale']) {
      document.getElementById(other)?.classList.toggle('sel', other === id);
    }
  });
}
gizmo.setMode('translate');

/* ---------------- render ---------------- */

function frame() {
  requestAnimationFrame(frame);
  orbit.update();
  renderer.render(scene, camera);
  if (state.showGravity) placeLabels();
}
frame();
