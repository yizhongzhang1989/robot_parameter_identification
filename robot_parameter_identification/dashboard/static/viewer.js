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
scene.add(linkGroup, frameGroup, obstacleGroup, ghostGroup);

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
  obstacles: [],
  selected: null,
  showMesh: true,
  showFrames: false,
  showLabels: false,
  showGhost: true,
  meshCache: new Map(),
};

window.__viewer = {
  select: selectObstacle,
  selected: () => state.selected,
  setMode: (mode) => gizmo.setMode(mode),
  frames: () => Object.keys(state.frames),
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
    state.obstacles = data.obstacles || [];
    syncObstacles(state.obstacles);
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
}
frame();
