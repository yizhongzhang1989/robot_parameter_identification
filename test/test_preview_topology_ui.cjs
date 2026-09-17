const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname,
  '../robot_parameter_identification/dashboard/static/viewer.js'), 'utf8');
class Vector3 {
  constructor(...values) { this.values = values; }
  copy(other) { this.values = [...other.values]; return this; }
}
class Material {
  constructor(options) {
    Object.assign(this, options);
    this.color = { value: options.color, setHex(value) { this.value = value; } };
  }
}
class Geometry {
  setFromPoints(points) { this.points = points; return this; }
}
class RenderObject {
  constructor(geometry, material) {
    this.geometry = geometry;
    this.material = material;
    this.position = new Vector3();
    this.scale = { value: 0, setScalar(value) { this.value = value; } };
  }
}
let preview = { groups: [] };
const context = vm.createContext({
  THREE: { Vector3, BufferGeometry: Geometry, Line: RenderObject, Mesh: RenderObject,
    LineBasicMaterial: Material, MeshBasicMaterial: Material },
  ballGeometry: {},
  state: { ghostNodes: [], flying: false },
  ghostGroup: { children: [], clear() { this.children = []; },
    add(...objects) { this.children.push(...objects); } },
  stopFlying() {}, fitScene() {}, flyTour() {},
  fetch: async (route, options) => {
    assert.equal(route, '/api/preview');
    assert.equal(options.cache, 'no-store');
    return { json: async () => preview };
  },
});
vm.runInContext(source.slice(source.indexOf('const GHOST_COLORS'),
  source.indexOf('/* ---------------- the tour')), context);

async function main() {
  const paths = [
    [[0, 0.2, 0], [0, 0.2, 0.5]],
    [[0, -0.2, 0], [0.3, -0.2, 0.4]],
  ];
  preview = { groups: [{ phase: 'A_gravity', poses: [{
    index: 1, pose_deg: [10, 20], paths, points: [[99, 99, 99], [100, 100, 100]],
  }] }] };
  await vm.runInContext('loadPreview()', context);
  assert.equal(context.state.ghostNodes.length, 1);
  const node = context.state.ghostNodes[0];
  assert.equal(node.parts.length, 2);
  assert.equal(context.ghostGroup.children.length, 4);
  node.parts.forEach((part, index) => {
    assert.deepEqual(part.line.geometry.points.map(point => point.values), paths[index]);
    assert.deepEqual(part.tip.position.values, paths[index].at(-1));
  });
  vm.runInContext("highlightPose('A_gravity', 1)", context);
  for (const part of node.parts) {
    assert.equal(part.line.material.opacity, 1);
    assert.equal(part.tip.scale.value, 0.024);
  }
  vm.runInContext("highlightPose('', 0, { A_gravity: [1] })", context);
  assert.ok(node.parts.every(part => part.line.material.opacity === 0.68));
  preview = { groups: [{ phase: 'D_validation', poses: [
    { index: 1, pose_deg: [30], points: paths[0] },
    { index: 2, pose_deg: [40], points: paths[1] },
  ] }] };
  await vm.runInContext('loadPreview()', context);
  assert.equal(context.state.ghostNodes.length, 2);
  assert.ok(context.state.ghostNodes.every(pose => pose.parts.length === 1));
  preview = { available: false };
  await vm.runInContext('loadPreview()', context);
  assert.equal(context.state.ghostNodes.length, 0);
  assert.equal(context.ghostGroup.children.length, 0);
  console.log('PASS: controller paths, separate tips, one tour entry per pose, legacy points, and invalidation');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
