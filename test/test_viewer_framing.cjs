const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const staticDir = path.resolve(__dirname, '../robot_parameter_identification/dashboard/static');
const source = fs.readFileSync(path.join(staticDir, 'viewer.js'), 'utf8');
const threeSource = fs.readFileSync(path.join(staticDir, 'vendor/three.module.js'), 'utf8');
const threeReady = import(`data:text/javascript;base64,${Buffer.from(threeSource).toString('base64')}`);

test('world geometry and preview bounds fit both FOV axes with overlay padding', async () => {
  const THREE = await threeReady;
  const fitBounds = vm.runInNewContext(
    source.slice(source.indexOf('function fitBounds('), source.indexOf('function fitScene('))
      + '\nfitBounds;', { THREE });
  const model = new THREE.Group();
  model.position.set(0.4, -0.3, 1.1);
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.3, 1.4));
  mesh.position.z = 0.7;
  model.add(mesh);
  const ghost = new THREE.Line(new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-1.2, 0.8, 1.1), new THREE.Vector3(0.9, 0.8, 2.9),
  ]));
  const bounds = new THREE.Box3().setFromObject(model, true).expandByObject(ghost, true);
  assert.ok(bounds.max.z > 2.8);
  for (const aspect of [2.4, 1, 390 / 420, 0.4]) {
    for (const padding of [
      { left: 0.08, right: 0.08, top: 0.3, bottom: 0.34 },
      { left: 0.2, right: 0.06, top: 0.08, bottom: 0.3 },
    ]) {
      const camera = new THREE.PerspectiveCamera(45, aspect, 0.01, 100);
      camera.up.set(0, 0, 1);
      camera.position.set(1.1, -1.1, 0.9);
      const orbit = { target: new THREE.Vector3(0, 0, 0.35) };
      const direction = camera.position.clone().sub(orbit.target).normalize();
      assert.equal(fitBounds(camera, orbit, bounds, padding), true);
      camera.updateMatrixWorld(true);
      assert.ok(camera.position.clone().sub(orbit.target).normalize().distanceTo(direction) < 1e-10);
      for (const horizontal of [bounds.min.x, bounds.max.x]) {
        for (const vertical of [bounds.min.y, bounds.max.y]) {
          for (const depth of [bounds.min.z, bounds.max.z]) {
            const point = new THREE.Vector3(horizontal, vertical, depth).project(camera);
            assert.ok(point.x >= -1 + 2 * padding.left - 1e-9);
            assert.ok(point.x <= 1 - 2 * padding.right + 1e-9);
            assert.ok(point.y >= -1 + 2 * padding.bottom - 1e-9);
            assert.ok(point.y <= 1 - 2 * padding.top + 1e-9);
            assert.ok(point.z > -1 && point.z < 1);
          }
        }
      }
      const position = camera.position.clone();
      fitBounds(camera, orbit, bounds, padding);
      assert.ok(position.distanceTo(camera.position) < 1e-9);
      assert.equal(fitBounds(camera, orbit, new THREE.Box3(), padding), false);
      assert.ok(position.distanceTo(camera.position) < 1e-9);
    }
  }
});

test('initial fit waits for synchronized geometry and does not repeat on telemetry', async () => {
  const THREE = await threeReady;
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  camera.up.set(0, 0, 1);
  camera.position.set(1.1, -1.1, 0.9);
  const orbit = { target: new THREE.Vector3(0, 0, 0.35) };
  const model = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.3, 1.4));
  model.position.z = 1.8;
  const state = {
    modelReady: false, modelFramed: false, ghostNodes: [], frames: {},
    linkNodes: new Map([['arm', model], ['missing', new THREE.Group()]]),
  };
  const fitScene = vm.runInNewContext(
    source.slice(source.indexOf('function fitBounds('), source.indexOf('function resize('))
      + '\nfitScene;', {
      THREE, camera, orbit, state,
      host: { getBoundingClientRect: () => ({ width: 500, height: 500, top: 0, bottom: 500 }) },
      document: { getElementById: () => null },
    });
  assert.equal(fitScene(), false);
  state.modelReady = true;
  assert.equal(fitScene(), false);
  assert.equal(state.modelFramed, false);
  state.frames.arm = new THREE.Matrix4();
  assert.equal(fitScene(), true);
  assert.equal(state.modelFramed, true);
  assert.ok(orbit.target.z > 1.1);
  camera.position.add(new THREE.Vector3(1, 1, 0));
  const userPosition = camera.position.clone();
  model.position.z += 0.2;
  const telemetryFit = source.match(/if \(!state\.modelFramed\) fitScene\(\);/);
  assert.ok(telemetryFit);
  vm.runInNewContext(telemetryFit[0], { state, fitScene });
  assert.deepEqual(camera.position, userPosition);
  state.ghostNodes.push({
    parts: [new THREE.Vector3(3, 3, 4), new THREE.Vector3(-4, -2, 3)].map(endpoint => {
      const tip = new THREE.Mesh(new THREE.SphereGeometry(0.012));
      tip.position.copy(endpoint);
      return {
        line: new THREE.Line(new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(0, 0, 1.1), endpoint,
        ])),
        tip,
      };
    }),
  });
  fitScene();
  assert.ok(camera.position.distanceTo(userPosition) > 0.1);
  const previewPosition = camera.position.clone();
  camera.aspect = 0.4;
  fitScene();
  assert.ok(camera.position.distanceTo(previewPosition) > 0.1);
  camera.updateMatrixWorld(true);
  for (const { tip } of state.ghostNodes[0].parts) {
    const projected = tip.position.clone().project(camera);
    assert.ok(Math.abs(projected.x) < 1 && Math.abs(projected.y) < 1);
    assert.ok(projected.z > -1 && projected.z < 1);
  }
});