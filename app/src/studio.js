import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

/** Visual-only scene: it makes no inferences about workouts or muscle targets. */
export async function createStudio(host) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'low-power' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
  renderer.setClearColor(0x141411, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const canvas = renderer.domElement;
  canvas.tabIndex = 0;
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', 'Interactive graphite human figure. Use left and right arrow keys to rotate, plus or minus to zoom.');
  host.append(canvas);

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x141411, 0.09);
  const pmrem = new THREE.PMREMGenerator(renderer);
  const room = new RoomEnvironment();
  const environment = pmrem.fromScene(room, 0.04);
  scene.environment = environment.texture;
  scene.environmentIntensity = 0.35;
  room.dispose();
  pmrem.dispose();

  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 40);
  camera.position.set(0.4, 1.22, 5.3);
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0, 1.12, 0);
  controls.enablePan = false;
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.minDistance = 3.3;
  controls.maxDistance = 7;
  controls.minPolarAngle = Math.PI * 0.33;
  controls.maxPolarAngle = Math.PI * 0.58;
  controls.autoRotateSpeed = 0.35;
  // One-finger page scrolling on phones; use two fingers or view buttons to orbit.
  controls.touches.ONE = null;
  controls.touches.TWO = THREE.TOUCH.DOLLY_ROTATE;

  scene.add(new THREE.HemisphereLight(0xcdd5bb, 0x292015, 1.3));
  const key = new THREE.DirectionalLight(0xffddba, 2.6);
  key.position.set(-2, 3.8, 3);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  key.shadow.camera.left = -2;
  key.shadow.camera.right = 2;
  key.shadow.camera.top = 3;
  key.shadow.camera.bottom = -1;
  key.shadow.normalBias = 0.025;
  key.shadow.bias = -0.0002;
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xff8d4a, 4.5);
  rim.position.set(1.8, 2.6, -2);
  scene.add(rim);
  const fill = new THREE.DirectionalLight(0xa8baac, 1.4);
  fill.position.set(2.5, 1.6, 3);
  scene.add(fill);
  const groundLight = new THREE.PointLight(0xed8f48, 1.8, 3.5, 2);
  groundLight.position.set(-0.4, 0.2, 0.4);
  scene.add(groundLight);

  const ground = new THREE.Mesh(new THREE.PlaneGeometry(30, 30), new THREE.MeshStandardMaterial({
    color: 0x191c14, roughness: 0.95, metalness: 0.15, transparent: true, opacity: 0.55,
  }));
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.025;
  ground.receiveShadow = true;
  scene.add(ground);
  const grid = new THREE.GridHelper(8, 24, 0x536144, 0x36432d);
  grid.material.transparent = true;
  grid.material.opacity = 0.12;
  grid.position.y = -0.02;
  scene.add(grid);
  const pedestal = new THREE.Mesh(new THREE.CylinderGeometry(0.53, 0.57, 0.025, 80), new THREE.MeshStandardMaterial({
    color: 0x23271c, metalness: 0.4, roughness: 0.65,
  }));
  pedestal.position.y = -0.004;
  pedestal.receiveShadow = true;
  scene.add(pedestal);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.55, 0.0025, 6, 96), new THREE.MeshBasicMaterial({ color: 0xa27d4c, transparent: true, opacity: 0.5 }));
  ring.rotation.x = Math.PI / 2;
  ring.position.y = 0.008;
  scene.add(ring);

  let figure;
  let frame = 0;
  let previousTime = 0;
  let disposed = false;
  let visible = true;
  let transition = null;
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const api = {
    canvas,
    rotating: false,
    onMotionChange: null,
    onViewChange: null,
    onContextLost: null,
    setMotion(active) {
      api.rotating = Boolean(active);
      controls.autoRotate = api.rotating;
      api.onMotionChange?.(api.rotating);
      invalidate();
    },
    setView(view) {
      api.setMotion(false);
      const angle = view === 'back' ? Math.PI + 0.14 : 0.14;
      const target = new THREE.Vector3(Math.sin(angle) * 5.3, 1.22, Math.cos(angle) * 5.3);
      if (reducedMotion.matches) {
        camera.position.copy(target);
        controls.update();
      } else {
        transition = { start: camera.position.clone(), end: target, at: performance.now() };
      }
      api.onViewChange?.(view);
      invalidate();
    },
    reset() {
      controls.target.set(0, 1.12, 0);
      api.setView('front');
    },
    setLighting(mode) {
      const warm = mode === 'warm';
      key.color.set(warm ? 0xffddba : 0xe0e8eb);
      rim.color.set(warm ? 0xff8d4a : 0xa8bbca);
      groundLight.color.set(warm ? 0xed8f48 : 0xa5b9cb);
      ring.material.color.set(warm ? 0xa27d4c : 0x9cabb6);
      invalidate();
    },
    dispose,
  };

  function invalidate() {
    if (!frame && !disposed && visible && !document.hidden) frame = requestAnimationFrame(render);
  }
  function render(now) {
    frame = 0;
    const delta = Math.min((now - previousTime) / 1000 || 0, 0.05);
    previousTime = now;
    if (transition) {
      const fraction = THREE.MathUtils.clamp((now - transition.at) / 650, 0, 1);
      const eased = 1 - Math.pow(1 - fraction, 3);
      camera.position.lerpVectors(transition.start, transition.end, eased);
      if (fraction >= 1) transition = null;
    }
    const changed = controls.update(delta);
    renderer.render(scene, camera);
    if (api.rotating || transition || changed) invalidate();
  }
  function resize() {
    const width = Math.max(host.clientWidth, 1);
    const height = Math.max(host.clientHeight, 1);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    const shift = window.innerWidth > 760 ? -width * 0.13 : -width * 0.04;
    camera.setViewOffset(width, height, shift, 0, width, height);
    camera.updateProjectionMatrix();
    invalidate();
  }
  function onInteractionStart() {
    transition = null;
    api.setMotion(false);
    api.onViewChange?.('custom');
    canvas.classList.add('dragging');
  }
  function onInteractionEnd() { canvas.classList.remove('dragging'); }
  function onKeyboard(event) {
    if (!['ArrowLeft', 'ArrowRight', '+', '=', '-', 'Home'].includes(event.key)) return;
    event.preventDefault();
    api.setMotion(false);
    transition = null;
    if (event.key === 'Home') return api.reset();
    const direction = camera.position.clone().sub(controls.target);
    if (event.key.startsWith('Arrow')) {
      direction.applyAxisAngle(new THREE.Vector3(0, 1, 0), event.key === 'ArrowLeft' ? -0.15 : 0.15);
    } else {
      const distance = direction.length();
      direction.setLength(THREE.MathUtils.clamp(distance + (event.key === '-' ? 0.25 : -0.25), controls.minDistance, controls.maxDistance));
    }
    camera.position.copy(controls.target).add(direction);
    api.onViewChange?.('custom');
    invalidate();
  }
  function onPreferenceChange() {
    if (reducedMotion.matches) {
      transition = null;
      api.setMotion(false);
    }
  }
  function onVisibilityChange() {
    if (document.hidden && frame) { cancelAnimationFrame(frame); frame = 0; }
    else invalidate();
  }
  function onContextLost(event) {
    event.preventDefault();
    api.setMotion(false);
    if (frame) { cancelAnimationFrame(frame); frame = 0; }
    api.onContextLost?.();
  }
  function dispose() {
    if (disposed) return;
    disposed = true;
    cancelAnimationFrame(frame);
    observer.disconnect();
    intersection.disconnect();
    controls.dispose();
    reducedMotion.removeEventListener('change', onPreferenceChange);
    document.removeEventListener('visibilitychange', onVisibilityChange);
    canvas.removeEventListener('keydown', onKeyboard);
    canvas.removeEventListener('webglcontextlost', onContextLost);
    const geometries = new Set();
    const materials = new Set();
    scene.traverse((object) => {
      if (object.geometry) geometries.add(object.geometry);
      if (object.material) (Array.isArray(object.material) ? object.material : [object.material]).forEach((material) => materials.add(material));
      if (object.isSkinnedMesh) object.skeleton.dispose();
    });
    geometries.forEach((geometry) => geometry.dispose());
    materials.forEach((material) => material.dispose());
    environment.dispose();
    renderer.dispose();
    canvas.remove();
  }
  const observer = new ResizeObserver(resize);
  const intersection = new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    if (visible) invalidate();
    else if (frame) { cancelAnimationFrame(frame); frame = 0; }
  });
  observer.observe(host);
  intersection.observe(host);
  controls.addEventListener('change', invalidate);
  controls.addEventListener('start', onInteractionStart);
  controls.addEventListener('end', onInteractionEnd);
  canvas.addEventListener('keydown', onKeyboard);
  canvas.addEventListener('webglcontextlost', onContextLost);
  reducedMotion.addEventListener('change', onPreferenceChange);
  document.addEventListener('visibilitychange', onVisibilityChange);
  resize();

  try {
    const loaded = await new GLTFLoader().loadAsync('/models/studio-figure.glb');
    figure = loaded.scene;
    // Lower the source's modelling A-pose into a relaxed studio stance.
    // This is a static display pose, not an exercise animation.
    const bones = new Map();
    figure.traverse((object) => {
      if (object.isBone) bones.set(object.name.replace(/[^a-z0-9]/gi, ''), object);
    });
    figure.updateMatrixWorld(true);
    for (const side of ['L', 'R']) {
      const upper = bones.get(`upperarm01${side}`);
      const elbow = bones.get(`lowerarm01${side}`);
      if (!upper || !elbow) continue;
      const origin = upper.getWorldPosition(new THREE.Vector3());
      const direction = elbow.getWorldPosition(new THREE.Vector3()).sub(origin).normalize();
      const relaxed = new THREE.Vector3(Math.sign(direction.x) * 0.24, -0.97, 0).normalize();
      const delta = new THREE.Quaternion().setFromUnitVectors(direction, relaxed);
      const desired = delta.multiply(upper.getWorldQuaternion(new THREE.Quaternion()));
      const parent = upper.parent.getWorldQuaternion(new THREE.Quaternion()).invert();
      upper.quaternion.copy(parent.multiply(desired));
      figure.updateMatrixWorld(true);
    }
    // Keep all source surfaces: the body and high-poly detail meshes are separate.
    const meshes = [];
    figure.traverse((object) => { if (object.isMesh) meshes.push(object); });
    const graphite = new THREE.MeshPhysicalMaterial({ color: 0x383a35, roughness: 0.64, metalness: 0.38, clearcoat: 0.05 });
    const discardedMaterials = new Set();
    meshes.forEach((mesh) => {
      (Array.isArray(mesh.material) ? mesh.material : [mesh.material]).forEach((material) => discardedMaterials.add(material));
      mesh.material = graphite;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    });
    discardedMaterials.forEach((material) => {
      Object.values(material).forEach((value) => { if (value?.isTexture) value.dispose(); });
      material.dispose();
    });
    figure.updateMatrixWorld(true);
    const bounds = new THREE.Box3().setFromObject(figure);
    const size = bounds.getSize(new THREE.Vector3());
    const scale = 2.28 / size.y;
    const center = bounds.getCenter(new THREE.Vector3());
    const model = new THREE.Group();
    figure.position.set(-center.x, -bounds.min.y, -center.z);
    model.add(figure);
    model.scale.setScalar(scale);
    model.rotation.y = 0.12;
    model.position.y = 0.011;
    scene.add(model);
    host.dataset.loaded = 'true';
    api.setMotion(!reducedMotion.matches);
    invalidate();
    return api;
  } catch (error) {
    dispose();
    throw error;
  }
}
