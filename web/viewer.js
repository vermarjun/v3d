// Three.js scene viewer: loading, photo-accurate rendering, camera motion, walk mode, grid, dimensions.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { computeBoundsTree, disposeBoundsTree, acceleratedRaycast } from 'three-mesh-bvh';

THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;

export const ACCENT = 0xffb547;
const EYE_HEIGHT = 1.6;

// ---------- shaders ----------
// Vertex colours are sRGB photo colours stored as bytes. Decode to linear here; the renderer's
// sRGB output encoding (colorspace_fragment) turns them back into exactly the original values.
const SRGB_DECODE = /* glsl */`
vec3 srgbToLinear(vec3 c) {
  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}`;
const FOG_UNIFORMS = /* glsl */`
uniform vec3 uFogColor; uniform float uFogNear; uniform float uFogFar; uniform float uFogMax;
uniform float uSat; uniform float uContrast;
vec3 grade(vec3 c, float dist) {
  float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
  c = mix(vec3(l), c, uSat);
  c = max(vec3(0.0), (c - 0.18) * uContrast + 0.18);
  return mix(c, uFogColor, smoothstep(uFogNear, uFogFar, dist) * uFogMax);
}`;

const MESH_VS = /* glsl */`
attribute vec4 aColor;
varying vec3 vColor; varying float vDist;
${SRGB_DECODE}
void main() {
  vColor = srgbToLinear(aColor.rgb);
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vDist = length(mv.xyz);
  gl_Position = projectionMatrix * mv;
}`;
const MESH_FS = /* glsl */`
varying vec3 vColor; varying float vDist;
${FOG_UNIFORMS}
void main() {
  gl_FragColor = vec4(grade(vColor, vDist), 1.0);
  #include <colorspace_fragment>
}`;

// Splat-like points: perspective-sized discs (world size = local vertex spacing), Gaussian falloff,
// alpha-to-coverage so the soft edges need no sorting and still write depth.
const POINTS_VS = /* glsl */`
attribute vec4 aColor; attribute float aSize;
uniform float uProj; uniform float uSizeScale; uniform float uMaxSize;
varying vec3 vColor; varying float vDist;
${SRGB_DECODE}
void main() {
  if (aSize <= 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  float s = length(modelViewMatrix[0].xyz);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = clamp(aSize * s * uSizeScale * uProj / max(-mv.z, 1e-3), 1.0, uMaxSize);
  vColor = srgbToLinear(aColor.rgb);
  vDist = length(mv.xyz);
}`;
const POINTS_FS = /* glsl */`
uniform float uSharp;
varying vec3 vColor; varying float vDist;
${FOG_UNIFORMS}
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(c, c);
  if (r2 > 1.0) discard;
  // Opaque core, soft rim: neighbours overlap fully, the rim is resolved by alpha-to-coverage.
  float a = 1.0 - smoothstep(uSharp, 1.0, r2);
  gl_FragColor = vec4(grade(vColor, vDist), a);
  #include <colorspace_fragment>
}`;

const GRID_VS = /* glsl */`
varying vec3 vWorld;
void main() {
  vec4 w = modelMatrix * vec4(position, 1.0);
  vWorld = w.xyz;
  gl_Position = projectionMatrix * viewMatrix * w;
}`;
const GRID_FS = /* glsl */`
uniform vec3 uColor; uniform float uFade; uniform float uOpacity;
varying vec3 vWorld;
float gridLine(vec2 p, float step) {
  vec2 g = p / step;
  vec2 d = abs(fract(g - 0.5) - 0.5) / fwidth(g);
  return 1.0 - min(min(d.x, d.y), 1.0);
}
void main() {
  float major = gridLine(vWorld.xz, 1.0);
  float minor = gridLine(vWorld.xz, 0.1) * 0.35;
  float dist = length(vWorld - cameraPosition);
  float fade = 1.0 - smoothstep(uFade * 0.35, uFade, dist);
  float a = max(major, minor) * fade * uOpacity;
  if (a < 0.003) discard;
  gl_FragColor = vec4(uColor, a);
  #include <colorspace_fragment>
}`;

// ---------- helpers ----------
const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);

export function formatLength(m) {
  if (!isFinite(m)) return '–';
  if (Math.abs(m) >= 1) return `${m.toFixed(2)} m`;
  if (Math.abs(m) >= 0.1) return `${(m * 100).toFixed(1)} cm`;
  return `${(m * 100).toFixed(1)} cm`;
}

// Filters flying-pixel triangles and computes a per-vertex splat size (mean of the kept incident edges).
function prepareGeometry(geom, edgeRatio) {
  const pos = geom.attributes.position.array;
  const n = pos.length / 3;
  let index = geom.index ? geom.index.array : null;
  if (!index) {
    index = new Uint32Array(n);
    for (let i = 0; i < n; i++) index[i] = i;
  }
  const minE = new Float32Array(n).fill(Infinity);
  const triCount = index.length / 3;
  const edges = new Float32Array(triCount * 3);
  const len = (a, b) => {
    const dx = pos[3 * a] - pos[3 * b], dy = pos[3 * a + 1] - pos[3 * b + 1], dz = pos[3 * a + 2] - pos[3 * b + 2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  };
  for (let t = 0; t < triCount; t++) {
    const a = index[3 * t], b = index[3 * t + 1], c = index[3 * t + 2];
    const ab = len(a, b), bc = len(b, c), ca = len(c, a);
    edges[3 * t] = ab; edges[3 * t + 1] = bc; edges[3 * t + 2] = ca;
    if (ab < minE[a]) minE[a] = ab; if (ab < minE[b]) minE[b] = ab;
    if (bc < minE[b]) minE[b] = bc; if (bc < minE[c]) minE[c] = bc;
    if (ca < minE[c]) minE[c] = ca; if (ca < minE[a]) minE[a] = ca;
  }
  const kept = new Uint32Array(index.length);
  const sum = new Float32Array(n);
  const cnt = new Uint16Array(n);
  let k = 0;
  const nrm = new Float64Array(3);
  const azBins = new Float64Array(36);
  for (let t = 0; t < triCount; t++) {
    const a = index[3 * t], b = index[3 * t + 1], c = index[3 * t + 2];
    const ab = edges[3 * t], bc = edges[3 * t + 1], ca = edges[3 * t + 2];
    const maxE = Math.max(ab, bc, ca);
    const ref = Math.max(minE[a], minE[b], minE[c]);
    if (!(maxE <= edgeRatio * ref) || maxE === 0) continue;
    kept[k++] = a; kept[k++] = b; kept[k++] = c;
    // Area-weighted face normal. Depth-map triangles all share one winding relative to their
    // camera, so the sum points back toward the side the video was shot from.
    const ux = pos[3 * b] - pos[3 * a], uy = pos[3 * b + 1] - pos[3 * a + 1], uz = pos[3 * b + 2] - pos[3 * a + 2];
    const vx = pos[3 * c] - pos[3 * a], vy = pos[3 * c + 1] - pos[3 * a + 1], vz = pos[3 * c + 2] - pos[3 * a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    nrm[0] += nx; nrm[1] += ny; nrm[2] += nz;
    // Area histogram of wall-like faces by azimuth: the biggest wall faces the camera.
    const h2 = nx * nx + nz * nz;
    if (ny * ny < 0.25 * h2) azBins[(Math.floor((Math.atan2(nx, nz) + Math.PI) / (2 * Math.PI) * 36) + 36) % 36] += Math.sqrt(h2 + ny * ny);
    const m = (ab + bc + ca) / 3;
    sum[a] += m; sum[b] += m; sum[c] += m;
    cnt[a]++; cnt[b]++; cnt[c]++;
  }
  const size = new Float32Array(n);
  for (let i = 0; i < n; i++) size[i] = cnt[i] ? sum[i] / cnt[i] : 0;
  const IndexArray = n > 65535 ? Uint32Array : Uint16Array;
  geom.setIndex(new THREE.BufferAttribute(new IndexArray(kept.subarray(0, k)), 1));
  geom.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
  return { triangles: triCount, kept: k / 3, normal: new THREE.Vector3(nrm[0], nrm[1], nrm[2]), azBins };
}

function percentile(sorted, p) {
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(p * (sorted.length - 1))))];
}

// ---------- viewer ----------
export class Viewer {
  constructor(container) {
    this.container = container;
    this.listeners = new Set();
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance', preserveDrawingBuffer: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.renderer.domElement.className = 'gl';
    container.appendChild(this.renderer.domElement);

    this.labels = new CSS2DRenderer();
    this.labels.domElement.className = 'labels';
    container.appendChild(this.labels.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.01, 2000);
    this.root = new THREE.Group();
    this.root.name = 'root';
    this.content = new THREE.Group();
    this.root.add(this.content);
    this.scene.add(this.root);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.zoomToCursor = true;
    this.controls.screenSpacePanning = true;
    this.controls.rotateSpeed = 0.7;
    this.controls.zoomSpeed = 0.9;

    this.meshes = [];
    this.points = [];
    this.mode = 'mesh';
    this.scale = 1;
    this.anim = null;
    this.walk = null;
    this.keys = new Set();
    this.clock = new THREE.Clock();
    this.uniforms = {
      uFogColor: { value: new THREE.Color(0x202020) },
      uFogNear: { value: 50 }, uFogFar: { value: 100 }, uFogMax: { value: 0.0 },
      uSat: { value: 1.0 }, uContrast: { value: 1.0 },
    };
    this.pointUniforms = {
      ...this.uniforms,
      uProj: { value: 500 }, uSizeScale: { value: 1.9 }, uMaxSize: { value: 64 }, uSharp: { value: 0.45 },
    };
    const range = this.renderer.getContext().getParameter(this.renderer.getContext().ALIASED_POINT_SIZE_RANGE);
    this.pointUniforms.uMaxSize.value = Math.min(96, range ? range[1] : 64);

    this.raycaster = new THREE.Raycaster();
    this.raycaster.firstHitOnly = true;

    this._onResize = () => this.resize();
    window.addEventListener('resize', this._onResize);
    this._bindInput();
    this.resize();
    this.renderer.setAnimationLoop(() => this._frame());
  }

  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  emit(type, data) { for (const fn of this.listeners) fn(type, data); }

  resize() {
    const w = this.container.clientWidth || window.innerWidth;
    const h = this.container.clientHeight || window.innerHeight;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = `${w}px`;
    this.renderer.domElement.style.height = `${h}px`;
    this.labels.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    for (const m of this.lineMaterials()) m.resolution.set(w, h);
  }

  lineMaterials() {
    const out = [];
    this.scene.traverse((o) => { if (o.material && o.material.isLineMaterial) out.push(o.material); });
    return out;
  }

  // ---------- loading ----------
  async load({ glbUrl, meta, onProgress }) {
    this.meta = meta;
    this.raw = !!meta.raw;
    const loader = new GLTFLoader();
    loader.setMeshoptDecoder(MeshoptDecoder);
    const gltf = await loader.loadAsync(glbUrl, (e) => {
      if (onProgress && e.total) onProgress(e.loaded / e.total);
    });
    onProgress && onProgress(1);
    gltf.scene.updateMatrixWorld(true);

    const meshMat = new THREE.ShaderMaterial({
      uniforms: this.uniforms, vertexShader: MESH_VS, fragmentShader: MESH_FS,
      side: THREE.DoubleSide, polygonOffset: true,
    });
    this.meshMaterial = meshMat;
    this.pointsMaterial = new THREE.ShaderMaterial({
      uniforms: this.pointUniforms, vertexShader: POINTS_VS, fragmentShader: POINTS_FS,
      alphaToCoverage: true, transparent: false, depthWrite: true,
    });

    let tris = 0, kept = 0, verts = 0, i = 0;
    this.normalSum = new THREE.Vector3();
    this.azBins = new Float64Array(36);
    const t0 = performance.now();
    gltf.scene.traverse((o) => {
      if (!o.isMesh && !o.isPoints) return;
      const g = o.geometry.clone();
      g.applyMatrix4(o.matrixWorld);
      const color = g.getAttribute('color');
      if (color) g.setAttribute('aColor', color);
      else g.setAttribute('aColor', new THREE.Float32BufferAttribute(new Float32Array(g.attributes.position.count * 4).fill(0.8), 4));
      verts += g.attributes.position.count;
      if (o.isMesh) {
        const r = prepareGeometry(g, this.raw ? 6 : 10);
        tris += r.triangles; kept += r.kept;
        this.normalSum.add(r.normal);
        for (let b = 0; b < 36; b++) this.azBins[b] += r.azBins[b];
        // Deterministic per-layer depth offset: overlapping per-frame depth maps resolve the same way every frame (no shimmer).
        const mat = meshMat.clone();
        mat.uniforms = this.uniforms;
        mat.polygonOffsetFactor = 0;
        mat.polygonOffsetUnits = i * 2;
        const mesh = new THREE.Mesh(g, mat);
        mesh.name = `layer_${i}`;
        this.content.add(mesh);
        this.meshes.push(mesh);
      }
      const pg = new THREE.BufferGeometry();
      pg.setAttribute('position', g.attributes.position);
      pg.setAttribute('aColor', g.attributes.aColor);
      pg.setAttribute('aSize', g.attributes.aSize || new THREE.Float32BufferAttribute(new Float32Array(g.attributes.position.count).fill(0.01), 1));
      const pts = new THREE.Points(pg, this.pointsMaterial);
      pts.frustumCulled = false;
      pts.visible = false;
      pts.raycast = () => {};
      this.content.add(pts);
      this.points.push(pts);
      i++;
    });
    this.stats = { vertices: verts, triangles: tris, keptTriangles: kept, prepareMs: Math.round(performance.now() - t0) };
    console.info('[video3d] geometry', JSON.stringify(this.stats), 'normalSum', this.normalSum.clone().normalize().toArray().map((x) => x.toFixed(3)).join(','));

    this._analyse();
    this._setupBackground();
    this._setupGrid();
    this.setScale(meta.scale_factor || 1, { keepView: false });
    this.setMode(this.mode);
    this._computeHomeView();
    this.playIntro();
    // BVH after the first frames so the intro starts immediately.
    this.bvhReady = new Promise((resolve) => {
      setTimeout(() => {
        const tb = performance.now();
        for (const m of this.meshes) m.geometry.computeBoundsTree({ maxLeafTris: 8 });
        this.stats.bvhMs = Math.round(performance.now() - tb);
        console.info('[video3d] bvh ms', this.stats.bvhMs);
        this.emit('bvh');
        resolve();
      }, 250);
    });
  }

  // Robust bbox (percentiles), average colour, centre.
  _analyse() {
    const xs = [], ys = [], zs = [];
    let r = 0, g = 0, b = 0, n = 0;
    for (const m of this.meshes.length ? this.meshes : this.points) {
      const pos = m.geometry.attributes.position.array;
      const col = m.geometry.attributes.aColor;
      const size = m.geometry.attributes.aSize?.array;
      const count = pos.length / 3;
      const step = Math.max(1, Math.floor(count / 40000));
      for (let i = 0; i < count; i += step) {
        if (size && size[i] <= 0) continue;
        xs.push(pos[3 * i]); ys.push(pos[3 * i + 1]); zs.push(pos[3 * i + 2]);
        r += col.getX(i); g += col.getY(i); b += col.getZ(i); n++;
      }
    }
    const sortNum = (a) => Float64Array.from(a).sort();
    const sx = sortNum(xs), sy = sortNum(ys), sz = sortNum(zs);
    const lo = this.raw ? 0.01 : 0.002, hi = 1 - lo;
    this.robustBox = new THREE.Box3(
      new THREE.Vector3(percentile(sx, lo), percentile(sy, lo), percentile(sz, lo)),
      new THREE.Vector3(percentile(sx, hi), percentile(sy, hi), percentile(sz, hi)),
    );
    const m = this.meta;
    this.box = (m.bbox_min && m.bbox_max && !this.raw)
      ? new THREE.Box3(new THREE.Vector3(...m.bbox_min), new THREE.Vector3(...m.bbox_max))
      : this.robustBox.clone();
    this.avgColor = n ? [r / n, g / n, b / n] : [0.3, 0.3, 0.3]; // sRGB 0..1
    this.radius = this.robustBox.getSize(new THREE.Vector3()).length() / 2;
    this.floorY = (m.floor_y !== null && m.floor_y !== undefined && !this.raw) ? m.floor_y : null;
    // Indoor: the video was shot from inside the reconstruction (a room walkthrough), not at an object.
    const src = this._sourceCamera();
    const rb = this.robustBox;
    this.indoor = !!src && src.pos.x > rb.min.x && src.pos.x < rb.max.x && src.pos.z > rb.min.z && src.pos.z < rb.max.z;
  }

  _setupBackground() {
    const [r, g, b] = this.avgColor;
    const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    // Keep the hue of the scene, but sit a little darker than it so scan borders recede.
    // Indoors, unseen gaps behind furniture show the background: a dark one reads as shadow, grey as cut-outs.
    const k = Math.min(1, (this.indoor ? 0.13 : 0.32) / Math.max(lum, 0.05));
    const mid = [r * k, g * k, b * k].map((v) => Math.min(1, v));
    const inner = mid.map((v) => Math.min(1, v * 1.35 + 0.02));
    const outer = mid.map((v) => v * 0.28);
    const css = (c, a = 1) => `rgba(${c.map((v) => Math.round(v * 255)).join(',')},${a})`;
    const cv = document.createElement('canvas');
    cv.width = 1024; cv.height = 1024;
    const ctx = cv.getContext('2d');
    const grad = ctx.createRadialGradient(512, 430, 40, 512, 512, 760);
    grad.addColorStop(0, css(inner));
    grad.addColorStop(0.55, css(mid));
    grad.addColorStop(1, css(outer));
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 1024, 1024);
    // Fine noise against gradient banding on projectors.
    const img = ctx.getImageData(0, 0, 1024, 1024);
    for (let i = 0; i < img.data.length; i += 4) {
      const d = (Math.random() - 0.5) * 3;
      img.data[i] += d; img.data[i + 1] += d; img.data[i + 2] += d;
    }
    ctx.putImageData(img, 0, 0);
    const tex = new THREE.CanvasTexture(cv);
    tex.colorSpace = THREE.SRGBColorSpace;
    this.scene.background = tex;
    this.uniforms.uFogColor.value.setRGB(mid[0], mid[1], mid[2], THREE.SRGBColorSpace);
    this.bgColors = { inner, mid, outer };
    document.documentElement.style.setProperty('--scene-bg', css(mid));
  }

  _setupGrid() {
    const mat = new THREE.ShaderMaterial({
      uniforms: { uColor: { value: new THREE.Color(0xffffff) }, uFade: { value: 12 }, uOpacity: { value: 0.32 } },
      vertexShader: GRID_VS, fragmentShader: GRID_FS,
      transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -8,
      extensions: { derivatives: true },
    });
    this.grid = new THREE.Mesh(new THREE.PlaneGeometry(400, 400).rotateX(-Math.PI / 2), mat);
    this.grid.renderOrder = 2;
    this.grid.visible = false;
    this.grid.raycast = () => {};
    this.scene.add(this.grid);
  }

  _updateGrid() {
    if (!this.grid) return;
    const y = (this.floorY ?? this.robustBox.min.y) * this.scale + 0.004;
    this.grid.position.set(0, y, 0);
    this.grid.material.uniforms.uFade.value = Math.max(6, this.radius * this.scale * 2.2);
  }

  setGrid(on) { this.grid.visible = on; this.emit('change'); }

  // ---------- render style ----------
  setMode(mode) {
    if (mode === 'photoreal' && !this.splat) mode = 'mesh';
    this.mode = mode;
    // Hidden meshes still take raycasts, so measuring works in every mode. In Photoreal the mesh is
    // hidden outright: as a depth-only occluder it would cull the splats that sit on its surface.
    for (const m of this.meshes) m.visible = mode === 'mesh';
    for (const p of this.points) p.visible = mode === 'points';
    if (this.splat) this.splat.setVisible(mode === 'photoreal');
    this.emit('change');
  }

  async mountSplat() {
    if (!this.meta.splat) return false;
    try {
      const m = await import('./splat.js');
      this.splat = await m.mountSplat({ THREE, scene: this.scene, root: this.root, camera: this.camera, renderer: this.renderer, meta: this.meta });
      this.splat.setVisible(this.mode === 'photoreal');
      this.emit('change');
      return true;
    } catch (err) {
      console.warn('[video3d] splat unavailable', err);
      return false;
    }
  }

  // ---------- scale ----------
  setScale(s, { keepView = true } = {}) {
    const ratio = s / this.scale;
    this.scale = s;
    this.root.scale.setScalar(s);
    this.root.updateMatrixWorld(true);
    if (keepView && isFinite(ratio) && ratio !== 1) {
      this.camera.position.multiplyScalar(ratio);
      this.controls.target.multiplyScalar(ratio);
      if (this.home) { this.home.pos.multiplyScalar(ratio); this.home.target.multiplyScalar(ratio); }
    }
    this.camera.near = Math.max(0.005, 0.01 * s);
    this.camera.far = Math.max(200, this.radius * s * 30);
    this.camera.updateProjectionMatrix();
    const R = this.radius * s;
    this.uniforms.uFogNear.value = R * 1.4;
    this.uniforms.uFogFar.value = R * 4.5;
    this.uniforms.uFogMax.value = 0.6;
    this.controls.maxDistance = R * 8;
    this.controls.minDistance = 0.02 * s;
    this._updateGrid();
    this.emit('scale', s);
    this.emit('change');
  }

  // ---------- picking ----------
  pick(clientX, clientY) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(ndc, this.camera);
    return this.raycastWorld(this.raycaster.ray.origin, this.raycaster.ray.direction);
  }

  raycastWorld(origin, dir) {
    this.raycaster.set(origin, dir.clone().normalize());
    this.raycaster.near = 0; this.raycaster.far = Infinity;
    const hits = this.raycaster.intersectObjects(this.meshes, false);
    if (!hits.length) return null;
    hits.sort((a, b) => a.distance - b.distance);
    const h = hits[0];
    return { world: h.point.clone(), local: this.root.worldToLocal(h.point.clone()), distance: h.distance };
  }

  // ---------- camera ----------
  _sourceCamera() {
    const cams = this.meta.cameras || [];
    if (cams.length) {
      return { pos: new THREE.Vector3(...cams[0].position), look: new THREE.Vector3(...cams[0].look_at) };
    }
    if (this.raw) return { pos: new THREE.Vector3(0, 0, 0), look: new THREE.Vector3(0, 0, -1) };
    return null;
  }


  _computeHomeView() {
    const center = this.robustBox.getCenter(new THREE.Vector3());
    const size = this.robustBox.getSize(new THREE.Vector3());
    const src = this._sourceCamera();
    let target, sph;
    if (src && this.indoor) {
      // Room walkthrough: stand where the video started, a touch higher, turned 20° towards the
      // dominant wall (more of the room in frame), looking slightly down. Stay close to the
      // recorded viewpoint: stepping away exposes the gaps behind furniture.
      const fwd = src.look.clone().sub(src.pos).setY(0).normalize();
      const right = new THREE.Vector3(-fwd.z, 0, fwd.x);
      const side = Math.sign(right.dot(this._dominantWallNormal().negate())) || 1;
      const yaw = THREE.MathUtils.degToRad(20) * side, pitch = THREE.MathUtils.degToRad(-10);
      const dir = fwd.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), -yaw);
      dir.multiplyScalar(Math.cos(pitch)).setY(Math.sin(pitch)).normalize();
      const pos = src.pos.clone().addScaledVector(fwd, -0.25);
      if (this.floorY !== null) pos.y = Math.max(pos.y, this.floorY + 1.2);
      const hit = this._rayLocal(pos, dir);
      const depth = THREE.MathUtils.clamp(hit ? hit.distanceTo(pos) : 3, 1.5, 4);
      target = pos.clone().addScaledVector(dir, depth);
      this.home = { pos: pos.multiplyScalar(this.scale), target: target.multiplyScalar(this.scale) };
      return;
    }
    if (src) {
      // Aim where the first recorded frame looked, at the depth of the surface it saw.
      const dir = src.look.clone().sub(src.pos).normalize();
      const hit = this._rayLocal(src.pos, dir);
      const depth = hit ? hit.distanceTo(src.pos) : center.distanceTo(src.pos);
      target = src.pos.clone().addScaledVector(dir, Math.min(depth, this.radius * 1.5) * 0.8);
      sph = new THREE.Spherical().setFromVector3(src.pos.clone().sub(target));
      sph.radius *= 1.12; // a step behind the recorded viewpoint shows a little more context
    } else {
      // No camera path: the summed face normal points back to where the video was shot from.
      const dirH = this._dominantWallNormal();
      target = center.clone();
      if (this.floorY !== null) target.y = this.floorY + Math.min(size.y * 0.35, 1.2);
      const fit = (Math.max(size.x, size.z) * 0.5) / Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
      sph = new THREE.Spherical().setFromVector3(dirH.multiplyScalar(Math.max(fit * 0.85, 1)));
      sph.phi = THREE.MathUtils.degToRad(80);
    }
    if (this.floorY !== null && target.y < this.floorY + 0.3) target.y = this.floorY + 0.3;
    // A 3/4 view: swing round a little and look slightly down, staying near the recorded viewpoint
    // (that is where the reconstruction is complete and looks most like the photo).
    sph.theta += THREE.MathUtils.degToRad(src ? 22 : -14);
    sph.phi = THREE.MathUtils.clamp(sph.phi - THREE.MathUtils.degToRad(10), 0.35, Math.PI / 2 - 0.02);
    sph.radius = Math.max(sph.radius, 0.5);
    const pos = target.clone().add(new THREE.Vector3().setFromSpherical(sph));
    this.home = { pos: pos.multiplyScalar(this.scale), target: target.multiplyScalar(this.scale) };
  }

  // Facing direction of the largest wall-like surface (smoothed azimuth histogram), oriented
  // to agree with the summed normal. Falls back to the summed normal, then +Z.
  _dominantWallNormal() {
    const n = this.normalSum;
    let best = -1, bestW = 0;
    for (let b = 0; b < 36; b++) {
      const w = this.azBins[(b + 35) % 36] * 0.5 + this.azBins[b] + this.azBins[(b + 1) % 36] * 0.5;
      if (w > bestW) { bestW = w; best = b; }
    }
    const dir = new THREE.Vector3(n.x, 0, n.z);
    if (best >= 0) {
      const az = ((best + 0.5) / 36) * 2 * Math.PI - Math.PI;
      const wall = new THREE.Vector3(Math.sin(az), 0, Math.cos(az));
      if (dir.lengthSq() === 0 || wall.dot(dir) > -0.2) return wall;
    }
    return dir.lengthSq() > 1e-12 ? dir.normalize() : new THREE.Vector3(0, 0, 1);
  }

  // Depth of the surface along a local-space ray, before the BVH exists: take vertices inside a
  // narrow cone and use a low percentile of their distances (robust to stray points).
  _rayLocal(origin, dir) {
    const cosMax = Math.cos(THREE.MathUtils.degToRad(2));
    const v = new THREE.Vector3();
    const depths = [];
    for (const m of this.meshes) {
      const pos = m.geometry.attributes.position.array;
      const size = m.geometry.attributes.aSize.array;
      for (let i = 0; i < pos.length / 3; i += 7) {
        if (size[i] <= 0) continue;
        v.set(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]).sub(origin);
        const d = v.length();
        if (d > 1e-6 && v.dot(dir) / d > cosMax) depths.push(d);
      }
    }
    if (!depths.length) return null;
    depths.sort((a, b) => a - b);
    return origin.clone().addScaledVector(dir, depths[Math.floor(depths.length * 0.3)]);
  }

  animateTo(pos, target, duration = 1.2, ease = easeInOutCubic) {
    this.cancelAnim();
    const p0 = this.camera.position.clone(), t0 = this.controls.target.clone();
    const s0 = new THREE.Spherical().setFromVector3(p0.clone().sub(t0));
    const s1 = new THREE.Spherical().setFromVector3(pos.clone().sub(target));
    let dTheta = s1.theta - s0.theta;
    if (dTheta > Math.PI) dTheta -= 2 * Math.PI;
    if (dTheta < -Math.PI) dTheta += 2 * Math.PI;
    return new Promise((resolve) => {
      this.anim = {
        t: 0, duration, resolve,
        step: (k) => {
          const e = ease(k);
          const tt = t0.clone().lerp(target, e);
          const s = new THREE.Spherical(
            THREE.MathUtils.lerp(s0.radius, s1.radius, e),
            THREE.MathUtils.lerp(s0.phi, s1.phi, e),
            s0.theta + dTheta * e,
          );
          this.controls.target.copy(tt);
          this.camera.position.copy(tt).add(new THREE.Vector3().setFromSpherical(s));
          this.camera.lookAt(tt);
        },
      };
    });
  }

  cancelAnim() {
    if (this.anim) { const r = this.anim.resolve; this.anim = null; r && r(); }
    if (this.fly) { this.fly = null; this.emit('change'); }
  }

  playIntro() {
    const { pos, target } = this.home;
    const sph = new THREE.Spherical().setFromVector3(pos.clone().sub(target));
    sph.theta -= THREE.MathUtils.degToRad(this.indoor ? 25 : 40);
    sph.phi = Math.max(0.25, sph.phi - THREE.MathUtils.degToRad(this.indoor ? 14 : 18));
    sph.radius *= this.indoor ? 1.5 : 1.8;
    this.controls.target.copy(target);
    this.camera.position.copy(target).add(new THREE.Vector3().setFromSpherical(sph));
    this.camera.lookAt(target);
    return this.animateTo(pos, target, 3.2, easeInOutCubic);
  }

  resetView() {
    if (this.walk) this.exitWalk();
    return this.animateTo(this.home.pos, this.home.target, 1.4);
  }

  flyToPoint(world) {
    const dir = this.camera.position.clone().sub(this.controls.target).normalize();
    const dist = Math.max(this.camera.position.distanceTo(world) * 0.6, 0.25 * this.scale);
    return this.animateTo(world.clone().addScaledVector(dir, dist), world, 1.0);
  }

  get canFlyThrough() { return (this.meta?.cameras || []).length >= 2; }

  flyThrough() {
    const cams = this.meta.cameras || [];
    if (cams.length < 2) return;
    if (this.walk) this.exitWalk();
    this.cancelAnim();
    const s = this.scale;
    const posCurve = new THREE.CatmullRomCurve3(cams.map((c) => new THREE.Vector3(...c.position).multiplyScalar(s)), false, 'centripetal');
    const lookCurve = new THREE.CatmullRomCurve3(cams.map((c) => new THREE.Vector3(...c.look_at).multiplyScalar(s)), false, 'centripetal');
    const duration = Math.max(8, Math.min(30, posCurve.getLength() / (0.6 * s)));
    this.fly = { t: 0, duration, posCurve, lookCurve };
    this.controls.enabled = false;
    this.emit('change');
  }

  // ---------- walk mode ----------
  enterWalk() {
    this.cancelAnim();
    const dir = new THREE.Vector3();
    this.camera.getWorldDirection(dir);
    // World units are metres (root is scaled), so the eye sits 1.6 m above the scaled floor.
    const eyeY = this.floorY !== null ? this.floorY * this.scale + EYE_HEIGHT
      : (this._sourceCamera()?.pos.y ?? this.camera.position.y) * this.scale;
    this.camera.position.y = eyeY;
    const yaw = Math.atan2(-dir.x, -dir.z);
    this.walk = { yaw, pitch: 0, vel: new THREE.Vector3() };
    this._applyWalkLook();
    this.controls.enabled = false;
    try {
      const p = this.renderer.domElement.requestPointerLock?.();
      if (p && p.catch) p.catch(() => {});
    } catch { /* headless or denied: drag-to-look fallback */ }
    this.emit('walk', true);
    this.emit('change');
  }

  exitWalk() {
    if (!this.walk) return;
    this.walk = null;
    if (document.pointerLockElement) document.exitPointerLock();
    const dir = new THREE.Vector3();
    this.camera.getWorldDirection(dir);
    this.controls.target.copy(this.camera.position).addScaledVector(dir, Math.max(1, this.radius * 0.5) * this.scale);
    this.controls.enabled = true;
    this.emit('walk', false);
    this.emit('change');
  }

  _applyWalkLook() {
    const e = new THREE.Euler(this.walk.pitch, this.walk.yaw, 0, 'YXZ');
    this.camera.quaternion.setFromEuler(e);
  }

  _walkStep(dt) {
    const w = this.walk;
    const speed = this.keys.has('ShiftLeft') || this.keys.has('ShiftRight') ? 3.2 : 1.4; // m/s
    const f = new THREE.Vector3(-Math.sin(w.yaw), 0, -Math.cos(w.yaw));
    const r = new THREE.Vector3(-f.z, 0, f.x);
    const want = new THREE.Vector3();
    if (this.keys.has('KeyW') || this.keys.has('ArrowUp')) want.add(f);
    if (this.keys.has('KeyS') || this.keys.has('ArrowDown')) want.sub(f);
    if (this.keys.has('KeyD') || this.keys.has('ArrowRight')) want.add(r);
    if (this.keys.has('KeyA') || this.keys.has('ArrowLeft')) want.sub(r);
    if (this.keys.has('KeyE')) want.y += 1;
    if (this.keys.has('KeyQ')) want.y -= 1;
    if (want.lengthSq() > 0) want.normalize().multiplyScalar(speed);
    w.vel.lerp(want, 1 - Math.exp(-dt * 10));
    this.camera.position.addScaledVector(w.vel, dt);
  }

  // ---------- input ----------
  _bindInput() {
    const el = this.renderer.domElement;
    const stop = () => { if (this.anim || this.fly) { this.cancelAnim(); this.controls.enabled = !this.walk; } };
    el.addEventListener('pointerdown', (e) => {
      stop();
      this._down = { x: e.clientX, y: e.clientY, t: performance.now(), button: e.button };
      if (this.walk && !document.pointerLockElement) el.setPointerCapture(e.pointerId);
    });
    el.addEventListener('pointermove', (e) => {
      if (this.walk) {
        let dx = 0, dy = 0;
        if (document.pointerLockElement === el) {
          dx = e.movementX; dy = e.movementY;
          if (Math.abs(dx) > 250 || Math.abs(dy) > 250) return; // Chrome sometimes reports a bogus jump right after locking
        }
        else if (this._down && e.buttons) { dx = e.movementX; dy = e.movementY; }
        this.walk.yaw -= dx * 0.0022;
        this.walk.pitch = THREE.MathUtils.clamp(this.walk.pitch - dy * 0.0022, -1.45, 1.45);
        this._applyWalkLook();
        return;
      }
      this.emit('hover', e);
    });
    el.addEventListener('pointerup', (e) => {
      const d = this._down;
      this._down = null;
      if (!d || this.walk) return;
      const moved = Math.hypot(e.clientX - d.x, e.clientY - d.y);
      if (moved < 5 && d.button === 0 && performance.now() - d.t < 600) this.emit('click', e);
    });
    el.addEventListener('dblclick', async (e) => {
      if (this.walk || this.measuring) return;
      await this.bvhReady;
      const hit = this.pick(e.clientX, e.clientY);
      if (hit) this.flyToPoint(hit.world);
    });
    el.addEventListener('wheel', stop, { passive: true });
    window.addEventListener('keydown', (e) => {
      if (e.target.closest && e.target.closest('input,textarea')) return;
      this.keys.add(e.code);
      if (e.code === 'Escape' && this.walk) this.exitWalk();
    });
    window.addEventListener('keyup', (e) => this.keys.delete(e.code));
    window.addEventListener('blur', () => this.keys.clear());
    document.addEventListener('pointerlockchange', () => {
      if (!document.pointerLockElement && this.walk && this._hadLock) this.exitWalk();
      this._hadLock = !!document.pointerLockElement;
    });
  }

  // ---------- dimensions ----------
  setDims(on) {
    if (!this.dims) this._buildDims();
    this.dims.visible = on;
    this.emit('change');
  }

  _buildDims() {
    const b = this.box;
    const { min: a, max: c } = b;
    const corners = [
      [a.x, a.y, a.z], [c.x, a.y, a.z], [c.x, a.y, c.z], [a.x, a.y, c.z],
      [a.x, c.y, a.z], [c.x, c.y, a.z], [c.x, c.y, c.z], [a.x, c.y, c.z],
    ];
    const E = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
    const arr = [];
    for (const [i, j] of E) arr.push(...corners[i], ...corners[j]);
    const geom = new LineSegmentsGeometry().setPositions(arr);
    const mat = new LineMaterial({ color: ACCENT, linewidth: 2, transparent: true, opacity: 0.9, depthTest: false, dashed: false });
    mat.resolution.set(this.container.clientWidth, this.container.clientHeight);
    const lines = new LineSegments2(geom, mat);
    lines.renderOrder = 10;
    lines.raycast = () => {};
    this.dims = new THREE.Group();
    this.dims.add(lines);
    this.root.add(this.dims);
    const mk = (cls) => {
      const el = document.createElement('div');
      el.className = `tag dim ${cls}`;
      const o = new CSS2DObject(el);
      this.dims.add(o);
      return o;
    };
    this.dimLabels = [mk('w'), mk('d'), mk('h')];
    this._placeDimLabels();
    this.updateDimLabels();
    this.on((t) => { if (t === 'scale') this.updateDimLabels(); });
  }

  // Labels sit on the far edges (as seen from the camera): those stay on screen and read cleanly
  // even when the camera is inside the box.
  _placeDimLabels() {
    const { min: a, max: c } = this.box;
    const cam = this.root.worldToLocal(this.camera.position.clone());
    const mid = this.box.getCenter(new THREE.Vector3());
    const farZ = cam.z > mid.z ? a.z : c.z;
    const farX = cam.x > mid.x ? a.x : c.x;
    const [w, d, h] = this.dimLabels;
    w.position.set(mid.x, a.y, farZ);
    d.position.set(farX, a.y, mid.z);
    h.position.set(farX, mid.y, farZ);
  }

  dimensions() {
    const s = this.box.getSize(new THREE.Vector3()).multiplyScalar(this.scale);
    return { width: s.x, depth: s.z, height: s.y };
  }

  updateDimLabels() {
    if (!this.dimLabels) return;
    const d = this.dimensions();
    const [w, dd, h] = this.dimLabels;
    w.element.innerHTML = `<b>W</b> ${d.width.toFixed(2)} m`;
    dd.element.innerHTML = `<b>D</b> ${d.depth.toFixed(2)} m`;
    h.element.innerHTML = `<b>H</b> ${d.height.toFixed(2)} m`;
  }

  // ---------- loop ----------
  _frame() {
    const dt = Math.min(this.clock.getDelta(), 0.1);
    if (this.anim) {
      this.anim.t += dt;
      const k = Math.min(1, this.anim.t / this.anim.duration);
      this.anim.step(k);
      if (k >= 1) { const r = this.anim.resolve; this.anim = null; r && r(); }
    } else if (this.fly) {
      const f = this.fly;
      f.t += dt;
      const k = easeInOutCubic(Math.min(1, f.t / f.duration));
      this.camera.position.copy(f.posCurve.getPointAt(k));
      const look = f.lookCurve.getPointAt(k);
      this.camera.lookAt(look);
      this.controls.target.copy(look);
      if (f.t >= f.duration) { this.fly = null; this.controls.enabled = true; this.emit('change'); }
    } else if (this.walk) {
      this._walkStep(dt);
    } else {
      this.controls.update();
    }
    const h = this.renderer.getDrawingBufferSize(new THREE.Vector2()).y;
    this.pointUniforms.uProj.value = h * this.camera.projectionMatrix.elements[5] * 0.5;
    if (this.dims?.visible) this._placeDimLabels();
    this.emit('frame', dt);
    this.renderer.render(this.scene, this.camera);
    this.labels.render(this.scene, this.camera);
  }

  debugHandle() {
    return { scene: this.scene, camera: this.camera, renderer: this.renderer, root: this.root, meta: this.meta, viewer: this, THREE };
  }
}
