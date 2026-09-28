/* dexcore viewer -- side-by-side source vs target, per frame.
 *
 * THE VIEWER DOES NO KINEMATICS. The server sends, per frame, the world position and quaternion of
 * every object part and every hand link, already composed (see scene.py). This file only copies
 * those into three.js objects. No hinge, no FK, no articulation angle, no wxyz quaternion -- so a
 * convention bug cannot originate in the browser.
 *
 * LOADING A TARGET IS ENOUGH. A generated demo records the source it came from and the absolute
 * frames it covers, so the left pane is fetched automatically at the matching window. No one has
 * to remember which file, or which frame range, went into a run.
 *
 * THE TWO PANES ARE ALIGNED BY ABSOLUTE SOURCE FRAME, never by row. A 200-frame reconstruction and
 * its 20-frame keyframe set are both valid things to put on the left, and row 5 of one is not row 5
 * of the other. A frame the left pane does not cover is hidden and said so, not held stale.
 *
 * THE SCENE IS Z-UP. ARCTIC world coordinates put the table at z ~ 1.0-1.4 m; three.js defaults to
 * Y-up. Left at the default the scene renders on its side AND OrbitControls locks its azimuth to
 * the wrong axis, which is what makes rotation feel stuck.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TrackballControls } from 'three/addons/controls/TrackballControls.js';

const $ = (id) => document.getElementById(id);
const UP = new THREE.Vector3(0, 0, 1);

/* ------------------------------------------------------------------ data helpers */
function unpack(o) {
  const bin = atob(o.b64);
  const buf = new ArrayBuffer(bin.length);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return {
    data: o.dtype === 'float32' ? new Float32Array(buf)
        : o.dtype === 'uint32'  ? new Uint32Array(buf)
        : new Uint8Array(buf),
    shape: o.shape,
  };
}

function makeMesh(payload, material) {
  const v = unpack(payload.verts), f = unpack(payload.faces);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(v.data, 3));
  g.setIndex(new THREE.BufferAttribute(f.data, 1));
  g.computeVertexNormals();
  return new THREE.Mesh(g, material);
}

const MAT = () => ({
  objTop:  new THREE.MeshStandardMaterial({ color: 0xffa040, roughness: 0.75 }),
  objBot:  new THREE.MeshStandardMaterial({ color: 0xd9822b, roughness: 0.8 }),
  left:    new THREE.MeshStandardMaterial({ color: 0x4c9aff, roughness: 0.6, transparent: true, opacity: 0.9 }),
  right:   new THREE.MeshStandardMaterial({ color: 0x3ddc97, roughness: 0.6, transparent: true, opacity: 0.9 }),
  contact: new THREE.MeshStandardMaterial({ color: 0xff6b5e, roughness: 0.45, emissive: 0x501510 }),
});
const KP_COLOR = { left: 0x9ecbff, right: 0xa8f0cf };

/* ------------------------------------------------------------------ one pane */
class View {
  constructor(hostId) {
    this.host = $(hostId);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.host.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x14161a);
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    this.camera.up.copy(UP);

    this.scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x20242c, 2.0));
    const k = new THREE.DirectionalLight(0xffffff, 1.7);
    k.position.set(1.2, -1.6, 1.4);
    this.scene.add(k);

    this.orbit = new OrbitControls(this.camera, this.renderer.domElement);
    this.orbit.enableDamping = true;
    this.orbit.dampingFactor = 0.08;
    this.orbit.minPolarAngle = 0;
    this.orbit.maxPolarAngle = Math.PI;      // over the top and underneath
    this.orbit.screenSpacePanning = true;

    this.trackball = new TrackballControls(this.camera, this.renderer.domElement);
    this.trackball.rotateSpeed = 3.0;
    this.trackball.dynamicDampingFactor = 0.15;
    this.trackball.enabled = false;

    this.controls = this.orbit;
    this.mat = MAT();
    this.grid = null;
    this.S = null;
    this.G = null;
    new ResizeObserver(() => this.resize()).observe(this.host);
  }

  setFreeRotate(free) {
    const t = this.controls.target.clone();
    this.orbit.enabled = !free;
    this.trackball.enabled = free;
    this.controls = free ? this.trackball : this.orbit;
    this.controls.target.copy(t);
    if (!free) this.camera.up.copy(UP);      // orbit needs an up; trackball has none
    this.controls.update();
  }

  dispose() {
    if (!this.G) return;
    this.scene.remove(this.G.root);
    this.G.root.traverse((o) => { if (o.geometry) o.geometry.dispose(); });
    this.G = null;
    this.S = null;
  }

  build(sd) {
    this.dispose();
    this.S = sd;
    const root = new THREE.Group();
    const parts = [], hands = {};

    for (const p of sd.object.parts) {
      const m = makeMesh(p.mesh, p.name === 'top' ? this.mat.objTop : this.mat.objBot);
      root.add(m);
      parts.push({ mesh: m, pos: unpack(p.pos), quat: unpack(p.quat) });
    }

    for (const side of ['left', 'right']) {
      const h = sd.hands[side];
      if (!h) continue;
      const group = new THREE.Group(); root.add(group);
      const baseMat = side === 'left' ? this.mat.left : this.mat.right;
      const links = h.meshes.map((mp) => { const m = makeMesh(mp, baseMat); group.add(m); return m; });

      const kp = new THREE.InstancedMesh(new THREE.SphereGeometry(0.0055, 10, 8),
        new THREE.MeshStandardMaterial({ color: KP_COLOR[side] }), 21);
      kp.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      group.add(kp);

      const skelGeo = new THREE.BufferGeometry();
      skelGeo.setAttribute('position',
        new THREE.BufferAttribute(new Float32Array(sd.skeleton.length * 6), 3));
      const skel = new THREE.LineSegments(skelGeo,
        new THREE.LineBasicMaterial({ color: KP_COLOR[side] }));
      group.add(skel);

      const kpts = unpack(h.keypoints);
      const tp = new Float32Array(sd.num_frames * 3);
      for (let t = 0; t < sd.num_frames; t++) {
        const o = (t * 21 + 20) * 3;
        tp[t * 3] = kpts.data[o]; tp[t * 3 + 1] = kpts.data[o + 1]; tp[t * 3 + 2] = kpts.data[o + 2];
      }
      const trailGeo = new THREE.BufferGeometry();
      trailGeo.setAttribute('position', new THREE.BufferAttribute(tp, 3));
      const trail = new THREE.Line(trailGeo,
        new THREE.LineBasicMaterial({ color: KP_COLOR[side], transparent: true, opacity: 0.5 }));
      group.add(trail);

      hands[side] = { group, links, kp, skel, trail, pos: unpack(h.pos), quat: unpack(h.quat),
                      kpts, contact: unpack(h.contact), names: h.links, baseMat };
    }

    this.scene.add(root);
    this.G = { root, parts, hands };
    this.rowOf = new Map();
    sd.frame_index.forEach((f, i) => this.rowOf.set(f, i));
    this.resize();
  }

  frameCamera(c, r) {
    r = Math.max(r, 0.25);
    this.orbit.target.set(c[0], c[1], c[2]);
    this.trackball.target.set(c[0], c[1], c[2]);
    // -y is "in front of" the scene and +z is above it, in this Z-up world
    this.camera.position.set(c[0] + r * 0.6, c[1] - r * 1.5, c[2] + r * 0.9);
    this.camera.up.copy(UP);
    this.camera.lookAt(c[0], c[1], c[2]);
    this.camera.near = r / 100; this.camera.far = r * 60;
    this.camera.updateProjectionMatrix();
    this.rebuildGrid(c, r);
  }

  rebuildGrid(centre, radius) {
    if (this.grid) { this.scene.remove(this.grid); this.grid.geometry.dispose(); }
    const span = Math.max(radius * 2.4, 0.5);
    this.grid = new THREE.GridHelper(span, 12, 0x323945, 0x252a32);
    this.grid.rotation.x = Math.PI / 2;      // GridHelper lies in XZ; this ground plane is XY
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.35;
    this.grid.position.set(centre[0], centre[1], centre[2] - radius * 0.95);
    this.grid.visible = $('t-grid').checked;
    this.scene.add(this.grid);
  }

  /* pose by ABSOLUTE source frame; -> {row, touching} or null when this pane has no such frame */
  pose(absFrame, opts) {
    if (!this.G) return null;
    const t = this.rowOf.has(absFrame) ? this.rowOf.get(absFrame) : -1;
    this.G.root.visible = t >= 0;
    if (this.grid) this.grid.visible = $('t-grid').checked;
    if (t < 0) return null;

    for (const p of this.G.parts) {
      p.mesh.position.set(p.pos.data[t * 3], p.pos.data[t * 3 + 1], p.pos.data[t * 3 + 2]);
      p.mesh.quaternion.set(p.quat.data[t * 4], p.quat.data[t * 4 + 1],
                            p.quat.data[t * 4 + 2], p.quat.data[t * 4 + 3]);
      p.mesh.visible = opts.object;
    }

    const touching = [];
    for (const side of ['left', 'right']) {
      const h = this.G.hands[side];
      if (!h) continue;
      h.group.visible = opts[side];
      const L = h.names.length;
      for (let i = 0; i < L; i++) {
        const po = (t * L + i) * 3, qo = (t * L + i) * 4;
        h.links[i].position.set(h.pos.data[po], h.pos.data[po + 1], h.pos.data[po + 2]);
        h.links[i].quaternion.set(h.quat.data[qo], h.quat.data[qo + 1],
                                  h.quat.data[qo + 2], h.quat.data[qo + 3]);
        const on = h.contact.data[t * L + i] === 1;
        h.links[i].material = (on && opts.contact) ? this.mat.contact : h.baseMat;
        h.links[i].visible = opts.hand;
        if (on) touching.push(`${side[0]}:${h.names[i]}`);
      }
      for (let j = 0; j < 21; j++) {
        const o = (t * 21 + j) * 3;
        _v.set(h.kpts.data[o], h.kpts.data[o + 1], h.kpts.data[o + 2]);
        h.kp.setMatrixAt(j, _m.compose(_v, _q.identity(), _s));
      }
      h.kp.instanceMatrix.needsUpdate = true;
      h.kp.visible = opts.kp;

      const sp = h.skel.geometry.attributes.position;
      this.S.skeleton.forEach(([a, b], k) => {
        const oa = (t * 21 + a) * 3, ob = (t * 21 + b) * 3;
        sp.array[k * 6]     = h.kpts.data[oa];
        sp.array[k * 6 + 1] = h.kpts.data[oa + 1];
        sp.array[k * 6 + 2] = h.kpts.data[oa + 2];
        sp.array[k * 6 + 3] = h.kpts.data[ob];
        sp.array[k * 6 + 4] = h.kpts.data[ob + 1];
        sp.array[k * 6 + 5] = h.kpts.data[ob + 2];
      });
      sp.needsUpdate = true;
      h.skel.visible = opts.skel;
      h.trail.visible = opts.trail;
    }
    return { row: t, touching };
  }

  resize() {
    const w = this.host.clientWidth, h = this.host.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.trackball.handleResize();
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  render() { this.controls.update(); this.renderer.render(this.scene, this.camera); }
}

const _q = new THREE.Quaternion(), _v = new THREE.Vector3(),
      _s = new THREE.Vector3(1, 1, 1), _m = new THREE.Matrix4();

/* ------------------------------------------------------------------ app state */
const A = new View('viewA');       // left  = source (or a chosen comparison)
const B = new View('viewB');       // right = target, and the one that drives the timeline
let frame = 0, playing = false, lastT = 0, manualLeft = null;

function opts() {
  return { object: $('t-object').checked, hand: $('t-hand').checked, kp: $('t-kp').checked,
           skel: $('t-skel').checked, contact: $('t-contact').checked, trail: $('t-trail').checked,
           left: $('t-left').checked, right: $('t-right').checked };
}

function keypointDelta(absFrame) {
  if (!A.G || !B.G) return null;
  const ta = A.rowOf.get(absFrame), tb = B.rowOf.get(absFrame);
  if (ta === undefined || tb === undefined) return null;
  let sum = 0, n = 0, max = 0;
  for (const side of ['left', 'right']) {
    const a = A.G.hands[side], b = B.G.hands[side];
    if (!a || !b) continue;
    for (let j = 0; j < 21; j++) {
      const oa = (ta * 21 + j) * 3, ob = (tb * 21 + j) * 3;
      const d = Math.hypot(a.kpts.data[oa] - b.kpts.data[ob],
                           a.kpts.data[oa + 1] - b.kpts.data[ob + 1],
                           a.kpts.data[oa + 2] - b.kpts.data[ob + 2]);
      sum += d; n++; if (d > max) max = d;
    }
  }
  return n ? { mean: (sum / n) * 1000, max: max * 1000 } : null;
}

function apply(t) {
  if (!B.S) return;
  frame = t;
  const abs = B.S.frame_index[t];
  const o = opts();
  const rb = B.pose(abs, o);
  const ra = A.pose(abs, o);

  const d = keypointDelta(abs);
  const isKey = B.S.selected_frames.includes(t);
  $('frameno').textContent = `${t} / ${B.S.num_frames - 1}`;
  $('hud').textContent =
    `frame ${t}  (source ${abs})   t=${(t / B.S.fps).toFixed(3)}s` +
    (isKey ? '   [SELECTED KEYFRAME]' : '') +
    (d ? `   Δ mean ${d.mean.toFixed(2)} mm  max ${d.max.toFixed(2)} mm`
       : (A.G && !ra ? '   [left pane has no frame here]' : ''));
  const touching = (rb && rb.touching) || [];
  $('contactlist').innerHTML = touching.length
    ? touching.map((x) => `<span class="on">${x}</span>`).join('  ')
    : '<span>no contact</span>';
}

/* ------------------------------------------------------------------ timeline ticks */
function drawTicks() {
  const cv = $('ticks'), ctx = cv.getContext('2d');
  const w = cv.clientWidth, h = cv.height;
  cv.width = w * devicePixelRatio; cv.height = h * devicePixelRatio;
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!B.S || !B.S.selected_frames.length) return;
  ctx.fillStyle = '#ff6b5e';
  for (const f of B.S.selected_frames) {
    ctx.fillRect((f / (B.S.num_frames - 1)) * (w - 2), 3, 1.5, h - 6);
  }
}

/* ------------------------------------------------------------------ loading */
const showError = (m) => { const e = $('err'); e.hidden = false; e.textContent = m; };
const clearError = () => { $('err').hidden = true; };

async function loadJSON(url, o) {
  const r = await fetch(url, o);
  const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

function windowQuery() {
  const a = $('fstart').value.trim(), b = $('fend').value.trim();
  return (a ? `&start=${encodeURIComponent(a)}` : '') + (b ? `&end=${encodeURIComponent(b)}` : '');
}

function setSplit(on) {
  $('panes').classList.toggle('single', !on);
  A.resize(); B.resize();
}

/* Load the LEFT pane: whatever the user picked, else the source the target itself recorded. */
async function loadLeft(sd) {
  const src = sd.source;
  if (manualLeft) {
    const j = await loadJSON('/api/scene?path=' + encodeURIComponent(manualLeft));
    A.build(j);
    $('labelA').textContent = `left · ${j.name} (manual)`;
    $('ovinfo').innerHTML = `<b>${j.name}</b> · ${j.num_frames} frames (manual)`;
    return true;
  }
  if (!src || !src.available) {
    A.dispose();
    $('ovinfo').textContent = src
      ? 'this IS a source demo — nothing to compare against'
      : 'no source recorded in this file';
    setSplit(false);
    return false;
  }
  const j = await loadJSON('/api/scene?path=' + encodeURIComponent(src.path)
                           + `&start=${src.start}&end=${src.end}`);
  A.build(j);
  $('labelA').textContent = `source · ${src.sequence} [${src.object}] frames ${src.start}-${src.end}`;
  $('ovinfo').innerHTML =
    `<b>${src.sequence}</b> · auto-loaded from the target's own record<br>frames ${src.start}–${src.end}`;
  return true;
}

async function loadDemo(sd) {
  B.build(sd);
  $('labelB').textContent = `target · ${sd.name}`;
  const c = sd.camera.centre, r = sd.camera.radius;
  B.frameCamera(c, r);

  let haveLeft = false;
  try { haveLeft = await loadLeft(sd); }
  catch (e) { showError('left pane: ' + e); A.dispose(); }
  if (haveLeft) { A.frameCamera(c, r); setSplit($('t-split').checked); }

  $('frame').max = sd.num_frames - 1;
  $('frame').value = 0;
  frame = 0;
  $('meta').textContent = sd.summary;
  $('prov').textContent = JSON.stringify(sd.provenance, null, 1);
  drawTicks();
  apply(0);
}

async function refreshList() {
  try {
    const { demos } = await loadJSON('/api/demos');
    for (const id of ['demos', 'ovdemos']) {
      const sel = $(id);
      sel.innerHTML = '';
      if (!demos.length) {
        sel.innerHTML = '<option value="">(no .npy found — run scripts/synthesize.py)</option>';
        continue;
      }
      if (id === 'ovdemos') sel.appendChild(new Option('— auto: recorded source —', ''));
      for (const d of demos) {
        sel.appendChild(new Option(
          `${d.group}/${d.rel}${d.label ? '  · ' + d.label : ''}`, d.path));
      }
    }
  } catch (e) { showError(String(e)); }
}

async function loadSelected() {
  const p = $('demos').value;
  if (!p) return;
  clearError();
  $('load').textContent = 'loading…';
  try { await loadDemo(await loadJSON('/api/scene?path=' + encodeURIComponent(p) + windowQuery())); }
  catch (e) { showError(String(e)); }
  finally { $('load').textContent = 'load'; }
}

/* ------------------------------------------------------------------ wiring */
$('load').addEventListener('click', loadSelected);
$('demos').addEventListener('change', loadSelected);
for (const id of ['fstart', 'fend'])
  $(id).addEventListener('keydown', (e) => { if (e.key === 'Enter') loadSelected(); });

$('ovload').addEventListener('click', async () => {
  manualLeft = $('ovdemos').value || null;
  if (B.S) { clearError(); try { await loadDemo(B.S); } catch (e) { showError(String(e)); } }
});
$('ovclear').addEventListener('click', async () => {
  manualLeft = null; $('ovdemos').value = '';
  if (B.S) { clearError(); try { await loadDemo(B.S); } catch (e) { showError(String(e)); } }
});

$('file').addEventListener('change', async (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  clearError();
  const fd = new FormData(); fd.append('file', f, f.name);
  try { await loadDemo(await loadJSON('/api/upload?x=1' + windowQuery(), { method: 'POST', body: fd })); }
  catch (e) { showError(String(e)); }
  ev.target.value = '';
});

$('t-split').addEventListener('change', (e) => setSplit(e.target.checked && !!A.G));
$('t-free').addEventListener('change', (e) => { A.setFreeRotate(e.target.checked); B.setFreeRotate(e.target.checked); });
for (const id of ['t-object', 't-hand', 't-kp', 't-skel', 't-contact', 't-trail',
                  't-left', 't-right', 't-grid'])
  $(id).addEventListener('change', () => apply(frame));

$('frame').addEventListener('input', (e) => apply(+e.target.value));
$('play').addEventListener('click', () => {
  playing = !playing;
  $('play').textContent = playing ? '❚❚' : '▶';
  lastT = performance.now();
});
addEventListener('keydown', (e) => {
  if (['SELECT', 'INPUT'].includes(e.target.tagName)) return;
  if (e.code === 'Space') { e.preventDefault(); $('play').click(); }
  const n = B.S ? B.S.num_frames - 1 : 0;
  if (e.code === 'ArrowLeft')  { $('frame').value = Math.max(0, frame - 1); apply(+$('frame').value); }
  if (e.code === 'ArrowRight') { $('frame').value = Math.min(n, frame + 1); apply(+$('frame').value); }
});
addEventListener('resize', () => { A.resize(); B.resize(); drawTicks(); });

/* B is the pane the user drives; A mirrors it while the cameras are linked. */
function syncCameras() {
  if (!$('t-link').checked || !A.G) return;
  A.camera.position.copy(B.camera.position);
  A.camera.quaternion.copy(B.camera.quaternion);
  A.camera.up.copy(B.camera.up);
  A.camera.zoom = B.camera.zoom;
  A.camera.updateProjectionMatrix();
  A.controls.target.copy(B.controls.target);
}

function tick(now) {
  requestAnimationFrame(tick);
  if (playing && B.S) {
    const speed = parseFloat($('speed').value);
    if (now - lastT >= 1000 / (B.S.fps * speed)) {
      lastT = now;
      const next = (frame + 1) % B.S.num_frames;
      $('frame').value = next;
      apply(next);
    }
  }
  B.render();
  syncCameras();
  A.render();
}

setSplit(false);
requestAnimationFrame(tick);
refreshList();
