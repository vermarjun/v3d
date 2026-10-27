// App shell: gallery + upload, reconstruction progress, and the viewer UI around viewer.js.
import { Viewer, formatLength } from './viewer.js';
import { Measure } from './measure.js';

const $ = (s) => document.querySelector(s);
const params = new URLSearchParams(location.search);
window.video3d = {};

// ---------- icons (inline SVG, lucide-style) ----------
const ICONS = {
  cube: '<path d="M12 2 3 7v10l9 5 9-5V7z"/><path d="m3 7 9 5 9-5M12 12v10"/>',
  upload: '<path d="M12 16V4M6 10l6-6 6 6"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
  back: '<path d="M15 18l-6-6 6-6"/>',
  orbit: '<circle cx="12" cy="12" r="3"/><path d="M3 12c0-3 4-5.5 9-5.5s9 2.5 9 5.5-4 5.5-9 5.5c-2 0-3.8-.4-5.3-1"/><path d="m8 14-1.5 2.5L9 18"/>',
  walk: '<circle cx="13" cy="4" r="2"/><path d="m9 20 3-6 3 3v4M7 12l3-4 4 1 3 3"/><path d="m12 14-1-5"/>',
  ruler: '<path d="M3 17 17 3l4 4L7 21z"/><path d="m7 13 2 2M10 10l2 2M13 7l2 2"/>',
  box: '<path d="M21 8 12 3 3 8v8l9 5 9-5z"/><path d="M3 8l9 5 9-5M12 13v8"/>',
  grid: '<path d="M3 9h18M3 15h18M9 3v18M15 3v18"/><rect x="3" y="3" width="18" height="18" rx="2"/>',
  reset: '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
  play: '<path d="M7 4v16l13-8z"/>',
  check: '<path d="M5 12l5 5L20 7"/>',
  film: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 4v16M17 4v16M3 9h4M3 15h4M17 9h4M17 15h4"/>',
};
const icon = (name) => `<svg class="ic" viewBox="0 0 24 24">${ICONS[name] || ''}</svg>`;
function paintIcons(rootEl = document) {
  for (const el of rootEl.querySelectorAll('[data-icon]')) el.insertAdjacentHTML('afterbegin', icon(el.dataset.icon));
}

function toast(text, ms = 2600) {
  const t = $('#toast');
  t.textContent = text;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.hidden = true; }, ms);
}

function show(id) {
  for (const v of document.querySelectorAll('.view')) v.hidden = v.id !== id;
}

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

// ---------- home ----------
async function showHome() {
  show('home');
  const gallery = $('#gallery');
  gallery.innerHTML = '';
  let scenes = null;
  try {
    scenes = await api('/api/scenes');
  } catch {
    $('#gallery-note').textContent = 'Backend offline';
    const devLink = location.pathname.includes('/web/')
      ? '<br>Dev: <a href="?glb=../fixtures/sample_raw.glb" style="color:var(--accent)">open the raw sample GLB</a>' : '';
    gallery.innerHTML = `<div class="empty">The reconstruction server is not reachable, so there are no scenes to list.
      Start it with <code>./run.sh</code>, or open any GLB directly with <code>?glb=&lt;url&gt;&amp;meta=&lt;url&gt;</code>.${devLink}</div>`;
    return;
  }
  $('#gallery-note').textContent = scenes.length ? `${scenes.length} cached` : '';
  if (!scenes.length) {
    gallery.innerHTML = '<div class="empty">No scenes yet. Drop a video above to build the first one.</div>';
    return;
  }
  for (const s of scenes) {
    const card = document.createElement('button');
    card.className = 'card';
    const date = s.created_at ? new Date(typeof s.created_at === 'number' ? s.created_at * 1000 : s.created_at) : null;
    const status = s.status === 'done' ? '' : `<span class="chip ${s.status === 'failed' ? 'bad' : ''}">${esc(s.status)}</span>`;
    card.innerHTML = `
      <div class="thumb">${s.thumbnail_url ? `<img src="${esc(s.thumbnail_url)}" alt="" loading="lazy">` : icon('cube')}</div>
      <div class="meta"><div class="name">${esc(s.name || s.id)}</div>
        <div class="sub">${date && !isNaN(date) ? date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) : ''}
        ${s.has_splat ? '<span class="chip accent">Photoreal</span>' : ''}${status}</div></div>`;
    card.addEventListener('click', () => { location.hash = `#/scene/${s.id}`; });
    gallery.appendChild(card);
  }
}

function setupDrop() {
  const drop = $('#drop'), file = $('#file'), over = $('#dragover');
  file.addEventListener('change', () => { handleFiles([...file.files]); file.value = ''; });
  let depth = 0;
  const isFile = (e) => [...(e.dataTransfer?.types || [])].includes('Files');
  window.addEventListener('dragenter', (e) => { if (!isFile(e)) return; e.preventDefault(); depth++; over.hidden = false; drop.classList.add('over'); });
  window.addEventListener('dragleave', () => { depth = Math.max(0, depth - 1); if (!depth) { over.hidden = true; drop.classList.remove('over'); } });
  window.addEventListener('dragover', (e) => { if (isFile(e)) e.preventDefault(); });
  window.addEventListener('drop', (e) => {
    if (!isFile(e)) return;
    e.preventDefault(); depth = 0; over.hidden = true; drop.classList.remove('over');
    handleFiles([...e.dataTransfer.files]);
  });
}

const isVideo = (x) => x.type.startsWith('video/') || /\.(mp4|mov|m4v|webm)$/i.test(x.name);
const isPhoto = (x) => /^image\/(jpeg|png)$/.test(x.type) || /\.(jpe?g|png)$/i.test(x.name);

function handleFiles(files) {
  const v = files.find(isVideo);
  if (v) return upload(v);
  const photos = files.filter(isPhoto);
  if (photos.length >= 3) return upload(photos);
  if (files.some((x) => /\.hei[cf]$/i.test(x.name))) return toast('HEIC photos are not supported: export them as JPG');
  toast(photos.length ? 'Add at least 3 photos taken from different angles' : 'Drop a video, or JPG/PNG photos');
}

// ---------- upload + progress ----------
let pollTimer = null, elapsedTimer = null;

function startProgressView({ name, videoUrl, thumbUrl }) {
  show('progress');
  $('#progress-name').textContent = name || '';
  const v = $('#preview-video'), img = $('#preview-img');
  // Portrait phone clips: letterbox instead of cropping away most of the frame.
  v.onloadedmetadata = () => v.classList.toggle('portrait', v.videoHeight > v.videoWidth);
  img.onload = () => img.classList.toggle('portrait', img.naturalHeight > img.naturalWidth);
  if (videoUrl) { v.hidden = false; img.hidden = true; v.src = videoUrl; v.play().catch(() => {}); }
  else if (thumbUrl) { v.hidden = true; img.hidden = false; img.src = thumbUrl; }
  $('#progress-error').hidden = true;
  const t0 = Date.now();
  clearInterval(elapsedTimer);
  elapsedTimer = setInterval(() => {
    const s = Math.floor((Date.now() - t0) / 1000);
    $('#elapsed').textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  }, 500);
  setStage(0, 0, 'Starting…');
}

const STAGE_OF = { queued: 0, uploading: 0, reconstructing: 1, postprocessing: 2, done: 3, failed: -1 };

function setStage(stage, progress, message, failed = false) {
  for (const li of document.querySelectorAll('#stages li')) {
    const i = Number(li.dataset.stage);
    li.classList.toggle('done', i < stage || (stage === 3 && i === 3));
    li.classList.toggle('active', i === stage && stage < 3 && !failed);
    li.classList.toggle('failed', failed && i === stage);
    const ic = li.querySelector('.st-icon');
    ic.innerHTML = li.classList.contains('done') ? icon('check') : '';
  }
  $('#progress-bar').style.width = `${Math.round(Math.max(0, Math.min(1, progress)) * 100)}%`;
  if (message) $('#stage-message').textContent = message;
}

function upload(input) {
  const photos = Array.isArray(input) ? input : null;
  const first = photos ? photos[0] : input;
  const name = photos ? `${photos.length} photos` : first.name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ');
  startProgressView(photos ? { name, thumbUrl: URL.createObjectURL(first) } : { name, videoUrl: URL.createObjectURL(first) });
  const fd = new FormData();
  if (photos) photos.forEach((p) => fd.append('photos', p)); else fd.append('video', first);
  fd.append('name', name);
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/scenes');
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) setStage(0, 0.1 * (e.loaded / e.total), `Uploading ${Math.round((100 * e.loaded) / e.total)}%`);
  };
  xhr.onload = () => {
    if (xhr.status !== 200 && xhr.status !== 202) return failProgress(`Upload failed (${xhr.status}). ${xhr.responseText.slice(0, 200)}`, 0);
    const res = JSON.parse(xhr.responseText);
    if (res.cached || res.status === 'done') {
      setStage(3, 1, 'Already reconstructed. Opening the cached scene.');
      setTimeout(() => { location.hash = `#/scene/${res.id}`; }, 700);
    } else pollScene(res.id);
  };
  xhr.onerror = () => failProgress('Could not reach the server. Is ./run.sh running?', 0);
  xhr.send(fd);
}

function failProgress(msg, stage = 1) {
  clearInterval(elapsedTimer);
  setStage(stage, 1, 'Failed', true);
  const e = $('#progress-error');
  e.textContent = msg;
  e.hidden = false;
}

function pollScene(id) {
  clearTimeout(pollTimer);
  const tick = async () => {
    let s;
    try { s = await api(`/api/scenes/${id}`); } catch (err) { pollTimer = setTimeout(tick, 2000); return; }
    if (s.video_url && $('#preview-video').hidden && !$('#preview-video').src) {
      $('#preview-video').hidden = false; $('#preview-video').src = s.video_url;
    }
    if (s.name) $('#progress-name').textContent = s.name;
    if (s.status === 'failed') return failProgress(s.error || s.stage_message || 'Reconstruction failed', STAGE_OF[s.status] ?? 1);
    const stage = STAGE_OF[s.status] ?? 0;
    const backend = s.backend === 'replicate' ? ' (Replicate)' : s.backend === 'hf-space' ? ' (HF Space)' : '';
    setStage(stage, s.progress ?? 0, `${s.stage_message || s.status}${stage === 1 ? backend : ''}`);
    if (s.status === 'done') {
      clearInterval(elapsedTimer);
      setTimeout(() => { if (location.hash === `#/scene/${id}`) location.reload(); else location.hash = `#/scene/${id}`; }, 900);
      return;
    }
    pollTimer = setTimeout(tick, 2000);
  };
  tick();
}

// Scripted progress for rehearsals and screenshots: ?mock=progress
function mockProgress() {
  startProgressView({ name: 'Robot lab', videoUrl: params.get('video') || '../fixtures/sample.mp4' });
  const steps = [
    [0, 0.06, 'Uploading 100%'], [1, 0.18, 'Extracting 24 frames'], [1, 0.42, 'MapAnything is regressing metric depth on the GPU'],
    [1, 0.7, 'Downloading the reconstruction'], [2, 0.86, 'Gravity-aligning and removing flying pixels'], [3, 1, 'Ready'],
  ];
  const at = Number(params.get('step') ?? -1);
  if (at >= 0) { const s = steps[Math.min(at, steps.length - 1)]; setStage(...s); return; }
  steps.forEach((s, i) => setTimeout(() => setStage(...s), 1500 * i));
}

// ---------- viewer ----------
async function openScene({ id = null, glbUrl, meta }) {
  show('stage');
  paintIcons($('#stage'));
  const viewer = new Viewer($('#stage'));
  const measure = new Measure(viewer);
  const ui = { id, viewer, measure, meta, calibrated: (meta.scale_factor ?? 1) !== 1 };
  Object.assign(window.video3d, viewer.debugHandle(), { measure, ui });

  renderHud(ui);
  $('#back').onclick = () => { location.href = location.pathname; };

  try {
    await viewer.load({
      glbUrl, meta,
      onProgress: (p) => { $('#loading-text').textContent = p < 1 ? `Loading scene… ${Math.round(p * 100)}%` : 'Preparing geometry…'; },
    });
  } catch (err) {
    console.error(err);
    $('#loading-text').textContent = `Could not load the scene: ${err.message || err}`;
    $('#loading .spinner').hidden = true;
    return;
  }
  Object.assign(window.video3d, viewer.debugHandle(), { measure, ui });
  $('#loading').classList.add('gone');
  setTimeout(() => $('#hint').classList.add('fade'), 9000);

  if (meta.splat) {
    viewer.mountSplat().then((ok) => {
      const b = document.querySelector('[data-mode="photoreal"]');
      b.hidden = !ok;
      if (ok && params.get('mode') !== 'mesh' && params.get('mode') !== 'points') toast('Photoreal splats available (3)');
    });
  }
  bindToolbar(ui);
  bindMeasurePanel(ui);
  viewer.on((type) => { if (type === 'change' || type === 'scale' || type === 'walk') syncToolbar(ui); if (type === 'scale') renderHud(ui); });
  if (params.get('mode')) viewer.setMode(params.get('mode'));
  syncToolbar(ui);
  window.video3d.ready = true;
}

function renderHud(ui) {
  const m = ui.meta;
  $('#hud-name').textContent = m.name || m.id || 'Scene';
  const rows = [];
  const model = (m.model || '').replace(/\s*\(.*\)/, '') || (m.raw ? 'Raw GLB' : '');
  const backend = { 'hf-space': 'HF Space', replicate: 'Replicate' }[m.backend] || m.backend;
  if (model) rows.push(['Model', `${esc(model)}${backend ? ` · ${esc(backend)}` : ''}`]);
  if (m.num_frames) rows.push(['Frames', m.num_frames]);
  if (m.timings_s?.reconstruct) rows.push(['Reconstruct', `${Math.round(m.timings_s.reconstruct)} s`]);
  const s = ui.viewer?.scale ?? m.scale_factor ?? 1;
  rows.push(['Scale', `×${s.toFixed(3)}${ui.calibrated ? ' <span class="ok">Calibrated ✓</span>' : ' <span class="muted">metric</span>'}`]);
  if (ui.viewer?.stats?.vertices) rows.push(['Points', `${(ui.viewer.stats.vertices / 1e6).toFixed(2)} M`]);
  $('#hud-rows').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
}

function bindToolbar(ui) {
  const { viewer, measure } = ui;
  const state = ui.state = { dims: false, grid: false };
  const actions = {
    orbit: () => { if (viewer.walk) viewer.exitWalk(); measure.setActive(false); },
    walk: () => { if (viewer.walk) viewer.exitWalk(); else { measure.setActive(false); viewer.enterWalk(); } },
    measure: () => { if (viewer.walk) viewer.exitWalk(); measure.setActive(!measure.active); },
    dims: () => { state.dims = !state.dims; viewer.setDims(state.dims); },
    grid: () => { state.grid = !state.grid; viewer.setGrid(state.grid); },
    reset: () => viewer.resetView(),
    fly: () => (viewer.fly ? viewer.cancelAnim() : viewer.flyThrough()),
  };
  ui.actions = actions;
  for (const b of document.querySelectorAll('#toolbar button')) {
    b.addEventListener('click', () => {
      if (b.dataset.mode) viewer.setMode(b.dataset.mode);
      else actions[b.dataset.tool || b.dataset.toggle || b.dataset.action]?.();
      syncToolbar(ui);
      b.blur();
    });
  }
  document.querySelector('[data-action="fly"]').hidden = !viewer.canFlyThrough;
  window.addEventListener('keydown', (e) => {
    if (e.target.closest('input,select,textarea') || e.metaKey || e.ctrlKey && e.code !== 'KeyZ') return;
    if (viewer.walk && e.code !== 'KeyP') return;
    const k = {
      KeyO: 'orbit', KeyP: 'walk', KeyM: 'measure', KeyB: 'dims', KeyG: 'grid', KeyR: 'reset', KeyF: 'fly',
    }[e.code];
    if (k === 'fly' && !viewer.canFlyThrough) return;
    if (k) { actions[k](); syncToolbar(ui); return; }
    if (e.code === 'Digit1') viewer.setMode('mesh');
    if (e.code === 'Digit2') viewer.setMode('points');
    if (e.code === 'Digit3' && viewer.splat) viewer.setMode('photoreal');
    if (e.code === 'Backspace' || (e.code === 'KeyZ' && e.ctrlKey)) measure.undo();
    if (e.code === 'Escape' && measure.active) { measure.setActive(false); syncToolbar(ui); }
  });
  measure.on(() => syncToolbar(ui));
}

function syncToolbar(ui) {
  const { viewer, measure, state } = ui;
  if (!state) return;
  const tool = viewer.walk ? 'walk' : measure.active ? 'measure' : 'orbit';
  for (const b of document.querySelectorAll('#toolbar [data-tool]')) b.classList.toggle('on', b.dataset.tool === tool);
  for (const b of document.querySelectorAll('#toolbar [data-toggle]')) b.classList.toggle('on', !!state[b.dataset.toggle]);
  for (const b of document.querySelectorAll('#toolbar [data-mode]')) b.classList.toggle('on', b.dataset.mode === viewer.mode);
  const fly = document.querySelector('[data-action="fly"]');
  fly.classList.toggle('on', !!viewer.fly);
  $('#walk-hint').hidden = !viewer.walk;
  $('#hint').hidden = !!viewer.walk;
  $('#mpanel').hidden = !(measure.active || measure.items.length);
  renderMeasureList(ui);
}

function renderMeasureList(ui) {
  const { measure } = ui;
  const list = $('#m-list');
  $('#m-hint').textContent = measure.pending ? 'Now click the second point.' : measure.active ? 'Click two points on a surface.' : 'Measure is off. Press M to add more.';
  list.innerHTML = '';
  measure.items.forEach((it, i) => {
    const li = document.createElement('li');
    li.className = it.id === measure.selected ? 'selected' : '';
    li.innerHTML = `<span class="idx">${i + 1}</span><span class="len">${formatLength(measure.lengthOf(it))}</span><span class="cal">${it.id === measure.selected ? 'selected' : 'calibrate…'}</span>`;
    li.onclick = () => measure.select(it.id);
    list.appendChild(li);
  });
  const sel = measure.items.find((i) => i.id === measure.selected);
  $('#m-cal').hidden = !sel;
  if (sel) $('#m-cal-name').textContent = `#${measure.items.indexOf(sel) + 1} (${formatLength(measure.lengthOf(sel))})`;
}

function bindMeasurePanel(ui) {
  const { measure, viewer } = ui;
  $('#m-undo').onclick = () => measure.undo();
  $('#m-clear').onclick = () => measure.clear();
  const apply = async () => {
    const v = parseFloat($('#m-cal-value').value);
    const unit = parseFloat($('#m-cal-unit').value);
    const s = measure.calibrationFor(measure.selected, v * unit);
    if (!s || !isFinite(s)) { toast('Type the real length first'); return; }
    await calibrate(ui, s);
    $('#m-cal-value').value = '';
    measure.select(measure.selected);
  };
  $('#m-cal-apply').onclick = apply;
  $('#m-cal-value').addEventListener('keydown', (e) => { if (e.key === 'Enter') apply(); });
}

async function calibrate(ui, s) {
  ui.viewer.setScale(s);
  ui.calibrated = true;
  ui.meta.scale_factor = s;
  renderHud(ui);
  if (!ui.id) { toast(`Calibrated ✓ ×${s.toFixed(3)} (not saved: no server scene)`); return; }
  try {
    const meta = await api(`/api/scenes/${ui.id}/calibration`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ scale_factor: s }),
    });
    ui.meta.scale_factor = meta.scale_factor ?? s;
    toast(`Calibrated ✓ scale ×${s.toFixed(3)} saved`);
  } catch (err) {
    toast(`Calibrated ✓ locally (save failed: ${err.message})`);
  }
}
window.video3d.calibrate = (s) => calibrate(window.video3d.ui, s);

// ---------- routing ----------
async function route() {
  paintIcons(document.querySelector('#home'));
  setupDrop();
  $('#progress-back').onclick = () => { location.href = location.pathname; };
  const glb = params.get('glb');
  if (glb) {
    let meta = { name: decodeURIComponent(glb.split('/').pop().replace(/\.glb$/i, '').replace(/[_-]+/g, ' ')), raw: true, cameras: [], scale_factor: 1 };
    if (params.get('meta')) {
      try { meta = await api(params.get('meta')); } catch (err) { toast(`meta.json failed: ${err.message}`); }
    }
    if (params.get('raw') === '1') meta.raw = true;
    if (params.get('raw') === '0') meta.raw = false;
    return openScene({ glbUrl: glb, meta });
  }
  if (params.get('mock') === 'progress') return mockProgress();
  const m = location.hash.match(/^#\/scene\/([\w-]+)/);
  if (m) {
    const id = m[1];
    let s;
    try { s = await api(`/api/scenes/${id}`); } catch (err) { toast(`Scene ${id} not found`); return showHome(); }
    if (s.status !== 'done') {
      startProgressView({ name: s.name, videoUrl: s.video_url, thumbUrl: s.thumbnail_url });
      return pollScene(id);
    }
    let meta;
    try { meta = await api(s.meta_url); } catch { meta = { name: s.name, cameras: [], scale_factor: 1 }; }
    return openScene({ id, glbUrl: s.glb_url, meta });
  }
  return showHome();
}

window.addEventListener('hashchange', () => location.reload());
route();
