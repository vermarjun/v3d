// Point-to-point measurements and scale calibration. Points live in root-local space, so the
// visuals follow root.scale and every length is localDistance × scale_factor.
import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { ACCENT, formatLength } from './viewer.js';

let nextId = 1;

export class Measure {
  constructor(viewer) {
    this.viewer = viewer;
    this.items = [];
    this.active = false;
    this.pending = null;
    this.selected = null;
    this.listeners = new Set();
    this.group = new THREE.Group();
    this.group.name = 'measurements';
    viewer.root.add(this.group);

    viewer.on((type, e) => {
      if (type === 'click' && this.active) this._onClick(e);
      else if (type === 'hover' && this.active && this.pending) this._onHover(e);
      else if (type === 'scale') this._relabel();
    });
  }

  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  _emit() { for (const fn of this.listeners) fn(); }

  setActive(on) {
    this.active = on;
    this.viewer.measuring = on;
    if (!on) this._cancelPending();
    this.viewer.renderer.domElement.classList.toggle('measuring', on);
    this._emit();
  }

  lengthOf(item) { return item.a.distanceTo(item.b) * this.viewer.scale; }

  async _onClick(e) {
    await this.viewer.bvhReady;
    const hit = this.viewer.pick(e.clientX, e.clientY);
    if (!hit) return;
    if (!this.pending) this._startPending(hit.local);
    else {
      const a = this.pending.a;
      this._cancelPending();
      this.add(a, hit.local);
    }
  }

  _onHover(e) {
    if (this._hoverQueued) { this._lastHover = e; return; }
    this._lastHover = e;
    this._hoverQueued = true;
    requestAnimationFrame(() => {
      this._hoverQueued = false;
      if (!this.pending) return;
      const hit = this.viewer.pick(this._lastHover.clientX, this._lastHover.clientY);
      if (!hit) return;
      const p = this.pending;
      p.b = hit.local;
      p.line.geometry.setPositions([...p.a.toArray(), ...p.b.toArray()]);
      p.line.visible = true;
      p.label.position.copy(p.a).lerp(p.b, 0.5);
      p.label.element.textContent = formatLength(p.a.distanceTo(p.b) * this.viewer.scale);
      p.label.visible = true;
    });
  }

  _makeLine(width, color, opacity) {
    const mat = new LineMaterial({ color, linewidth: width, transparent: true, opacity, depthTest: false, depthWrite: false });
    const r = this.viewer.renderer.domElement;
    mat.resolution.set(r.clientWidth, r.clientHeight);
    const line = new LineSegments2(new LineSegmentsGeometry().setPositions([0, 0, 0, 0, 0, 0]), mat);
    line.renderOrder = 20;
    line.frustumCulled = false;
    line.raycast = () => {};
    return line;
  }

  _makeTag(cls, text = '') {
    const el = document.createElement('div');
    el.className = cls;
    el.textContent = text;
    return new CSS2DObject(el);
  }

  _startPending(a) {
    const line = this._makeLine(2, 0xffffff, 0.85);
    line.visible = false;
    const dot = this._makeTag('mdot pending');
    dot.position.copy(a);
    const label = this._makeTag('tag live');
    label.visible = false;
    this.group.add(line, dot, label);
    this.pending = { a, line, dot, label };
    this._emit();
  }

  _cancelPending() {
    const p = this.pending;
    if (!p) return;
    this.group.remove(p.line, p.dot, p.label);
    p.line.geometry.dispose(); p.line.material.dispose();
    p.dot.element.remove(); p.label.element.remove();
    this.pending = null;
    this._emit();
  }

  // a, b: root-local Vector3. Returns the new item.
  add(a, b) {
    const shadow = this._makeLine(6, 0x000000, 0.35);
    const line = this._makeLine(3, ACCENT, 1);
    const arr = [...a.toArray(), ...b.toArray()];
    shadow.geometry.setPositions(arr);
    line.geometry.setPositions(arr);
    const da = this._makeTag('mdot'); da.position.copy(a);
    const db = this._makeTag('mdot'); db.position.copy(b);
    const label = this._makeTag('tag measure');
    label.position.copy(a).lerp(b, 0.5);
    const item = { id: nextId++, a: a.clone(), b: b.clone(), objects: [shadow, line, da, db, label], label };
    label.element.addEventListener('pointerdown', (ev) => { ev.stopPropagation(); this.select(item.id); });
    this.group.add(...item.objects);
    this.items.push(item);
    this._relabel();
    this._emit();
    return item;
  }

  select(id) {
    this.selected = this.selected === id ? null : id;
    this._relabel();
    this._emit();
  }

  _remove(item) {
    for (const o of item.objects) {
      this.group.remove(o);
      if (o.isCSS2DObject) o.element.remove();
      else { o.geometry.dispose(); o.material.dispose(); }
    }
    if (this.selected === item.id) this.selected = null;
  }

  undo() {
    if (this.pending) return this._cancelPending();
    const item = this.items.pop();
    if (item) this._remove(item);
    this._emit();
  }

  clear() {
    this._cancelPending();
    for (const item of this.items) this._remove(item);
    this.items = [];
    this._emit();
  }

  _relabel() {
    for (const item of this.items) {
      item.label.element.textContent = formatLength(this.lengthOf(item));
      item.label.element.classList.toggle('selected', item.id === this.selected);
    }
  }

  // Scale so that this measurement reads `realMeters`; scale_factor = real / measured × current.
  calibrationFor(id, realMeters) {
    const item = this.items.find((i) => i.id === id);
    if (!item || !(realMeters > 0)) return null;
    return (realMeters / this.lengthOf(item)) * this.viewer.scale;
  }
}
