/**
 * MeshEditor — Three.js editor for GLB mesh scenes (TripoSR output).
 *
 * Mirrors the SplatEditor interface (records/selected/select/deleteObject/…) so
 * main.js can drive either viewer. Each object is a loaded GLB wrapped in a
 * Group; edits are transforms on that Group.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';

const CLICK_SLOP_PX = 4; // pointer travel under this counts as a click, not a drag

export class MeshEditor {
  constructor(container, callbacks = {}) {
    this.container = container;
    this.cb = callbacks;
    this.statusCallback = callbacks.onStatus ?? null;

    this.records = new Map(); // id -> { info, group, deleted }
    this.selected = null;
    this.transformMode = 'translate';

    const w = container.clientWidth || window.innerWidth;
    const h = container.clientHeight || window.innerHeight;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 1000);
    this.camera.position.set(0, 0.4, 2.5);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(this.renderer.domElement);

    // Lights live for the editor's lifetime (loadScene must NOT clear them).
    this.rig = new THREE.Group();
    const key = new THREE.DirectionalLight(0xffffff, 2.4);
    key.position.set(3, 5, 4);
    const fill = new THREE.DirectionalLight(0xffffff, 0.9);
    fill.position.set(-4, 1, -3);
    this.rig.add(key, fill, new THREE.AmbientLight(0xffffff, 0.55));
    this.scene.add(this.rig);

    this.orbit = new OrbitControls(this.camera, this.renderer.domElement);
    this.orbit.enableDamping = true;

    this.tc = new TransformControls(this.camera, this.renderer.domElement);
    this.tc.addEventListener('dragging-changed', (e) => { this.orbit.enabled = !e.value; });
    // r160–r168: TransformControls is an Object3D; r169+: add its helper.
    this.scene.add(typeof this.tc.getHelper === 'function' ? this.tc.getHelper() : this.tc);

    this.loader = new GLTFLoader();
    this._raycaster = new THREE.Raycaster();
    this._pointer = new THREE.Vector2();
    this._downXY = null;

    this._onPointerDown = (e) => { this._downXY = [e.clientX, e.clientY]; };
    this._onPointerUp = (e) => this._handlePointerUp(e);
    this.renderer.domElement.addEventListener('pointerdown', this._onPointerDown);
    this.renderer.domElement.addEventListener('pointerup', this._onPointerUp);
    this._onResize = () => this._resize();
    window.addEventListener('resize', this._onResize);

    this._animate = this._animate.bind(this);
    this._raf = requestAnimationFrame(this._animate);
  }

  _status(msg, kind) { this.statusCallback?.(msg, kind); }

  async loadScene(manifest) {
    for (const rec of this.records.values()) this.scene.remove(rec.group);
    this.records.clear();
    this.tc.detach();
    this.selected = null;

    const box = new THREE.Box3();
    const objects = manifest.objects ?? [];
    for (let i = 0; i < objects.length; i++) {
      const info = objects[i];
      this._status(`Loading ${info.label} (${i + 1}/${objects.length})…`);
      try {
        const gltf = await this.loader.loadAsync(info.mesh_url);
        const group = new THREE.Group();
        group.add(gltf.scene);
        group.userData.objectId = info.id;
        group.traverse((o) => { if (o.isMesh) o.userData.objectId = info.id; });
        this.scene.add(group);
        box.expandByObject(group);
        this.records.set(info.id, { info, group, deleted: false });
      } catch (e) {
        this._status(`Failed to load ${info.label}: ${e.message}`, 'error');
      }
    }

    if (!box.isEmpty()) {
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z) || 1;
      const dist = (maxDim / (2 * Math.tan((this.camera.fov * Math.PI) / 360))) * 1.5;
      this.camera.near = maxDim / 100;
      this.camera.far = maxDim * 100;
      this.camera.position.copy(center).add(new THREE.Vector3(0.4 * dist, 0.35 * dist, dist));
      this.camera.updateProjectionMatrix();
      this.orbit.target.copy(center);
      this.orbit.update();
    }
    const n = this.records.size;
    this._status(`Loaded ${n} object${n === 1 ? '' : 's'} · click one to select`, 'ok');
  }

  _handlePointerUp(e) {
    if (this._downXY) {
      const [dx, dy] = [Math.abs(e.clientX - this._downXY[0]), Math.abs(e.clientY - this._downXY[1])];
      this._downXY = null;
      if (dx > CLICK_SLOP_PX || dy > CLICK_SLOP_PX) return; // was an orbit drag
    }
    const rect = this.renderer.domElement.getBoundingClientRect();
    this._pointer.set(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    );
    this._raycaster.setFromCamera(this._pointer, this.camera);
    const groups = [...this.records.values()].filter((r) => !r.deleted).map((r) => r.group);
    const hits = this._raycaster.intersectObjects(groups, true);
    if (hits.length) {
      let o = hits[0].object;
      while (o && o.userData.objectId === undefined) o = o.parent;
      if (o) { this.select(o.userData.objectId); return; }
    }
    this.deselect();
  }

  select(id) {
    if (id === null || id === undefined) { this.deselect(); return; }
    const rec = this.records.get(id);
    if (!rec || rec.deleted) return;
    this.selected = id;
    this.tc.attach(rec.group);
    this.tc.setMode(this.transformMode);
    this.cb.onSelectionChanged?.(id);
  }
  selectObject(id) { this.select(id); }

  deselect() {
    this.selected = null;
    this.tc.detach();
    this.cb.onSelectionChanged?.(null);
  }

  setTransformMode(mode) {
    this.transformMode = mode;
    if (this.selected !== null) this.tc.setMode(mode);
  }
  setMode(mode) { this.setTransformMode(mode); }

  deleteObject(id) {
    const rec = this.records.get(id);
    if (!rec) return;
    rec.deleted = true;
    rec.group.visible = false;
    if (this.selected === id) this.deselect();
    this.cb.onObjectStateChanged?.(id, true);
  }

  restoreObject(id) {
    const rec = this.records.get(id);
    if (!rec) return;
    rec.deleted = false;
    rec.group.visible = true;
    this.cb.onObjectStateChanged?.(id, false);
  }

  getEditState() {
    return [...this.records.entries()].map(([id, rec]) => ({
      id,
      deleted: rec.deleted,
      position: rec.group.position.toArray(),
      quaternion: rec.group.quaternion.toArray(),
      scale: rec.group.scale.x,
    }));
  }

  _resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  _animate() {
    this._raf = requestAnimationFrame(this._animate);
    this.orbit.update();
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    cancelAnimationFrame(this._raf);
    window.removeEventListener('resize', this._onResize);
    this.renderer.domElement.removeEventListener('pointerdown', this._onPointerDown);
    this.renderer.domElement.removeEventListener('pointerup', this._onPointerUp);
    this.tc.detach();
    this.tc.dispose?.();
    this.orbit.dispose?.();
    this.scene.traverse((o) => {
      if (o.isMesh) {
        o.geometry?.dispose?.();
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        mats.forEach((m) => m?.dispose?.());
      }
    });
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.records.clear();
  }
}
