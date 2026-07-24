/**
 * SplatEditor — object-level editing on top of GaussianSplats3D.
 *
 * Strategy: the backend splits a labeled scene into one PLY per object
 * (positions recentred at the object's centroid). Each object is loaded as its
 * own splat scene in a dynamic-mode viewer, so:
 *   select = OBB raycast against per-object bounding boxes
 *   move   = set position/quaternion/scale on that object's splat scene
 *   delete = remove that splat scene from the viewer
 * No retraining, no shader surgery — exactly the "edit = transform or filter
 * Gaussian subsets" design from the architecture doc.
 */

import * as THREE from 'three';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

const ACCENT = 0xf2a93b;

export class SplatEditor {
  /**
   * @param {HTMLElement} rootElement container for the viewer canvas
   * @param {{
   *   onSelectionChanged?: (objectId: number|null) => void,
   *   onObjectStateChanged?: (objectId: number, deleted: boolean) => void,
   *   onStatus?: (message: string, kind?: 'info'|'error'|'ok') => void,
   * }} callbacks
   */
  constructor(rootElement, callbacks = {}) {
    this.root = rootElement;
    this.callbacks = callbacks;
    this.viewer = null;
    this.transformControls = null;
    this.manifest = null;
    /** @type {Map<number, object>} objectId -> record */
    this.records = new Map();
    /** @type {number[]} objectIds in splat-scene index order */
    this.loadedOrder = [];
    this.selectedId = null;
    this.mode = 'translate';
    this.sceneRadius = 1;
    this._opChain = Promise.resolve();
    this._raycaster = new THREE.Raycaster();
    this._pointerDown = null;

    this._onPointerDown = (e) => {
      this._pointerDown = { x: e.clientX, y: e.clientY };
    };
    this._onPointerUp = (e) => {
      const down = this._pointerDown;
      this._pointerDown = null;
      if (!down || e.button !== 0) return;
      if (Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) return; // drag, not click
      if (this.transformControls?.dragging) return;
      this._handlePick(e.clientX, e.clientY);
    };
  }

  _status(message, kind = 'info') {
    this.callbacks.onStatus?.(message, kind);
  }

  /** Serialize splat-scene add/remove ops; the library forbids concurrency. */
  _enqueue(op) {
    this._opChain = this._opChain.then(op, (err) => {
      console.error('scene op failed:', err);
      this._status(`Scene operation failed: ${err.message}`, 'error');
    });
    return this._opChain;
  }

  async loadScene(manifest) {
    await this.disposeViewer();
    this.manifest = manifest;
    this.records.clear();
    this.loadedOrder = [];
    this.selectedId = null;

    // Frame the camera from the union of object bounding boxes.
    const bounds = new THREE.Box3();
    for (const o of manifest.objects) {
      bounds.expandByPoint(new THREE.Vector3(...o.bbox_min));
      bounds.expandByPoint(new THREE.Vector3(...o.bbox_max));
    }
    const center = bounds.getCenter(new THREE.Vector3());
    const radius = Math.max(bounds.getSize(new THREE.Vector3()).length() / 2, 1);
    this.sceneRadius = radius;
    const camPos = center.clone().add(new THREE.Vector3(0.75, 0.55, 0.95).multiplyScalar(radius));

    this.viewer = new GaussianSplats3D.Viewer({
      rootElement: this.root,
      cameraUp: [0, 1, 0],
      initialCameraPosition: camPos.toArray(),
      initialCameraLookAt: center.toArray(),
      selfDrivenMode: true,
      useBuiltInControls: true,
      dynamicScene: true,
      sharedMemoryForWorkers: false,
      sphericalHarmonicsDegree: 0,
      sceneRevealMode: GaussianSplats3D.SceneRevealMode.Instant,
    });
    this.viewer.start();

    for (let i = 0; i < manifest.objects.length; i++) {
      const info = manifest.objects[i];
      this._status(`Loading ${info.label} (${i + 1}/${manifest.objects.length})…`);
      await this._enqueue(() =>
        this.viewer.addSplatScene(info.ply_url, {
          showLoadingUI: false,
          progressiveLoad: false,
          position: info.centroid,
          splatAlphaRemovalThreshold: 5,
        })
      );
      this.loadedOrder.push(info.id);
      this._makeRecord(info);
    }

    if (this.viewer.controls) {
      this.viewer.controls.target.copy(center);
      this.viewer.controls.update();
    }
    this._setupTransformControls();

    const dom = this.viewer.renderer.domElement;
    dom.addEventListener('pointerdown', this._onPointerDown);
    dom.addEventListener('pointerup', this._onPointerUp);

    this._status(`Loaded ${manifest.objects.length} objects · click one to select`, 'ok');
  }

  _makeRecord(info) {
    // Pivot lives at the object's world centroid; per-object PLYs are already
    // recentred, so rotate/scale naturally pivot around the object itself.
    const pivot = new THREE.Object3D();
    pivot.position.set(...info.centroid);

    const bboxLocal = new THREE.Box3(
      new THREE.Vector3(...info.bbox_min).sub(pivot.position),
      new THREE.Vector3(...info.bbox_max).sub(pivot.position)
    );
    const helper = new THREE.Box3Helper(bboxLocal, ACCENT);
    helper.visible = false;
    pivot.add(helper);
    this.viewer.threeScene.add(pivot);

    const diag = bboxLocal.getSize(new THREE.Vector3()).length();
    this.records.set(info.id, {
      info,
      pivot,
      helper,
      bboxLocal,
      pickBox: bboxLocal.clone().expandByScalar(0.015 * diag),
      volume: Math.max(
        bboxLocal.getSize(new THREE.Vector3()).toArray().reduce((a, b) => a * Math.max(b, 1e-4), 1),
        1e-9
      ),
      deleted: false,
    });
  }

  _setupTransformControls() {
    const tc = new TransformControls(this.viewer.camera, this.viewer.renderer.domElement);
    tc.setMode(this.mode);
    tc.setSize(0.9);
    tc.addEventListener('dragging-changed', (e) => {
      if (this.viewer.controls) this.viewer.controls.enabled = !e.value;
    });
    tc.addEventListener('objectChange', () => this._syncSelectedTransform());
    // The gizmo ships with depthTest/depthWrite off ("always on top"), but the
    // splat mesh renders after the three scene and would paint over it. Making
    // the gizmo write depth lets splats depth-test against it instead.
    tc.traverse((child) => {
      if (!child.material) return;
      const mats = Array.isArray(child.material) ? child.material : [child.material];
      for (const m of mats) {
        m.depthTest = true;
        m.depthWrite = true;
      }
    });
    // r160-r168: TransformControls is an Object3D; r169+: add its helper instead.
    this.viewer.threeScene.add(typeof tc.getHelper === 'function' ? tc.getHelper() : tc);
    this.transformControls = tc;
  }

  _sceneIndexOf(objectId) {
    return this.loadedOrder.indexOf(objectId);
  }

  _syncSelectedTransform() {
    const record = this.records.get(this.selectedId);
    if (!record || record.deleted) return;
    if (this.mode === 'scale') {
      const s = record.pivot.scale;
      s.setScalar(Math.max((s.x + s.y + s.z) / 3, 1e-3));
    }
    const idx = this._sceneIndexOf(record.info.id);
    if (idx < 0) return;
    const splatScene = this.viewer.getSplatScene(idx);
    splatScene.position.copy(record.pivot.position);
    splatScene.quaternion.copy(record.pivot.quaternion);
    splatScene.scale.copy(record.pivot.scale);
  }

  _handlePick(clientX, clientY) {
    const dom = this.viewer.renderer.domElement;
    const rect = dom.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1
    );
    this._raycaster.setFromCamera(ndc, this.viewer.camera);
    const ray = this._raycaster.ray;

    const hits = [];
    const hitPoint = new THREE.Vector3();
    for (const record of this.records.values()) {
      if (record.deleted) continue;
      record.pivot.updateMatrixWorld(true);
      const inv = record.pivot.matrixWorld.clone().invert();
      const localRay = ray.clone().applyMatrix4(inv);
      if (localRay.intersectBox(record.pickBox, hitPoint)) {
        const world = hitPoint.clone().applyMatrix4(record.pivot.matrixWorld);
        const s = record.pivot.scale.x;
        hits.push({
          id: record.info.id,
          distance: world.distanceTo(ray.origin),
          volume: record.volume * s * s * s,
        });
      }
    }

    if (hits.length === 0) {
      this.selectObject(null);
      return;
    }
    // Prefer the nearest hit, but when boxes overlap (lamp standing on the
    // table), let the smaller object win the tie so it stays clickable.
    hits.sort((a, b) => a.distance - b.distance);
    const tolerance = 0.08 * this.sceneRadius;
    const near = hits.filter((h) => h.distance <= hits[0].distance + tolerance);
    near.sort((a, b) => a.volume - b.volume);
    this.selectObject(near[0].id);
  }

  selectObject(objectId) {
    if (objectId === this.selectedId) return;
    const prev = this.records.get(this.selectedId);
    if (prev) prev.helper.visible = false;
    this.transformControls?.detach();
    this.selectedId = objectId;

    const record = this.records.get(objectId);
    if (record && !record.deleted) {
      record.helper.visible = true;
      this.transformControls?.attach(record.pivot);
    } else {
      this.selectedId = record ? objectId : null;
    }
    this.callbacks.onSelectionChanged?.(this.selectedId);
  }

  setMode(mode) {
    if (!['translate', 'rotate', 'scale'].includes(mode)) return;
    this.mode = mode;
    this.transformControls?.setMode(mode);
  }

  async deleteObject(objectId) {
    const record = this.records.get(objectId);
    if (!record || record.deleted) return;
    record.deleted = true;
    record.helper.visible = false;
    if (this.selectedId === objectId) this.transformControls?.detach();

    await this._enqueue(async () => {
      const idx = this._sceneIndexOf(objectId);
      if (idx < 0) return;
      await this.viewer.removeSplatScene(idx, false);
      this.loadedOrder.splice(idx, 1);
    });
    this.callbacks.onObjectStateChanged?.(objectId, true);
    this._status(`Deleted ${record.info.label} — note the hole it leaves (see README)`, 'info');
  }

  async restoreObject(objectId) {
    const record = this.records.get(objectId);
    if (!record || !record.deleted) return;

    await this._enqueue(async () => {
      await this.viewer.addSplatScene(record.info.ply_url, {
        showLoadingUI: false,
        progressiveLoad: false,
        position: record.pivot.position.toArray(),
        rotation: record.pivot.quaternion.toArray(),
        scale: record.pivot.scale.toArray(),
        splatAlphaRemovalThreshold: 5,
      });
      this.loadedOrder.push(objectId);
      record.deleted = false;
    });
    this.callbacks.onObjectStateChanged?.(objectId, false);
    this._status(`Restored ${record.info.label}`, 'ok');
  }

  /** Edit state for the export endpoint (world-space, three.js quaternion order). */
  getEditState() {
    return [...this.records.values()].map((record) => ({
      id: record.info.id,
      deleted: record.deleted,
      position: record.pivot.position.toArray(),
      quaternion: record.pivot.quaternion.toArray(),
      scale: (record.pivot.scale.x + record.pivot.scale.y + record.pivot.scale.z) / 3,
    }));
  }

  async disposeViewer() {
    if (!this.viewer) return;
    const viewer = this.viewer;
    const dom = viewer.renderer?.domElement;
    dom?.removeEventListener('pointerdown', this._onPointerDown);
    dom?.removeEventListener('pointerup', this._onPointerUp);
    this.transformControls?.detach();
    this.transformControls?.dispose?.();
    this.transformControls = null;
    this.viewer = null;
    await this._enqueue(() => viewer.dispose());
  }
}
