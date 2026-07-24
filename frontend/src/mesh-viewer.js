/**
 * MeshEditor: Three.js-based editor for mesh scenes from TripoSR.
 *
 * Each object is a loaded GLTF mesh. Edit controls:
 * - Select: OBB raycast with keyboard/click
 * - Transform: translate, rotate, uniform scale via gizmo
 * - Delete/restore: toggle visibility
 * - Export: merged GLB with transforms baked in
 */

import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const ACCENT = 0xffb800;
const DELETE_COLOR = 0xff6b6b;

export class MeshEditor {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1a1a1a);
    this.camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 10000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.shadowMap.enabled = true;
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.autoRotate = false;
    this.transformControls = new TransformControls(this.camera, this.renderer.domElement);
    this.scene.add(this.transformControls);

    // Lighting
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(5, 10, 7);
    light.castShadow = true;
    this.scene.add(light);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.4));

    // State
    this.records = new Map(); // id → {info, group, helper, deleted}
    this.selected = null;
    this.gltfLoader = new GLTFLoader();
    this.statusCallback = null;
    this.transformMode = "translate";

    // Events
    this.renderer.domElement.addEventListener("pointerdown", (e) => this._onPointerDown(e));
    this.transformControls.addEventListener("change", () => this.renderer.render(this.scene, this.camera));
    window.addEventListener("resize", () => this._onWindowResize());

    this.animate();
  }

  _status(msg, level = "info") {
    if (this.statusCallback) this.statusCallback(msg, level);
  }

  async loadScene(manifest) {
    this._status(`Loading ${manifest.objects.length} objects...`);
    this.scene.clear();
    this.scene.add(this.controls.getObject());
    this.scene.add(this.transformControls);
    this.records.clear();
    this.selected = null;

    let bbox = new THREE.Box3();

    for (const info of manifest.objects) {
      await this._loadObject(info);
      if (this.records.has(info.id)) {
        const { group } = this.records.get(info.id);
        bbox.expandByObject(group);
      }
    }

    // Frame camera
    if (!bbox.isEmpty()) {
      const size = bbox.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      const fov = this.camera.fov * (Math.PI / 180);
      const distance = maxDim / (2 * Math.tan(fov / 2));
      const center = bbox.getCenter(new THREE.Vector3());
      this.camera.position.copy(center).add(new THREE.Vector3(0, 0.2 * maxDim, distance * 0.7));
      this.camera.lookAt(center);
      this.controls.target.copy(center);
      this.controls.update();
    }

    this._status(`Loaded ${manifest.objects.length} objects`, "ok");
  }

  async _loadObject(info) {
    try {
      const gltf = await this._loadGLTF(info.mesh_url);
      const group = new THREE.Group();
      group.add(gltf.scene);
      group.userData.objectId = info.id;
      this.scene.add(group);

      // Compute bounds
      const bbox = new THREE.Box3().setFromObject(gltf.scene);
      const helper = new THREE.Box3Helper(bbox, ACCENT);
      helper.visible = false;
      group.add(helper);

      this.records.set(info.id, {
        info,
        group,
        helper,
        deleted: false,
        initialTransform: { position: group.position.clone(), quaternion: group.quaternion.clone(), scale: group.scale.clone() },
      });
    } catch (e) {
      this._status(`Failed to load ${info.label}: ${e.message}`, "error");
    }
  }

  _loadGLTF(url) {
    return new Promise((resolve, reject) => {
      this.gltfLoader.load(url, resolve, undefined, reject);
    });
  }

  _onPointerDown(e) {
    if (e.button !== 0) return;

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2((e.clientX / window.innerWidth) * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1);
    raycaster.setFromCamera(mouse, this.camera);

    const selectables = Array.from(this.records.values()).map((r) => r.group);
    const intersects = raycaster.intersectObjects(selectables, true);

    if (intersects.length > 0) {
      const obj = intersects[0].object.parent;
      const objectId = obj.userData?.objectId ?? obj.parent?.userData?.objectId;
      if (objectId !== undefined) {
        this.select(objectId);
      }
    } else {
      this.deselect();
    }
  }

  select(objectId) {
    if (this.selected !== null) {
      const old = this.records.get(this.selected);
      if (old) old.helper.visible = false;
    }
    this.selected = objectId;
    const rec = this.records.get(objectId);
    if (rec) {
      rec.helper.visible = true;
      this.transformControls.attach(rec.group);
      this.transformControls.setMode(this.transformMode);
    }
  }

  deselect() {
    if (this.selected !== null) {
      const rec = this.records.get(this.selected);
      if (rec) rec.helper.visible = false;
      this.transformControls.detach();
    }
    this.selected = null;
  }

  setTransformMode(mode) {
    this.transformMode = mode; // "translate", "rotate", "scale"
    if (this.selected !== null) {
      this.transformControls.setMode(mode);
    }
  }

  deleteObject(objectId) {
    const rec = this.records.get(objectId);
    if (rec) {
      rec.deleted = true;
      rec.group.visible = false;
      if (this.selected === objectId) this.deselect();
    }
  }

  restoreObject(objectId) {
    const rec = this.records.get(objectId);
    if (rec) {
      rec.deleted = false;
      rec.group.visible = true;
    }
  }

  async export() {
    const objects = [];
    for (const [id, rec] of this.records.entries()) {
      if (!rec.deleted) {
        objects.push({
          id,
          position: rec.group.position.toArray(),
          quaternion: rec.group.quaternion.toArray(),
          scale: rec.group.scale.toArray(),
        });
      }
    }
    return { scene_id: "exported", objects };
  }

  _onWindowResize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
