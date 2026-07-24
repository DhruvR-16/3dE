/** Sidebar / status-bar DOM management. Pure view layer — no editor logic. */

/** Deterministic pleasant color for an object with no explicit color (mesh scenes). */
function colorForObject(o) {
  if (Array.isArray(o.color) && o.color.length >= 3) {
    const [r, g, b] = o.color.map((c) => Math.round(c * 255));
    return `rgb(${r} ${g} ${b})`;
  }
  const hue = (Number(o.id) * 137.508) % 360; // golden angle → well-spread hues
  return `hsl(${hue} 62% 58%)`;
}

/** Right-aligned count chip: splats for PLY scenes, triangles for GLB scenes. */
function countForObject(o) {
  if (typeof o.splat_count === 'number') return `${(o.splat_count / 1000).toFixed(1)}k`;
  if (typeof o.triangle_count === 'number') return `${(o.triangle_count / 1000).toFixed(1)}k△`;
  return '';
}

export class UI {
  /**
   * @param {{
   *   onSceneChange: (sceneId: string) => void,
   *   onSelect: (objectId: number) => void,
   *   onModeChange: (mode: string) => void,
   *   onDelete: () => void,
   *   onRestore: () => void,
   *   onExport: () => void,
   *   onUpload: (sceneId: string, file: File) => void,
   *   onReconstruct: (sceneId: string, images: File[]) => Promise<void>,
   * }} handlers
   */
  constructor(handlers) {
    this.handlers = handlers;
    this.sceneSelect = document.getElementById('scene-select');
    this.objectList = document.getElementById('object-list');
    this.statusEl = document.getElementById('status');
    this.deleteBtn = document.getElementById('btn-delete');
    this.restoreBtn = document.getElementById('btn-restore');
    this.exportBtn = document.getElementById('btn-export');
    this.exportLabeled = document.getElementById('export-labeled');
    this.modeButtons = [...document.querySelectorAll('#mode-buttons button')];
    this._statusTimer = null;

    this.sceneSelect.addEventListener('change', () =>
      handlers.onSceneChange(this.sceneSelect.value)
    );
    this.modeButtons.forEach((btn) =>
      btn.addEventListener('click', () => handlers.onModeChange(btn.dataset.mode))
    );
    this.deleteBtn.addEventListener('click', handlers.onDelete);
    this.restoreBtn.addEventListener('click', handlers.onRestore);
    this.exportBtn.addEventListener('click', handlers.onExport);

    document.getElementById('btn-upload').addEventListener('click', () => {
      const sceneId = document.getElementById('upload-id').value.trim();
      const file = document.getElementById('upload-file').files[0];
      if (!sceneId || !file) {
        this.status('Pick a scene id and a .ply file first', 'error');
        return;
      }
      handlers.onUpload(sceneId, file);
    });

    document.getElementById('reconstruct-btn').addEventListener('click', async () => {
      const sceneId = document.getElementById('reconstruct-scene-id').value.trim();
      const images = Array.from(document.getElementById('reconstruct-images').files);
      if (!images.length) {
        this.status('Add 1–20 images first (drag & drop or browse)', 'error');
        return;
      }
      await handlers.onReconstruct(sceneId, images);
    });
  }

  setScenes(scenes, activeId) {
    this.sceneSelect.replaceChildren(
      ...scenes.map((s) => {
        const opt = document.createElement('option');
        opt.value = s.id;
        opt.textContent = `${s.name} · ${s.object_count} objects`;
        opt.selected = s.id === activeId;
        return opt;
      })
    );
  }

  setObjects(objects) {
    this.objectList.replaceChildren(
      ...objects.map((o) => {
        const li = document.createElement('li');
        li.dataset.objectId = String(o.id);

        const swatch = document.createElement('span');
        swatch.className = 'swatch';
        swatch.style.background = colorForObject(o);

        const label = document.createElement('span');
        label.className = 'obj-label';
        label.textContent = o.label ?? `object ${o.id}`;

        const count = document.createElement('span');
        count.className = 'obj-count';
        count.textContent = countForObject(o);

        li.append(swatch, label, count);
        li.addEventListener('click', () => this.handlers.onSelect(o.id));
        return li;
      })
    );
  }

  _rows() {
    return [...this.objectList.querySelectorAll('li')];
  }

  setSelection(objectId, isDeleted) {
    for (const li of this._rows()) {
      li.classList.toggle('selected', li.dataset.objectId === String(objectId));
    }
    const hasSelection = objectId !== null && objectId !== undefined;
    this.deleteBtn.disabled = !hasSelection || isDeleted;
    this.restoreBtn.disabled = !hasSelection || !isDeleted;
  }

  setObjectDeleted(objectId, deleted) {
    for (const li of this._rows()) {
      if (li.dataset.objectId === String(objectId)) {
        li.classList.toggle('deleted', deleted);
      }
    }
  }

  setMode(mode) {
    this.modeButtons.forEach((btn) =>
      btn.classList.toggle('active', btn.dataset.mode === mode)
    );
  }

  status(message, kind = 'info', timeoutMs = 4500) {
    clearTimeout(this._statusTimer);
    this.statusEl.textContent = message;
    this.statusEl.className = `status show ${kind === 'info' ? '' : kind}`.trim();
    if (timeoutMs > 0) {
      this._statusTimer = setTimeout(() => {
        this.statusEl.classList.remove('show');
      }, timeoutMs);
    }
  }
}
