import './style.css';
import { api } from './api.js';
import { SplatEditor } from './viewer.js';
import { MeshEditor } from './mesh-viewer.js';
import { UI } from './ui.js';

let editor = null;
let ui = null;
let currentManifest = null;
let isUsingMeshViewer = false;
let availableScenes = [];

/* ----------------------------- view switching ---------------------------- */

function showView(view) {
  document.documentElement.setAttribute('data-view', view);
  const homeBtn = document.getElementById('btn-home');
  if (homeBtn) homeBtn.hidden = view !== 'editor';
  if (view === 'home') {
    editor?.deselect?.();
    editor?.selectObject?.(null);
  }
}

/* --------------------------- editor abstraction -------------------------- */
// SplatEditor (PLY) and MeshEditor (GLB) expose slightly different method
// names; these helpers call whichever exists so the app works with both.

function editorSetMode(mode) {
  if (editor?.setTransformMode) editor.setTransformMode(mode);
  else editor?.setMode?.(mode);
}
function editorSelect(id) {
  if (editor?.select) editor.select(id);
  else editor?.selectObject?.(id);
}
function editorDeselect() {
  if (editor?.deselect) editor.deselect();
  else editor?.selectObject?.(null);
}

function selectedRecordState() {
  if (!editor) return { id: null, deleted: false };
  const id = editor.selectedId ?? editor.selected ?? null;
  if (id === null || id === undefined) return { id: null, deleted: false };
  return { id, deleted: editor.records?.get(id)?.deleted ?? false };
}

function editorCallbacks() {
  return {
    onSelectionChanged: () => {
      const { id, deleted } = selectedRecordState();
      ui.setSelection(id, deleted);
    },
    onObjectStateChanged: (objectId, deleted) => {
      ui.setObjectDeleted(objectId, deleted);
      const sel = selectedRecordState();
      ui.setSelection(sel.id, sel.deleted);
    },
    onStatus: (msg, kind) => ui.status(msg, kind),
  };
}

function makeSplatEditor() {
  return new SplatEditor(document.getElementById('viewer-root'), editorCallbacks());
}

function makeMeshEditor() {
  return new MeshEditor(document.getElementById('viewer-root'), editorCallbacks());
}

async function disposeEditor() {
  if (!editor) return;
  if (editor.disposeViewer) await editor.disposeViewer(); // SplatEditor
  else editor.dispose?.();                                 // MeshEditor
}

/* ------------------------------ scene loading ---------------------------- */

async function loadScene(sceneId) {
  try {
    const manifest = await api.getScene(sceneId);
    currentManifest = manifest;
    ui.setObjects(manifest.objects);
    ui.setSelection(null, false);

    const isMeshScene = manifest.objects.some((o) => o.mesh_url?.endsWith('.glb'));
    if (isMeshScene && !isUsingMeshViewer) {
      await disposeEditor();
      editor = makeMeshEditor();
      isUsingMeshViewer = true;
    } else if (!isMeshScene && isUsingMeshViewer) {
      await disposeEditor();
      editor = makeSplatEditor();
      isUsingMeshViewer = false;
    }

    editor.statusCallback = (msg, kind) => ui.status(msg, kind);
    window.__editor = editor; // dev console access (kept in sync across viewer swaps)
    await editor.loadScene(manifest);
    ui.setScenes(availableScenes, manifest.id ?? sceneId);
    showView('editor');
  } catch (err) {
    console.error(err);
    ui.status(`Failed to load scene: ${err.message}`, 'error', 0);
  }
}

async function refreshSceneList(activeId) {
  availableScenes = await api.listScenes();
  ui.setScenes(availableScenes, activeId);
  return availableScenes;
}

/* --------------------------------- export -------------------------------- */

async function handleExport() {
  if (!currentManifest) return;
  ui.status('Exporting…', 'info', 0);
  try {
    const editState = editor.getEditState ? editor.getEditState() : await editor.export?.();
    const blob = await api.exportScene(currentManifest.id, {
      objects: editState,
      include_object_id: ui.exportLabeled.checked,
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentManifest.id}_edited.${isUsingMeshViewer ? 'glb' : 'ply'}`;
    a.click();
    URL.revokeObjectURL(url);
    ui.status(`Exported ${(blob.size / 1e6).toFixed(1)} MB`, 'ok');
  } catch (err) {
    ui.status(`Export failed: ${err.message}`, 'error');
  }
}

/* --------------------------- import labeled .ply ------------------------- */

async function handleUpload(sceneId, file) {
  ui.status(`Uploading ${file.name}…`, 'info', 0);
  try {
    const { scene } = await api.uploadScene(sceneId, file);
    await refreshSceneList(scene.id);
    ui.status(`Imported ${scene.id}: ${scene.objects.length} objects`, 'ok');
    await loadScene(scene.id);
  } catch (err) {
    ui.status(`Upload failed: ${err.message}`, 'error');
  }
}

/* ---------------------- reconstruct from images (TripoSR) ---------------- */

async function handleReconstruct(rawSceneId, images) {
  const sceneId = (rawSceneId || `scene-${Date.now().toString(36)}`)
    .toLowerCase()
    .replace(/[^a-z0-9-]+/g, '-')
    .replace(/^-+|-+$/g, '') || `scene-${Date.now().toString(36)}`;

  const progressEl = document.getElementById('reconstruct-progress');
  const progressBar = document.getElementById('progress-bar');
  const progressText = document.getElementById('progress-text');
  const btn = document.getElementById('reconstruct-btn');

  progressEl.hidden = false;
  progressBar.style.width = '8%';
  progressText.textContent = `Uploading ${images.length} image(s)…`;
  btn.disabled = true;
  ui.status(`Reconstructing “${sceneId}” from ${images.length} image(s)…`, 'info', 0);

  try {
    const formData = new FormData();
    formData.append('scene_id', sceneId);
    images.forEach((img) => formData.append('images', img));

    progressBar.style.width = '35%';
    progressText.textContent = 'Building 3D objects…';

    const response = await fetch('/api/scenes/reconstruct', {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      let detail = response.statusText;
      try { detail = (await response.json()).detail ?? detail; } catch { /* keep */ }
      throw new Error(detail);
    }

    const result = await response.json();
    progressBar.style.width = '100%';
    progressText.textContent = `Done in ${result.processing_time_sec?.toFixed?.(1) ?? '?'}s`;

    await refreshSceneList(result.scene.name);
    ui.status(`Reconstructed ${result.scene.name}: ${result.scene.object_count} objects`, 'ok');
    await loadScene(result.scene.name);
  } catch (err) {
    console.error(err);
    ui.status(`Reconstruction failed: ${err.message}`, 'error', 0);
  } finally {
    btn.disabled = false;
    setTimeout(() => { progressEl.hidden = true; progressBar.style.width = '0%'; }, 900);
  }
}

/* --------------------------- drag & drop on home ------------------------- */

function renderFileChips(files) {
  const list = document.getElementById('dropzone-files');
  list.replaceChildren(
    ...[...files].slice(0, 20).map((f) => {
      const li = document.createElement('li');
      li.textContent = f.name;
      return li;
    })
  );
}

function bindDropzone() {
  const dz = document.getElementById('dropzone');
  const input = document.getElementById('reconstruct-images');
  if (!dz || !input) return;

  dz.addEventListener('click', (e) => {
    if (e.target.closest('input')) return;
    input.click();
  });
  input.addEventListener('change', () => renderFileChips(input.files));

  ['dragenter', 'dragover'].forEach((evt) =>
    dz.addEventListener(evt, (e) => {
      e.preventDefault();
      dz.classList.add('dragging');
    })
  );
  ['dragleave', 'dragend'].forEach((evt) =>
    dz.addEventListener(evt, (e) => {
      if (evt === 'dragleave' && dz.contains(e.relatedTarget)) return;
      dz.classList.remove('dragging');
    })
  );
  dz.addEventListener('drop', (e) => {
    e.preventDefault();
    dz.classList.remove('dragging');
    const dropped = [...(e.dataTransfer?.files ?? [])].filter((f) =>
      /image\/(jpeg|png)/.test(f.type)
    );
    if (!dropped.length) {
      ui.status('Drop JPG or PNG images', 'error');
      return;
    }
    const dt = new DataTransfer();
    dropped.forEach((f) => dt.items.add(f));
    input.files = dt.files;
    renderFileChips(input.files);
  });
}

/* -------------------------------- keyboard ------------------------------- */

function bindKeyboard() {
  window.addEventListener('keydown', (e) => {
    if (document.documentElement.getAttribute('data-view') !== 'editor') return;
    const tag = e.target?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    switch (e.code) {
      case 'KeyG': editorSetMode('translate'); ui.setMode('translate'); break;
      case 'KeyR': editorSetMode('rotate'); ui.setMode('rotate'); break;
      case 'KeyS': editorSetMode('scale'); ui.setMode('scale'); break;
      case 'KeyX':
      case 'Delete':
      case 'Backspace': {
        const { id, deleted } = selectedRecordState();
        if (id !== null && !deleted) editor.deleteObject?.(id);
        break;
      }
      case 'Escape': editorDeselect(); break;
      default: return;
    }
    e.preventDefault();
  });
}

/* ---------------------------------- init --------------------------------- */

async function init() {
  ui = new UI({
    onSceneChange: loadScene,
    onSelect: (objectId) => editorSelect(objectId),
    onModeChange: (mode) => { editorSetMode(mode); ui.setMode(mode); },
    onDelete: () => {
      const { id, deleted } = selectedRecordState();
      if (id !== null && !deleted) editor.deleteObject?.(id);
    },
    onRestore: () => {
      const { id, deleted } = selectedRecordState();
      if (id !== null && deleted) editor.restoreObject?.(id);
    },
    onExport: handleExport,
    onUpload: handleUpload,
    onReconstruct: handleReconstruct,
  });

  editor = makeSplatEditor();
  isUsingMeshViewer = false;

  bindKeyboard();
  bindDropzone();
  showView('home');
  window.__editor = editor;

  document.getElementById('btn-home')?.addEventListener('click', () => showView('home'));
  document.getElementById('btn-try-sample')?.addEventListener('click', () => {
    if (availableScenes.length) loadScene(availableScenes[0].id);
    else ui.status('No sample scene found on the backend', 'error');
  });

  try {
    const scenes = await refreshSceneList();
    if (scenes.length === 0) {
      ui.status('No scenes yet — reconstruct from images to get started', 'info', 6000);
    }
  } catch (err) {
    ui.status(
      `Backend unreachable (${err.message}) — start it: uvicorn app.main:app --app-dir backend --port 8000`,
      'error',
      0
    );
  }
}

init();
