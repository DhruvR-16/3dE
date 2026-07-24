import './style.css';
import { api } from './api.js';
import { SplatEditor } from './viewer.js';
import { UI } from './ui.js';

let editor = null;
let ui = null;
let currentManifest = null;

async function loadScene(sceneId) {
  try {
    const manifest = await api.getScene(sceneId);
    currentManifest = manifest;
    ui.setObjects(manifest.objects);
    ui.setSelection(null, false);
    await editor.loadScene(manifest);
  } catch (err) {
    console.error(err);
    ui.status(`Failed to load scene: ${err.message}`, 'error', 0);
  }
}

async function refreshSceneList(activeId) {
  const scenes = await api.listScenes();
  ui.setScenes(scenes, activeId);
  return scenes;
}

function selectedRecordState() {
  const id = editor.selectedId;
  if (id === null) return { id: null, deleted: false };
  return { id, deleted: editor.records.get(id)?.deleted ?? false };
}

async function handleExport() {
  if (!currentManifest) return;
  ui.status('Exporting…', 'info', 0);
  try {
    const blob = await api.exportScene(currentManifest.id, {
      objects: editor.getEditState(),
      include_object_id: ui.exportLabeled.checked,
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentManifest.id}_edited.ply`;
    a.click();
    URL.revokeObjectURL(url);
    ui.status(`Exported ${(blob.size / 1e6).toFixed(1)} MB .ply`, 'ok');
  } catch (err) {
    ui.status(`Export failed: ${err.message}`, 'error');
  }
}

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

function bindKeyboard() {
  window.addEventListener('keydown', (e) => {
    const tag = e.target?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    switch (e.code) {
      case 'KeyG': editor.setMode('translate'); ui.setMode('translate'); break;
      case 'KeyR': editor.setMode('rotate'); ui.setMode('rotate'); break;
      case 'KeyS': editor.setMode('scale'); ui.setMode('scale'); break;
      case 'KeyX':
      case 'Delete':
      case 'Backspace': {
        const { id, deleted } = selectedRecordState();
        if (id !== null && !deleted) editor.deleteObject(id);
        break;
      }
      case 'Escape': editor.selectObject(null); break;
      default: return;
    }
    e.preventDefault();
  });
}

async function init() {
  ui = new UI({
    onSceneChange: loadScene,
    onSelect: (objectId) => editor.selectObject(objectId),
    onModeChange: (mode) => { editor.setMode(mode); ui.setMode(mode); },
    onDelete: () => {
      const { id, deleted } = selectedRecordState();
      if (id !== null && !deleted) editor.deleteObject(id);
    },
    onRestore: () => {
      const { id, deleted } = selectedRecordState();
      if (id !== null && deleted) editor.restoreObject(id);
    },
    onExport: handleExport,
    onUpload: handleUpload,
  });

  editor = new SplatEditor(document.getElementById('viewer-root'), {
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
  });

  bindKeyboard();
  window.__editor = editor; // dev console access

  try {
    const scenes = await refreshSceneList();
    if (scenes.length === 0) {
      ui.status(
        'No scenes found — run "python scripts/make_sample_scene.py" then reload',
        'error',
        0
      );
      return;
    }
    await loadScene(scenes[0].id);
  } catch (err) {
    ui.status(
      `Backend unreachable (${err.message}) — start it with: uvicorn app.main:app --app-dir backend --port 8000`,
      'error',
      0
    );
  }
}

init();
