import './style.css';
import { api } from './api.js';
import { SplatEditor } from './viewer.js';
import { MeshEditor } from './mesh-viewer.js';
import { UI } from './ui.js';

let editor = null;
let ui = null;
let currentManifest = null;
let isUsingMeshViewer = false;

async function loadScene(sceneId) {
  try {
    const manifest = await api.getScene(sceneId);
    currentManifest = manifest;
    ui.setObjects(manifest.objects);
    ui.setSelection(null, false);

    // Detect if this is a mesh scene (GLB) or splat scene (PLY)
    const isMeshScene = manifest.objects.some(o => o.mesh_url?.endsWith('.glb'));
    if (isMeshScene && !isUsingMeshViewer) {
      // Switch to mesh viewer
      console.log('Switching to mesh viewer...');
      if (editor) editor.dispose?.();
      editor = new MeshEditor(document.getElementById('viewer-root'));
      isUsingMeshViewer = true;
    } else if (!isMeshScene && isUsingMeshViewer) {
      // Switch to splat viewer
      console.log('Switching to splat viewer...');
      if (editor) editor.dispose?.();
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
      isUsingMeshViewer = false;
    }

    editor.statusCallback = (msg, kind) => ui.status(msg, kind);
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

async function handleReconstruct(sceneId, images) {
  ui.status(`Reconstructing from ${images.length} image(s)…`, 'info', 0);
  const progressEl = document.getElementById('reconstruct-progress');
  const progressBar = document.getElementById('progress-bar');
  const progressText = document.getElementById('progress-text');

  progressEl.style.display = 'block';
  progressBar.style.width = '0%';

  try {
    const formData = new FormData();
    formData.append('scene_id', sceneId);
    images.forEach((img) => {
      formData.append('images', img);
    });

    const response = await fetch('/api/scenes/reconstruct', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || response.statusText);
    }

    progressText.textContent = 'Processing...';
    progressBar.style.width = '30%';

    const result = await response.json();
    progressBar.style.width = '80%';

    await refreshSceneList(result.scene.name);
    progressBar.style.width = '100%';
    progressText.textContent = `Done in ${result.processing_time_sec.toFixed(1)}s`;

    setTimeout(() => {
      progressEl.style.display = 'none';
      ui.status(`Reconstructed ${result.scene.name}: ${result.scene.object_count} objects`, 'ok');
      loadScene(result.scene.name);
    }, 500);
  } catch (err) {
    console.error(err);
    progressEl.style.display = 'none';
    ui.status(`Reconstruction failed: ${err.message}`, 'error', 0);
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
    onSelect: (objectId) => editor.select?.(objectId) || editor.selectObject?.(objectId),
    onModeChange: (mode) => {
      editor.setTransformMode?.(mode) || editor.setMode?.(mode);
      ui.setMode(mode);
    },
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
