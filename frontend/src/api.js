/** Thin typed-ish wrapper over the backend REST API. */

async function requestJson(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body.detail) detail = String(body.detail);
    } catch { /* non-JSON error body; keep the status code */ }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  /** @returns {Promise<Array<{id: string, name: string, splat_count: number, object_count: number}>>} */
  listScenes: () => requestJson('/api/scenes'),

  /** Full manifest with per-object centroid/bbox/color/ply_url. */
  getScene: (sceneId) => requestJson(`/api/scenes/${encodeURIComponent(sceneId)}`),

  /**
   * Merge edits server-side and return the exported PLY as a Blob.
   * @param {string} sceneId
   * @param {{objects: Array, include_object_id: boolean}} payload
   */
  async exportScene(sceneId, payload) {
    const res = await fetch(`/api/scenes/${encodeURIComponent(sceneId)}/export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try { detail = (await res.json()).detail ?? detail; } catch { /* keep code */ }
      throw new Error(detail);
    }
    return res.blob();
  },

  /** Upload a labeled scene.ply; backend splits it into per-object PLYs. */
  async uploadScene(sceneId, file) {
    const form = new FormData();
    form.append('scene_id', sceneId);
    form.append('file', file);
    return requestJson('/api/scenes', { method: 'POST', body: form });
  },
};
