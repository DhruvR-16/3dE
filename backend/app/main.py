"""FastAPI app for the Photo -> Editable 3D Scene editor.

Endpoints:
    GET  /api/health
    GET  /api/scenes                     -> list of scenes
    GET  /api/scenes/{sid}               -> full manifest (objects, bboxes, ...)
    POST /api/scenes/{sid}/rebuild       -> force re-split from scene.ply
    POST /api/scenes/{sid}/export        -> merged PLY download from edit state
    POST /api/scenes                     -> upload a labeled scene.ply
    GET  /files/scenes/{sid}/objects/... -> static per-object PLYs
"""

from __future__ import annotations

import logging
import re

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from . import scene_store
from .models import ExportRequest, SceneManifest, SceneSummary, UploadResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Editable 3D Scene API", version="0.1.0")

# Dev origins only; the Vite dev server proxies /api and /files so requests are
# usually same-origin anyway. Tighten before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_OBJECT_PLY_RE = re.compile(r"^obj_\d{3}\.ply$")


def _manifest_or_404(scene_id: str) -> SceneManifest:
    try:
        return scene_store.build_manifest(scene_id)
    except FileNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"scene {scene_id!r} not found")
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/scenes", response_model=list[SceneSummary])
def get_scenes() -> list[SceneSummary]:
    return scene_store.list_scenes()


@app.get("/api/scenes/{scene_id}", response_model=SceneManifest)
def get_scene(scene_id: str) -> SceneManifest:
    return _manifest_or_404(scene_id)


@app.post("/api/scenes/{scene_id}/rebuild", response_model=SceneManifest)
def rebuild_scene(scene_id: str) -> SceneManifest:
    try:
        return scene_store.build_manifest(scene_id, force=True)
    except FileNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"scene {scene_id!r} not found")
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))


@app.post("/api/scenes/{scene_id}/export")
def export_scene(scene_id: str, request: ExportRequest) -> FileResponse:
    _manifest_or_404(scene_id)
    try:
        out_path = scene_store.export_scene(scene_id, request)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return FileResponse(
        out_path,
        media_type="application/octet-stream",
        filename=f"{scene_id}_{out_path.name}",
    )


@app.post(
    "/api/scenes",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_scene(
    scene_id: str = Form(...),
    file: UploadFile = File(...),
) -> UploadResponse:
    try:
        scene_store.validate_scene_id(scene_id)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))

    ply_bytes = await file.read()
    if len(ply_bytes) < 16 or not ply_bytes.startswith(b"ply"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "file is not a PLY")
    try:
        manifest = scene_store.import_labeled_ply(scene_id, ply_bytes)
    except FileExistsError as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e))
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"invalid splat PLY: {e}")
    logger.info("Imported scene %s (%d splats)", scene_id, manifest.splat_count)
    return UploadResponse(scene=manifest)


@app.get("/files/scenes/{scene_id}/objects/{filename}")
def get_object_ply(scene_id: str, filename: str) -> FileResponse:
    if not _OBJECT_PLY_RE.match(filename):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid object filename")
    _manifest_or_404(scene_id)  # also validates scene_id against traversal
    path = scene_store.scene_dir(scene_id) / "objects" / filename
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{filename} not found")
    return FileResponse(path, media_type="application/octet-stream")
