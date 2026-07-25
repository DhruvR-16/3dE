"""FastAPI app for the Photo -> Editable 3D Scene editor.

Endpoints:
    GET  /api/health
    GET  /api/scenes                        -> list of scenes
    GET  /api/scenes/{sid}                  -> full manifest (objects, bboxes, ...)
    POST /api/scenes/reconstruct            -> image upload -> TripoSR -> GLB scene
    POST /api/scenes/{sid}/rebuild          -> force re-split from scene.ply
    POST /api/scenes/{sid}/export           -> merged export download from edit state
    POST /api/scenes                        -> upload a labeled scene.ply
    GET  /files/scenes/{sid}/objects/...    -> static per-object files (GLB or PLY)
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from . import scene_store, triposr_api, mesh_scene, art_to_3d
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


@app.post("/api/scenes/reconstruct", response_model=dict, status_code=status.HTTP_201_CREATED)
async def reconstruct_scene(
    scene_id: str = Form(...),
    images: list[UploadFile] = File(...),
) -> dict:
    """Reconstruct 3D scene from images using TripoSR.

    Each image becomes one object in the final scene.
    Processing takes ~10-30 sec per image.

    Args:
        scene_id: new scene name (validated)
        images: list of image files (jpg/png)

    Returns:
        {"scene": SceneManifest, "processing_time_sec": float}
    """
    try:
        scene_store.validate_scene_id(scene_id)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))

    # Validate images
    if not images or len(images) == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "at least 1 image required")
    if len(images) > 20:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "max 20 images per scene")

    scene_path = scene_store.scene_dir(scene_id)
    if scene_path.exists():
        raise HTTPException(status.HTTP_409_CONFLICT, f"scene {scene_id!r} already exists")

    scene_path.mkdir(parents=True, exist_ok=True)
    objects_dir = scene_path / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)

    labels = {}
    import time as time_mod
    start = time_mod.time()

    try:
        for i, upload in enumerate(images):
            if not upload.filename:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, f"image {i} has no filename")

            ext = Path(upload.filename).suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png"}:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    f"image {i}: unsupported format {ext}. Use jpg/png.",
                )

            # Save image temp, run TripoSR
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(await upload.read())
                tmp_path = Path(tmp.name)

            try:
                logger.info("TripoSR: processing %s (%d/%d)...", upload.filename, i + 1, len(images))
                glb_bytes = await run_in_threadpool(triposr_api.reconstruct_image_to_glb, tmp_path)
                obj_path = objects_dir / f"{i:03d}.glb"
                obj_path.write_bytes(glb_bytes)
                labels[str(i)] = upload.filename.rsplit(".", 1)[0]
                logger.info("TripoSR: saved %s", obj_path)
            except Exception as e:
                raise HTTPException(
                    status.HTTP_502_BAD_GATEWAY,
                    f"TripoSR failed on image {i} ({upload.filename}): {e}",
                )
            finally:
                tmp_path.unlink(missing_ok=True)

        # Save labels
        (scene_path / "labels.json").write_text(__import__("json").dumps(labels, indent=2))

        # Build manifest
        manifest = mesh_scene.build_mesh_manifest(scene_path)
        logger.info("Reconstructed scene %s: %d objects", scene_id, manifest["object_count"])

        elapsed = time_mod.time() - start
        return {"scene": manifest, "processing_time_sec": elapsed}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Reconstruction failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"reconstruction failed: {e}")


@app.post("/api/scenes/art", status_code=status.HTTP_201_CREATED)
async def art_to_3d_scene(
    scene_id: str = Form(...),
    images: list[UploadFile] = File(...),
    thickness: float = Form(0.9),
    parts: bool = Form(True),
) -> dict:
    """Turn flat 2D artwork into an editable, volumetric 3D scene.

    Silhouette inflation (see art_to_3d): depth is constructed from the outline
    rather than inferred, so it runs on CPU in well under a second and cannot
    collapse to a flat sheet the way learned image->3D models do on flat art.

    Each drawing is split into its natural parts; multiple drawings are laid out
    side by side so the result is one editable scene.
    """
    try:
        scene_store.validate_scene_id(scene_id)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    if not images:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "at least 1 image required")
    if len(images) > 12:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "max 12 images per scene")

    scene_path = scene_store.scene_dir(scene_id)
    if scene_path.exists():
        raise HTTPException(status.HTTP_409_CONFLICT, f"scene {scene_id!r} already exists")
    objects_dir = scene_path / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)

    import time as _time

    started = _time.time()
    labels: dict[str, str] = {}
    index = 0
    cursor_x = 0.0          # lay successive drawings out along X
    GAP = 0.15

    try:
        for upload in images:
            ext = Path(upload.filename or "art.png").suffix.lower()
            if ext not in art_to_3d.SUPPORTED_EXTS:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    f"unsupported format {ext!r}; use {sorted(art_to_3d.SUPPORTED_EXTS)}",
                )
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(await upload.read())
                tmp_path = Path(tmp.name)
            try:
                pieces = await run_in_threadpool(
                    art_to_3d.art_to_parts, tmp_path, parts=parts, thickness=thickness
                )
            except ValueError as e:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, f"{upload.filename}: {e}")
            finally:
                tmp_path.unlink(missing_ok=True)

            if not pieces:
                continue
            stem = Path(upload.filename or "art").stem

            lo = min(m.bounds[0][0] for m, _ in pieces)
            hi = max(m.bounds[1][0] for m, _ in pieces)
            shift = cursor_x - lo if len(images) > 1 else 0.0
            for j, (mesh, _mask) in enumerate(pieces):
                if shift:
                    mesh.apply_translation([shift, 0.0, 0.0])
                mesh.export(str(objects_dir / f"{index:03d}.glb"))
                labels[str(index)] = stem if len(pieces) == 1 else f"{stem} · part {j + 1}"
                index += 1
            cursor_x += (hi - lo) + GAP

        if index == 0:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "no usable silhouette found")

        (scene_path / "labels.json").write_text(json.dumps(labels, indent=2))
        manifest = mesh_scene.build_mesh_manifest(scene_path)
        logger.info("art->3D scene %s: %d object(s)", scene_id, manifest["object_count"])
        return {"scene": manifest, "processing_time_sec": _time.time() - started}

    except HTTPException:
        shutil.rmtree(scene_path, ignore_errors=True)   # don't leave a half-built scene
        raise
    except Exception as e:
        shutil.rmtree(scene_path, ignore_errors=True)
        logger.exception("art->3D failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"art->3D failed: {e}")


@app.get("/api/scenes", response_model=list[SceneSummary])
def get_scenes() -> list[SceneSummary]:
    return scene_store.list_scenes()


@app.get("/api/scenes/{scene_id}")
def get_scene(scene_id: str):
    """Return a scene manifest — splat (SceneManifest) or mesh (dict) shaped."""
    try:
        scene_store.validate_scene_id(scene_id)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    if scene_store.is_mesh_scene(scene_id):
        return mesh_scene.build_mesh_manifest(scene_store.scene_dir(scene_id))
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
    try:
        scene_store.validate_scene_id(scene_id)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))

    try:
        if scene_store.is_mesh_scene(scene_id):
            out_path = mesh_scene.export_mesh_scene(
                scene_store.scene_dir(scene_id), request.objects
            )
            media_type = "model/gltf-binary"
        else:
            _manifest_or_404(scene_id)
            out_path = scene_store.export_scene(scene_id, request)
            media_type = "application/octet-stream"
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return FileResponse(
        out_path,
        media_type=media_type,
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
def get_object_file(scene_id: str, filename: str) -> FileResponse:
    """Serve per-object files (PLY or GLB).

    Accepts both naming conventions to prevent directory traversal:
      - `obj_001.ply`  (splat scenes split by scene_store)
      - `001.glb`      (mesh scenes from the TripoSR reconstruct path)
    """
    if not re.match(r"^(obj_)?\d{3}\.(ply|glb)$", filename):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid object filename")
    try:
        sdir = scene_store.scene_dir(scene_id)  # validates scene_id against traversal
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    path = sdir / "objects" / filename
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{filename} not found")

    # Infer media type from extension
    media_type = "application/octet-stream"
    if filename.endswith(".glb"):
        media_type = "model/gltf-binary"
    elif filename.endswith(".ply"):
        media_type = "text/plain"

    return FileResponse(path, media_type=media_type)
