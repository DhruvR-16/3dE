"""Scene registry: split labeled splat scenes into per-object PLYs and export edits.

On-disk layout, one directory per scene under DATA_DIR:

    data/scenes/<scene_id>/
        scene.ply        # source scene, every splat carries object_id
        labels.json      # optional {"1": "floor", ...} human labels
        manifest.json    # generated: per-object metadata (centroid, bbox, ...)
        objects/
            obj_001.ply  # per-object splats, positions recentred at centroid
        exports/
            edited_*.ply # merged exports produced from client edit state
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import numpy as np

from .gaussian_ply import (
    GaussianCloud,
    concat_clouds,
    load_gaussian_ply,
    save_gaussian_ply,
    transform_cloud,
)
from .models import ExportRequest, ObjectInfo, SceneManifest, SceneSummary

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "scenes"

_SCENE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def validate_scene_id(scene_id: str) -> str:
    """Reject ids that could escape DATA_DIR (path traversal) or collide."""
    if not _SCENE_ID_RE.match(scene_id):
        raise ValueError(
            f"invalid scene id {scene_id!r}: use lowercase letters, digits, '-', '_'"
        )
    return scene_id


def scene_dir(scene_id: str) -> Path:
    return DATA_DIR / validate_scene_id(scene_id)


def _object_ply_name(object_id: int) -> str:
    return f"obj_{object_id:03d}.ply"


def _load_labels(sdir: Path) -> dict[int, str]:
    labels_path = sdir / "labels.json"
    if not labels_path.exists():
        return {}
    try:
        raw = json.loads(labels_path.read_text())
        return {int(k): str(v) for k, v in raw.items()}
    except (ValueError, TypeError) as e:
        logger.warning("Ignoring malformed labels.json in %s: %s", sdir, e)
        return {}


def build_manifest(scene_id: str, *, force: bool = False) -> SceneManifest:
    """Split scene.ply by object_id into per-object PLYs and write manifest.json.

    Per-object PLYs are recentred at each object's centroid so the frontend can
    rotate/scale around a sensible pivot; the centroid is recorded in the
    manifest and applied as the object's initial position.
    """
    sdir = scene_dir(scene_id)
    src = sdir / "scene.ply"
    if not src.exists():
        raise FileNotFoundError(f"scene {scene_id!r}: missing {src}")

    manifest_path = sdir / "manifest.json"
    if manifest_path.exists() and not force:
        if manifest_path.stat().st_mtime >= src.stat().st_mtime:
            return SceneManifest.model_validate_json(manifest_path.read_text())

    logger.info("Building manifest for scene %s", scene_id)
    cloud = load_gaussian_ply(src)
    if cloud.object_ids is None:
        # Unlabeled scene: treat the whole thing as one editable object.
        cloud.object_ids = np.ones(len(cloud), dtype=np.uint32)

    labels = _load_labels(sdir)
    objects_dir = sdir / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)
    for stale in objects_dir.glob("obj_*.ply"):
        stale.unlink()

    infos: list[ObjectInfo] = []
    for oid in sorted(int(v) for v in np.unique(cloud.object_ids)):
        part = cloud.select(cloud.object_ids == oid)
        centroid = part.xyz.mean(axis=0)
        recentred = part.translated(-centroid)
        recentred.object_ids = None
        save_gaussian_ply(objects_dir / _object_ply_name(oid), recentred)

        infos.append(
            ObjectInfo(
                id=oid,
                label=labels.get(oid, "background" if oid == 0 else f"object {oid}"),
                splat_count=len(part),
                centroid=[float(c) for c in centroid],
                bbox_min=[float(c) for c in part.xyz.min(axis=0)],
                bbox_max=[float(c) for c in part.xyz.max(axis=0)],
                color=[float(c) for c in part.colors_rgb().mean(axis=0)],
                ply_url=f"/files/scenes/{scene_id}/objects/{_object_ply_name(oid)}",
            )
        )

    manifest = SceneManifest(
        id=scene_id,
        name=scene_id.replace("-", " ").replace("_", " ").title(),
        splat_count=len(cloud),
        sh_rest_dim=cloud.sh_rest_dim,
        objects=infos,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    return manifest


def list_scenes() -> list[SceneSummary]:
    if not DATA_DIR.exists():
        return []
    summaries: list[SceneSummary] = []
    for sdir in sorted(DATA_DIR.iterdir()):
        if not (sdir / "scene.ply").exists():
            continue
        try:
            m = build_manifest(sdir.name)
        except (ValueError, FileNotFoundError) as e:
            logger.warning("Skipping scene %s: %s", sdir.name, e)
            continue
        summaries.append(
            SceneSummary(
                id=m.id,
                name=m.name,
                splat_count=m.splat_count,
                object_count=len(m.objects),
            )
        )
    return summaries


def export_scene(scene_id: str, request: ExportRequest) -> Path:
    """Merge per-object PLYs with the client's edit transforms into one PLY."""
    sdir = scene_dir(scene_id)
    manifest = build_manifest(scene_id)
    known_ids = {o.id for o in manifest.objects}

    parts: list[GaussianCloud] = []
    for edit in request.objects:
        if edit.deleted:
            continue
        if edit.id not in known_ids:
            raise ValueError(f"unknown object id {edit.id} for scene {scene_id!r}")
        part = load_gaussian_ply(sdir / "objects" / _object_ply_name(edit.id))
        part = transform_cloud(
            part,
            translation=np.array(edit.position),
            quaternion_xyzw=np.array(edit.quaternion),
            uniform_scale=edit.scale,
        )
        if request.include_object_id:
            part.object_ids = np.full(len(part), edit.id, dtype=np.uint32)
        parts.append(part)

    if not parts:
        raise ValueError("export would be empty: every object is deleted")

    merged = concat_clouds(parts)
    exports_dir = sdir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    out_path = exports_dir / f"edited_{time.strftime('%Y%m%d_%H%M%S')}.ply"
    save_gaussian_ply(out_path, merged, include_object_id=request.include_object_id)
    logger.info("Exported %d splats -> %s", len(merged), out_path)
    return out_path


def import_labeled_ply(scene_id: str, ply_bytes: bytes) -> SceneManifest:
    """Create a new scene from an uploaded labeled (or unlabeled) 3DGS PLY."""
    sdir = scene_dir(scene_id)
    if (sdir / "scene.ply").exists():
        raise FileExistsError(f"scene {scene_id!r} already exists")
    sdir.mkdir(parents=True, exist_ok=True)
    dst = sdir / "scene.ply"
    dst.write_bytes(ply_bytes)
    try:
        return build_manifest(scene_id, force=True)
    except ValueError:
        # Not a valid splat PLY — roll back so a bad upload doesn't leave a
        # broken scene directory behind.
        dst.unlink(missing_ok=True)
        raise
