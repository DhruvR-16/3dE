"""Mesh-based scene management (GLB format).

Handles:
- Per-object GLB meshes from TripoSR
- Scene manifest with object metadata
- Export as merged GLB or separate per-object files
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import trimesh


def _mesh_stats(loaded) -> tuple[int, list[list[float]]]:
    """Triangle count + AABB for a trimesh result, which may be a Trimesh or Scene.

    TripoSR GLBs load as a Scene, whose `.faces` is empty — the geometry lives in
    `.geometry`. Returns (triangle_count, [[minx,miny,minz],[maxx,maxy,maxz]]).
    """
    if isinstance(loaded, trimesh.Scene):
        tris = sum(len(g.faces) for g in loaded.geometry.values())
    else:
        tris = len(getattr(loaded, "faces", []))
    bounds = loaded.bounds
    if bounds is None:
        bounds = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    return tris, [bounds[0].tolist(), bounds[1].tolist()]


def build_mesh_manifest(scene_dir: Path) -> dict:
    """Build scene manifest from GLB objects.

    Scans scene_dir/objects/*.glb and creates metadata.

    Args:
        scene_dir: path to scene directory (data/scenes/<id>/)

    Returns:
        {"name": str, "object_count": int, "objects": [
            {"id": int, "label": str, "mesh_url": str, "bounds": {...}}
        ]}
    """
    display_name = scene_dir.name.replace("-", " ").replace("_", " ").title()
    objects_dir = scene_dir / "objects"
    if not objects_dir.exists():
        return {
            "id": scene_dir.name,
            "name": display_name,
            "object_count": 0,
            "objects": [],
            "total_triangles": 0,
        }

    objects = []
    total_tris = 0

    labels_path = scene_dir / "labels.json"
    labels = {}
    if labels_path.exists():
        try:
            labels = json.loads(labels_path.read_text())
        except (ValueError, OSError):
            labels = {}

    for i, glb_path in enumerate(sorted(objects_dir.glob("*.glb"))):
        try:
            mesh = trimesh.load(glb_path, process=False)
            tri_count, bounds = _mesh_stats(mesh)
            total_tris += tri_count

            objects.append({
                "id": i,
                "label": labels.get(str(i), f"object {i}"),
                "mesh_url": f"/files/scenes/{scene_dir.name}/objects/{glb_path.name}",
                "bounds": {"min": bounds[0], "max": bounds[1]},
                "triangle_count": tri_count,
            })
        except Exception as e:  # noqa: BLE001 — surface but don't abort the whole scene
            print(f"Failed to load mesh {glb_path}: {e}")

    return {
        "id": scene_dir.name,
        "name": display_name,
        "object_count": len(objects),
        "objects": objects,
        "total_triangles": total_tris,
    }


def _trs_matrix(position, quaternion_xyzw, uniform_scale: float) -> np.ndarray:
    """Compose a 4x4 transform: translate * rotate * scale (matches Three.js)."""
    x, y, z, w = (float(v) for v in quaternion_xyzw)
    rot = trimesh.transformations.quaternion_matrix([w, x, y, z])  # trimesh wants wxyz
    scale = np.diag([uniform_scale, uniform_scale, uniform_scale, 1.0])
    translate = trimesh.transformations.translation_matrix([float(v) for v in position])
    return translate @ rot @ scale


def export_mesh_scene(scene_dir: Path, edits) -> Path:
    """Merge per-object GLBs into one GLB with the client's edit transforms baked in.

    TripoSR meshes carry per-vertex colors (no textures), so a plain
    concatenate preserves appearance losslessly.

    Args:
        scene_dir: the scene directory (data/scenes/<id>/)
        edits: iterable of edit records with .id, .deleted, .position (xyz),
               .quaternion (xyzw), and .scale (uniform).

    Returns:
        Path to the written GLB under <scene_dir>/exports/.
    """
    objects_dir = scene_dir / "objects"
    parts: list[trimesh.Trimesh] = []
    for edit in edits:
        if edit.deleted:
            continue
        glb_path = objects_dir / f"{int(edit.id):03d}.glb"
        if not glb_path.exists():
            raise ValueError(f"unknown object id {edit.id} for scene {scene_dir.name!r}")
        # force='mesh' flattens the GLB's Scene into one Trimesh (colors kept).
        mesh = trimesh.load(glb_path, process=False, force="mesh")
        mesh.apply_transform(_trs_matrix(edit.position, edit.quaternion, edit.scale))
        parts.append(mesh)

    if not parts:
        raise ValueError("export would be empty: every object is deleted")

    merged = parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)
    exports_dir = scene_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    out_path = exports_dir / f"edited_{time.strftime('%Y%m%d_%H%M%S')}.glb"
    merged.export(str(out_path), file_type="glb")
    return out_path
