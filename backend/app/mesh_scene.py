"""Mesh-based scene management (GLB format).

Handles:
- Per-object GLB meshes from TripoSR
- Scene manifest with object metadata
- Export as merged GLB or separate per-object files
"""

from __future__ import annotations

import json
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


def merge_meshes_to_glb(scene_dir: Path, output_path: Path) -> None:
    """Merge all per-object GLB files into a single GLB.

    Args:
        scene_dir: path to scene directory (data/scenes/<id>/)
        output_path: where to write merged GLB
    """
    objects_dir = scene_dir / "objects"
    if not objects_dir.exists():
        raise ValueError(f"No objects directory: {objects_dir}")

    # Load and combine all meshes
    combined = None
    for glb_path in sorted(objects_dir.glob("*.glb")):
        mesh = trimesh.load(glb_path, process=False)
        if combined is None:
            combined = mesh
        else:
            combined = trimesh.util.concatenate([combined, mesh])

    if combined is None:
        raise ValueError("No meshes to merge")

    combined.export(str(output_path), file_type="glb")
    print(f"Merged {len(list(objects_dir.glob('*.glb')))} meshes -> {output_path}")
