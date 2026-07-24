"""Mesh-based scene management (GLB format).

Handles:
- Per-object GLB meshes from TripoSR
- Scene manifest with object metadata
- Export as merged GLB or separate per-object files
"""

from __future__ import annotations

import json
from pathlib import Path

import trimesh


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
    objects_dir = scene_dir / "objects"
    if not objects_dir.exists():
        return {
            "name": scene_dir.name,
            "object_count": 0,
            "objects": [],
            "total_triangles": 0,
        }

    objects = []
    total_tris = 0

    for i, glb_path in enumerate(sorted(objects_dir.glob("*.glb"))):
        try:
            mesh = trimesh.load(glb_path, process=False)
            bounds = mesh.bounds if hasattr(mesh, "bounds") else [[0, 0, 0], [1, 1, 1]]
            tri_count = len(mesh.faces) if hasattr(mesh, "faces") else 0
            total_tris += tri_count

            # Load label from labels.json if exists
            labels_path = scene_dir / "labels.json"
            label = "object"
            if labels_path.exists():
                try:
                    labels = json.loads(labels_path.read_text())
                    label = labels.get(str(i), f"object {i}")
                except Exception:
                    pass

            objects.append({
                "id": i,
                "label": label,
                "mesh_url": f"/api/files/{scene_dir.name}/objects/{glb_path.name}",
                "bounds": {
                    "min": bounds[0].tolist(),
                    "max": bounds[1].tolist(),
                },
                "triangle_count": tri_count,
            })
        except Exception as e:
            print(f"Failed to load mesh {glb_path}: {e}")

    return {
        "name": scene_dir.name,
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
