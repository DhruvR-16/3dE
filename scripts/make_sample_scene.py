"""Generate a synthetic labeled Gaussian-splat scene ("sample-room").

Builds a small Y-up room — floor, wall, table, chair, ball, lamp — out of
surfel-style Gaussians (disks aligned to each surface), assigns every splat an
object_id, and writes it as data/scenes/sample-room/scene.ply plus labels.json.
Finally it runs the backend's manifest/split step so the scene is immediately
servable.

This exists so the select/move/delete/export loop can be developed and demoed
without running the GPU reconstruction pipeline at all.

Usage:
    python scripts/make_sample_scene.py [--seed 7] [--scene-id sample-room]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.gaussian_ply import SH_C0, GaussianCloud, concat_clouds, save_gaussian_ply  # noqa: E402
from app import scene_store  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("make_sample_scene")

Vec3 = tuple[float, float, float]


# ---------------------------------------------------------------------------
# Surface samplers: each returns (points (N,3), normals (N,3), area)
# ---------------------------------------------------------------------------

def sample_plane(
    rng: np.random.Generator,
    center: Vec3,
    u_dir: Vec3,
    v_dir: Vec3,
    half_u: float,
    half_v: float,
    normal: Vec3,
    n: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    u = rng.uniform(-half_u, half_u, n)
    v = rng.uniform(-half_v, half_v, n)
    pts = (
        np.asarray(center)
        + u[:, None] * np.asarray(u_dir)
        + v[:, None] * np.asarray(v_dir)
    )
    normals = np.tile(np.asarray(normal, dtype=np.float64), (n, 1))
    return pts, normals, 4.0 * half_u * half_v


def sample_box(
    rng: np.random.Generator, center: Vec3, size: Vec3, n: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample a box surface with per-face area weighting; normals face outward."""
    sx, sy, sz = size
    faces = [  # (axis, sign, area)
        (0, +1, sy * sz), (0, -1, sy * sz),
        (1, +1, sx * sz), (1, -1, sx * sz),
        (2, +1, sx * sy), (2, -1, sx * sy),
    ]
    areas = np.array([f[2] for f in faces])
    counts = rng.multinomial(n, areas / areas.sum())
    half = np.array([sx, sy, sz]) / 2.0

    all_pts, all_nrm = [], []
    for (axis, sign, _), cnt in zip(faces, counts):
        if cnt == 0:
            continue
        pts = rng.uniform(-half, half, size=(cnt, 3))
        pts[:, axis] = sign * half[axis]
        nrm = np.zeros((cnt, 3))
        nrm[:, axis] = sign
        all_pts.append(pts)
        all_nrm.append(nrm)
    pts = np.concatenate(all_pts) + np.asarray(center)
    return pts, np.concatenate(all_nrm), float(areas.sum())


def sample_sphere(
    rng: np.random.Generator, center: Vec3, radius: float, n: int
) -> tuple[np.ndarray, np.ndarray, float]:
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.asarray(center) + radius * d, d, 4.0 * np.pi * radius**2


def sample_cylinder(
    rng: np.random.Generator,
    base_center: Vec3,
    radius: float,
    height: float,
    n: int,
    *,
    caps: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Vertical (Y-axis) cylinder; base_center is the center of the bottom cap."""
    lateral_area = 2.0 * np.pi * radius * height
    cap_area = np.pi * radius**2
    total = lateral_area + (2 * cap_area if caps else 0.0)
    n_lat = int(round(n * lateral_area / total))

    theta = rng.uniform(0, 2 * np.pi, n_lat)
    y = rng.uniform(0, height, n_lat)
    pts = np.stack([radius * np.cos(theta), y, radius * np.sin(theta)], axis=1)
    nrm = np.stack([np.cos(theta), np.zeros(n_lat), np.sin(theta)], axis=1)

    if caps:
        n_cap = max(1, (n - n_lat) // 2)
        for y_val, ny in ((0.0, -1.0), (height, 1.0)):
            r = radius * np.sqrt(rng.uniform(0, 1, n_cap))
            th = rng.uniform(0, 2 * np.pi, n_cap)
            cap_pts = np.stack([r * np.cos(th), np.full(n_cap, y_val), r * np.sin(th)], axis=1)
            cap_nrm = np.tile([0.0, ny, 0.0], (n_cap, 1))
            pts = np.concatenate([pts, cap_pts])
            nrm = np.concatenate([nrm, cap_nrm])
    return pts + np.asarray(base_center), nrm, total


def sample_cone_frustum(
    rng: np.random.Generator,
    base_center: Vec3,
    r_bottom: float,
    r_top: float,
    height: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Lateral surface of a vertical frustum (lampshade), normals outward."""
    t = rng.uniform(0, 1, n)  # 0 = bottom, 1 = top
    theta = rng.uniform(0, 2 * np.pi, n)
    r = r_bottom + t * (r_top - r_bottom)
    pts = np.stack([r * np.cos(theta), t * height, r * np.sin(theta)], axis=1)

    slope = (r_bottom - r_top) / height  # outward tilt of the surface normal
    nrm = np.stack([np.cos(theta), np.full(n, slope), np.sin(theta)], axis=1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    slant = np.hypot(height, r_bottom - r_top)
    area = np.pi * (r_bottom + r_top) * slant
    return pts + np.asarray(base_center), nrm, area


# ---------------------------------------------------------------------------
# Points+normals -> surfel Gaussians
# ---------------------------------------------------------------------------

def quats_z_to_normals(normals: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz quaternions rotating local +Z onto each normal."""
    z = np.array([0.0, 0.0, 1.0])
    dots = normals @ z
    w = 1.0 + dots
    axis = np.cross(np.tile(z, (len(normals), 1)), normals)
    q = np.concatenate([w[:, None], axis], axis=1)

    # Antiparallel normals (n ≈ -z) give a zero quaternion; use 180° about X.
    degenerate = w < 1e-8
    q[degenerate] = [0.0, 1.0, 0.0, 0.0]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.astype(np.float32)


def surfel_cloud(
    rng: np.random.Generator,
    points: np.ndarray,
    normals: np.ndarray,
    area: float,
    color: Vec3,
    object_id: int,
    *,
    radius_factor: float = 1.9,
    color_jitter: float = 0.035,
) -> GaussianCloud:
    """Turn sampled surface points into disk-shaped Gaussians lying on the surface."""
    n = len(points)
    spacing = float(np.sqrt(area / n))
    radii = spacing * radius_factor * rng.uniform(0.8, 1.25, n)
    scales = np.log(
        np.stack([radii, radii, radii * 0.12], axis=1).astype(np.float32)
    )

    rgb = np.clip(np.asarray(color) + rng.normal(0, color_jitter, (n, 3)), 0.02, 0.98)
    f_dc = ((rgb - 0.5) / SH_C0).astype(np.float32)

    alpha = rng.uniform(0.85, 0.97, n)
    opacity = np.log(alpha / (1.0 - alpha)).astype(np.float32)

    # Slight jitter along the normal hides the perfectly flat synthetic look.
    points = points + normals * rng.normal(0, 0.1 * spacing, (n, 1))

    return GaussianCloud(
        xyz=points.astype(np.float32),
        normals=np.zeros((n, 3), dtype=np.float32),
        f_dc=f_dc,
        f_rest=np.zeros((n, 0), dtype=np.float32),
        opacity=opacity,
        scales=scales,
        rots=quats_z_to_normals(normals),
        object_ids=np.full(n, object_id, dtype=np.uint32),
    )


# ---------------------------------------------------------------------------
# The room
# ---------------------------------------------------------------------------

def build_room(rng: np.random.Generator) -> tuple[GaussianCloud, dict[int, str]]:
    parts: list[GaussianCloud] = []
    labels: dict[int, str] = {}

    def add(object_id: int, label: str, samples: list, color: Vec3) -> None:
        labels[object_id] = label
        for pts, nrm, area in samples:
            parts.append(surfel_cloud(rng, pts, nrm, area, color, object_id))

    # 1. floor — light oak, 7 x 7 m
    add(1, "floor",
        [sample_plane(rng, (0, 0, 0), (1, 0, 0), (0, 0, 1), 3.5, 3.5, (0, 1, 0), 20000)],
        (0.72, 0.60, 0.46))

    # 2. back wall — soft sage
    add(2, "wall",
        [sample_plane(rng, (0, 1.4, -3.5), (1, 0, 0), (0, 1, 0), 3.5, 1.4, (0, 0, 1), 11000)],
        (0.80, 0.83, 0.78))

    # 3. table — walnut top + 4 legs, top surface at y = 0.78
    table_samples = [sample_box(rng, (0, 0.74, -1.2), (1.6, 0.08, 0.9), 8000)]
    for lx in (-0.72, 0.72):
        for lz in (-1.58, -0.82):
            table_samples.append(
                sample_cylinder(rng, (lx, 0.0, lz), 0.04, 0.70, 900)
            )
    add(3, "table", table_samples, (0.44, 0.30, 0.20))

    # 4. ball — coral, resting on the floor
    add(4, "ball", [sample_sphere(rng, (1.3, 0.22, 0.4), 0.22, 4200)], (0.90, 0.35, 0.31))

    # 5. lamp — dark base + stem, amber shade, standing on the table
    lamp_samples = [
        sample_cylinder(rng, (0.45, 0.78, -1.35), 0.09, 0.03, 500, caps=True),
        sample_cylinder(rng, (0.45, 0.81, -1.35), 0.015, 0.30, 700),
    ]
    add(5, "lamp", lamp_samples, (0.16, 0.16, 0.18))
    parts.append(  # amber shade shares the lamp's object_id

        surfel_cloud(
            rng,
            *sample_cone_frustum(rng, (0.45, 1.06, -1.35), 0.15, 0.06, 0.17, 2600),
            (0.95, 0.71, 0.34),
            5,
        )
    )

    # 6. chair — teal seat + backrest + 4 legs, tucked near the table
    chair_x, chair_z = -0.55, -0.35
    chair_samples = [
        sample_box(rng, (chair_x, 0.44, chair_z), (0.44, 0.06, 0.44), 2400),
        sample_box(rng, (chair_x, 0.72, chair_z - 0.20), (0.44, 0.52, 0.05), 2400),
    ]
    for lx in (chair_x - 0.18, chair_x + 0.18):
        for lz in (chair_z - 0.18, chair_z + 0.18):
            chair_samples.append(sample_cylinder(rng, (lx, 0.0, lz), 0.025, 0.41, 450))
    add(6, "chair", chair_samples, (0.20, 0.53, 0.53))

    return concat_clouds(parts), labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scene-id", default="sample-room")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cloud, labels = build_room(rng)
    logger.info("Generated %d splats across %d objects", len(cloud), len(labels))

    sdir = scene_store.scene_dir(args.scene_id)
    sdir.mkdir(parents=True, exist_ok=True)
    save_gaussian_ply(sdir / "scene.ply", cloud, include_object_id=True)
    (sdir / "labels.json").write_text(
        json.dumps({str(k): v for k, v in labels.items()}, indent=2)
    )

    manifest = scene_store.build_manifest(args.scene_id, force=True)
    for obj in manifest.objects:
        logger.info("  #%d %-8s %6d splats", obj.id, obj.label, obj.splat_count)
    logger.info("Scene ready: %s", sdir)


if __name__ == "__main__":
    main()
