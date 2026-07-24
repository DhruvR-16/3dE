"""Read/write/transform 3D Gaussian Splatting PLY files.

The on-disk layout follows the INRIA 3DGS convention:
    x, y, z, nx, ny, nz,
    f_dc_0..2,                      # SH degree-0 color coefficients
    f_rest_0..(3*((d+1)^2-1)-1),    # higher-order SH (optional, K columns)
    opacity,                        # logit-space
    scale_0..2,                     # log-space
    rot_0..3                        # quaternion, (w, x, y, z), unnormalized

We additionally support one extra per-splat property, `object_id` (uint32),
which is what makes a scene "editable": every Gaussian knows which object it
belongs to. 0 is reserved for "background / unassigned".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814  # Y_0^0 spherical harmonics constant

_F_REST_RE = re.compile(r"^f_rest_(\d+)$")


@dataclass
class GaussianCloud:
    """In-memory Gaussian splat scene. All arrays share the same length N."""

    xyz: np.ndarray  # (N, 3) float32 positions
    normals: np.ndarray  # (N, 3) float32, conventionally zeros
    f_dc: np.ndarray  # (N, 3) float32 SH DC coefficients
    f_rest: np.ndarray  # (N, K) float32 higher-order SH, K may be 0
    opacity: np.ndarray  # (N,)  float32 logit opacities
    scales: np.ndarray  # (N, 3) float32 log scales
    rots: np.ndarray  # (N, 4) float32 quaternions (w, x, y, z)
    object_ids: np.ndarray | None = None  # (N,) uint32, or None if unlabeled

    def __len__(self) -> int:
        return self.xyz.shape[0]

    @property
    def sh_rest_dim(self) -> int:
        return self.f_rest.shape[1]

    def colors_rgb(self) -> np.ndarray:
        """Approximate per-splat base color in [0, 1] from the SH DC term."""
        return np.clip(0.5 + SH_C0 * self.f_dc, 0.0, 1.0)

    def select(self, mask: np.ndarray) -> "GaussianCloud":
        """Return a new cloud containing only splats where mask is True."""
        return GaussianCloud(
            xyz=self.xyz[mask],
            normals=self.normals[mask],
            f_dc=self.f_dc[mask],
            f_rest=self.f_rest[mask],
            opacity=self.opacity[mask],
            scales=self.scales[mask],
            rots=self.rots[mask],
            object_ids=None if self.object_ids is None else self.object_ids[mask],
        )

    def translated(self, offset: np.ndarray) -> "GaussianCloud":
        return replace(self, xyz=(self.xyz + offset.astype(np.float32)))


def load_gaussian_ply(path: str | Path) -> GaussianCloud:
    """Parse a 3DGS PLY file (with or without an object_id property)."""
    ply = PlyData.read(str(path))
    if "vertex" not in [el.name for el in ply.elements]:
        raise ValueError(f"{path}: PLY has no 'vertex' element; not a splat file")
    v = ply["vertex"].data
    names = v.dtype.names or ()

    required = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")
    missing = [n for n in required if n not in names]
    if missing:
        raise ValueError(f"{path}: missing 3DGS properties: {missing}")

    n = len(v)

    def col3(a: str, b: str, c: str) -> np.ndarray:
        return np.stack(
            [v[a].astype(np.float32), v[b].astype(np.float32), v[c].astype(np.float32)],
            axis=1,
        )

    rest_indices = sorted(
        (int(m.group(1)) for name in names if (m := _F_REST_RE.match(name))),
    )
    if rest_indices and rest_indices != list(range(len(rest_indices))):
        raise ValueError(f"{path}: non-contiguous f_rest_* properties")
    f_rest = (
        np.stack([v[f"f_rest_{i}"].astype(np.float32) for i in rest_indices], axis=1)
        if rest_indices
        else np.zeros((n, 0), dtype=np.float32)
    )

    normals = (
        col3("nx", "ny", "nz")
        if all(k in names for k in ("nx", "ny", "nz"))
        else np.zeros((n, 3), dtype=np.float32)
    )
    object_ids = v["object_id"].astype(np.uint32) if "object_id" in names else None

    return GaussianCloud(
        xyz=col3("x", "y", "z"),
        normals=normals,
        f_dc=col3("f_dc_0", "f_dc_1", "f_dc_2"),
        f_rest=f_rest,
        opacity=v["opacity"].astype(np.float32),
        scales=col3("scale_0", "scale_1", "scale_2"),
        rots=np.stack(
            [v[f"rot_{i}"].astype(np.float32) for i in range(4)], axis=1
        ),
        object_ids=object_ids,
    )


def save_gaussian_ply(
    path: str | Path,
    cloud: GaussianCloud,
    *,
    include_object_id: bool = False,
) -> None:
    """Write a cloud back to binary-little-endian 3DGS PLY."""
    fields: list[tuple[str, str]] = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    fields += [(f"f_rest_{i}", "f4") for i in range(cloud.sh_rest_dim)]
    fields += [
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    write_ids = include_object_id and cloud.object_ids is not None
    if write_ids:
        fields.append(("object_id", "u4"))

    data = np.empty(len(cloud), dtype=fields)
    data["x"], data["y"], data["z"] = cloud.xyz.T
    data["nx"], data["ny"], data["nz"] = cloud.normals.T
    data["f_dc_0"], data["f_dc_1"], data["f_dc_2"] = cloud.f_dc.T
    for i in range(cloud.sh_rest_dim):
        data[f"f_rest_{i}"] = cloud.f_rest[:, i]
    data["opacity"] = cloud.opacity
    data["scale_0"], data["scale_1"], data["scale_2"] = cloud.scales.T
    for i in range(4):
        data[f"rot_{i}"] = cloud.rots[:, i]
    if write_ids:
        data["object_id"] = cloud.object_ids

    el = PlyElement.describe(data, "vertex")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([el], text=False).write(str(path))


def concat_clouds(clouds: list[GaussianCloud]) -> GaussianCloud:
    """Concatenate clouds, zero-padding f_rest to the widest SH band present."""
    if not clouds:
        raise ValueError("concat_clouds: empty input")
    max_k = max(c.sh_rest_dim for c in clouds)

    def pad_rest(c: GaussianCloud) -> np.ndarray:
        if c.sh_rest_dim == max_k:
            return c.f_rest
        padded = np.zeros((len(c), max_k), dtype=np.float32)
        padded[:, : c.sh_rest_dim] = c.f_rest
        return padded

    has_ids = all(c.object_ids is not None for c in clouds)
    return GaussianCloud(
        xyz=np.concatenate([c.xyz for c in clouds]),
        normals=np.concatenate([c.normals for c in clouds]),
        f_dc=np.concatenate([c.f_dc for c in clouds]),
        f_rest=np.concatenate([pad_rest(c) for c in clouds]),
        opacity=np.concatenate([c.opacity for c in clouds]),
        scales=np.concatenate([c.scales for c in clouds]),
        rots=np.concatenate([c.rots for c in clouds]),
        object_ids=(
            np.concatenate([c.object_ids for c in clouds]) if has_ids else None
        ),
    )


# ---------------------------------------------------------------------------
# Rigid transforms
# ---------------------------------------------------------------------------

def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert a three.js-style (x, y, z, w) quaternion to (w, x, y, z)."""
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[3], q[0], q[1], q[2]])


def quat_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix from a (w, x, y, z) quaternion (normalized here)."""
    w, x, y, z = q_wxyz / np.linalg.norm(q_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_multiply_left(q_edit_wxyz: np.ndarray, quats_wxyz: np.ndarray) -> np.ndarray:
    """Left-multiply every quaternion in (N, 4) `quats_wxyz` by one edit quaternion.

    R(q_edit * q) == R(q_edit) @ R(q), so this rotates each Gaussian's frame
    by the edit rotation.
    """
    w1, x1, y1, z1 = q_edit_wxyz / np.linalg.norm(q_edit_wxyz)
    w2, x2, y2, z2 = quats_wxyz[:, 0], quats_wxyz[:, 1], quats_wxyz[:, 2], quats_wxyz[:, 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    ).astype(np.float32)


def transform_cloud(
    cloud: GaussianCloud,
    *,
    translation: np.ndarray,
    quaternion_xyzw: np.ndarray,
    uniform_scale: float,
) -> GaussianCloud:
    """Apply a similarity transform: p' = s * R @ p + t.

    Splat orientations are composed with the edit rotation and log-scales are
    shifted by log(s). Note: f_rest (view-dependent SH) is intentionally NOT
    rotated — exact SH rotation is out of scope for v1 and only affects
    view-dependent shading, not geometry.
    """
    if uniform_scale <= 0:
        raise ValueError(f"uniform_scale must be > 0, got {uniform_scale}")
    q_wxyz = quat_xyzw_to_wxyz(np.asarray(quaternion_xyzw))
    rot = quat_to_rotmat(q_wxyz)
    t = np.asarray(translation, dtype=np.float64)

    xyz = (uniform_scale * (cloud.xyz.astype(np.float64) @ rot.T) + t).astype(np.float32)
    rots = quat_multiply_left(q_wxyz, cloud.rots.astype(np.float64))
    scales = (cloud.scales + np.float32(np.log(uniform_scale))).astype(np.float32)
    return replace(cloud, xyz=xyz, rots=rots, scales=scales)
