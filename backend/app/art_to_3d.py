"""2D art (cartoon / clipart / character) -> volumetric 3D, by construction.

Why this exists
---------------
Generic image->3D models (TripoSR, Hunyuan3D, TRELLIS) are trained on renders of
real 3D assets, so they rely on shading and perspective cues. Flat vector art has
neither, and they collapse it into a near-flat sheet — measured: Hunyuan3D turned
a party-hat clipart into a mesh whose thinnest/longest axis ratio was 0.030.

Flat art, however, gives us something photos don't: a clean, closed silhouette.
So instead of *guessing* depth we *construct* it — the classic silhouette
inflation idea (Igarashi's "Teddy", SIGGRAPH '99): height at each pixel grows
with its distance from the outline, so edges stay thin and the middle puffs out.
Mirroring that height field front/back gives a closed, watertight, genuinely
volumetric solid. It cannot come out flat, and nothing is hallucinated.

CPU only, no model, no GPU — a typical drawing takes well under a second.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage
from skimage import measure

# Voxel depth of the occupancy volume. 64 is plenty: the silhouette carries the
# detail, Z only carries the smooth inflation profile.
Z_RES = 64
# Marching cubes needs empty space around the solid to close the surface.
PAD = 2


def load_art(path: str | Path, white_thresh: int = 245) -> tuple[np.ndarray, np.ndarray]:
    """Load artwork and its foreground mask.

    Uses real alpha when present. Otherwise removes only the near-white region
    *connected to the image border*, so white INSIDE the drawing (a white stripe,
    an eye highlight) stays part of the object instead of being punched out.

    Returns:
        (rgb uint8 [H,W,3], mask bool [H,W])
    """
    img = Image.open(path).convert("RGBA")
    arr = np.asarray(img)
    rgb = arr[..., :3].copy()

    if arr[..., 3].min() < 250:                      # genuine alpha channel
        return rgb, arr[..., 3] > 127

    white = (rgb > white_thresh).all(axis=2)
    labels, _ = ndimage.label(white)
    border = set(labels[0]) | set(labels[-1]) | set(labels[:, 0]) | set(labels[:, -1])
    border.discard(0)
    background = np.isin(labels, list(border)) if border else np.zeros_like(white)
    return rgb, ~background


def _height_field(mask: np.ndarray, profile: str, roundness: float) -> np.ndarray:
    """Normalised inflation height in [0,1] — 0 at the outline, 1 at the core."""
    dist = ndimage.distance_transform_edt(mask)
    peak = dist.max()
    if peak <= 0:
        return np.zeros_like(dist, dtype=float)
    h = dist / peak
    if profile == "round":       # spherical cap: bulges fast, flattens on top
        h = np.sqrt(np.clip(h * (2 - h), 0, 1))
    elif profile == "sqrt":      # softer dome
        h = np.sqrt(h)
    # "linear" leaves h as-is: a faceted, tent-like look
    return np.clip(h, 0, 1) ** max(roundness, 1e-3)


def inflate_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    max_dim: int = 256,
    thickness: float = 0.9,   # 1.0 inflates a disc into a full ball
    profile: str = "round",
    roundness: float = 1.0,
    smooth: int = 1,
    center: bool = True,
) -> trimesh.Trimesh:
    """Inflate one silhouette into a watertight, vertex-coloured 3D solid.

    Args:
        rgb: source artwork, uint8 [H,W,3] (used for colour).
        mask: foreground, bool [H,W].
        max_dim: working resolution for the long side (detail vs speed).
        thickness: depth as a fraction of the object's longest side.
        profile: "round" | "sqrt" | "linear" inflation curve.
        roundness: >1 sharpens toward the core, <1 makes it fuller.
        smooth: Laplacian smoothing passes on the final mesh.

    Returns:
        A watertight trimesh, centred, with vertex colours from the artwork.
    """
    if not mask.any():
        raise ValueError("empty silhouette: nothing to inflate")

    # Work at a bounded resolution; scale mask and colours together.
    h0, w0 = mask.shape
    scale = min(1.0, max_dim / max(h0, w0))
    if scale < 1.0:
        size = (max(1, int(w0 * scale)), max(1, int(h0 * scale)))
        mask_s = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127
        rgb_s = np.asarray(Image.fromarray(rgb).resize(size, Image.BILINEAR))
    else:
        mask_s, rgb_s = mask, rgb
    if not mask_s.any():
        raise ValueError("silhouette vanished at working resolution — raise max_dim")

    dist = ndimage.distance_transform_edt(mask_s)
    dist_peak = float(dist.max())          # the part's own "radius", in pixels
    height = _height_field(mask_s, profile, roundness)

    # Occupancy volume: symmetric about the mid-plane, so front and back mirror
    # and the solid closes itself at the outline (height -> 0 there).
    half_voxels = np.rint(height * (Z_RES / 2 - 1)).astype(int)
    z_idx = np.arange(Z_RES)[None, None, :]
    centre = Z_RES // 2
    volume = np.abs(z_idx - centre) <= half_voxels[..., None]
    volume &= mask_s[..., None]
    volume = np.pad(volume, PAD, mode="constant", constant_values=False)

    verts, faces, _normals, _vals = measure.marching_cubes(volume.astype(np.float32), level=0.5)

    # Volume index space (row, col, z) -> world (x right, y up, z depth),
    # scaled so the artwork keeps its aspect ratio and requested thickness.
    rows, cols = mask_s.shape
    longest = max(rows, cols)
    px = 1.0 / longest                                   # world units per pixel
    # Depth scales with the part's OWN size: a disc of radius r inflates to a
    # ball of radius r at thickness=1. Without this a small part (a pom-pom)
    # would get the same absolute depth as the whole drawing and become a rod.
    half_depth_world = thickness * dist_peak * px
    depth_per_voxel = half_depth_world / max(Z_RES / 2 - 1, 1)

    x = (verts[:, 1] - PAD - cols / 2.0) * px
    y = -(verts[:, 0] - PAD - rows / 2.0) * px           # image rows go down
    z = (verts[:, 2] - PAD - centre) * depth_per_voxel
    vertices = np.column_stack([x, y, z])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    # Negating the Y axis above mirrors the mesh, which inverts face winding and
    # leaves normals pointing inward (negative volume). Put them back outward.
    mesh.fix_normals()

    # Colour: sample the artwork at each vertex's (row, col). Front and back get
    # the same colour, which is what you want for a two-sided cartoon figure.
    rr = np.clip(np.rint(verts[:, 0] - PAD).astype(int), 0, rows - 1)
    cc = np.clip(np.rint(verts[:, 1] - PAD).astype(int), 0, cols - 1)
    colors = rgb_s[rr, cc]
    # process=True may weld/reorder vertices; remap colours through that.
    if len(mesh.vertices) != len(vertices):
        tree_src = trimesh.proximity.ProximityQuery(
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        )
        _d, idx = tree_src.vertex(mesh.vertices)
        colors = colors[idx]
    mesh.visual = trimesh.visual.color.ColorVisuals(
        mesh, vertex_colors=np.column_stack([colors, np.full(len(colors), 255, np.uint8)])
    )

    if smooth > 0:
        trimesh.smoothing.filter_taubin(mesh, iterations=smooth)

    # Parts must stay in the shared image frame so they reassemble into the
    # original drawing; only a standalone mesh gets centred.
    if center:
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
    return mesh


def split_parts(
    mask: np.ndarray,
    rgb: np.ndarray | None = None,
    *,
    min_area_frac: float = 0.02,
    neck_frac: float = 0.02,
    by_color: bool = True,
    color_levels: int = 6,
) -> list[np.ndarray]:
    """Split one silhouette into separately editable parts.

    Two cues, both cheap and reliable on flat art:

    1. **Thin necks** — eroding the mask pulls apart pieces joined by a narrow
       bridge (a pom-pom on its stem, a head on a neck, a balloon on a string).
       The surviving blobs are "cores"; every pixel is then handed back to its
       nearest core, so the parts tile the original silhouette exactly.
    2. **Flat colour regions** — cartoons use large uniform fills, so a big
       distinctly-coloured area (a purple brim against a striped cone) is
       usually a real part. Applied inside each geometric part.

    Returns a list of boolean masks. Falls back to ``[mask]`` when the artwork
    is a single indivisible shape — one honest object beats fake fragments.
    """
    total = int(mask.sum())
    if total == 0:
        return []

    # --- 1. geometric split at thin necks -------------------------------
    erode_px = max(1, int(neck_frac * max(mask.shape)))
    cores, n = ndimage.label(ndimage.binary_erosion(mask, iterations=erode_px))
    keep = [i for i in range(1, n + 1) if (cores == i).sum() >= min_area_frac * total]

    if len(keep) > 1:
        core = np.zeros_like(cores)
        for new, old in enumerate(keep, start=1):
            core[cores == old] = new
        # nearest-core assignment fills the eroded rim back in
        _dist, (rr, cc) = ndimage.distance_transform_edt(core == 0, return_indices=True)
        assigned = core[rr, cc]
        parts = [(assigned == i) & mask for i in range(1, len(keep) + 1)]
    else:
        parts = [mask]

    if not by_color or rgb is None:
        return parts

    # --- 2. colour split inside each geometric part ----------------------
    out: list[np.ndarray] = []
    for part in parts:
        area = int(part.sum())
        if area < min_area_frac * total:
            out.append(part)
            continue
        # coarse colour quantisation -> dominant flat fills
        q = (rgb.astype(np.uint16) * (color_levels - 1) // 255)
        key = (q[..., 0] * color_levels + q[..., 1]) * color_levels + q[..., 2]
        sub: list[np.ndarray] = []
        for val in np.unique(key[part]):
            region = part & (key == val)
            if region.sum() < max(min_area_frac * total, 0.15 * area):
                continue                     # too small / a thin stripe: leave it
            lbl2, m2 = ndimage.label(region)
            for j in range(1, m2 + 1):
                blob = lbl2 == j
                if blob.sum() >= max(min_area_frac * total, 0.10 * area):
                    sub.append(blob)
        # Only accept a colour split that explains most of the part; otherwise
        # keep the part whole rather than shedding pixels into slivers.
        if len(sub) > 1 and sum(int(s.sum()) for s in sub) > 0.6 * area:
            covered = np.zeros_like(part)
            for s in sub:
                covered |= s
            leftover = part & ~covered
            if leftover.any():                # hand strays to the nearest sub-part
                seed = np.zeros_like(part, dtype=int)
                for i, s in enumerate(sub, start=1):
                    seed[s] = i
                _d2, (r2, c2) = ndimage.distance_transform_edt(seed == 0, return_indices=True)
                grown = seed[r2, c2]
                sub = [((grown == i) & part) for i in range(1, len(sub) + 1)]
            out.extend(sub)
        else:
            out.append(part)

    return [p for p in out if p.sum() >= min_area_frac * total] or [mask]


def art_to_parts(
    path: str | Path,
    *,
    parts: bool = True,
    **kwargs,
) -> list[tuple[trimesh.Trimesh, np.ndarray]]:
    """Artwork file -> list of (inflated mesh, its mask), one per editable part.

    Each part is inflated in place, so the pieces stay in their original relative
    positions and reassemble into the original drawing.
    """
    rgb, mask = load_art(path)
    masks = split_parts(mask, rgb) if parts else [mask]

    result: list[tuple[trimesh.Trimesh, np.ndarray]] = []
    for m in masks:
        try:
            result.append((inflate_mask(rgb, m, center=False, **kwargs), m))
        except ValueError:
            continue
    if not result:
        return result

    # Centre the assembly as a whole, preserving every part's relative position.
    lo = np.min([mesh.bounds[0] for mesh, _ in result], axis=0)
    hi = np.max([mesh.bounds[1] for mesh, _ in result], axis=0)
    offset = -(lo + hi) / 2.0
    for mesh, _ in result:
        mesh.apply_translation(offset)
    # Biggest part first — it reads as the "main" object in the sidebar.
    result.sort(key=lambda mm: -len(mm[0].faces))
    return result


def volume_ratio(mesh: trimesh.Trimesh) -> float:
    """thinnest/longest bbox axis. ~0 = flat sheet, higher = properly solid."""
    extents = sorted(mesh.bounds[1] - mesh.bounds[0])
    return float(extents[0] / max(extents[2], 1e-9))


def art_to_mesh(path: str | Path, **kwargs) -> trimesh.Trimesh:
    """Convenience: artwork file -> inflated 3D mesh."""
    rgb, mask = load_art(path)
    return inflate_mask(rgb, mask, **kwargs)
