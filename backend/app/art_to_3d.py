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

import colorsys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage
from skimage import measure

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

# Voxel depth of the occupancy volume. 64 is plenty: the silhouette carries the
# detail, Z only carries the smooth inflation profile.
Z_RES = 64
# Marching cubes needs empty space around the solid to close the surface.
PAD = 2


def load_art(
    path: str | Path,
    white_thresh: int = 245,
    holes: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Load artwork and its foreground mask.

    Uses real alpha when present. Otherwise the near-white region *connected to
    the image border* is background. What to do with white **enclosed** by the
    drawing is genuinely ambiguous — it can be a real hole (a donut, the counter
    of an 'e') or a design element the same colour as the page (a white stripe,
    an eye highlight). ``holes`` decides:

      "auto"  – punch it out only if it is *blobby*; elongated regions like
                stripes stay solid. A donut hole is roughly as tall as it is
                wide, a stripe is not, which separates the two cases well.
      "punch" – every enclosed light region becomes a hole.
      "keep"  – enclosed light regions stay part of the object.

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

    if holes != "keep":
        enclosed = white & ~background
        min_area = 0.0005 * white.size          # ignore specks / anti-aliasing
        lbl, n = ndimage.label(enclosed)
        for sl, i in zip(ndimage.find_objects(lbl), range(1, n + 1)):
            if sl is None:
                continue
            region = lbl[sl] == i
            area = int(region.sum())
            if area < min_area:
                continue
            h = sl[0].stop - sl[0].start
            w = sl[1].stop - sl[1].start
            # ~1 for a real hole (donut, letter counter), low for a stripe.
            # 0.5 keeps short stripes near a cone's apex solid (measured 0.37)
            # while still punching genuine holes.
            compact = min(h, w) / max(h, w)
            if holes == "punch" or compact >= 0.5:
                background[sl] |= region

    return rgb, ~background


def _signed_field(
    mask: np.ndarray, profile: str, roundness: float, antialias: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Continuous inflation field.

    Returns ``(signed, height, peak)`` where *signed* is the signed distance to
    the outline (positive inside, negative outside, zero on it) and *height* is
    the normalised inflation profile in [0,1].

    The signed distance is what buys smoothness: a binary in/out mask forces
    marching cubes onto voxel midpoints, which is exactly what produced the
    terraced look. A continuous field lets it interpolate the crossing, and a
    light blur removes pixel-level jaggies from the traced outline without
    noticeably moving the zero level set (a distance field is near-linear there).
    """
    inside = ndimage.distance_transform_edt(mask)
    outside = ndimage.distance_transform_edt(~mask)
    signed = inside - outside
    if antialias > 0:
        signed = ndimage.gaussian_filter(signed, antialias)

    peak = float(inside.max())            # true radius, for physical depth scaling
    if peak <= 0:
        return signed, np.zeros_like(signed), 0.0

    # Normalise the profile by the *blurred* peak, not the true one: blurring
    # rounds off the medial ridge, and dividing by the sharper unblurred peak
    # would stop the profile ever reaching 1 and quietly thin the model.
    profile_peak = max(float(signed.max()), 1e-6)
    h = np.clip(signed, 0, None) / profile_peak
    if profile == "round":       # spherical cap: a disc inflates to a ball
        h = np.sqrt(np.clip(h * (2 - h), 0, 1))
    elif profile == "sqrt":      # softer dome
        h = np.sqrt(h)
    # "linear" leaves h as-is: a faceted, tent-like look
    return signed, np.clip(h, 0, 1) ** max(roundness, 1e-3), peak


def inflate_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    max_dim: int = 256,
    thickness: float = 0.9,   # 1.0 inflates a disc into a full ball
    profile: str = "round",
    roundness: float = 1.0,
    smooth: int = 1,
    antialias: float = 0.8,
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

    signed, height, dist_peak = _signed_field(mask_s, profile, roundness, antialias)

    # Scalar field, symmetric about the mid-plane, whose zero level set IS the
    # inflated surface: F = half-thickness(x,y) - |z - centre|. Outside the
    # silhouette the half-thickness goes negative (the signed distance), so the
    # solid closes itself exactly on the outline. Marching cubes at level 0 then
    # interpolates the crossing to sub-voxel precision -- no terracing.
    half_voxels = height * (Z_RES / 2 - 1)
    half_voxels = np.where(signed > 0, half_voxels, signed)
    z_idx = np.arange(Z_RES)[None, None, :]
    centre = Z_RES // 2
    field = half_voxels[..., None] - np.abs(z_idx - centre)
    field = np.pad(field, PAD, mode="constant", constant_values=-1e4)

    verts, faces, _normals, _vals = measure.marching_cubes(field.astype(np.float32), level=0.0)

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


def _is_texture_not_parts(regions: list[tuple[int, np.ndarray]], area: int) -> bool:
    """Decide whether colour variation means separate parts, or just surface.

    Three signals, each aimed at a case seen in real artwork:

    * **many small regions** — stripes on a party hat produce ~16 fragments that
      together cover under half the shape. That is a pattern, not parts.
    * **poor coverage** by the largest few regions — same story.
    * **one hue, many lightnesses** — a smoothly shaded ball quantises into
      concentric bands that are all the same colour, just lighter and darker.
      Real parts (skin arm vs blue torso) differ in *hue*, not only brightness.
    """
    if not regions:
        return True
    if len(regions) > 8:
        return True
    if sum(s for s, _ in regions[:4]) / max(area, 1) < 0.6:
        return True

    hues = []
    for _size, mean_rgb in regions:
        r, g, b = (float(v) / 255.0 for v in mean_rgb)
        hi, lo = max(r, g, b), min(r, g, b)
        if hi - lo < 0.12:          # near-grey: hue is meaningless, skip it
            continue
        hues.append(colorsys.rgb_to_hsv(r, g, b)[0] * 2 * np.pi)
    if len(hues) >= 2:
        ang = np.asarray(hues)
        concentration = float(np.hypot(np.cos(ang).mean(), np.sin(ang).mean()))
        if concentration > 0.93:    # all essentially one hue -> shading
            return True
    return False


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
        # Thresholds are deliberately low: a character's arms are a small
        # fraction of the body but are exactly the parts worth separating.
        # _is_texture_not_parts then rejects the whole split if the colour
        # variation turns out to be a pattern or shading rather than parts.
        candidates: list[tuple[int, np.ndarray, np.ndarray]] = []
        for val in np.unique(key[part]):
            region = part & (key == val)
            if region.sum() < max(min_area_frac * total, 0.05 * area):
                continue                     # too small to be a real part
            lbl2, m2 = ndimage.label(region)
            for j in range(1, m2 + 1):
                blob = lbl2 == j
                size = int(blob.sum())
                if size >= max(0.5 * min_area_frac * total, 0.03 * area):
                    candidates.append((size, rgb[blob].mean(axis=0), blob))
        candidates.sort(key=lambda c: -c[0])
        if _is_texture_not_parts([(s, c) for s, c, _ in candidates], area):
            out.append(part)
            continue
        sub = [blob for _s, _c, blob in candidates]
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
