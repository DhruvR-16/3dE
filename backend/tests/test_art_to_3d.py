"""Tests for art_to_3d — the 2D-art -> volumetric 3D core.

Test shapes are drawn in-code so the suite is self-contained and deterministic.
Every assertion here pins a property that actually broke during development:

* inflation must never collapse to a sheet (the whole reason this module exists —
  Hunyuan3D scored 0.030 on the same kind of input)
* a disc must inflate to a ball of the requested thickness
* real holes (donut, letter counters) must survive
* white *inside* the drawing (a stripe) must not be punched out
* stripes and smooth shading must not be mistaken for separate parts
* genuinely distinct parts must still split
* surfaces must stay smooth (no voxel terracing)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import art_to_3d as a3  # noqa: E402

SIZE = 240  # small keeps the suite fast; the maths is resolution independent


def _save(img: Image.Image, tmp_path: Path, name: str) -> Path:
    p = tmp_path / f"{name}.png"
    img.save(p)
    return p


def _blank() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (SIZE, SIZE), "white")
    return img, ImageDraw.Draw(img)


def disc(tmp_path: Path, name: str = "disc") -> Path:
    img, d = _blank()
    d.ellipse([40, 40, SIZE - 40, SIZE - 40], fill=(200, 60, 60))
    return _save(img, tmp_path, name)


def ring(tmp_path: Path) -> Path:
    img, d = _blank()
    d.ellipse([25, 25, SIZE - 25, SIZE - 25], fill=(60, 120, 200))
    d.ellipse([90, 90, SIZE - 90, SIZE - 90], fill=(255, 255, 255))
    return _save(img, tmp_path, "ring")


def striped(tmp_path: Path) -> Path:
    """A blob with white stripes: white here is decoration, not background.

    The outline drawn last matters — as in real clip art it seals the silhouette
    so the stripes read as *enclosed* white rather than leaking into the page.
    """
    img, d = _blank()
    box = [30, 30, SIZE - 30, SIZE - 30]
    d.ellipse(box, fill=(60, 150, 220))
    for y in range(60, SIZE - 60, 30):
        d.rectangle([35, y, SIZE - 35, y + 12], fill=(255, 255, 255))
    d.ellipse(box, outline=(20, 70, 130), width=6)
    return _save(img, tmp_path, "striped")


def two_colour_blobs(tmp_path: Path) -> Path:
    """Two clearly different-coloured lobes joined by a thin neck."""
    img, d = _blank()
    d.ellipse([20, 70, 110, 170], fill=(220, 80, 60))
    d.ellipse([130, 70, 220, 170], fill=(60, 90, 210))
    d.rectangle([105, 112, 135, 128], fill=(120, 120, 120))
    return _save(img, tmp_path, "blobs")


def shaded_ball(tmp_path: Path) -> Path:
    """One object whose colour varies smoothly — shading, not parts."""
    arr = np.full((SIZE, SIZE, 3), 255, np.uint8)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    c, r = SIZE / 2, SIZE / 2 - 25
    inside = (xx - c) ** 2 + (yy - c) ** 2 <= r * r
    nx, ny = (xx - c) / r, (yy - c) / r
    nz = np.sqrt(np.clip(1 - nx**2 - ny**2, 0, 1))
    lam = np.clip(0.35 + 0.75 * (-0.5 * nx - 0.5 * ny + 0.7 * nz), 0, 1)
    for ch, base in enumerate((210, 70, 80)):
        plane = arr[..., ch]
        plane[inside] = (base * lam[inside]).astype(np.uint8)
        arr[..., ch] = plane
    return _save(Image.fromarray(arr), tmp_path, "shaded_ball")


def mean_dihedral_deg(mesh) -> float:
    return float(np.degrees(mesh.face_adjacency_angles.mean()))


# --------------------------------------------------------------------------- #
# core inflation
# --------------------------------------------------------------------------- #


def test_inflation_is_volumetric_not_a_sheet(tmp_path):
    """The entire point: flat art must not stay flat."""
    mesh = a3.art_to_mesh(disc(tmp_path))
    assert a3.volume_ratio(mesh) > 0.5, "collapsed toward a flat sheet"
    assert mesh.volume > 0, "normals inverted (negative volume)"


def test_mesh_is_watertight(tmp_path):
    mesh = a3.art_to_mesh(disc(tmp_path))
    assert mesh.is_watertight


def test_disc_inflates_to_a_ball_of_requested_thickness(tmp_path):
    """A disc is the case with a known answer: depth == thickness x width."""
    for thickness in (0.5, 0.9):
        mesh = a3.art_to_mesh(disc(tmp_path), thickness=thickness)
        extents = mesh.bounds[1] - mesh.bounds[0]
        width = max(extents[0], extents[1])
        assert extents[2] == pytest.approx(thickness * width, rel=0.08)


def test_thickness_control_changes_depth(tmp_path):
    p = disc(tmp_path)
    thin = a3.art_to_mesh(p, thickness=0.4).bounds
    fat = a3.art_to_mesh(p, thickness=1.2).bounds
    assert (fat[1] - fat[0])[2] > 2 * (thin[1] - thin[0])[2]


def test_vertices_carry_colour_from_the_artwork(tmp_path):
    mesh = a3.art_to_mesh(disc(tmp_path))
    colours = np.asarray(mesh.visual.vertex_colors)[:, :3]
    assert colours.mean(axis=0)[0] > colours.mean(axis=0)[2], "should read as red"


def test_surface_is_smooth_not_terraced(tmp_path):
    """Guards the continuous-field fix; the binary volume scored ~8 degrees."""
    mesh = a3.art_to_mesh(shaded_ball(tmp_path))
    assert mean_dihedral_deg(mesh) < 4.0


# --------------------------------------------------------------------------- #
# silhouette / holes
# --------------------------------------------------------------------------- #


def test_real_hole_survives(tmp_path):
    """A donut must keep its hole (genus 1), not be filled in."""
    mesh = a3.art_to_mesh(ring(tmp_path))
    genus = (2 - mesh.euler_number) // 2
    assert genus == 1


def test_interior_white_stripes_are_not_punched_out(tmp_path):
    """White used as decoration must stay solid, unlike a genuine hole."""
    _rgb, mask = a3.load_art(striped(tmp_path))
    mesh = a3.art_to_mesh(striped(tmp_path))
    assert (2 - mesh.euler_number) // 2 == 0, "stripes were punched into holes"
    assert mask.sum() > 0


def test_holes_mode_can_be_forced(tmp_path):
    p = striped(tmp_path)
    kept = a3.load_art(p, holes="keep")[1].sum()
    punched = a3.load_art(p, holes="punch")[1].sum()
    assert punched < kept


def test_alpha_channel_is_respected(tmp_path):
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse([50, 50, SIZE - 50, SIZE - 50], fill=(30, 200, 120, 255))
    p = tmp_path / "alpha.png"
    img.save(p)
    _rgb, mask = a3.load_art(p)
    assert 0.05 < mask.mean() < 0.6


# --------------------------------------------------------------------------- #
# part splitting
# --------------------------------------------------------------------------- #


def test_distinct_parts_split(tmp_path):
    parts = a3.art_to_parts(two_colour_blobs(tmp_path))
    assert len(parts) >= 2


def test_stripes_are_not_mistaken_for_parts(tmp_path):
    """Pattern is surface, not structure — this regressed twice."""
    parts = a3.art_to_parts(striped(tmp_path))
    assert len(parts) <= 2


def test_smooth_shading_is_not_mistaken_for_parts(tmp_path):
    """One hue varying only in lightness is shading, not separate parts."""
    parts = a3.art_to_parts(shaded_ball(tmp_path))
    assert len(parts) == 1


def test_parts_can_be_disabled(tmp_path):
    parts = a3.art_to_parts(two_colour_blobs(tmp_path), parts=False)
    assert len(parts) == 1


def test_parts_stay_assembled_in_place(tmp_path):
    """Parts must keep their relative positions, not pile up on the origin."""
    parts = a3.art_to_parts(two_colour_blobs(tmp_path))
    centres = [m.bounds.mean(axis=0) for m, _ in parts]
    spread = max(c[0] for c in centres) - min(c[0] for c in centres)
    assert spread > 0.1, "parts collapsed onto each other"


def test_small_parts_are_not_rods(tmp_path):
    """Depth scales with each part's own size, so a small part stays in scale."""
    img, d = _blank()
    d.ellipse([40, 120, 200, 210], fill=(80, 160, 90))      # large lobe
    d.ellipse([105, 30, 135, 60], fill=(200, 60, 160))      # small lobe
    d.rectangle([115, 55, 125, 125], fill=(120, 120, 120))  # thin neck
    p = _save(img, tmp_path, "rod")
    for mesh, _ in a3.art_to_parts(p):
        e = mesh.bounds[1] - mesh.bounds[0]
        assert e[2] <= 1.6 * max(e[0], e[1]), "part is deeper than it is wide"


# --------------------------------------------------------------------------- #
# part naming
# --------------------------------------------------------------------------- #


def test_parts_are_named_by_colour(tmp_path):
    """Distinct colours need no position word — "red", not "part 1"."""
    names = a3.describe_parts(a3.art_to_parts(two_colour_blobs(tmp_path)))
    assert any("red" in n for n in names)
    assert any("blue" in n for n in names)
    assert not any(n[0].isdigit() or "part" in n for n in names)


def test_same_colour_parts_get_position_words(tmp_path):
    """Two same-coloured lobes side by side should read left / right."""
    img, d = _blank()
    d.ellipse([20, 80, 100, 160], fill=(60, 90, 210))
    d.ellipse([140, 80, 220, 160], fill=(60, 90, 210))
    names = a3.describe_parts(a3.art_to_parts(_save(img, tmp_path, "twins")))
    joined = " ".join(names)
    assert "left" in joined and "right" in joined


def test_part_names_are_unique(tmp_path):
    names = a3.describe_parts(a3.art_to_parts(two_colour_blobs(tmp_path)))
    assert len(set(names)) == len(names)


def test_colour_naming_ignores_page_coloured_filler(tmp_path):
    """A part padded with kept page-colour is named for its paint, not the page."""
    img, d = _blank()
    d.rectangle([60, 40, 180, 100], fill=(40, 60, 150))     # painted block
    d.rectangle([60, 100, 180, 200], fill=(255, 255, 255))  # page-coloured tail
    d.rectangle([58, 38, 182, 202], outline=(20, 30, 90), width=5)
    names = a3.describe_parts(a3.art_to_parts(_save(img, tmp_path, "filler")))
    assert not any("white" in n for n in names), names


def test_empty_input_names_cleanly():
    assert a3.describe_parts([]) == []


# --------------------------------------------------------------------------- #
# robustness
# --------------------------------------------------------------------------- #


def test_blank_image_raises_cleanly(tmp_path):
    p = _save(Image.new("RGB", (SIZE, SIZE), "white"), tmp_path, "blank")
    with pytest.raises(ValueError):
        a3.art_to_mesh(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises((ValueError, OSError, FileNotFoundError)):
        a3.art_to_mesh(tmp_path / "nope.png")
