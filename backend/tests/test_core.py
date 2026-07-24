"""Tests for the Gaussian PLY core and scene store split/export math."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.gaussian_ply import (  # noqa: E402
    GaussianCloud,
    concat_clouds,
    load_gaussian_ply,
    save_gaussian_ply,
    transform_cloud,
)
from app import scene_store  # noqa: E402
from app.models import ExportRequest, ObjectEdit  # noqa: E402


def make_cloud(n: int, object_id: int | None = None, offset: float = 0.0) -> GaussianCloud:
    rng = np.random.default_rng(42 + n)
    return GaussianCloud(
        xyz=(rng.uniform(-1, 1, (n, 3)) + offset).astype(np.float32),
        normals=np.zeros((n, 3), dtype=np.float32),
        f_dc=rng.normal(0, 1, (n, 3)).astype(np.float32),
        f_rest=np.zeros((n, 0), dtype=np.float32),
        opacity=rng.normal(2, 0.5, n).astype(np.float32),
        scales=rng.normal(-4, 0.3, (n, 3)).astype(np.float32),
        rots=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1)),
        object_ids=(
            None if object_id is None else np.full(n, object_id, dtype=np.uint32)
        ),
    )


class TestPlyRoundtrip:
    def test_roundtrip_preserves_all_fields(self, tmp_path: Path) -> None:
        cloud = make_cloud(100, object_id=3)
        path = tmp_path / "t.ply"
        save_gaussian_ply(path, cloud, include_object_id=True)
        loaded = load_gaussian_ply(path)

        np.testing.assert_allclose(loaded.xyz, cloud.xyz, rtol=1e-6)
        np.testing.assert_allclose(loaded.f_dc, cloud.f_dc, rtol=1e-6)
        np.testing.assert_allclose(loaded.opacity, cloud.opacity, rtol=1e-6)
        np.testing.assert_allclose(loaded.scales, cloud.scales, rtol=1e-6)
        np.testing.assert_allclose(loaded.rots, cloud.rots, rtol=1e-6)
        assert loaded.object_ids is not None
        assert (loaded.object_ids == 3).all()

    def test_roundtrip_without_object_id(self, tmp_path: Path) -> None:
        cloud = make_cloud(10)
        path = tmp_path / "t.ply"
        save_gaussian_ply(path, cloud)
        assert load_gaussian_ply(path).object_ids is None

    def test_rejects_non_splat_ply(self, tmp_path: Path) -> None:
        # A bare xyz point cloud is a valid PLY but not a splat file.
        from plyfile import PlyData, PlyElement

        data = np.zeros(5, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        path = tmp_path / "points.ply"
        PlyData([PlyElement.describe(data, "vertex")]).write(str(path))
        with pytest.raises(ValueError, match="missing 3DGS properties"):
            load_gaussian_ply(path)


class TestTransform:
    def test_translation_only(self) -> None:
        cloud = make_cloud(50)
        moved = transform_cloud(
            cloud,
            translation=np.array([1.0, 2.0, 3.0]),
            quaternion_xyzw=np.array([0, 0, 0, 1.0]),
            uniform_scale=1.0,
        )
        np.testing.assert_allclose(moved.xyz, cloud.xyz + [1, 2, 3], atol=1e-5)
        np.testing.assert_allclose(moved.scales, cloud.scales, atol=1e-6)

    def test_rotation_90deg_about_y(self) -> None:
        cloud = make_cloud(1)
        cloud.xyz = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        half = np.sqrt(0.5)
        rotated = transform_cloud(
            cloud,
            translation=np.zeros(3),
            quaternion_xyzw=np.array([0, half, 0, half]),  # +90° about Y
            uniform_scale=1.0,
        )
        # +X rotates to -Z under a +90° yaw
        np.testing.assert_allclose(rotated.xyz[0], [0, 0, -1], atol=1e-6)

    def test_uniform_scale_shifts_log_scales(self) -> None:
        cloud = make_cloud(20)
        scaled = transform_cloud(
            cloud,
            translation=np.zeros(3),
            quaternion_xyzw=np.array([0, 0, 0, 1.0]),
            uniform_scale=2.0,
        )
        np.testing.assert_allclose(
            scaled.scales, cloud.scales + np.log(2.0), rtol=1e-5
        )
        np.testing.assert_allclose(scaled.xyz, cloud.xyz * 2.0, rtol=1e-5)

    def test_rejects_nonpositive_scale(self) -> None:
        with pytest.raises(ValueError, match="uniform_scale"):
            transform_cloud(
                make_cloud(1),
                translation=np.zeros(3),
                quaternion_xyzw=np.array([0, 0, 0, 1.0]),
                uniform_scale=0.0,
            )


class TestSceneStore:
    @pytest.fixture()
    def scene(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
        monkeypatch.setattr(scene_store, "DATA_DIR", tmp_path / "scenes")
        merged = concat_clouds([make_cloud(60, 1, offset=0.0), make_cloud(40, 2, offset=5.0)])
        sdir = scene_store.scene_dir("test-scene")
        sdir.mkdir(parents=True)
        save_gaussian_ply(sdir / "scene.ply", merged, include_object_id=True)
        return "test-scene"

    def test_split_recentres_objects(self, scene: str) -> None:
        manifest = scene_store.build_manifest(scene)
        assert [o.id for o in manifest.objects] == [1, 2]
        assert manifest.splat_count == 100

        obj2 = next(o for o in manifest.objects if o.id == 2)
        part = load_gaussian_ply(
            scene_store.scene_dir(scene) / "objects" / "obj_002.ply"
        )
        # Recentred: local centroid at origin, world centroid recorded in manifest.
        np.testing.assert_allclose(part.xyz.mean(axis=0), 0, atol=1e-4)
        assert np.allclose(obj2.centroid, 5.0, atol=0.5)

    def test_export_identity_reassembles_scene(self, scene: str) -> None:
        manifest = scene_store.build_manifest(scene)
        edits = [
            ObjectEdit(id=o.id, position=o.centroid) for o in manifest.objects
        ]
        out = scene_store.export_scene(scene, ExportRequest(objects=edits))
        merged = load_gaussian_ply(out)
        original = load_gaussian_ply(scene_store.scene_dir(scene) / "scene.ply")
        assert len(merged) == len(original)
        # Same point set up to reordering: compare sorted coordinates.
        np.testing.assert_allclose(
            np.sort(merged.xyz, axis=0), np.sort(original.xyz, axis=0), atol=1e-4
        )

    def test_export_with_deletion_drops_object(self, scene: str) -> None:
        manifest = scene_store.build_manifest(scene)
        edits = [
            ObjectEdit(id=1, position=manifest.objects[0].centroid, deleted=True),
            ObjectEdit(id=2, position=manifest.objects[1].centroid),
        ]
        out = scene_store.export_scene(scene, ExportRequest(objects=edits))
        assert len(load_gaussian_ply(out)) == 40

    def test_export_all_deleted_raises(self, scene: str) -> None:
        manifest = scene_store.build_manifest(scene)
        edits = [
            ObjectEdit(id=o.id, position=o.centroid, deleted=True)
            for o in manifest.objects
        ]
        with pytest.raises(ValueError, match="empty"):
            scene_store.export_scene(scene, ExportRequest(objects=edits))

    def test_traversal_scene_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid scene id"):
            scene_store.scene_dir("../etc")
