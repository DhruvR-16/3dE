"""Pydantic request/response models for the scene API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ObjectInfo(BaseModel):
    id: int
    label: str
    splat_count: int
    centroid: list[float]  # [x, y, z] world-space
    bbox_min: list[float]
    bbox_max: list[float]
    color: list[float]  # mean base color, [r, g, b] in 0..1
    ply_url: str


class SceneManifest(BaseModel):
    id: str
    name: str
    splat_count: int
    sh_rest_dim: int
    objects: list[ObjectInfo]


class SceneSummary(BaseModel):
    id: str
    name: str
    splat_count: int
    object_count: int


class ObjectEdit(BaseModel):
    """Client-side edit state for one object, in world space.

    The transform is interpreted as: p_world = s * R(q) @ p_local + t, where
    p_local are the object's splat positions recentred at its original
    centroid (which is exactly how the frontend loads them).
    """

    id: int = Field(ge=0)
    deleted: bool = False
    position: list[float] = Field(min_length=3, max_length=3)
    quaternion: list[float] = Field(  # (x, y, z, w) — three.js order
        default=[0.0, 0.0, 0.0, 1.0], min_length=4, max_length=4
    )
    scale: float = Field(default=1.0, gt=0.0, le=100.0)

    @field_validator("quaternion")
    @classmethod
    def quaternion_nonzero(cls, q: list[float]) -> list[float]:
        if sum(c * c for c in q) < 1e-12:
            raise ValueError("quaternion must be non-zero")
        return q


class ExportRequest(BaseModel):
    objects: list[ObjectEdit]
    include_object_id: bool = False


class UploadResponse(BaseModel):
    scene: SceneManifest
