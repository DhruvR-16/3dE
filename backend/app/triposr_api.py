"""Real image -> 3D reconstruction via the TripoSR HuggingFace Space.

Uses the official `gradio_client` against `stabilityai/TripoSR`, whose API is:

    /preprocess(image, remove_background: bool, foreground_ratio: float) -> image
    /generate(processed_image, marching_cubes_resolution: float)  -> (obj, glb)

We run preprocess (background removal) then generate, and return the GLB bytes.
A single image becomes a single 3D object (~15-25 s end to end).

Verified working 2026-07-24: a real photo returns a valid 2-3 MB glTF binary.
"""

from __future__ import annotations

import threading
from pathlib import Path

from gradio_client import Client, handle_file

TRIPOSR_SPACE = "stabilityai/TripoSR"

# Tuning knobs (see the Space's UI for ranges).
REMOVE_BACKGROUND = True      # rembg-based cutout so the object floats free
FOREGROUND_RATIO = 0.85       # how much of the frame the object should fill (0.5-1.0)
MC_RESOLUTION = 256           # marching-cubes grid (32-320): higher = finer + slower

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

_client: Client | None = None
_client_lock = threading.Lock()


def _get_client(force_new: bool = False) -> Client:
    """Lazily create and cache the Gradio client (connecting takes a few seconds)."""
    global _client
    with _client_lock:
        if _client is None or force_new:
            _client = Client(TRIPOSR_SPACE, verbose=False)
        return _client


def _run_once(image_path: Path, force_new: bool) -> bytes:
    client = _get_client(force_new=force_new)
    processed = client.predict(
        handle_file(str(image_path)),
        REMOVE_BACKGROUND,
        FOREGROUND_RATIO,
        api_name="/preprocess",
    )
    _obj_path, glb_path = client.predict(
        handle_file(processed),
        MC_RESOLUTION,
        api_name="/generate",
    )
    data = Path(glb_path).read_bytes()
    if data[:4] != b"glTF":
        raise ValueError("TripoSR returned data that is not a valid GLB")
    return data


def reconstruct_image_to_glb(image_path: Path) -> bytes:
    """Convert a single image to a GLB mesh via TripoSR.

    Args:
        image_path: path to a JPG/PNG/WEBP image of one object.

    Returns:
        GLB (glTF binary) bytes, ready to write to disk.

    Raises:
        ValueError: bad input, or the Space returned something unusable.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise ValueError(f"Image not found: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(
            f"Unsupported format {image_path.suffix!r}; use one of {sorted(SUPPORTED_EXTS)}"
        )

    try:
        return _run_once(image_path, force_new=False)
    except Exception:
        # The public Space may have restarted / dropped the connection. Retry
        # once with a fresh client before giving up (let the 2nd error surface).
        return _run_once(image_path, force_new=True)
