"""TripoSR API integration via HuggingFace Spaces.

TripoSR is Meta's fast image-to-3D model. This module calls the free
HF Spaces endpoint to convert images → GLB meshes.

API: https://huggingface.co/spaces/stabilityai/TripoSR
"""

from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path

import requests
from PIL import Image

HF_API_URL = "https://stabilityai-tripor.hf.space/call/predict"
TIMEOUT = 120  # 2 min per image


def image_to_base64(image_path: Path) -> str:
    """Convert image file to base64 string."""
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def reconstruct_image_to_glb(image_path: Path) -> bytes:
    """Call TripoSR API to convert image → GLB mesh.

    Args:
        image_path: path to input image (jpg/png)

    Returns:
        GLB bytes ready to write to disk

    Raises:
        requests.RequestException: if API fails or times out
        ValueError: if input is invalid or output parse fails
    """
    if not image_path.exists():
        raise ValueError(f"Image not found: {image_path}")

    ext = image_path.suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(f"Unsupported format: {ext}. Use jpg/png.")

    image_b64 = image_to_base64(image_path)
    payload = {"data": [f"data:image/png;base64,{image_b64}"]}

    try:
        # Queue the job
        resp = requests.post(HF_API_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        job_id = resp.json()["call_id"]
    except (requests.RequestException, KeyError) as e:
        raise ValueError(f"Failed to queue TripoSR job: {e}")

    # Poll for completion (~10-30 sec)
    status_url = f"{HF_API_URL}/status/{job_id}"
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        try:
            resp = requests.get(status_url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") == "DONE":
                # Extract GLB from output
                output = data.get("data", {})
                if isinstance(output, list) and len(output) > 0:
                    result = output[0]
                    if isinstance(result, dict) and "name" in result:
                        glb_url = result["name"]
                        # Download GLB
                        glb_resp = requests.get(glb_url, timeout=60)
                        glb_resp.raise_for_status()
                        return glb_resp.content
                raise ValueError(f"Unexpected output format: {output}")

            if data.get("status") == "FAILED":
                raise ValueError(f"TripoSR failed: {data.get('error', 'unknown')}")

        except requests.RequestException as e:
            if time.time() >= deadline:
                raise ValueError(f"TripoSR timeout after {TIMEOUT}s")
            time.sleep(2)
            continue

    raise ValueError(f"TripoSR did not complete within {TIMEOUT}s")
