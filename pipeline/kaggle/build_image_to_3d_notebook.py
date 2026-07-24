"""Generate image_to_3d.ipynb — a Kaggle GPU notebook that turns each input
image into a genuinely volumetric 3D mesh with Hunyuan3D-2 (shape generation),
output in the local app's mesh-scene format (objects/000.glb + labels.json).

Why Hunyuan3D-2 shape-gen: TripoSR (the fast web path) is weak and collapses
flat/illustration inputs to near-flat sheets. Hunyuan3D-2's DiT shape model is
pure-PyTorch (no exotic CUDA build), fits a T4 (~6 GB), and produces real,
watertight, volumetric geometry from a single image.

Cell order is deliberate: the pip install runs BEFORE the GPU guard, so a P100
(what the Kaggle *API* always assigns) still exercises + verifies the install;
only the actual generation needs a T4 (selected in the web UI).

Run:  python pipeline/kaggle/build_image_to_3d_notebook.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

OUT_PATH = Path(__file__).parent / "image_to_3d.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip()}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": textwrap.dedent(source).strip(),
    }


CELLS = [
    md("""
    # Image → volumetric 3D (Hunyuan3D-2) — Kaggle GPU

    Turns each input **image** into a real, volumetric 3D **mesh** using
    Hunyuan3D-2 shape generation — far better than the fast TripoSR path for
    flat/illustration inputs, which TripoSR collapses to a flat sheet.

    **Before running**
    1. *Session options → Accelerator*: **GPU T4 x2** (sm_75). ⚠️ **Not P100** —
       Kaggle's PyTorch dropped Pascal (sm_60), so a P100 crashes on the first
       CUDA op. API-launched runs always get a P100; pick the T4 in the **web UI**
       and *Run All* (or *Save & Run All*). The install runs first so a P100 run
       still verifies the environment before the GPU guard stops it.
    2. *Session options → Internet*: **On** (downloads the model + pip packages).
    3. *Add Input* → attach your images as a Kaggle **Dataset** (1–20 images).

    **Output:** `/kaggle/working/editable_scene.zip` with `objects/000.glb …`
    and `labels.json` — import into the local app as a mesh scene.
    """),

    code("""
    # ---------------- Config ----------------
    from pathlib import Path
    INPUT_DIR = Path("/kaggle/input")     # searched recursively for images
    WORK = Path("/kaggle/working")
    OUT = WORK / "out"; OBJ = OUT / "objects"
    for d in (OUT, OBJ): d.mkdir(parents=True, exist_ok=True)

    MAX_IMAGES = 20        # cap objects per scene
    REMOVE_BG = True       # cut the object out of its background first
    STEPS = 30             # DiT sampling steps (quality vs time)
    OCTREE_RES = 256       # marching-cubes resolution of the shape decoder
    PROJECT_COLOR = True   # best-effort: paint front-facing verts from the image
    """),

    code("""
    %%bash
    # ---------------- Install Hunyuan3D-2 shape gen (~2 min; no GPU needed) ----------------
    # Installed BEFORE the GPU guard so a P100 run tests the whole cell (including
    # the hy3dgen.shapegen import). Cloned to /tmp so kernel OUTPUT stays tiny.
    #
    # CRITICAL: Kaggle already ships torch/numpy/scipy/transformers/opencv. We add
    # ONLY the missing deps and must NOT let pip upgrade numpy/scipy — doing so
    # (via rembg/scikit-image) broke scipy<->numpy at import ("_blas_supports_fpe").
    # `trimesh` needs only numpy (no scipy upgrade); diffusers/omegaconf/einops are
    # pure/light. hy3dgen is imported via sys.path (no `pip install -e .`), and
    # background removal uses a numpy threshold (no rembg) — nothing touches scipy.
    set -e
    cd /tmp
    [ -d Hunyuan3D-2 ] || git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
    echo ">>> clone OK"
    # pymeshlab is a self-contained wheel (only needs numpy) that hy3dgen.shapegen
    # imports at load time; the others are pure/light. None upgrade numpy/scipy.
    pip -q install trimesh diffusers omegaconf einops pymeshlab || { echo ">>> deps FAILED"; exit 1; }
    echo ">>> deps OK"
    python - <<'PY'
    import sys; sys.path.insert(0, "/tmp/Hunyuan3D-2")
    import numpy, scipy, trimesh            # verify the numpy/scipy/trimesh stack is intact
    print(">>> numpy", numpy.__version__, "| scipy", scipy.__version__, "| trimesh", trimesh.__version__)
    import hy3dgen
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    print(">>> shapegen import OK")
    PY
    """),

    code("""
    # ---------------- GPU guard (fail fast on an unusable accelerator) ----------------
    # Import the scientific stack IN THE KERNEL first — this is exactly what the
    # model cell needs, and it reproduces the numpy/scipy break here (on any GPU)
    # rather than after the model download.
    import torch, numpy, scipy, trimesh
    print(f"stack: numpy {numpy.__version__} | scipy {scipy.__version__} | "
          f"trimesh {trimesh.__version__} | torch {torch.__version__}")
    if not torch.cuda.is_available():
        raise SystemExit("No GPU. Session options -> Accelerator -> 'GPU T4 x2', then Run All.")
    cap = torch.cuda.get_device_capability(); tag = f"sm_{cap[0]}{cap[1]}"
    name = torch.cuda.get_device_name(0); arch = torch.cuda.get_arch_list()
    print(f"GPU: {name} ({tag}) | torch {torch.__version__}")
    print(f"torch supports: {arch}")
    if tag not in arch:
        raise SystemExit(
            f"\\n{'='*68}\\nThis session got a {name} ({tag}), unsupported by this PyTorch.\\n"
            f"Set Accelerator to 'GPU T4 x2' in the Kaggle UI, then Run All.\\n"
            f"(API-launched runs always get a P100; only the UI can pick the T4.)\\n{'='*68}"
        )
    _ = (torch.zeros(16, device='cuda') + 1).sum().item()
    print("GPU OK — install verified and a CUDA kernel launched.")
    """),

    code("""
    # ---------------- Load model + collect images ----------------
    import sys, glob, numpy as np
    sys.path.insert(0, "/tmp/Hunyuan3D-2")   # hy3dgen imported from the clone
    from PIL import Image
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    def to_rgba(img, white_thresh=245):
        # Hunyuan3D uses the alpha channel as the foreground mask. If the image
        # has no real alpha, make near-white pixels transparent (great for clip
        # art / product shots on white; harmless otherwise). Pure numpy -> never
        # pulls scipy/scikit-image.
        if img.mode == 'RGBA' and np.asarray(img)[..., 3].min() < 250:
            return img
        rgb = np.asarray(img.convert('RGB'))
        alpha = np.where((rgb > white_thresh).all(2), 0, 255).astype('uint8')
        return Image.fromarray(np.dstack([rgb, alpha]), 'RGBA')

    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')

    exts = ('*.png', '*.jpg', '*.jpeg', '*.webp', '*.PNG', '*.JPG', '*.JPEG')
    paths = []
    for e in exts:
        paths += glob.glob(str(INPUT_DIR / '**' / e), recursive=True)
    paths = sorted(set(paths))[:MAX_IMAGES]
    print(f"found {len(paths)} image(s):", [Path(p).name for p in paths])
    assert paths, "No images found — Add Input -> attach an image dataset."
    """),

    code("""
    # ---------------- Optional: bake front-view vertex colors from the image ----------------
    def project_colors(mesh, pil_img):
        # Hunyuan3D returns a mesh normalized ~[-1,1] facing the camera (+Z).
        # Paint front-facing vertices by sampling the image at their (x,y).
        try:
            rgb = np.asarray(pil_img.convert('RGB'))
            H, W = rgb.shape[:2]
            v = np.asarray(mesh.vertices); n = np.asarray(mesh.vertex_normals)
            xy = v[:, :2].copy()
            mn, mx = xy.min(0), xy.max(0)
            uv = (xy - mn) / np.clip(mx - mn, 1e-6, None)     # 0..1
            u = np.clip((uv[:, 0] * (W - 1)).astype(int), 0, W - 1)
            vv = np.clip(((1 - uv[:, 1]) * (H - 1)).astype(int), 0, H - 1)
            cols = rgb[vv, u]
            front = n[:, 2] >= 0
            base = np.full_like(cols, 180)                    # gray for back faces
            base[front] = cols[front]
            mesh.visual = __import__('trimesh').visual.color.ColorVisuals(
                mesh, vertex_colors=np.concatenate([base, np.full((len(base), 1), 255, np.uint8)], 1))
        except Exception as e:
            print("colour projection skipped:", e)
        return mesh
    """),

    code("""
    # ---------------- Generate a mesh per image ----------------
    import json, trimesh
    labels = {}
    for i, p in enumerate(paths):
        name = Path(p).stem
        print(f"[{i+1}/{len(paths)}] {name} …", flush=True)
        img = Image.open(p).convert('RGB')
        img_in = to_rgba(img) if REMOVE_BG else img
        # Be tolerant of pipeline kwarg differences across hy3dgen versions.
        try:
            mesh = pipe(image=img_in, num_inference_steps=STEPS, octree_resolution=OCTREE_RES)[0]
        except TypeError:
            mesh = pipe(image=img_in)[0]
        if PROJECT_COLOR:
            mesh = project_colors(mesh, img)
        ext = mesh.bounds[1] - mesh.bounds[0]
        thin = sorted(ext)[0] / max(sorted(ext)[2], 1e-6)
        print(f"  verts={len(mesh.vertices)} bbox={np.round(ext,3)} thinnest/longest={thin:.3f}")
        mesh.export(str(OBJ / f"{i:03d}.glb"))
        labels[str(i)] = name
    (OUT / "labels.json").write_text(json.dumps(labels, indent=2))
    print("wrote", len(labels), "objects to", OBJ)
    """),

    code("""
    # ---------------- Package for the local app ----------------
    import shutil
    zip_path = shutil.make_archive(str(WORK / "editable_scene"), "zip", root_dir=OUT)
    print("done ->", zip_path)
    for f in sorted(OBJ.glob("*.glb")):
        print(f"  {f.name}  ({f.stat().st_size/1e6:.2f} MB)")
    """),

    md("""
    ## Troubleshooting

    | Symptom | Fix |
    |---|---|
    | `SystemExit` at the GPU guard / `sm_60` / `no kernel image` | You're on a **P100**. Session options → Accelerator → **GPU T4 x2**, then Run All. |
    | Out of memory | Lower `OCTREE_RES` to 192 and `MAX_IMAGES`; process fewer images per run. |
    | Result still looks flat | Use a cleaner single-object image; very thin/2D inputs have little shape to infer. |
    | `hy3dgen` import errors | Re-run the install cell once (first `import` can lazily compile small helpers). |
    """),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT_PATH.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
