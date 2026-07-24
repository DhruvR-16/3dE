# Photo → Editable 3D Scene

Reconstruct real photos into a 3D Gaussian-splat scene where **every splat
knows which object it belongs to** — then select, move, delete, and export
objects like it's a 3D editor, not a frozen scan.

This is v1 of the build plan in
[photo-to-editable-3d-scene-architecture.md](photo-to-editable-3d-scene-architecture.md).
Everything runs locally on free tooling; GPU work happens on Kaggle's free T4s.

![stack](https://img.shields.io/badge/stack-FastAPI%20·%20Three.js%20·%20gsplat%20·%20SAM2-blue)

## What works today (v1)

| Verb | Status | How |
|---|---|---|
| Reconstruct | ✅ via Kaggle notebook | COLMAP → gsplat 3DGS training |
| Segment objects in 3D | ✅ via Kaggle notebook | SAM2 video propagation → projection majority vote → per-Gaussian `object_id` |
| **Select** | ✅ local app | click (OBB raycast) or sidebar |
| **Move / rotate / scale** | ✅ local app | transform gizmo per object, real-time |
| **Delete / restore** | ✅ local app | removes that object's splats from the render |
| **Export** | ✅ local app | merged `.ply` with edits baked in (optionally keeping `object_id`) |
| Replace with generated object | 🚧 v2 | planned: TripoSR/TRELLIS via HF Spaces API (per architecture doc §7) |
| Mesh export (SuGaR) | 🚧 v2 | splat `.ply` export works now; SuperSplat opens it directly |

A synthetic **sample scene** ships with the repo flow, so the whole edit loop
is demoable in ~2 minutes with zero GPU time.

## Quickstart (local editor, no GPU needed)

```bash
# 1. backend
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt

# 2. generate the sample scene (synthetic labeled room)
.venv/bin/python scripts/make_sample_scene.py

# 3. run backend (terminal 1)
.venv/bin/uvicorn app.main:app --app-dir backend --port 8000

# 4. run frontend (terminal 2)
cd frontend && npm install && npm run dev
```

Open http://localhost:5173 — the sample room loads automatically.

**Controls:** click an object to select · `G`/`R`/`S` switch move/rotate/scale ·
drag the gizmo · `X`/`Del` delete · `Esc` deselect · *Export edited .ply*
downloads the merged scene with your edits baked in.

Tests: `.venv/bin/python -m pytest backend/tests -q`

## Real scenes (Kaggle free GPU)

1. Capture 20–80 photos of a **static** scene (~70% overlap, loop around it),
   or a slow video. For video, optionally pre-extract sharp frames locally:
   `python pipeline/scripts/extract_frames.py capture.mp4 --out frames/`
2. Upload the images as a Kaggle **Dataset**.
3. Upload [pipeline/kaggle/photo_to_editable_3d.ipynb](pipeline/kaggle/photo_to_editable_3d.ipynb)
   as a Kaggle notebook. Settings: **GPU on, Internet on**. Run all
   (~10–40 min on a T4; well inside the free 30 h/week quota).
4. Download `editable_scene.zip`, unzip, then in the local app sidebar:
   **Import labeled .ply** → choose an id → pick `scene.ply` → *Upload & split*.
5. Edit and export like the sample scene.

## How it works

```
photos ─▶ COLMAP (poses) ─▶ gsplat 3DGS training ─▶ SAM2 masks ─▶ lift to 3D
                                                                     │
        scene.ply — every Gaussian carries object_id  ◀──────────────┘
                          │  (backend splits by object_id, recentres each object)
                          ▼
        FastAPI: /api/scenes · /export · /files/…/obj_00N.ply
                          │
                          ▼
        Three.js + GaussianSplats3D: one dynamic splat scene per object
        select = OBB raycast · move = scene transform · delete = remove scene
                          │
                          ▼
        export: server re-merges per-object PLYs with your transforms baked in
```

The core design decision (architecture doc §2): **3DGS as the scene
representation**, because a splat scene is just a big array — filtering and
transforming a labeled subset is trivial, no retraining needed.

## Repo layout

```
backend/app/            FastAPI + Gaussian PLY core (load/save/transform/split/export)
backend/tests/          pytest suite for PLY roundtrip + export math
frontend/               Vite + Three.js + GaussianSplats3D editor UI
scripts/                make_sample_scene.py — synthetic labeled scene, zero GPU
pipeline/kaggle/        photo_to_editable_3d.ipynb (+ generator script)
pipeline/scripts/       extract_frames.py — video → sharp frames helper
data/scenes/<id>/       scene.ply (labeled) · labels.json · objects/ · exports/
```

## Known limitations (v1, by design — see architecture doc §13)

- **Deleting exposes a hole**: geometry occluded by the object was never
  photographed. v2 options: 2D-inpaint + refuse, or leave the honest hole.
- **Mask lifting has no depth test**: occluded Gaussians can inherit a
  front-object's label. Gaussian Grouping / FlashSplat are the planned upgrades.
- **SH degree 0** in the pipeline (base color only, no view-dependent shine).
- **Uniform scale only** on objects — non-uniform scale isn't exactly
  representable for arbitrarily-rotated Gaussians on export.
- Rotating an object does not rotate higher-order SH (only matters for
  imported degree>0 scenes; geometry is unaffected).
- Reflective / transparent / thin objects reconstruct poorly (photogrammetry
  limitation, not specific to this project).

## Roadmap → v2

1. **Replace** verb: TripoSR (fast) → TRELLIS (quality) via free HF Spaces
   Gradio APIs, manual placement + crude shadow blob.
2. Depth-aware mask lifting (or full Gaussian Grouping training) on Kaggle.
3. SuGaR mesh extraction → per-object glTF export.
4. Hole handling after delete (2D inpaint → refuse into new Gaussians).
5. Optional hosting: HF Spaces (backend) + static frontend.
