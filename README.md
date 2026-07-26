# 3dE — Flat 2D art → editable 3D

Drop in a cartoon, character, mascot, sticker or logo and get a **solid 3D
model, split into parts you can take apart** — select, move, rotate, delete and
export each piece. Runs on your own machine in **under a second**. No GPU, no
model download, no API key.

![stack](https://img.shields.io/badge/stack-FastAPI%20·%20Three.js%20·%20trimesh%20·%20scikit--image-blue)

## Why this exists

Every general image→3D model (TripoSR, Hunyuan3D-2, TRELLIS) is trained on
renders of real 3D assets, so it depends on shading and perspective cues.
**Flat art has neither, and they all collapse it into a sheet.** Measured on the
same party-hat clipart, thinnest/longest bounding-box ratio:

| | Result | Time | Hardware |
|---|---|---|---|
| Hunyuan3D-2 | **0.030** — a flat sheet | ~25 s | T4 GPU |
| 3dE | **0.53** — a real cone | **0.26 s** | laptop CPU |

Flat art gives us something photos don't: a clean, closed **silhouette**. So
depth is *constructed* rather than guessed — height grows with distance from the
outline, mirrored front and back into a watertight solid
([Igarashi's "Teddy"](https://dl.acm.org/doi/10.1145/311535.311602) inflation).
It cannot come out flat, and nothing is hallucinated.

Then it goes further than a single blob: the drawing is **split into parts** —
at thin necks (a pom-pom's stem) and across flat colour fills — so each becomes
its own editable object. No other 2D→3D tool hands you pieces you can pull apart.

## What works today

| Verb | Status | How |
|---|---|---|
| **Art → 3D** | ✅ local, ~0.3 s | drop art in the app → silhouette inflation → watertight vertex-coloured mesh |
| **Split into parts** | ✅ local | thin-neck (erosion) + flat-colour-fill separation; stripes and shading are *not* mistaken for parts |
| **Select** | ✅ local app | click (raycast) or sidebar |
| **Move / rotate / scale** | ✅ local app | transform gizmo per object, real-time |
| **Delete / restore** | ✅ local app | hides that object from the render |
| **Export** | ✅ local app | merged `.glb` (art/mesh scenes) or `.ply` (splat scenes), edits baked in |
| **3D print check** | ✅ local app | solidity, real size in mm, material volume, wall thickness and bed adhesion, with actionable warnings |
| Photos → editable splat scene | ✅ Kaggle T4 | the original multi-view path: COLMAP → gsplat → SAM2 → per-Gaussian `object_id` |
| Rigging / articulation | 🚧 idea | parts are already separate, so posing is a natural next step |

### Where it holds up, and where it doesn't

**Good:** clean silhouettes, flat cartoon fills, outlined clip art, mascots,
stickers, logos and letterforms, drawings with separable elements.

**Honest limits:** it inflates the *outline*, so it cannot invent detail the
silhouette doesn't contain — a head drawn in profile inflates to a
profile-shaped solid, not a face. The back is a mirror of the front, which suits
stickers and cartoon figures but not strongly asymmetric objects. Photographs
are out of scope by design; use the multi-view path for those.

A synthetic **sample scene** also ships with the repo, so the edit loop is
demoable in ~2 minutes with zero GPU time.

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

Open http://localhost:5173 and **drag a piece of artwork onto the drop zone** —
a PNG or JPG of a cartoon, mascot, sticker or logo. It becomes an editable 3D
scene in about a third of a second.

Three knobs on the home screen:

- **Puffiness** — how far the shape inflates. `1.0` turns a circle into a full
  sphere; lower is flatter and more sticker-like.
- **Back** — the back profile, independent of the front. `full` mirrors the
  front (a rounded figure); `flat` gives a relief that sits flush against a
  surface — what you want for fridge magnets, badges, wall pieces, and for
  3D printing without supports.
- **Split into editable parts** — off gives one object, on separates the natural
  pieces.

**Controls:** click an object to select · `G`/`R`/`S` switch move/rotate/scale ·
drag the gizmo · `X`/`Del` delete · `Esc` deselect · *Export edited scene*
downloads the merged model with your edits baked in.

**3D print check** — enter the size you intend to print and press *Check*. It
reports whether each part is a closed solid a slicer will accept, the real
dimensions and material volume, and warns about walls thinner than your nozzle
can produce or a model with no flat face to sit on the bed. The advice is
verified against the tool: setting *Back* to `flat` clears the bed-adhesion
warning, and printing larger clears thin walls.

Tests: `.venv/bin/python -m pytest backend/tests -q` (30 tests)

## How the art pipeline works

```
artwork ─▶ silhouette          (border flood-fill; enclosed white kept unless
   │        extraction          it is blobby enough to be a real hole)
   │
   ├─▶ part split               thin necks (erosion + nearest-core regrow)
   │                            + flat colour fills, rejecting stripes/shading
   │
   └─▶ per part: signed distance to the outline ─▶ inflation profile
                     │
                     ▼
       F(x,y,z) = half_thickness(x,y) − |z − centre|      ◀── continuous field,
                     │                                        so marching cubes
                     ▼                                        interpolates to
       marching cubes at level 0 ─▶ watertight mesh            sub-voxel precision
                     │                                        (no terracing)
                     ▼
       vertex colours sampled from the artwork ─▶ objects/NNN.glb
```

Core module: [backend/app/art_to_3d.py](backend/app/art_to_3d.py) ·
endpoint `POST /api/scenes/art`.

## Photo scenes (Kaggle free GPU)

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

### How the photo pipeline works

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
backend/app/art_to_3d.py   silhouette inflation + part splitting (the art path)
backend/app/mesh_scene.py  GLB mesh scenes: manifest + merged export
backend/app/gaussian_ply.py  Gaussian PLY core (load/save/transform/split/export)
backend/tests/             pytest suite (30 tests: art core + PLY/export math)
frontend/                  Vite + Three.js editor UI (mesh + splat viewers)
scripts/                   make_sample_scene.py — synthetic labeled scene, zero GPU
pipeline/kaggle/           photo_to_editable_3d.ipynb (+ generator script)
pipeline/scripts/          extract_frames.py — video → sharp frames helper
data/scenes/<id>/          objects/*.glb or scene.ply · labels.json · exports/
```

## Known limitations of the photo pipeline (see architecture doc §13)

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

## Roadmap

**Art path (the focus)**

1. Better part names than "part 1/2/3" — derive them from colour and position.
2. Asymmetric depth: let the front and back profiles differ instead of mirroring.
3. Posing — the parts are already separate objects, so simple joints/rigging is
   the natural next step.
4. 3D-print sanity: manifold check, wall thickness, and a flat base option.

**Photo path**

5. Depth-aware mask lifting (or full Gaussian Grouping training) on Kaggle.
6. Hole handling after delete (2D inpaint → refuse into new Gaussians).
7. Optional hosting: HF Spaces (backend) + static frontend.
