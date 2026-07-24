# Photo → Editable 3D Scene
### Feasibility assessment + full architecture, stack, and build plan (100% open-source / free-compute)
*Compiled July 2026*

---

## 0. TL;DR Verdict

| Capability | Status | Confidence |
|---|---|---|
| Reconstruct real photos into 3D geometry | ✅ Solved, multiple mature options | High |
| **Select** individual objects in the 3D scene | ✅ Solved (2025-26 research, code available) | High |
| **Move** a selected object | ✅ Solved (trivial once selection works) | High |
| **Delete** a selected object | ✅ Solved, but leaves a geometric "hole" | Medium — hole-fill is the weak point |
| **Replace** with a generated object | ⚠️ Partially solved — generation is easy, seamless *compositing* (lighting/shadow/scale match) is open research | Low-Medium |
| **Export** to standard 3D formats | ✅ Solved (mesh + glTF, or native splat formats) | High |
| Run on free/no GPU | ✅ Yes — Colab/Kaggle free T4, or Hugging Face Spaces API offload | High |

**Bottom line:** A real, working, object-decomposable 3D scene editor from photos is a legitimate weekend-to-two-week hackathon build using entirely free/open tools. Scope your demo around **select → move → delete → export**, which is genuinely solid engineering on top of existing research code. Treat **replace** as a stretch/"wow" feature with an honest fallback (e.g., swap in a pre-made asset rather than a freshly generated one) if time runs short — photorealistic insertion is still a research problem, not a library you `pip install`.

---

## 1. Why this is harder than it sounds (read this before building)

Most "photo to 3D" demos you've seen (Luma, Polycam, Kiri Engine) reconstruct a scene as **one undifferentiated blob** — a point cloud, mesh, or Gaussian splat with no concept of "this is a chair" vs "this is a wall." That's the easy 80%.

The genuinely hard 20% — and the part that makes your project interesting rather than a Luma clone — is **object-level decomposition**: giving every 3D point/Gaussian an object identity so you can click on "the chair" and only the chair moves. This is an active 2024-2026 research area (see the survey citation in §9). It is *not* yet a one-command tool. You will be assembling 3-4 research repos into a pipeline yourself. That's very achievable, just be aware going in that "stitch together open-source research code" is the actual nature of the task, not "call an API."

---

## 2. High-Level Architecture

```
┌──────────────┐   ┌───────────────────┐   ┌────────────────────┐   ┌──────────────────────┐
│  1. CAPTURE  │──▶│ 2. RECONSTRUCTION │──▶│ 3. OBJECT           │──▶│ 4. EDITABLE SCENE     │
│  photos/video│   │ pose + geometry   │   │    SEGMENTATION     │   │    (labeled Gaussians)│
└──────────────┘   │ (VGGT / COLMAP+   │   │ SAM2 (2D) + lift    │   │ every splat has an    │
                    │  3DGS)            │   │ to 3D (Gaussian     │   │ object_id             │
                    └───────────────────┘   │ Grouping/FlashSplat)│   └──────────┬────────────┘
                                             └─────────────────────┘              │
                              ┌────────────────────────────────────────────────────┘
                              ▼
                    ┌───────────────────────┐        ┌──────────────────────┐
                    │ 5. EDIT ENGINE        │───────▶│ 6. GENERATIVE REPLACE │
                    │ select/move/delete =  │        │ image→3D (TRELLIS.2 / │
                    │ transform or filter   │        │ Hunyuan3D) + align +  │
                    │ Gaussian subsets       │        │ inpaint hole          │
                    └──────────┬────────────┘        └──────────┬────────────┘
                               └──────────────────┬──────────────┘
                                                   ▼
                                         ┌───────────────────────┐
                                         │ 7. EXPORT              │
                                         │ SuGaR mesh extraction  │
                                         │ → glTF/OBJ/USDZ, or    │
                                         │ native .ply/.splat     │
                                         └───────────────────────┘
```

Everything downstream of Stage 2 revolves around one design decision: **use 3D Gaussian Splatting (3DGS) as the core scene representation**, not a mesh and not a NeRF. Reasons:

- It trains in minutes on a free T4, not hours.
- Gaussians are just a big list of (position, rotation, scale, color, opacity) — trivially easy to filter, transform, or delete a subset of. A mesh requires re-topology to edit cleanly; a NeRF is a black-box network, nearly impossible to edit locally.
- The entire "select/move/delete" research literature (2024-2026) targets 3DGS specifically, so tooling and prior code exist.
- You only convert to mesh (via SuGaR) at the very end, for export/interop.

---

## 3. Stage 1 — Capture

No ML here, just good data hygiene, since reconstruction quality is capped by your input photos.

- **20-80 photos** of a static scene, ~70% overlap between adjacent shots, walking a full loop/sphere around the subject. More views = better; fewer than ~15 views degrades badly.
- Avoid moving objects, reflective/transparent surfaces (glass, mirrors, glossy metal), and pure texture-less walls — all break both SfM and segmentation.
- Consistent lighting across shots (don't mix flash and daylight).
- Phone video works too — just extract frames at ~2 fps and dedupe blurry ones (`ffmpeg` + a Laplacian-variance blur filter is enough).
- **Single-photo input is a different, much lower-fidelity problem** (monocular depth + generative completion, e.g. via Hunyuan3D/TRELLIS "scene" variants). It technically works but gives you a rough relightable diorama, not an accurate scene. Recommend requiring multi-photo capture as your primary supported mode, with single-photo as a degraded fallback if you have time.

---

## 4. Stage 2 — Geometry Reconstruction (pose + point cloud)

You need per-photo camera position/orientation before you can build 3D geometry. Two viable open-source routes:

### Option A (recommended for hackathon): **VGGT** — feed-forward, pose-free
- Repo: `facebookresearch` ecosystem / VGGT (Visual Geometry Grounded Transformer, CVPR 2025).
- Single forward pass over all your images → dense point cloud + camera poses + depth, no per-scene optimization loop. Minutes, not hours.
- Benchmarked as currently the strongest pose-free reconstruction model, ahead of DUSt3R/MASt3R, and it processes many views jointly rather than pairwise.
- Runs on a free Colab/Kaggle T4 for the ~20-80 image counts you'll be using.
- Sibling options if you want alternatives: **MASt3R** (better dense local features, pairwise), **Fast3R** (positional-interpolation for large multi-view sets).

### Option B (classic, more accurate, slower): **COLMAP** + **3D Gaussian Splatting**
- COLMAP (open-source, CPU-capable, no GPU strictly required) does Structure-from-Motion → sparse point cloud + camera poses. This is the standard, most battle-tested pipeline (used by essentially every 3DGS tutorial).
- Feed COLMAP's output into **gsplat** (the actively-maintained open-source Gaussian Splatting training library, successor to the original INRIA `graphdeco-inria/gaussian-splatting` repo) or **Nerfstudio's** `splatfacto` model to train the actual 3DGS scene.
- Slower (per-scene optimization: ~10-30 min on a T4 for a small scene) but more accurate/higher fidelity than the feed-forward route, and is the representation everything downstream (SuGaR, Gaussian Grouping) was originally designed against — so compatibility with Stage 3/4 tooling is smoother.

**Recommendation:** Use COLMAP + gsplat as your primary path (best tooling compatibility for the segmentation/editing stage), and keep VGGT in your back pocket as a fast fallback for scenes where COLMAP fails to find enough feature matches (common on texture-poor rooms).

---

## 5. Stage 3 — Per-Object 3D Segmentation (the hard, novel part)

This is what separates "editable scene" from "static splat viewer." Two-step process:

### 5a. 2D segmentation across views
- **SAM2** (Meta, open-source, Apache-licensed) — segment-and-track objects across a video/image sequence with a single click or box prompt per object. Gives you consistent 2D masks per object across all your input photos.
- Runs on a free T4; the base checkpoints are small enough to also run reasonably on CPU for a handful of images if you're truly GPU-starved.

### 5b. Lifting 2D masks into 3D object IDs on the Gaussians
This is the actual research-frontier step. Pick one implementation (all have public code):

- **Gaussian Grouping** ("Segment and Edit Anything in 3D Scenes", ECCV 2024) — trains an *identity encoding* per Gaussian jointly with the 3DGS optimization, supervised by your SAM2 masks + a video object tracker for cross-view consistency. Directly designed for exactly your use case (segment **and edit**). This is your primary target repo.
- **FlashSplat** (ECCV 2024) — reframes 2D→3D Gaussian segmentation as a solvable linear program instead of learned optimization; notably faster, worth trying if Gaussian Grouping's training step is too slow on your free GPU quota.
- **OmniSeg3D** — hierarchical contrastive learning for view-consistent 3D segmentation, good if you want click-based multi-granularity selection (e.g., "the whole sofa" vs "just the cushion").

**Output of this stage:** your Gaussian splat scene, where every single Gaussian now carries an extra `object_id` attribute alongside its position/rotation/scale/color/opacity. This is the data structure that makes everything else possible.

---

## 6. Stage 4 — The Editable Representation & Edit Operations

Once every Gaussian has an `object_id`, your four verbs become simple, well-defined operations on a big array — no more ML required for these:

| Operation | Implementation |
|---|---|
| **Select** | Ray-cast from a screen click into the splat scene, find nearest-hit Gaussian, read its `object_id`, highlight all Gaussians sharing that ID (e.g. via an outline shader or opacity boost). |
| **Move** | Apply a rigid transform (translation + rotation, optionally uniform scale) to the `position` and `rotation` fields of every Gaussian with the matching `object_id`. This is literally a matrix multiply over a filtered array — real-time, no retraining. |
| **Delete** | Filter the render set to exclude Gaussians with that `object_id`. Trivial. **The catch:** this exposes a hole — geometry that was occluded by the object was never photographed, so there's nothing there. |
| **Hole-filling after delete** | This is the honest weak point of the whole pipeline. Three tiers of solution, pick based on time budget: **(1) Cheap/hackathon-safe:** just leave the hole or fill with a simple plane/color-matched patch — acceptable for a demo, and arguably more honest than a bad fake. **(2) Medium:** 2D inpainting (e.g. an open diffusion inpainting model) applied to the "hole" region in each original camera view, then re-fuse those inpainted pixels back into new Gaussians for that region. **(3) Research-grade:** dedicated 3DGS object-removal-with-inpainting methods (e.g. papers like "Learning 3D geometry and feature consistent gaussian splatting for object removal," GPGS) — code availability varies, treat as a stretch goal, not a dependency. |

Build your edit engine as a thin layer (JS/Python) operating directly on the Gaussian buffer — this is genuinely simple once Stage 3 hands you labeled data. Don't over-engineer this part; it's the part of the pipeline you have full control over.

---

## 7. Stage 5 — Generative Replace

Once an object is deleted (and its hole handled), "replace" = generate a new 3D asset and insert it.

### Generating the new object
Current (July 2026) best open-source image-to-3D options, all Hugging Face-hosted with free Spaces you can call via API instead of running locally:

| Model | Best for | License | Local hardware |
|---|---|---|---|
| **TRELLIS.2** (Microsoft) | Best overall fidelity/PBR materials; current open academic baseline | MIT (fully open commercial use) | ~12-16GB VRAM locally, or use the free HF Space |
| **Hunyuan3D 2.1 / 2.5** (Tencent) | Best texture quality, PBR materials, production-shaped assets | Tencent Community License (commercial OK w/ conditions) | Similar range; heavier "Paint" texture stage |
| **TripoSR** | Fastest (<1 sec/object), lowest VRAM (6-8GB), best starting point to prototype your pipeline plumbing | MIT | Runs fine on a free T4 |
| **Hi3DGen** | Best pure geometric accuracy if texture matters less | Open | Moderate |
| **Stable Fast 3D (SF3D)** | Sub-second generation, free for orgs under $1M revenue | Semi-open | Low |

**Recommendation:** prototype your insertion pipeline against **TripoSR** first (fast, light, forgiving), then swap in **TRELLIS.2** for your actual demo quality once the mechanics work.

### The part nobody's fully solved: seamless compositing
Generating a nice-looking isolated 3D object is the easy half. Making it look like it was *always in the photographed scene* requires:
- **Scale/placement matching** — you control this manually or heuristically (snap to the floor plane, match the deleted object's bounding box).
- **Lighting/shadow matching** — genuinely open research. Realistic options for a hackathon: bake a rough shadow blob under the inserted object using the scene's dominant light direction (estimate this crudely from existing shadows in your source photos), and don't chase full relighting — it's not solved well enough in open-source tooling yet to be worth your time budget.
- **Set expectations accordingly** in your demo: "replace" will look like a competent AR sticker, not an invisible edit. That's a fine, honest outcome for a hackathon.

---

## 8. Stage 6 — Export

Two parallel export paths depending on target use case:

### A. Mesh export (for Blender / Unity / Unreal / standard 3D pipelines)
- **SuGaR** (Surface-Aligned Gaussian Splatting, CVPR 2024) extracts a clean textured mesh directly from your trained Gaussians via a surface-alignment regularization + Poisson reconstruction. Ships with a Blender import script.
- Export the mesh as **.glTF/.glb** (best modern interop, PBR-ready) or **.OBJ+.MTL** (universal fallback).
- Because your Gaussians already carry `object_id`, you can export **per-object sub-meshes** — i.e. actually deliver "every object as a separate, selectable mesh" in the exported file, not just a fused blob. This is the feature that makes your export genuinely "editable" downstream too.

### B. Native splat export (for web viewers / lightweight sharing)
- Export directly as **.ply** or **.splat** — no mesh conversion needed.
- **SuperSplat** (PlayCanvas, MIT-licensed, free, browser-based) — drag in your .ply, do final cleanup (remove stray "floater" Gaussians, crop, compress 70-90%), re-export. Also usable as a lightweight secondary editor/QA step in your own pipeline.
- For embedding in your own app: **GaussianSplats3D** (Three.js), `antimatter15/splat` (plain WebGL, great mobile support) — both open-source, both give you a real-time in-browser splat viewer for your final "app" deliverable without needing Unity/Unreal at all.

---

## 9. Full Tech Stack (start to end)

| Layer | Tool | Why |
|---|---|---|
| Capture preprocessing | `ffmpeg` + OpenCV blur-detection | Frame extraction, dedup |
| SfM / pose | **COLMAP** (primary) / **VGGT** (fast fallback) | Camera poses + sparse geometry |
| Core 3D representation | **3D Gaussian Splatting** via **gsplat** (or Nerfstudio `splatfacto`) | Fast, editable, mature ecosystem |
| Mesh extraction (optional/export) | **SuGaR** | Best current open Gaussians→mesh path |
| 2D segmentation | **SAM2** | Object masks + cross-frame tracking |
| 3D object decomposition | **Gaussian Grouping** (primary) / FlashSplat / OmniSeg3D | Bakes `object_id` into every Gaussian |
| Generative object replace | **TripoSR** (prototype) → **TRELLIS.2** (final) | Image → 3D asset |
| Splat cleanup/QA editor | **SuperSplat** | Free browser tool, also good for manual polish |
| 3D viewer / your app frontend | **Three.js + GaussianSplats3D**, or **React** wrapper | Real-time splat rendering in-browser |
| Backend / orchestration | **FastAPI** (Python) | Wraps the ML pipeline as job endpoints |
| Job/GPU execution | Colab/Kaggle notebook workers, or HF Spaces API calls | See §10 |
| Storage | Local disk / S3-compatible bucket for scene assets (.ply, .glb) | Straightforward |
| DB (if multi-user/project) | Postgres (Supabase/Neon free tier) | Scene metadata, object graph, user projects |

---

## 10. Compute Strategy on Zero Budget

You do **not** need to own a GPU. Two complementary strategies:

**1. Free notebook GPUs for the pipeline stages you must run yourself (3DGS training, Gaussian Grouping):**
- **Google Colab free tier** — NVIDIA T4, 16GB VRAM, ~15-30 hrs/week, 12-hour session cap.
- **Kaggle Notebooks** — T4 or P100, 16GB VRAM, a more generous/predictable **30 hrs/week**, 9-hour sessions. Good complement to Colab — rotate between the two when one is rate-limited.
- Both are sufficient for training a 3DGS scene on a hackathon-sized capture (20-80 images) — this typically takes single-digit minutes to ~30 min on a T4, well within free quotas.

**2. Offload the heaviest generative models entirely — don't run them locally at all:**
- **Hugging Face Spaces** hosts free, GPU-backed demo instances (many using "ZeroGPU" shared-A100 allocation) for SAM2, TRELLIS.2, Hunyuan3D, TripoSR, etc. Many expose a **Gradio API** you can call from your FastAPI backend with a few lines of code — effectively giving you free access to hardware far beyond a T4 for the generation step specifically, without managing any GPU yourself.
- This is the single highest-leverage trick for your constraint: keep 3DGS training + Gaussian Grouping on your own free Colab/Kaggle T4 (these need per-scene optimization so you want a stable session), and route TripoSR/TRELLIS.2 calls through free HF Spaces APIs (these are single feed-forward calls, perfect for stateless API use).

---

## 11. Suggested Application Architecture

```
┌───────────────────┐     ┌─────────────────────────┐     ┌──────────────────────┐
│  Next.js frontend  │────▶│  FastAPI backend         │────▶│ Colab/Kaggle GPU      │
│  - upload photos   │     │  - job orchestration     │     │ worker (long-lived    │
│  - Three.js splat   │     │  - calls HF Spaces API   │     │ notebook, polls a     │
│    viewer w/ click- │◀────│    for TripoSR/TRELLIS   │     │ job queue, runs       │
│    to-select        │     │  - stores results in     │     │ COLMAP→3DGS→Gaussian  │
│  - move/delete/     │     │    Postgres + object     │     │ Grouping)             │
│    replace UI       │     │    storage               │     └──────────────────────┘
└───────────────────┘     └─────────────────────────┘
```

- **Frontend:** Next.js + Three.js (via `GaussianSplats3D`) for the actual editable viewer — click-to-select, drag-to-move, delete key, "replace" panel triggering the generation call.
- **Backend:** FastAPI exposing `/reconstruct`, `/segment`, `/edit`, `/generate`, `/export` endpoints. Keep the heavy long-running jobs (COLMAP+3DGS+Gaussian Grouping) async — push to a queue your Colab/Kaggle notebook polls, since those environments aren't reachable as always-on servers.
- **Database:** Postgres (Supabase or Neon free tier) for scene/project metadata and the object graph (`object_id` → bounding box, label, transform history).
- **Storage:** flat files (.ply per scene, .glb per export) in object storage or even local disk for a hackathon demo.

---

## 12. Realistic Build Roadmap

**Phase 1 — Core pipeline (days 1-3):** COLMAP → gsplat 3DGS training on one test scene. Get a static, viewable splat working end-to-end in a Three.js viewer. This alone is a legitimate demo checkpoint.

**Phase 2 — Segmentation (days 3-5):** SAM2 masks on your captured views → Gaussian Grouping training → verify every Gaussian has a sane `object_id`. This is your riskiest phase — budget the most slack here, and have FlashSplat as a fallback if Gaussian Grouping's training proves too slow/finicky on free-tier GPU time.

**Phase 3 — Edit engine (days 5-6):** Select/move/delete as described in §6 — pure engineering on top of Phase 2's output, should move fast.

**Phase 4 — Export (day 6-7):** SuGaR mesh extraction + glTF export, plus native .ply export via SuperSplat cleanup.

**Phase 5 — Replace (stretch, remaining time):** TripoSR integration first for plumbing, TRELLIS.2 swap-in if time allows, manual placement/scale UI, crude shadow baking. Explicitly scope this as "nice to have" in your pitch — a working select/move/delete/export pipeline is already a strong, honest, differentiated demo without it.

---

## 13. Known Limitations (state these honestly in your pitch — judges will respect it more than overclaiming)

- Reflective, transparent, and thin/wiry objects reconstruct poorly with any photogrammetry-based method (Gaussian splatting included).
- Deleted-object holes are visibly imperfect without heavy additional engineering — pick a graceful failure mode (soft blur/patch) rather than trying to hide this.
- "Replace" will look composited, not seamlessly relit, with fully open-source tooling as of mid-2026.
- Reconstruction accuracy scales directly with photo count/coverage — a 5-photo scene will look noticeably worse than a 50-photo scene; set capture guidance expectations in your UI.
- Free-tier GPU session limits (12hr Colab / 9hr Kaggle) mean you should checkpoint long training runs, not assume one continuous session.

---

## 14. Reference Repos & Links

- COLMAP: `colmap/colmap`
- gsplat (3DGS training library): `nerfstudio-project/gsplat`
- Original 3DGS reference implementation: `graphdeco-inria/gaussian-splatting`
- VGGT: Meta AI's Visual Geometry Grounded Transformer (search "VGGT 3D reconstruction")
- SAM2: `facebookresearch/sam2`
- Gaussian Grouping: "Segment and Edit Anything in 3D Scenes" (ECCV 2024) — search `lkeab/gaussian-grouping`
- FlashSplat: ECCV 2024, "2D to 3D Gaussian Splatting Segmentation Solved Optimally"
- SuGaR: `Anttwo/SuGaR`
- SuperSplat: PlayCanvas, `playcanvas/supersplat`, also at superspl.at/editor
- GaussianSplats3D (Three.js viewer): `mkkellogg/GaussianSplats3D`
- TripoSR: Stability AI + Tripo, on Hugging Face
- TRELLIS.2: Microsoft, on Hugging Face
- Hunyuan3D 2.1/2.5: Tencent, on Hugging Face

---

*This doc reflects the open-source 3D reconstruction/editing landscape as of July 2026. This is a genuinely fast-moving research area — re-check Hugging Face's "image-to-3D" model listings and the `Awesome-3DGS-Applications` survey repo every few weeks during your build for newer, better-performing drop-in replacements.*
