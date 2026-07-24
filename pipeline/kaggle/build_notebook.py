"""Generate photo_to_editable_3d.ipynb — the Kaggle GPU pipeline notebook.

The notebook is fully self-contained (no repo clone of this project needed):
photos/video -> COLMAP SfM -> 3DGS training (gsplat public API) -> SAM2
segmentation -> per-Gaussian object_id -> scene.ply, ready to import into the
local Splat Editor app via "Import labeled .ply".

Run:  python pipeline/kaggle/build_notebook.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

OUT_PATH = Path(__file__).parent / "photo_to_editable_3d.ipynb"


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
    # Photo → Editable 3D Scene — Kaggle GPU pipeline

    Turns a folder of photos (or one video) of a **static scene** into a
    **labeled Gaussian-splat scene**: every Gaussian gets an `object_id`, so the
    scene can be edited object-by-object (select / move / delete / export) in the
    local Splat Editor app.

    **Pipeline:** COLMAP SfM → 3DGS training (gsplat) → SAM2 video segmentation →
    lift 2D masks to per-Gaussian `object_id` (projection majority vote).

    **Before running**
    1. *Settings → Accelerator*: **GPU T4 x2** (sm_75). ⚠️ **Not P100** — Kaggle's
       current PyTorch dropped Pascal (sm_60) support, so a P100 crashes at the
       first CUDA op. Note: notebooks launched via the Kaggle **API always get a
       P100**; you must pick the T4 in the **web UI** (Session options →
       Accelerator) and *Run All* there. The first cell guards against this.
    2. *Settings → Internet*: **On** (downloads SAM2 checkpoint + pip packages).
    3. Attach your capture as a Kaggle **Dataset** (Add Input): 20–80 photos of a
       static scene walking around it with ~70% overlap, **or** a single video.
       Update `INPUT_DIR` below.

    **Output:** `/kaggle/working/editable_scene.zip` containing `scene.ply`
    (labeled), `scene_raw.ply` (unlabeled backup) and `labels.json`.
    Download it, then in the local app: *Import labeled .ply* → pick `scene.ply`.

    Runtime on a T4: roughly 10–40 min depending on image count and steps.
    """),

    code("""
    # ---------------- Config ----------------
    from pathlib import Path

    INPUT_DIR = Path("/kaggle/input")   # searched recursively for images/video
    WORK = Path("/kaggle/working")

    MAX_IMAGES = 60          # cap frames fed to COLMAP/3DGS (quality vs time)
    MAX_LONG_SIDE = 1600     # downscale images to this long side
    TRAIN_STEPS = 7000       # 3DGS optimization steps (7000 ≈ good preview quality)
    K_OBJECTS = 8            # max objects to segment with SAM2
    VIDEO_FPS = 2            # frame extraction rate when input is a video

    FRAMES = WORK / "frames"; COLMAP_WS = WORK / "colmap"; OUT = WORK / "out"
    for d in (FRAMES, COLMAP_WS, OUT): d.mkdir(parents=True, exist_ok=True)
    """),

    code("""
    # ---------------- GPU guard (fail fast on unsupported accelerator) ----------------
    # Kaggle's current PyTorch build supports sm_70+ only. A Tesla P100 (sm_60)
    # therefore cannot run any CUDA op — every gsplat/torch call would crash.
    # IMPORTANT: notebooks launched via the Kaggle *API* always get a P100; only
    # the *web UI* lets you pick "GPU T4 x2" (sm_75), which works. This cell uses
    # the pre-installed torch to catch that in seconds, before the long install
    # and COLMAP steps waste ~15 min.
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU. Settings -> Accelerator -> 'GPU T4 x2', then Run All."
        )
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    supported = torch.cuda.get_arch_list()
    print(f"GPU: {name} (sm_{cap[0]}{cap[1]}) | torch {torch.__version__}")
    print(f"torch supports: {supported}")

    if f"sm_{cap[0]}{cap[1]}" not in supported:
        raise SystemExit(
            f"\\n{'='*70}\\n"
            f"This session got a {name} (sm_{cap[0]}{cap[1]}), which this PyTorch\\n"
            f"build does not support — training would crash at the first CUDA op.\\n\\n"
            f"FIX: In the Kaggle editor, open the right-hand panel ->\\n"
            f"     Session options -> Accelerator -> 'GPU T4 x2' -> then Run All.\\n"
            f"(API-launched runs always get a P100; only the UI can pick the T4.)\\n"
            f"{'='*70}"
        )
    # Force a real kernel launch so an incompatible device fails here, not later.
    _ = (torch.zeros(16, device="cuda") + 1).sum().item()
    print("GPU OK — a CUDA kernel launched successfully.")
    """),

    code("""
    %%bash
    # ---------------- Install dependencies (~3-5 min) ----------------
    set -e
    apt-get -qq update && apt-get -qq install -y colmap ffmpeg > /dev/null
    pip -q install gsplat plyfile opencv-python-headless scipy torchmetrics
    pip -q install "git+https://github.com/facebookresearch/sam2.git"
    mkdir -p /kaggle/working/ckpts
    [ -f /kaggle/working/ckpts/sam2.1_hiera_small.pt ] || \\
      wget -q -O /kaggle/working/ckpts/sam2.1_hiera_small.pt \\
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
    colmap -h > /dev/null && echo "colmap OK"
    python -c "import gsplat, torch; print('gsplat', gsplat.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
    """),

    code("""
    # ---------------- Collect frames (photos or video) ----------------
    import shutil, subprocess, cv2, numpy as np

    IMG_EXT = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
    VID_EXT = {".mp4", ".mov", ".avi", ".mkv"}

    images = sorted(p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in IMG_EXT)
    videos = sorted(p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in VID_EXT)

    tmp = WORK / "raw_frames"; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
    if videos and not images:
        print("Extracting frames from", videos[0].name)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(videos[0]),
                        "-vf", f"fps={VIDEO_FPS}", str(tmp / "%05d.jpg")], check=True)
        images = sorted(tmp.glob("*.jpg"))
    assert images, f"No images or videos found under {INPUT_DIR} — attach your capture as a Dataset"
    print(f"Found {len(images)} candidate frames")

    def sharpness(path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return cv2.Laplacian(img, cv2.CV_64F).var() if img is not None else 0.0

    # Drop the blurriest frames if over budget (keeps temporal order for SAM2).
    scored = [(p, sharpness(p)) for p in images]
    scored = [s for s in scored if s[1] > 5.0]
    if len(scored) > MAX_IMAGES:
        cutoff = sorted(s[1] for s in scored)[len(scored) - MAX_IMAGES]
        scored = [s for s in scored if s[1] >= cutoff][:MAX_IMAGES]

    shutil.rmtree(FRAMES, ignore_errors=True); FRAMES.mkdir()
    for i, (p, _) in enumerate(scored):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        scale = MAX_LONG_SIDE / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(FRAMES / f"{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Kept {len(scored)} frames -> {FRAMES}")
    """),

    code("""
    %%bash
    # ---------------- COLMAP: camera poses + sparse points (~5-15 min, CPU) ----------------
    set -e
    cd /kaggle/working/colmap
    rm -rf db.db sparse undist && mkdir -p sparse
    colmap feature_extractor --database_path db.db --image_path ../frames \\
        --ImageReader.camera_model SIMPLE_RADIAL --ImageReader.single_camera 1 \\
        --SiftExtraction.use_gpu 0 > /dev/null
    colmap sequential_matcher --database_path db.db --SiftMatching.use_gpu 0 \\
        --SequentialMatching.overlap 15 --SequentialMatching.loop_detection 0 > /dev/null
    colmap mapper --database_path db.db --image_path ../frames --output_path sparse > /dev/null
    # Undistort so cameras become simple PINHOLE (fx fy cx cy)
    colmap image_undistorter --image_path ../frames --input_path sparse/0 \\
        --output_path undist --output_type COLMAP > /dev/null
    colmap model_converter --input_path undist/sparse --output_path undist/sparse \\
        --output_type TXT > /dev/null
    echo "COLMAP done:"; ls undist/sparse
    """),

    code("""
    # ---------------- Parse COLMAP output ----------------
    import numpy as np

    SPARSE = COLMAP_WS / "undist" / "sparse"
    UNDIST_IMAGES = COLMAP_WS / "undist" / "images"

    def qvec2rotmat(q):
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]])

    cameras = {}
    for line in (SPARSE / "cameras.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip(): continue
        parts = line.split()
        cam_id, model, w, h = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        params = np.array(list(map(float, parts[4:])))
        assert model == "PINHOLE", f"expected PINHOLE after undistortion, got {model}"
        fx, fy, cx, cy = params[:4]
        cameras[cam_id] = dict(w=w, h=h, K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))

    views = []  # dicts: name, R (world->cam), t, K, w, h
    lines = [l for l in (SPARSE / "images.txt").read_text().splitlines()
             if not l.startswith("#") and l.strip()]
    for meta in lines[0::2]:  # every second line lists 2D points; skip those
        p = meta.split()
        q = np.array(list(map(float, p[1:5]))); t = np.array(list(map(float, p[5:8])))
        cam = cameras[int(p[8])]
        views.append(dict(name=p[9], R=qvec2rotmat(q), t=t, **cam))
    views.sort(key=lambda v: v["name"])

    pts, cols = [], []
    for line in (SPARSE / "points3D.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip(): continue
        p = line.split()
        pts.append(list(map(float, p[1:4]))); cols.append(list(map(int, p[4:7])))
    points_xyz = np.array(pts); points_rgb = np.array(cols) / 255.0

    cam_centers = np.stack([-v["R"].T @ v["t"] for v in views])
    scene_scale = float(np.linalg.norm(cam_centers - cam_centers.mean(0), axis=1).max()) * 1.1
    print(f"{len(views)} registered views, {len(points_xyz)} sparse points, scene_scale={scene_scale:.2f}")
    assert len(views) >= 10, "Too few registered views — capture more overlap / better texture"
    """),

    code("""
    # ---------------- Train 3DGS with gsplat (~10-25 min on T4) ----------------
    import cv2, torch, math
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy
    from scipy.spatial import cKDTree

    torch.manual_seed(0)
    device = "cuda"
    SH_C0 = 0.28209479177387814

    # --- init from sparse points
    n = len(points_xyz)
    tree = cKDTree(points_xyz)
    knn_dist = tree.query(points_xyz, k=4)[0][:, 1:].mean(1)  # mean dist to 3 NN
    params = torch.nn.ParameterDict({
        "means":     torch.nn.Parameter(torch.tensor(points_xyz, dtype=torch.float32)),
        "quats":     torch.nn.Parameter(torch.zeros(n, 4).index_fill_(1, torch.tensor(0), 1.0)),
        "scales":    torch.nn.Parameter(torch.log(torch.tensor(knn_dist, dtype=torch.float32))[:, None].repeat(1, 3)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.1))),
        "sh0":       torch.nn.Parameter(torch.tensor((points_rgb - 0.5) / SH_C0, dtype=torch.float32)[:, None, :]),
    }).to(device)

    lrs = {"means": 1.6e-4 * scene_scale, "quats": 1e-3, "scales": 5e-3,
           "opacities": 5e-2, "sh0": 2.5e-3}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
                  for k, lr in lrs.items()}

    strategy = DefaultStrategy(verbose=False)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    except ImportError:
        ssim = None

    gt_cache = {}
    def ground_truth(view):
        if view["name"] not in gt_cache:
            img = cv2.cvtColor(cv2.imread(str(UNDIST_IMAGES / view["name"])), cv2.COLOR_BGR2RGB)
            gt_cache[view["name"]] = torch.tensor(img / 255.0, dtype=torch.float32)
        return gt_cache[view["name"]].to(device)

    def render(view, means, quats, scales, opacities, colors):
        viewmat = torch.eye(4, device=device)
        viewmat[:3, :3] = torch.tensor(view["R"], dtype=torch.float32, device=device)
        viewmat[:3, 3] = torch.tensor(view["t"], dtype=torch.float32, device=device)
        K = torch.tensor(view["K"], dtype=torch.float32, device=device)
        return rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
            viewmats=viewmat[None], Ks=K[None], width=view["w"], height=view["h"],
            sh_degree=0, packed=False)

    means_decay = math.exp(math.log(0.01) / TRAIN_STEPS)  # decay means lr to 1% over run
    order = torch.randperm(len(views))
    for step in range(TRAIN_STEPS):
        view = views[order[step % len(views)].item()]
        if step % len(views) == 0 and step > 0:
            order = torch.randperm(len(views))

        renders, alphas, info = render(
            view, params["means"], params["quats"], torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), params["sh0"])
        rgb = renders[0].clamp(0, 1)
        gt = ground_truth(view)

        l1 = (rgb - gt).abs().mean()
        if ssim is not None:
            s = ssim(rgb.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
            loss = 0.8 * l1 + 0.2 * (1.0 - s)
        else:
            loss = l1

        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss.backward()
        strategy.step_post_backward(params, optimizers, state, step, info)
        for k, opt in optimizers.items():
            opt.step(); opt.zero_grad(set_to_none=True)
        optimizers["means"].param_groups[0]["lr"] *= means_decay

        if step % 500 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | gaussians {len(params['means']):,}")

    print(f"Training done: {len(params['means']):,} gaussians")
    """),

    code("""
    # ---------------- Save raw (unlabeled) 3DGS PLY ----------------
    import numpy as np
    from plyfile import PlyData, PlyElement

    def save_gs_ply(path, means, quats_wxyz, log_scales, logit_opacity, f_dc, object_id=None):
        n = len(means)
        fields = ([("x","f4"),("y","f4"),("z","f4"),("nx","f4"),("ny","f4"),("nz","f4"),
                   ("f_dc_0","f4"),("f_dc_1","f4"),("f_dc_2","f4"),("opacity","f4"),
                   ("scale_0","f4"),("scale_1","f4"),("scale_2","f4"),
                   ("rot_0","f4"),("rot_1","f4"),("rot_2","f4"),("rot_3","f4")]
                  + ([("object_id","u4")] if object_id is not None else []))
        v = np.zeros(n, dtype=fields)
        v["x"], v["y"], v["z"] = means.T
        v["f_dc_0"], v["f_dc_1"], v["f_dc_2"] = f_dc.T
        v["opacity"] = logit_opacity
        v["scale_0"], v["scale_1"], v["scale_2"] = log_scales.T
        for i in range(4): v[f"rot_{i}"] = quats_wxyz[:, i]
        if object_id is not None: v["object_id"] = object_id
        PlyData([PlyElement.describe(v, "vertex")], text=False).write(str(path))

    with torch.no_grad():
        MEANS = params["means"].cpu().numpy()
        QUATS = torch.nn.functional.normalize(params["quats"], dim=1).cpu().numpy()
        LOG_SCALES = params["scales"].cpu().numpy()
        LOGIT_OPACITY = params["opacities"].cpu().numpy()
        F_DC = params["sh0"][:, 0, :].cpu().numpy()

    save_gs_ply(OUT / "scene_raw.ply", MEANS, QUATS, LOG_SCALES, LOGIT_OPACITY, F_DC)
    print("wrote", OUT / "scene_raw.ply")
    """),

    code("""
    # ---------------- SAM2: segment objects & track across all frames ----------------
    import torch, cv2, numpy as np
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    CKPT = "/kaggle/working/ckpts/sam2.1_hiera_small.pt"
    CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

    # 1) Seed masks on the first frame (largest K distinct regions)
    frame0 = cv2.cvtColor(cv2.imread(str(FRAMES / "00000.jpg")), cv2.COLOR_BGR2RGB)
    amg = SAM2AutomaticMaskGenerator(
        build_sam2(CFG, CKPT, device="cuda"),
        points_per_side=16, pred_iou_thresh=0.8, stability_score_thresh=0.9,
        min_mask_region_area=int(frame0.shape[0] * frame0.shape[1] * 0.002))
    raw_masks = sorted(amg.generate(frame0), key=lambda m: -m["area"])

    seeds, taken = [], np.zeros(frame0.shape[:2], bool)
    for m in raw_masks:
        seg = m["segmentation"]
        if (seg & taken).sum() / seg.sum() > 0.5: continue  # mostly overlaps an earlier seed
        seeds.append(seg); taken |= seg
        if len(seeds) >= K_OBJECTS: break
    print(f"Seeded {len(seeds)} objects on frame 0")

    # 2) Propagate every seed through the frame sequence as a "video"
    predictor = build_sam2_video_predictor(CFG, CKPT, device="cuda")
    vstate = predictor.init_state(video_path=str(FRAMES), offload_video_to_cpu=True)
    for oid, seg in enumerate(seeds, start=1):
        predictor.add_new_mask(vstate, frame_idx=0, obj_id=oid,
                               mask=torch.from_numpy(seg))

    frame_names = sorted(p.name for p in FRAMES.glob("*.jpg"))
    label_maps = {}  # frame name -> uint16 label map at frame resolution
    with torch.no_grad():
        for fidx, obj_ids, mask_logits in predictor.propagate_in_video(vstate):
            lbl = np.zeros(frame0.shape[:2], np.uint16)
            for oid, logit in zip(obj_ids, mask_logits):
                lbl[(logit[0] > 0.0).cpu().numpy()] = oid
            label_maps[frame_names[fidx]] = lbl
    print(f"Propagated labels through {len(label_maps)} frames")
    """),

    code("""
    # ---------------- Lift 2D masks -> per-Gaussian object_id ----------------
    # Crude-but-effective FlashSplat-style baseline: project every Gaussian mean
    # into every frame and take a majority vote over the SAM2 labels it lands on.
    import json, numpy as np

    N = len(MEANS)
    votes = np.zeros((N, len(seeds) + 1), np.int32)

    # COLMAP frames and SAM2 frames are the same images (undistortion changes
    # intrinsics, not names). Vote in undistorted geometry, sample labels at the
    # same pixel scale.
    for view in views:
        lbl = label_maps.get(view["name"])
        if lbl is None: continue
        Xc = MEANS @ view["R"].T + view["t"]
        z = Xc[:, 2]
        front = z > 1e-3
        uv = (Xc[front] / z[front, None]) @ view["K"].T
        u = np.round(uv[:, 0]).astype(int); v = np.round(uv[:, 1]).astype(int)
        lh, lw = lbl.shape
        sx, sy = lw / view["w"], lh / view["h"]  # label map may differ in scale
        u = np.round(u * sx).astype(int); v = np.round(v * sy).astype(int)
        ok = (u >= 0) & (u < lw) & (v >= 0) & (v < lh)
        idx = np.flatnonzero(front)[ok]
        np.add.at(votes, (idx, lbl[v[ok], u[ok]].astype(int)), 1)

    seen = votes.sum(1)
    obj_votes = votes[:, 1:]
    best = obj_votes.argmax(1) + 1
    best_count = obj_votes.max(1)
    frac = np.divide(best_count, np.maximum(seen, 1))
    OBJECT_ID = np.where((best_count >= 3) & (frac > 0.4), best, 0).astype(np.uint32)

    counts = {int(i): int((OBJECT_ID == i).sum()) for i in np.unique(OBJECT_ID)}
    print("splats per object_id (0 = background):", counts)

    save_gs_ply(OUT / "scene.ply", MEANS, QUATS, LOG_SCALES, LOGIT_OPACITY, F_DC,
                object_id=OBJECT_ID)
    labels = {str(i): f"object {i}" for i in sorted(counts) if i != 0}
    labels["0"] = "background"
    (OUT / "labels.json").write_text(json.dumps(labels, indent=2))
    print("wrote", OUT / "scene.ply")
    """),

    code("""
    # ---------------- Package for download ----------------
    import shutil
    zip_path = shutil.make_archive(str(WORK / "editable_scene"), "zip", OUT)
    print("Download this file:", zip_path)
    """),

    md("""
    ## Import into the local Splat Editor

    1. Download `editable_scene.zip` (right panel → Output) and unzip it.
    2. Start the local app (`uvicorn` backend + `npm run dev` frontend — see the
       project README).
    3. In the sidebar: **Import labeled .ply** → choose a scene id → pick
       `scene.ply` → *Upload & split*. The backend splits it into per-object
       PLYs and the scene becomes editable immediately.
    4. Optional: rename objects by editing `labels.json` inside
       `data/scenes/<scene-id>/` locally, then *rebuild* (POST
       `/api/scenes/<id>/rebuild` or delete `manifest.json` and reload).

    ## Troubleshooting

    | Symptom | Fix |
    |---|---|
    | `SystemExit` at cell 2 / `no kernel image is available` / `sm_60 not compatible` | You're on a **P100**. Session options → Accelerator → **GPU T4 x2**, then Run All. API-launched runs always get a P100 — switch in the web UI. |
    | COLMAP registers < 10 views | More photo overlap; avoid textureless walls, glass, mirrors; shoot a slow loop video instead |
    | `CUDA out of memory` in training | Lower `MAX_LONG_SIDE` to 1200, or `MAX_IMAGES` to 40 |
    | Training slow / session timeout | Lower `TRAIN_STEPS` to 4000 (preview quality) — Kaggle gives ~9 h sessions, this fits easily |
    | SAM2 masks flicker across frames | Use a video capture (temporal continuity helps tracking); raise `VIDEO_FPS` to 3–4 |
    | Objects merged / background labeled as object | Lower/raise `K_OBJECTS`; re-seed on a frame where objects are well separated (change `frame_idx=0`) |
    | gsplat CUDA compile errors | First `import gsplat` JIT-compiles kernels (~2 min) — rerun the cell once before assuming failure |

    **Known v1 limitations** (by design, see the project architecture doc):
    no depth-aware vote (occluded Gaussians can pick up front-object labels),
    SH degree 0 only (no view-dependent shading), deleting an object exposes an
    unphotographed hole. All are listed as v2 work in the roadmap.
    """),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    # nbformat expects `source` as a string or list of lines; strings are fine.
    OUT_PATH.write_text(json.dumps(notebook, indent=1))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
