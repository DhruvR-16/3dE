"""Headless driver for the Hunyuan3D image->3D Kaggle notebook.

Two steps, because Kaggle's API always assigns a P100 (which this model can't
run on) while only the web UI can pick a T4:

    # 1) upload images + push the notebook, then run it on a T4 in the Kaggle UI
    python pipeline/kaggle/run_image_to_3d.py push --images ./my_images

    #    -> open the kernel, Session options -> Accelerator -> GPU T4 x2,
    #       then "Save & Run All (Commit)".

    # 2) once it finishes, pull the GLBs straight into the local app
    python pipeline/kaggle/run_image_to_3d.py fetch --scene-id my-hat

After `fetch`, the scene shows up in the app automatically (it becomes a mesh
scene under data/scenes/<scene-id>/objects/*.glb).

The Kaggle username comes from an authenticated `kaggle` CLI; no secret is read
or printed by this tool.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = Path(__file__).parent / "image_to_3d.ipynb"
DATASET_SLUG = "img3d-input"
KERNEL_SLUG = "image-to-3d"
KERNEL_TITLE = "Image to 3D"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd))
    return subprocess.run(cmd, text=True, **kw)


def kaggle_user() -> str:
    r = subprocess.run(
        ["kaggle", "kernels", "list", "--mine", "-p", "1", "--csv"],
        text=True, capture_output=True,
    )
    for line in r.stdout.splitlines():
        if "/" in line:
            return line.split("/", 1)[0].strip().strip('"')
    sys.exit("Could not determine Kaggle username — is the kaggle CLI authenticated?")


def cmd_push(args: argparse.Namespace) -> None:
    user = kaggle_user()
    print("kaggle user:", user)
    imgs = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not imgs:
        sys.exit(f"no images ({sorted(IMAGE_EXTS)}) in {args.images}")
    if len(imgs) > 20:
        print(f"note: {len(imgs)} images found; the notebook caps at 20 objects.")

    # stage + push dataset
    ds_stage = REPO_ROOT / ".kaggle_stage" / "img3d_dataset"
    ds_stage.mkdir(parents=True, exist_ok=True)
    for old in ds_stage.glob("*"):
        old.unlink()
    for p in imgs:
        (ds_stage / p.name).write_bytes(p.read_bytes())
    ds_ref = f"{user}/{DATASET_SLUG}"
    (ds_stage / "dataset-metadata.json").write_text(
        json.dumps({"title": DATASET_SLUG, "id": ds_ref, "licenses": [{"name": "CC0-1.0"}]}, indent=2)
    )
    status = subprocess.run(["kaggle", "datasets", "status", ds_ref], text=True, capture_output=True)
    exists = status.returncode == 0 and "error" not in (status.stdout + status.stderr).lower()
    if exists:
        sh(["kaggle", "datasets", "version", "-p", str(ds_stage), "-m", "update", "--dir-mode", "zip"])
    else:
        sh(["kaggle", "datasets", "create", "-p", str(ds_stage), "--dir-mode", "zip"])
    print(f"staged {len(imgs)} image(s) -> dataset {ds_ref}")

    # stage + push kernel
    k_stage = REPO_ROOT / ".kaggle_stage" / "img3d_kernel"
    k_stage.mkdir(parents=True, exist_ok=True)
    (k_stage / NOTEBOOK.name).write_bytes(NOTEBOOK.read_bytes())
    (k_stage / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{KERNEL_SLUG}",
        "title": KERNEL_TITLE,
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [ds_ref],
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=2))
    sh(["kaggle", "kernels", "push", "-p", str(k_stage)])

    print("\n" + "=" * 70)
    print(f"Pushed: https://www.kaggle.com/code/{user}/{KERNEL_SLUG}")
    print("Now, in the Kaggle web UI:")
    print("  1. Open the notebook (link above) -> Edit")
    print("  2. Session options -> Accelerator -> 'GPU T4 x2'")
    print("  3. Save Version -> 'Save & Run All (Commit)'")
    print("  (API runs always get a P100, which this model can't use — the T4")
    print("   must be picked in the UI.)")
    print("Then:  python pipeline/kaggle/run_image_to_3d.py fetch --scene-id <name>")
    print("=" * 70)


def cmd_fetch(args: argparse.Namespace) -> None:
    user = kaggle_user()
    ref = f"{user}/{KERNEL_SLUG}"
    dl = REPO_ROOT / ".kaggle_stage" / "img3d_output"
    dl.mkdir(parents=True, exist_ok=True)
    for old in dl.glob("*"):
        if old.is_file():
            old.unlink()
    r = sh(["kaggle", "kernels", "output", ref, "-p", str(dl)])
    if r.returncode != 0:
        sys.exit("fetch failed — has the run finished (COMPLETE) on Kaggle?")

    zips = list(dl.glob("editable_scene.zip"))
    scene_dir = REPO_ROOT / "data" / "scenes" / args.scene_id
    objects = scene_dir / "objects"
    objects.mkdir(parents=True, exist_ok=True)
    for old in objects.glob("*.glb"):
        old.unlink()

    if zips:
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall(dl / "unzipped")
        src = dl / "unzipped"
    else:
        src = dl  # some runs expose out/ directly

    glbs = sorted(src.rglob("*.glb"))
    if not glbs:
        sys.exit(f"no .glb files in the run output ({dl}) — check the kernel log for errors.")
    for i, g in enumerate(glbs):
        (objects / f"{i:03d}.glb").write_bytes(g.read_bytes())
    labels_src = next(src.rglob("labels.json"), None)
    labels = json.loads(labels_src.read_text()) if labels_src else {str(i): f"object {i}" for i in range(len(glbs))}
    (scene_dir / "labels.json").write_text(json.dumps(labels, indent=2))

    print(f"installed {len(glbs)} object(s) -> {scene_dir}")
    print(f"open the app and pick scene '{args.scene_id}'.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push", help="upload images + push the notebook")
    p.add_argument("--images", type=Path, required=True, help="folder of input images")
    p.set_defaults(func=cmd_push)
    f = sub.add_parser("fetch", help="download the finished run into the app")
    f.add_argument("--scene-id", required=True, help="local scene id to create (lowercase, - and _)")
    f.set_defaults(func=cmd_fetch)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
