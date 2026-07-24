"""Headless Kaggle driver: create dataset, push notebook, run on GPU, fetch output.

Uses the official Kaggle API (needs ~/.kaggle/kaggle.json). No browser required.
This is how we test the pipeline notebook on Kaggle's free GPU end to end:

    python pipeline/kaggle/run_on_kaggle.py --images <dir> [--wait] [--fetch <dir>]

Steps:
  1. `kaggle datasets create` from the image folder (or `version` if it exists).
  2. `kaggle kernels push` with GPU + internet on and the dataset attached — a
     push automatically queues a run.
  3. With --wait: poll `kaggle kernels status` until it finishes.
  4. With --fetch: download the run's /kaggle/working output (editable_scene.zip).

The Kaggle username is read from ~/.kaggle/kaggle.json (the key is never printed).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = Path(__file__).parent / "photo_to_editable_3d.ipynb"

DATASET_SLUG = "south-building-test50"      # <user>/south-building-test50
KERNEL_SLUG = "photo-to-editable-3d"        # <user>/photo-to-editable-3d
KERNEL_TITLE = "Photo to Editable 3D"       # >= 5 chars required by Kaggle


def kaggle_username(override: str | None = None) -> str:
    """Resolve the Kaggle username without reading any secret.

    Order: explicit --user, KAGGLE_USERNAME env, kaggle.json (classic creds),
    else derive from an authenticated `kaggle kernels list --mine` (works with
    KGAT_ access tokens, where no username file exists).
    """
    if override:
        return override
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if cred.exists():
        user = json.loads(cred.read_text()).get("username")
        if user:
            return user
    # Derive from the ref prefix of one's own kernels (e.g. "dhruvrama/foo").
    r = subprocess.run(
        ["kaggle", "kernels", "list", "--mine", "-p", "1", "--csv"],
        text=True, capture_output=True,
    )
    for line in r.stdout.splitlines():
        if "/" in line:
            return line.split("/", 1)[0].strip().strip('"')
    sys.exit(
        "Could not determine Kaggle username. Pass --user <name>, or set up "
        "~/.kaggle/access_token (Settings -> API -> Create New Token) / kaggle.json."
    )


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, text=True, **kw)


def dataset_exists(ref: str) -> bool:
    r = run(["kaggle", "datasets", "status", ref], capture_output=True)
    return r.returncode == 0 and "error" not in (r.stdout + r.stderr).lower()


def push_dataset(images_dir: Path, user: str) -> str:
    ref = f"{user}/{DATASET_SLUG}"
    stage = REPO_ROOT / ".kaggle_stage" / "dataset"
    stage.mkdir(parents=True, exist_ok=True)
    for old in stage.glob("*"):
        old.unlink()
    imgs = sorted(p for p in images_dir.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not imgs:
        sys.exit(f"no images found in {images_dir}")
    for p in imgs:
        (stage / p.name).write_bytes(p.read_bytes())
    print(f"staged {len(imgs)} images -> {stage}")

    meta = {
        "title": DATASET_SLUG,
        "id": ref,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (stage / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    if dataset_exists(ref):
        print(f"dataset {ref} exists -> pushing a new version")
        r = run(["kaggle", "datasets", "version", "-p", str(stage),
                 "-m", "update", "--dir-mode", "zip"])
    else:
        print(f"creating dataset {ref}")
        r = run(["kaggle", "datasets", "create", "-p", str(stage), "--dir-mode", "zip"])
    if r.returncode != 0:
        sys.exit("dataset push failed")
    return ref


def push_kernel(dataset_ref: str, user: str) -> str:
    ref = f"{user}/{KERNEL_SLUG}"
    stage = REPO_ROOT / ".kaggle_stage" / "kernel"
    stage.mkdir(parents=True, exist_ok=True)
    (stage / NOTEBOOK.name).write_bytes(NOTEBOOK.read_bytes())

    meta = {
        "id": ref,
        "title": KERNEL_TITLE,
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [dataset_ref],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"kernel metadata:\n{json.dumps(meta, indent=2)}")

    r = run(["kaggle", "kernels", "push", "-p", str(stage)])
    if r.returncode != 0:
        sys.exit("kernel push failed")
    return ref


def wait_for_kernel(ref: str, timeout_min: int = 90) -> str:
    print(f"waiting for {ref} (poll every 30s, timeout {timeout_min} min)...")
    deadline = time.time() + timeout_min * 60
    last = ""
    while time.time() < deadline:
        r = run(["kaggle", "kernels", "status", ref], capture_output=True)
        out = (r.stdout + r.stderr).strip()
        if out != last:
            print(out)
            last = out
        low = out.lower()
        if "complete" in low:
            return "complete"
        if "error" in low or "cancel" in low:
            return "error"
        time.sleep(30)
    return "timeout"


def fetch_output(ref: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    r = run(["kaggle", "kernels", "output", ref, "-p", str(dest)])
    if r.returncode != 0:
        print("output fetch failed (run may not have produced files)")
        return
    print(f"downloaded output -> {dest}")
    for p in sorted(dest.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(dest)}  ({p.stat().st_size/1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, required=True, help="folder of test images")
    ap.add_argument("--wait", action="store_true", help="poll until the run finishes")
    ap.add_argument("--fetch", type=Path, help="download run output to this folder")
    ap.add_argument("--skip-dataset", action="store_true",
                    help="reuse existing dataset, only push+run the kernel")
    ap.add_argument("--user", help="Kaggle username (else auto-detected)")
    args = ap.parse_args()

    user = kaggle_username(args.user)
    print(f"kaggle user: {user}")

    dataset_ref = f"{user}/{DATASET_SLUG}"
    if not args.skip_dataset:
        dataset_ref = push_dataset(args.images, user)
        # Dataset processing lags a few seconds behind create/version.
        time.sleep(10)

    kernel_ref = push_kernel(dataset_ref, user)
    print(f"\nkernel pushed: https://www.kaggle.com/code/{kernel_ref}")

    if args.wait:
        status = wait_for_kernel(kernel_ref)
        print(f"\nfinal status: {status}")
        if status == "complete" and args.fetch:
            fetch_output(kernel_ref, args.fetch)
        elif status != "complete":
            print("check logs: kaggle kernels output", kernel_ref)
            sys.exit(1)


if __name__ == "__main__":
    main()
