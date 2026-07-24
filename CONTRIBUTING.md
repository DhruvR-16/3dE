# Contributing to Photo → Editable 3D Scene

## Local Development Setup

### Backend
```bash
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
.venv/bin/pytest backend/tests -q
```

### Frontend
```bash
cd frontend
npm install
npm run dev  # Vite dev server on :5173
```

## Architecture Overview

See [photo-to-editable-3d-scene-architecture.md](photo-to-editable-3d-scene-architecture.md) for detailed design docs.

### Key Concepts
- **3DGS Representation**: Gaussian splats as the core scene format
- **Object Segmentation**: Per-Gaussian `object_id` attribute from SAM2 majority voting
- **Transform Math**: Per-object pivot points and local coordinate frames
- **Export**: Server re-merges edited objects with client transforms baked in

## Testing

Backend unit tests validate:
- PLY I/O and roundtrip correctness
- Quaternion and transform math
- Scene split/export logic
- Security (path traversal rejection)

```bash
.venv/bin/python -m pytest backend/tests -q
```

Frontend testing is manual (open http://localhost:5173):
1. Load sample scene (6 objects, synthetic)
2. Select objects (click or sidebar)
3. Move/rotate/scale with transform gizmo
4. Delete/restore objects
5. Export merged PLY
6. Import custom scene via sidebar

## Real-Photo GPU Pipeline

See [pipeline/kaggle/photo_to_editable_3d.ipynb](pipeline/kaggle/photo_to_editable_3d.ipynb).

To test locally:
```bash
python pipeline/kaggle/run_on_kaggle.py --images <dir> --wait --fetch output/
```

## Code Style

- **Backend**: PEP 8, type hints, docstrings for non-obvious logic
- **Frontend**: ESLint (Airbnb), meaningful variable names, minimal comments
- Prefer simplicity over cleverness; avoid premature abstraction

## Reporting Issues

If you find a bug or limitation:
1. Check the [Known Limitations](README.md#known-limitations-v1-by-design) section
2. Check [Kaggle Troubleshooting](pipeline/kaggle/photo_to_editable_3d.ipynb#troubleshooting)
3. Open an issue with: steps to reproduce, expected behavior, actual behavior

## Future Work (v2)

See [Roadmap → v2](README.md#roadmap--v2) in the main README.

