# Canonical USR dataset archive root

This directory is the sanctioned location for archived USR datasets and archive buckets.

## Rules

- Place archived datasets here instead of introducing alternate archive roots.
- Archived datasets remain outside the default live dataset-id namespace under `src/dnadesign/usr/datasets/`.
- Use an explicit dataset path here, or set `--root` to this directory when you need to inspect archived material.
- `promoter_misc_pytorch/` is the legacy promoter-focused PyTorch archive bucket; it is not a `records.parquet` dataset root.
- Use `src/dnadesign/usr/scripts/archived_pytorch_manager.py` for `.pt` inspection and metadata updates inside that bucket.
