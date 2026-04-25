# Legacy promoter-focused PyTorch archive bucket

This bucket stores historical `.pt` artifact directories plus their progress and summary sidecars.

## Scope

- Dense batch, sequence batch, shuffle batch, latent-DNA batch, and similar legacy PyTorch artifacts live here.
- This bucket is not a `records.parquet` dataset root.
- Use `src/dnadesign/usr/scripts/archived_pytorch_manager.py` when you need to validate or inspect these `.pt` files.
