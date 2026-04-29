## demo_dense_array_showcase workspace

Run from this directory:

```bash
# Start a clean generation pass (default mode if omitted).
./runbook.sh --mode fresh
# Continue generation without wiping prior outputs.
./runbook.sh --mode resume
# Rebuild plots/notebook/video from existing outputs only.
./runbook.sh --mode analysis
```

This local demo uses toy TFBS inputs, CBC, fixed-length 100 bp sequences, and a parquet-only output sink. It is meant to showcase dense array packing under three constraint regimes: no fixed anchors, one fixed anchor pair, and two fixed anchor pairs.

- Runbook: [runbook.md](runbook.md)
- Config: [config.yaml](config.yaml)
- All workspaces: [../README.md](../README.md)
