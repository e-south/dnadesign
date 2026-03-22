## study_stress_ethanol_cipro workspace

This workspace writes its USR sink directly to the shared USR root
`src/dnadesign/usr/datasets/`. Treat that shared root as the DenseGen producer
surface for this study and as the cross-tool handoff source for downstream
status.

Run from this directory:

```bash
# Start a clean generation pass (default mode if omitted).
./runbook.sh --mode fresh
# Continue generation without wiping prior outputs.
./runbook.sh --mode resume
# Rebuild plots/notebook from existing outputs only.
./runbook.sh --mode analysis
```

- Runbook: [runbook.md](runbook.md)
- Config: [config.yaml](config.yaml)
- All workspaces: [../README.md](../README.md)
