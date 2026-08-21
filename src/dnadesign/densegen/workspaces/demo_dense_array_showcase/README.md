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

### Publication playback

This workspace owns the two public, study-neutral playback tiers:

1. [`playback.yaml`](playback.yaml) teaches generic overlap packing using
   unpadded examples and one neutral binding-site treatment.
2. [`playback-constraints.yaml`](playback-constraints.yaml) teaches fixed
   anchors, their required span, and the RNAP-coupled promoter abstraction
   without introducing study-specific TF identities.

Both recipes consume persisted DenseGen records and publish ignored,
regenerable bundles under `outputs/publication/playback/`. Run either recipe
from the `dnadesign` repository root:

```bash
# Publish one configured endpoint.
uv run python -m dnadesign.densegen.src.integrations.dense_arrays playback-config.yaml
```

Replace `playback-config.yaml` with the recipe path. Pass `--replace` only when
intentionally replacing an existing generated bundle.

The public mechanics and authority language are owned by the `dense-arrays`
package. Study-specific selection, labels, and interpretation remain in the
owning research study.
