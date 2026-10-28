# SeqLedger CLI quickstart

**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28


Run commands from repo root with `uv run seqledger ...`.

## Command availability

```bash
# Show full CLI command surface.
uv run seqledger --help
```

## Minimal end-to-end sequence

```bash
# Use a scratch datasets root so this walkthrough does not touch tracked demo data.
ROOT="/tmp/usr_quickstart_$(date -u +%Y%m%d_%H%M%S)"
DATASET="quickstart_demo"
OUT_DIR="/tmp/usr_exports"
mkdir -p "$ROOT" "$OUT_DIR"

# 1) Register namespace contract for derived columns.
uv run seqledger --root "$ROOT" namespace register quickstart \
  --columns 'quickstart__X_value:list<float64>,quickstart__intensity_log2_offset_delta:float64'

# 2) Create dataset and import canonical sequence rows.
uv run seqledger --root "$ROOT" init "$DATASET" --source "docs quickstart"
uv run seqledger --root "$ROOT" import "$DATASET" --from csv \
  --path src/dnadesign/seqledger/assets/demo_material/demo_sequences.csv \
  --bio-type dna --alphabet dna_4

# 3) Attach derived overlays.
uv run seqledger --root "$ROOT" attach "$DATASET" \
  --path src/dnadesign/seqledger/assets/demo_material/demo_attachment_one.csv \
  --namespace quickstart --key sequence --key-col sequence --columns X_value
uv run seqledger --root "$ROOT" attach "$DATASET" \
  --path src/dnadesign/seqledger/assets/demo_material/demo_y_sfxi.csv \
  --namespace quickstart --key sequence --key-col sequence \
  --columns intensity_log2_offset_delta --allow-missing

# 4) Materialize overlays into records.parquet and keep a rollback snapshot.
uv run seqledger --root "$ROOT" materialize "$DATASET" --yes --snapshot-before

# 5) Inspect and export portable handoff files.
uv run seqledger --root "$ROOT" info "$DATASET"
uv run seqledger --root "$ROOT" export "$DATASET" --fmt parquet --out "$OUT_DIR"
uv run seqledger --root "$ROOT" export "$DATASET" --fmt csv --out "$OUT_DIR"
```

`materialize` mutates canonical `records.parquet`. `export` is the handoff step for files copied elsewhere.

## Check the imported data

Continue with the same scratch dataset created above:

```bash
uv run seqledger --root "$ROOT" ls
uv run seqledger --root "$ROOT" info "$DATASET"
uv run seqledger --root "$ROOT" head "$DATASET" -n 5
uv run seqledger --root "$ROOT" schema "$DATASET" --tree
uv run seqledger --root "$ROOT" events tail "$DATASET" --format json --n 5
uv run seqledger --root "$ROOT" validate "$DATASET" --strict
```

## Notes

- Developer-only mock generators are hidden from the default CLI surface. If you need them for local fixture work, make that explicit: `SEQLEDGER_SHOW_DEV_COMMANDS=1 uv run seqledger dev make-mock --help`.
- macOS: set `SEQLEDGER_SHOW_PYARROW_SYSCTL=1` to show PyArrow sysctl warnings.

## Next steps

- Remote and iterative batch workflows: [SeqLedger workflow map](../operations/routes/workflow-map.md)
- Schema and overlay contracts: [../reference/README.md](../reference/README.md)
