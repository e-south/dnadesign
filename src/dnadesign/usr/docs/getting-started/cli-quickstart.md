# USR CLI quickstart

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


Run commands from repo root with `uv run usr ...`.

## Command availability

```bash
# Show full CLI command surface.
uv run usr --help
```

## Minimal end-to-end sequence

```bash
# Use a scratch datasets root so this walkthrough does not touch tracked demo data.
ROOT="/tmp/usr_quickstart_$(date -u +%Y%m%d_%H%M%S)"
DATASET="densegen/quickstart_demo"
OUT_DIR="/tmp/usr_exports"
mkdir -p "$ROOT" "$OUT_DIR"

# 1) Register namespace contract for derived columns.
uv run usr --root "$ROOT" namespace register quickstart \
  --columns 'quickstart__X_value:list<float64>,quickstart__intensity_log2_offset_delta:float64'

# 2) Create dataset and import canonical sequence rows.
uv run usr --root "$ROOT" init "$DATASET" --source "docs quickstart"
uv run usr --root "$ROOT" import "$DATASET" --from csv \
  --path src/dnadesign/usr/demo_material/demo_sequences.csv \
  --bio-type dna --alphabet dna_4

# 3) Attach derived overlays.
uv run usr --root "$ROOT" attach "$DATASET" \
  --path src/dnadesign/usr/demo_material/demo_attachment_one.csv \
  --namespace quickstart --key sequence --key-col sequence --columns X_value
uv run usr --root "$ROOT" attach "$DATASET" \
  --path src/dnadesign/usr/demo_material/demo_y_sfxi.csv \
  --namespace quickstart --key sequence --key-col sequence \
  --columns intensity_log2_offset_delta --allow-missing

# 4) Materialize overlays into records.parquet and keep a rollback snapshot.
uv run usr --root "$ROOT" materialize "$DATASET" --yes --snapshot-before

# 5) Inspect and export portable handoff files.
uv run usr --root "$ROOT" info "$DATASET"
uv run usr --root "$ROOT" export "$DATASET" --fmt parquet --out "$OUT_DIR"
uv run usr --root "$ROOT" export "$DATASET" --fmt csv --out "$OUT_DIR"
```

`materialize` mutates canonical `records.parquet`. `export` is the handoff step for files copied elsewhere.

## Common inspection and validation commands

```bash
uv run usr ls
uv run usr info densegen/demo
uv run usr head densegen/demo -n 5
uv run usr schema densegen/demo --tree
uv run usr events tail densegen/demo --format json --n 5
uv run usr validate densegen/demo --strict
```

## Notes

- `src/dnadesign/usr/datasets/demo` is tracked; use `--root` scratch paths for disposable runs.
- macOS: set `USR_SHOW_PYARROW_SYSCTL=1` to show PyArrow sysctl warnings.

## Next steps

- Remote and iterative batch workflows: [../operations/workflow-map.md](../operations/workflow-map.md)
- Schema and overlay contracts: [../reference/README.md](../reference/README.md)
