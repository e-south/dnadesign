# USR CLI quickstart

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-24


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
DATASET="densegen_quickstart_demo"
OUT_DIR="/tmp/usr_exports"
mkdir -p "$ROOT" "$OUT_DIR"

# 1) Register namespace contract for derived columns.
uv run usr --root "$ROOT" namespace register quickstart \
  --columns 'quickstart__X_value:list<float64>,quickstart__intensity_log2_offset_delta:float64'

# 2) Create dataset and import canonical sequence rows.
uv run usr --root "$ROOT" init "$DATASET" --source "docs quickstart"
uv run usr --root "$ROOT" import "$DATASET" --from csv \
  --path src/dnadesign/usr/assets/demo_material/demo_sequences.csv \
  --bio-type dna --alphabet dna_4

# 3) Attach derived overlays.
uv run usr --root "$ROOT" attach "$DATASET" \
  --path src/dnadesign/usr/assets/demo_material/demo_attachment_one.csv \
  --namespace quickstart --key sequence --key-col sequence --columns X_value
uv run usr --root "$ROOT" attach "$DATASET" \
  --path src/dnadesign/usr/assets/demo_material/demo_y_sfxi.csv \
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

## Packaged demo smoke test

The shared `usr_demo_cli_examples` dataset is the read-only didactic fixture for CLI inspection. It exercises three registry-backed shapes:

- `mock__X_value`: vector-valued CSV attachment parsed as `list<float64>`.
- `mock__y_label`: vector-valued label attachment parsed as `list<float64>`.
- `opal__demo__*`: materialized OPAL demo campaign columns.

Use strict validation as the smoke test for the packaged fixture:

```bash
uv run usr validate usr_demo_cli_examples --strict
uv run usr schema usr_demo_cli_examples --tree
uv run usr head usr_demo_cli_examples -n 5
```

## Common inspection and validation commands

```bash
uv run usr ls
uv run usr info usr_demo_cli_examples
uv run usr head usr_demo_cli_examples -n 5
uv run usr schema usr_demo_cli_examples --tree
uv run usr events tail usr_demo_cli_examples --format json --n 5
uv run usr validate usr_demo_cli_examples --strict
uv run usr validate usr_demo_cli_examples --registry-mode namespace-current
```

## Notes

- `src/dnadesign/usr/datasets/usr_demo_cli_examples` is the packaged shared demo; use `--root` scratch paths for disposable runs.
- Developer-only mock generators are hidden from the default CLI surface. If you need them for local fixture work, make that explicit: `USR_SHOW_DEV_COMMANDS=1 uv run usr dev make-mock --help`.
- macOS: set `USR_SHOW_PYARROW_SYSCTL=1` to show PyArrow sysctl warnings.

## Next steps

- Remote and iterative batch workflows: [USR workflow map](../operations/routes/workflow-map.md)
- Schema and overlay contracts: [../reference/README.md](../reference/README.md)
