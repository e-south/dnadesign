![Universal Sequence Record banner](assets/usr-banner.svg)

## Contents
- [At a glance](#at-a-glance)
- [Start here (mental model and stack boundary)](#start-here-mental-model-and-stack-boundary)
- [Maintainer code map](#maintainer-code-map)
- [Command-line quickstart (run from anywhere)](#command-line-quickstart-run-from-anywhere)
- [Remote synchronization (secure shell)](#remote-synchronization-secure-shell)
- [Python application programming interface](#python-application-programming-interface)
- [Schema contract](#schema-contract)
- [Namespace registry (required)](#namespace-registry-required)
- [Event log schema](#event-log-schema)

## At a glance

**Intent:** Canonical, auditable sequence storage with explicit overlay contracts and mutation events.

**When to use:**
- Store canonical sequence datasets (generated and curated).
- Add derived metrics as append-only overlays.
- Materialize overlays into base records with reproducible semantics.
- Emit mutation events for external operators (Notify).

**When not to use:**
- Not a sequence generator (use DenseGen or other upstream tools).
- Not a webhook sender or alerting transport (use Notify).

**Boundary / contracts:**
- Base table is `records.parquet`; overlays live under `_derived/`.
- Derived columns must be `<namespace>__<field>` and namespace-registered.
- `.events.log` is the integration boundary consumed by Notify.
- Commands can target a dataset id or explicit file/path modes.

**Start here:**
- command-line quickstart: `#command-line-quickstart-run-from-anywhere`
- Overlay contract: `#namespace-registry-required`
- Event schema: `#event-log-schema`
- Remote sync: `docs/operations/sync.md`

---

## Start here (mental model and stack boundary)

Universal Sequence Record is the canonical store and the mutation and event boundary:

1) Base records live in `records.parquet` (atomic, audit-friendly).
2) Derived data is written as namespaced overlays in `_derived/` (append-only parts).
3) All mutations emit events to `.events.log` (newline-delimited JSON). This is the integration boundary.

When running the full stack:

- DenseGen can write sequences plus a `densegen` overlay namespace into a Universal Sequence Record dataset.
- Notify reads Universal Sequence Record `.events.log` and sends webhook notifications.
- DenseGen runtime telemetry (`outputs/meta/events.jsonl`) is not Notify input.

Relevant docs:
- DenseGen outputs plus event streams: `../densegen/docs/reference/outputs.md`
- Notify operators doc: `../../../docs/notify/usr-events.md`

## Doc map

- Quickstart command line: `#command-line-quickstart-run-from-anywhere`
- Overlay plus registry contract: `#namespace-registry-required`
- How overlays merge (conflict resolution): `#how-overlays-merge-conflict-resolution`
- Event log schema (Notify input): `#event-log-schema`
- Remote sync: `docs/operations/sync.md`
- Remote sync runbook uses progressive disclosure: quick path -> advanced path -> failure diagnosis.

## Maintainer code map

Core dataset orchestration:
- `src/dnadesign/usr/src/dataset.py`: public dataset methods and lifecycle entrypoints.
- `src/dnadesign/usr/src/dataset_activity.py`: metadata notes and event recording helpers.
- `src/dnadesign/usr/src/dataset_materialize.py`: overlay materialization engine.
- `src/dnadesign/usr/src/dataset_overlay_ops.py`: attach and overlay write operations.
- `src/dnadesign/usr/src/dataset_registry_modes.py`: registry-mode normalization and validation.
- `src/dnadesign/usr/src/dataset_dedupe.py`: dedupe execution flow.

Command-line decomposition:
- `src/dnadesign/usr/src/cli.py`: Typer command surface and command wiring.
- `src/dnadesign/usr/src/cli_commands/read.py`: read/info/schema handlers.
- `src/dnadesign/usr/src/cli_commands/write.py`: init/import/attach handlers.
- `src/dnadesign/usr/src/cli_commands/state.py`: delete/restore/state handlers.
- `src/dnadesign/usr/src/cli_commands/sync.py`: diff/pull/push handlers.
- `src/dnadesign/usr/src/cli_commands/remotes.py`: remotes list/show/add/wizard/doctor handlers.

---

### Layout

```text
src/dnadesign/usr/
├─ src/
├─ datasets/
│    ├─ <namespace>/
│    │    └─ <dataset_name>/
│    │         ├─ records.parquet     # base table
│    │         ├─ _derived/           # derived overlays (namespace.parquet or namespace/part-*.parquet)
│    │         ├─ meta.md             # notes + command snippets
│    │         ├─ .events.log         # append-only newline-delimited JSON event stream
│    │         ├─ _registry/          # frozen registry snapshots (optional)
│    │         └─ _snapshots/         # rolling copies of records.parquet
│    └─ _archive/
│         └─ <namespace>/<dataset_name>/...
└─ demo_material/                      # example CSVs used in this README
```

**Dataset ids** are typically `namespace/dataset`. If you pass an unqualified name (for example, `demo`), Universal Sequence Record resolves it only when the match is unique; ambiguous names require the full `namespace/dataset` id.
Legacy dataset ids under `archived/` are rejected with a hard error.
Path-first commands also hard-error on legacy archive locations (`datasets/archived/**` and `usr/archived/**`).

---

### Core schema

| column       | type               | notes                          |
|--------------|--------------------|--------------------------------|
| `id`         | string             | sha1(UTF‑8 `bio_type\|sequence_norm`) |
| `bio_type`   | string             | `"dna"` \| `"rna"` \| `"protein"` |
| `sequence`   | string             | case‑preserving (trimmed to `sequence_norm`) |
| `alphabet`   | string             | `dna_4`, `dna_5`, `rna_4`, `rna_5`, `protein_20`, `protein_21` |
| `length`     | int32              | `len(sequence_norm)`           |
| `source`     | string             | ingest provenance              |
| `created_at` | timestamp(us, coordinated universal time) | ingest time                    |

`sequence_norm` is `sequence.strip()` and is the value used for ID hashing. `bio_type` must not contain the `|` delimiter.

> **Contract:** exactly one `records.parquet` per dataset directory (base table). Derived overlays live in `_derived/` (either `<namespace>.parquet` or `_derived/<namespace>/part-*.parquet`) and are merged via `usr materialize`.
> **Derived columns must be namespaced** as `<tool>__<field>` (e.g., `mock__score`, `infer__llr`).

---

## How overlays merge (conflict resolution)

Universal Sequence Record overlays are append-only parts under `_derived/<namespace>/`.
When reading (or when materializing), Universal Sequence Record constructs an overlay view with deterministic last-writer-wins semantics across parts:

1) Overlay parts are ordered by `created_at` descending, then filename descending.
2) For each join key (`id` or `sequence`, depending on the operation), the newest value wins per column.

This makes retry and resume behavior deterministic (including DenseGen retries).

Operational implications:

- Within a single overlay part, the join key must be unique (duplicates are rejected).
- If you re-attach the same namespace and columns in a new part, the new part overrides earlier parts.
- For large runs, compact parts periodically (`usr maintenance overlay-compact ...`) to reduce read overhead while preserving semantics.

### Command-line interface availability

In this monorepo, run the command-line interface as:

```bash
# Show full command-line interface command surface.
uv run usr --help
```

Command snippets below use `usr ...` for readability; when running from this repo, prefix them with `uv run` (for example, `uv run usr ls`).

---

## Command-line quickstart (run from anywhere)

Demo inputs in this repo:

* Sequences: `src/dnadesign/usr/demo_material/demo_sequences.csv`
* Attachments: `src/dnadesign/usr/demo_material/demo_attachment_{one|two}.csv`
* OPAL labels (SFXI vec8): `src/dnadesign/usr/demo_material/demo_y_sfxi.csv` (includes `intensity_log2_offset_delta`)

### Minimal end-to-end sequence (copy/paste)

Use this when you want one linear path from empty root -> materialized dataset -> portable export.

```bash
# Use a scratch datasets root so this walkthrough does not touch tracked demo data.
ROOT="/tmp/usr_quickstart_$(date -u +%Y%m%d_%H%M%S)"
DATASET="densegen/quickstart_demo"
OUT_DIR="/tmp/usr_exports"
mkdir -p "$ROOT" "$OUT_DIR"

# 1) Register a namespace contract for derived columns (one-time per root).
usr --root "$ROOT" namespace register quickstart \
  --columns 'quickstart__X_value:list<float64>,quickstart__intensity_log2_offset_delta:float64'

# 2) Create base dataset and import canonical sequence rows.
usr --root "$ROOT" init "$DATASET" --source "readme quickstart"
usr --root "$ROOT" import "$DATASET" --from csv \
  --path src/dnadesign/usr/demo_material/demo_sequences.csv \
  --bio-type dna --alphabet dna_4

# 3) Attach derived overlays.
usr --root "$ROOT" attach "$DATASET" \
  --path src/dnadesign/usr/demo_material/demo_attachment_one.csv \
  --namespace quickstart --key sequence --key-col sequence --columns X_value
usr --root "$ROOT" attach "$DATASET" \
  --path src/dnadesign/usr/demo_material/demo_y_sfxi.csv \
  --namespace quickstart --key sequence --key-col sequence --columns intensity_log2_offset_delta --allow-missing

# 4) Materialize overlays into records.parquet and keep a rollback snapshot.
usr --root "$ROOT" materialize "$DATASET" --yes --snapshot-before

# 5) Inspect canonical dataset path and export portable handoff files.
usr --root "$ROOT" info "$DATASET"
usr --root "$ROOT" export "$DATASET" --fmt parquet --out "$OUT_DIR"
usr --root "$ROOT" export "$DATASET" --fmt csv --out "$OUT_DIR"
```

`usr materialize` rewrites canonical `records.parquet`; `usr export` is the handoff step for files you copy elsewhere.

**macOS note:** Universal Sequence Record suppresses PyArrow `sysctlbyname` warnings by default. Set `USR_SHOW_PYARROW_SYSCTL=1` to force showing warnings. Backward-compatible flag `USR_SUPPRESS_PYARROW_SYSCTL` still works and takes precedence when explicitly set (`1` suppress, `0` show).

**Demo dataset note:** `src/dnadesign/usr/datasets/demo` is tracked. If you want a scratch run, copy it first (or point `--root` to a scratch datasets folder) before running attach/materialize/snapshot.

**Subapps:** tool-specific utilities live under `usr maintenance`, `usr densegen`, `usr legacy`, `usr state`, and `usr dev` (dev commands are hidden unless `USR_SHOW_DEV_COMMANDS=1`).

**Register an overlay namespace** (required on fresh roots)

```bash
# Register namespace and allowed derived columns.
usr namespace register quickstart \
  --columns 'quickstart__X_value:list<float64>,quickstart__intensity_log2_offset_delta:float64'
```

> The first successful namespace registration creates `registry.yaml` and includes the reserved `usr_state` namespace automatically.

**Create a dataset** (namespace is recommended)

```bash
# Create dataset directory and base records file.
usr init densegen/demo --source "readme quickstart" --notes "hello, world"
```

**Import sequences** (only essential Universal Sequence Record columns are ingested; extra comma-separated value columns are ignored)

```bash
# Import sequence rows into base records.
usr import densegen/demo --from csv \
  --path src/dnadesign/usr/demo_material/demo_sequences.csv \
  --bio-type dna --alphabet dna_4
```

> Sequences must be non-empty. If you include `bio_type` or `alphabet` columns in your file, all rows must be filled; missing values are treated as errors.

**Attach namespaced metadata** (namespacing required)

```bash
# Attach first derived field set by sequence key.
usr attach densegen/demo \
  --path src/dnadesign/usr/demo_material/demo_attachment_one.csv \
  --namespace quickstart --key sequence --key-col sequence --columns X_value

# Attach second derived field set (allow unmatched rows).
usr attach densegen/demo \
  --path src/dnadesign/usr/demo_material/demo_y_sfxi.csv \
  --namespace quickstart --key sequence --key-col sequence --columns intensity_log2_offset_delta --allow-missing
```

Examples of resulting columns:

* `quickstart__X_value` → list<float64> (nullable)
* `quickstart__intensity_log2_offset_delta` → float64 (nullable)

> Re‑attaching the same columns requires `--allow-overwrite`.
> By default, unmatched ids/sequences raise an error; use `--allow-missing` to skip unmatched rows.
> JSON‑like strings are parsed by default; pass `--no-parse-json` to keep raw strings.
> Attachment files must have unique ids (or sequences); duplicates are rejected.
> For large parquet attachments, use `--backend duckdb` (parquet only) and pass `--no-parse-json`.
> **Registry is required:** register the namespace in `registry.yaml` before attaching.
> `demo_attachment_two.csv` has duplicate `sequence` rows by design (`tag` variants), so it is not a valid direct input for `--key sequence` without pre-aggregation.

**Materialize overlays** (merge derived columns into `records.parquet`)

```bash
# Materialize overlays into base table.
usr materialize densegen/demo

# Materialize with confirmation bypass and pre-write snapshot.
usr materialize densegen/demo --yes --snapshot-before
```

By default, overlays are **kept**. To remove or archive overlays after materialize:

```bash
# Materialize and then drop overlay parts.
usr materialize densegen/demo --drop-overlays

# Materialize and archive overlay parts.
usr materialize densegen/demo --archive-overlays
```

**Inspect & validate**

```bash
# List datasets in human-friendly table.
usr ls

# List datasets as JavaScript Object Notation.
usr ls --format json

# Show dataset summary (rows, columns, namespaces).
usr info densegen/demo
usr info densegen/demo --format json

# Preview first rows.
usr head densegen/demo -n 5
usr head densegen/demo -n 5 --columns id,sequence
usr head densegen/demo -n 5 --include-deleted
usr grep densegen/demo --pattern ATG --limit 10

# Show schema (plain and tree forms).
usr schema densegen/demo
usr schema densegen/demo --tree
usr schema densegen/demo --format json

# Tail mutation events.
usr events tail densegen/demo --format json --follow
usr events tail /abs/path/to/datasets/densegen/demo --format json --n 5

# Validate schema and registry contracts.
usr validate densegen/demo
usr validate densegen/demo --strict
usr validate densegen/demo --registry-mode frozen
usr validate densegen/demo --registry-mode either
usr validate /abs/path/to/datasets/densegen/demo --strict
```

**Maintenance (registry + overlays)**

```bash
# Freeze active registry into dataset artifact.
usr maintenance registry-freeze densegen/demo

# Compact overlay parts for one namespace.
usr maintenance overlay-compact densegen/demo --namespace densegen
```

**Delete & restore (tombstones)**

```bash
# Tombstone one record.
usr delete densegen/demo --id e153ebc4...

# Tombstone many records by id list.
usr delete densegen/demo --id-file /tmp/ids.txt --reason "bad input"

# Restore a tombstoned record.
usr restore densegen/demo --id e153ebc4...

# Set state fields.
usr state set densegen/demo --id e153ebc4... --masked
usr state set densegen/demo --id e153ebc4... --qc-status pass --split train

# Clear all reserved state fields.
usr state clear densegen/demo --id e153ebc4...
```

**Column-wise summary (types, null %, list stats)**
```bash
# Summarize columns and null rates from a sample.
usr describe densegen/demo --sample 2048
```

**Fetch a single record by id (pretty table)**

```bash
# Fetch one record by id and selected columns.
usr get densegen/demo --id e153ebc4... --columns id,sequence,densegen__used_tfbs
```

**Export**

```bash
# Export full dataset to comma-separated values.
usr export densegen/demo --fmt csv   --out src/dnadesign/usr/demo_material/out.csv

# Export selected columns.
usr export densegen/demo --fmt csv   --columns id,sequence --out src/dnadesign/usr/demo_material/out_small.csv

# Export newline-delimited JSON.
usr export densegen/demo --fmt jsonl --out src/dnadesign/usr/demo_material/out.jsonl

# Export parquet (preserves Arrow schema/types).
usr export densegen/demo --fmt parquet --out src/dnadesign/usr/demo_material/out.parquet

# Export into an existing directory (auto-name from dataset id + format).
usr export densegen/demo --fmt parquet --out /tmp/usr_exports

# Export including tombstoned rows.
usr export densegen/demo --fmt csv --out src/dnadesign/usr/demo_material/out_with_deleted.csv --include-deleted

# Dataset argument may be a dataset id or an absolute dataset directory path.
usr export /abs/path/to/datasets/densegen/demo --fmt parquet --out /tmp/usr_exports

# or if you're in the cwd of records.parquet
usr export --fmt csv --out records.csv
```

**Snapshots**

```bash
# Write timestamped snapshot under _snapshots/.
usr snapshot densegen/demo   # writes records-YYYYMMDDThhmmssffffff.parquet under _snapshots/
```

---

## Interactive notebook (marimo)

There is a marimo notebook for interactive exploration (filters + summaries):

```bash
# Install project deps (includes marimo).
uv sync --locked

# Open marimo explorer notebook.
uv run marimo edit --sandbox --watch src/dnadesign/usr/notebooks/usr_explorer.py
```

Use the widgets to choose a dataset, sample size, and panel configuration.

### Path‑first tools (work on files or directories anywhere)

These commands accept a dataset name **or** a file/directory path. When a directory contains multiple Parquet files, Universal Sequence Record presents an interactive picker.

```bash
# Preview current directory parquet (interactive picker if needed).
usr head .

# List columns for selected file.
usr cols

# Print one cell.
usr cell --row 0 --col sequence

# Explicit file path examples
usr head permuter/run42/records.parquet
usr cols ./some/dir --glob 'events*.parquet'
```

> When you run inside `src/dnadesign/usr/datasets/<namespace>/<dataset>` (or legacy `.../datasets/<dataset>`), commands default to that dataset.

---

## De‑duplication

Each `id` must map to exactly one sequence. De‑duplicate with an explicit key:

```bash
# Case-insensitive dedupe (dna_4 only).
usr maintenance dedupe <dataset> --key sequence_ci --keep keep-first

# Case-preserving dedupe.
usr maintenance dedupe <dataset> --key sequence --keep keep-last

# Preview dedupe impact.
usr maintenance dedupe <dataset> --dry-run

# DenseGen-focused cleanup helpers.
usr densegen repair --dedupe keep-first
usr densegen repair --filter-single-tf
usr densegen repair --drop-id-seq-only
```

`sequence_ci` is only valid for `dna_4` datasets; other keys preserve case.

---

## Merge datasets (Universal Sequence Record to Universal Sequence Record)

Align columns and control duplicates while merging rows from a source dataset into a destination dataset.

* Column alignment:

  * `--require-same-columns` (strict; names & types must match), or
  * `--union-columns` (default; missing columns are filled with NULLs)
* Duplicates by `id`:

  * `--if-duplicate {error|skip|prefer-src|prefer-dest}` (default `skip`)
* Overlapping column type coercion:

  * `--coerce-overlap to-dest` (default) or `--coerce-overlap none`
* By default, rows with the **same letters** (ignoring case) on `(bio_type, sequence)` **are not merged** from the source. Override with `--no-avoid-casefold-dups`.

**Example**

```bash
# Merge source dataset into destination dataset.
usr maintenance merge \
  --dest 60bp_dual_promoter_cpxR_LexA \
  --src  60bp_dual_promoter_cpxR_LexA_v2 \
  --union-columns \
  --if-duplicate skip
```

---

## Remote synchronization (secure shell)

Built-in secure shell plus `rsync` moves dataset folders and single files. `USR_REMOTES_PATH` is required.
Use this for shared cluster workflows where datasets are not Git-tracked.

```bash
# Point Universal Sequence Record tooling at remote config file.
export USR_REMOTES_PATH="$HOME/.config/dnadesign/usr-remotes.yaml"

# Create Boston University Shared Computing Cluster remote profile.
usr remotes wizard \
  --preset bu-scc \
  --name bu-scc \
  --user <cluster-user> \
  --host scc1.bu.edu \
  --base-dir /project/<cluster-user>/dnadesign/src/dnadesign/usr/workspaces/<workspace>/outputs/usr_datasets

# Validate remote connectivity and config.
usr remotes doctor --remote bu-scc

# Preview, then transfer
usr diff densegen/60bp_dual_promoter_cpxR_LexA bu-scc --verify auto
usr pull densegen/60bp_dual_promoter_cpxR_LexA bu-scc -y
usr push densegen/60bp_dual_promoter_cpxR_LexA bu-scc -y

# Strict full-fidelity transfer check for sidecars.
usr pull densegen/60bp_dual_promoter_cpxR_LexA bu-scc -y --verify-sidecars
```

Dataset pull/push transfers acquire the shared remote `.usr.lock` and print a post-action sync audit summary.

**Dataset directory mode** supports explicit dataset paths outside `--root`:

```bash
# Diff dataset directory path outside --root.
usr diff /path/to/src/dnadesign/usr/workspaces/<workspace>/outputs/usr_datasets/densegen/demo_hpc bu-scc

# Pull dataset directory by path.
usr pull /path/to/src/dnadesign/usr/workspaces/<workspace>/outputs/usr_datasets/densegen/demo_hpc bu-scc -y

# Push dataset directory by path.
usr push /path/to/src/dnadesign/usr/workspaces/<workspace>/outputs/usr_datasets/densegen/demo_hpc bu-scc -y
```

**File mode** lets you diff, pull, and push arbitrary files by path:

```bash
# Diff/pull one file by path.
usr diff permuter/run42/records.parquet bu-scc
usr pull permuter/run42/records.parquet bu-scc -y
```

See **docs/operations/sync.md** for full setup, storage-location guidance, and file-mode mapping options (`repo_root`, `local_repo_root`, `--repo-root`, `--remote-path`).

---

## Python application programming interface

Mutation methods require a registry at the dataset root. Bootstrap one first:

```bash
uv run usr --root src/dnadesign/usr/datasets namespace register mock \
  --columns 'mock__score:float64'
```

```python
from pathlib import Path
from dnadesign.usr import Dataset

# Run from repo root.
root = Path("src/dnadesign/usr/datasets").resolve()

ds = Dataset.open(root, "densegen/demo_py")
ds.init(source="python quickstart")

result = ds.add_sequences(
    [{"sequence": "ACGTACGTAC"}],
    bio_type="dna",
    alphabet="dna_4",
    source="unit-test",
)
print(result.added)

overlay_df = ds.head(1, include_deleted=True)[["id"]].assign(mock__score=1.0)
ds.write_overlay("mock", overlay_df, key="id")

print(ds.head(3))
```

---

### Schema contract

Required columns (non‑null):

| column     | type                       | description |
|------------|----------------------------|-------------|
| `id`       | `string`                   | `sha1(UTF-8 bio_type\|sequence_norm)` |
| `bio_type` | `string`                   | One of `dna`, `rna`, `protein` |
| `sequence` | `string`                   | Raw sequence (case preserved) |
| `alphabet` | `string`                   | `dna_4`, `dna_5`, `rna_4`, `rna_5`, `protein_20`, `protein_21` |
| `length`   | `int32`                    | `len(sequence_norm)` |
| `source`   | `string`                   | Source label or file |
| `created_at` | `timestamp[us, coordinated universal time]`     | Ingest time (coordinated universal time) |

`sequence_norm` is `sequence.strip()` and is the value used for ID hashing. IDs are computed with UTF‑8 encoding and the `|` delimiter; `bio_type` must not contain `|`.

Derived columns:

* Must be namespaced as `<namespace>__<field>` and **must not** overlap essential columns.
* Namespace regex: `^[a-z][a-z0-9_]*$`
* Reserved namespaces: `usr`
* Collision policy: error unless `--allow-overwrite` is explicitly provided.

Base table metadata (Parquet key/value):

* `usr:schema_version`
* `usr:dataset_created_at`
* `usr:id_hash`
* `usr:registry_hash`

### Namespace registry (required)

Universal Sequence Record enforces a strict registry for overlay namespaces. Create `registry.yaml` under your datasets root (default: `src/dnadesign/usr/datasets/registry.yaml`) and register each namespace before attaching overlays. **All dataset mutations (init/import/attach/snapshot/maintenance) require a registry.**

The registry must include the reserved `usr_state` namespace with the standardized columns listed above; Universal Sequence Record will fail fast if it is missing or modified.

Register a namespace:

```bash
# Register namespace and derived columns.
usr namespace register mock \
  --columns 'mock__score:float64,mock__vec:list<float64>' \
  --owner "your-name" \
  --description "example derived metrics"
```

List or show registered namespaces:

```bash
# List namespaces.
usr namespace list

# Inspect one namespace.
usr namespace show mock
```

Freeze the registry into a dataset (for historic compatibility):

```bash
# Freeze current registry into dataset.
usr maintenance registry-freeze densegen/demo
```

Universal Sequence Record also auto-freezes the registry on the first dataset mutation when a registry is present. Freezing writes `_registry/registry.<hash>.yaml` and stamps `usr:registry_hash` into `records.parquet`. Use `usr validate --registry-mode frozen` when you want to validate against the frozen registry rather than the repository-wide current registry.

### Design notes & contracts

* **Immutability of essentials:** `id`, `bio_type`, `sequence`, `alphabet`, `length`, `source`, `created_at` are canonical and stable.
* **Namespacing:** All derived values must be namespaced `<namespace>__<field>`. Non‑namespaced derived columns are rejected.
* **Deterministic IDs:** `id = sha1(f"{bio_type}|{sequence_norm}".encode("utf-8"))` with case preserved; `bio_type` must not contain `|`.
* **Safety:** Writes are atomic; snapshots are kept in `_snapshots/`; operations append to `.events.log`.
* **Tombstones:** Deletions are logical (`usr__deleted`, `usr__deleted_at`, `usr__deleted_reason`) and live in the reserved `usr` namespace. `usr__deleted_at` is `timestamp[us, coordinated universal time]`. Deleted rows are hidden by default unless `include_deleted=True`. Use `usr materialize --drop-deleted` to physically remove them.
* **Record state:** The reserved `usr_state` namespace defines standardized state fields: `usr_state__masked` (bool), `usr_state__qc_status` (string), `usr_state__split` (string), `usr_state__supersedes` (string), `usr_state__lineage` (list<string>). Allowed `usr_state__qc_status`: `pass`, `fail`, `warn`, `unknown`. Allowed `usr_state__split`: `train`, `val`, `test`, `holdout`. Unset values are `null`. These are registry-governed like any other namespace.
* **Update policy:** Base records are append-only. Updates are overlays only; base rewrites are maintenance operations. In the library, use `with ds.maintenance(reason=...): ds.materialize(...)` (the command-line interface handles this).
* **Registry:** Namespaces must be registered in `registry.yaml` before attaching or materializing overlays.
* **Registry hash:** `usr:registry_hash` is persisted in base and overlay metadata; overlay validation requires the hash to match the dataset's registry (current or frozen, depending on `--registry-mode`).
* **Overlay parts:** Overlays may be stored as append-only parts under `_derived/<namespace>/part-*.parquet`; compact with `usr maintenance overlay-compact`. **Compaction guidance:** compact at run end or when parts exceed ~200 files or the overlay exceeds ~1–2GB.

### Event log schema

Each line of `.events.log` is newline-delimited JSON with:

* `event_version` (integer)
* `timestamp_utc` (RFC 3339 Coordinated Universal Time string)
* `action` (string)
* `dataset` (object with `name` and `root`)
* `args` (object, key-based secret redaction applied)
* `metrics` (object; empty object if not applicable)
* `artifacts` (object; empty object if not applicable)
* `maintenance` (object; empty object if not applicable)
* `fingerprint` (object with `rows`, `cols`, `size_bytes`, and optional `sha256` when `USR_EVENT_SHA256=1`)
* `registry_hash` (string or null)
* `actor` (object with `tool`, `run_id`, `host`, `pid`)
* `version` (Universal Sequence Record package version)

Notify expects at minimum `event_version` and `action`. See: `../../../docs/notify/usr-events.md`.


---

@e-south
