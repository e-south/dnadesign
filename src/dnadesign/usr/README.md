## USR — Universal Sequence Record

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
- CLI quickstart: `#cli-quickstart-run-from-anywhere`
- Overlay contract: `#namespace-registry-required`
- Event schema: `#event-log-schema`
- Remote sync: `docs/operations/sync.md`

---

## Start here (mental model and stack boundary)

USR is the canonical store and the mutation/event boundary:

1) Base records live in `records.parquet` (atomic, audit-friendly).
2) Derived data is written as namespaced overlays in `_derived/` (append-only parts).
3) All mutations emit events to `.events.log` (JSONL). This is the integration boundary.

When running the full stack:

- DenseGen can write sequences plus a `densegen` overlay namespace into a USR dataset.
- Notify reads USR `.events.log` and sends webhook notifications.

Relevant docs:
- DenseGen outputs plus event streams: `../densegen/docs/reference/outputs.md`
- Notify operators doc: `../../../docs/notify/usr_events.md`

## Doc map

- Quickstart CLI: `#cli-quickstart-run-from-anywhere`
- Overlay plus registry contract: `#namespace-registry-required`
- How overlays merge (conflict resolution): `#how-overlays-merge-conflict-resolution`
- Event log schema (Notify input): `#event-log-schema`
- Remote sync: `docs/operations/sync.md`

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
│    │         ├─ .events.log         # append‑only JSONL event stream
│    │         ├─ _registry/          # frozen registry snapshots (optional)
│    │         └─ _snapshots/         # rolling copies of records.parquet
│    └─ _archive/
│         └─ <namespace>/<dataset_name>/...
└─ demo_material/                      # example CSVs used in this README
```

**Dataset ids** are typically `namespace/dataset`. If you pass an unqualified name (e.g., `demo`), USR resolves it only when the match is unique; ambiguous names require the full `namespace/dataset` id.
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
| `created_at` | timestamp(us, UTC) | ingest time                    |

`sequence_norm` is `sequence.strip()` and is the value used for ID hashing. `bio_type` must not contain the `|` delimiter.

> **Contract:** exactly one `records.parquet` per dataset directory (base table). Derived overlays live in `_derived/` (either `<namespace>.parquet` or `_derived/<namespace>/part-*.parquet`) and are merged via `usr materialize`.
> **Derived columns must be namespaced** as `<tool>__<field>` (e.g., `mock__score`, `infer__llr`).

---

## How overlays merge (conflict resolution)

USR overlays are append-only parts under `_derived/<namespace>/`.
When reading (or when materializing), USR constructs an overlay view with deterministic last-writer-wins semantics across parts:

1) Overlay parts are ordered by `created_at` descending, then filename descending.
2) For each join key (`id` or `sequence`, depending on the operation), the newest value wins per column.

This makes retry and resume behavior deterministic (including DenseGen retries).

Operational implications:

- Within a single overlay part, the join key must be unique (duplicates are rejected).
- If you re-attach the same namespace and columns in a new part, the new part overrides earlier parts.
- For large runs, compact parts periodically (`usr maintenance overlay-compact ...`) to reduce read overhead while preserving semantics.

### CLI availability

In this monorepo, run the CLI as:

```bash
uv run usr --help
```

---

## CLI quickstart (run from anywhere)

Demo inputs in this repo:

* Sequences: `src/dnadesign/usr/demo_material/demo_sequences.csv`
* Attachments: `src/dnadesign/usr/demo_material/demo_attachment_{one|two}.csv`
* OPAL labels (SFXI vec8): `src/dnadesign/usr/demo_material/demo_y_sfxi.csv` (includes `intensity_log2_offset_delta`)

**macOS note:** USR suppresses PyArrow `sysctlbyname` warnings by default. Set `USR_SHOW_PYARROW_SYSCTL=1` to force showing warnings. Back-compat flag `USR_SUPPRESS_PYARROW_SYSCTL` still works and takes precedence when explicitly set (`1` suppress, `0` show).

**Demo dataset note:** `src/dnadesign/usr/datasets/demo` is tracked. If you want a scratch run, copy it first (or point `--root` to a scratch datasets folder) before running attach/materialize/snapshot.

**Subapps:** tool-specific utilities live under `usr maintenance`, `usr densegen`, `usr legacy`, `usr state`, and `usr dev` (dev commands are hidden unless `USR_SHOW_DEV_COMMANDS=1`).

**Create a dataset** (namespace is recommended)

```bash
usr init densegen/demo --source "readme quickstart" --notes "hello, world"
```

**Import sequences** (only essential USR columns are ingested; extra CSV columns are ignored)

```bash
usr import densegen/demo --from csv \
  --path src/dnadesign/usr/demo_material/demo_sequences.csv \
  --bio-type dna --alphabet dna_4
```

> Sequences must be non-empty. If you include `bio_type` or `alphabet` columns in your file, all rows must be filled; missing values are treated as errors.

**Register an overlay namespace** (required on fresh roots)

```bash
usr namespace register quickstart \
  --columns quickstart__X_value:list<float64>,quickstart__intensity_log2_offset_delta:float64
```

> The first successful namespace registration creates `registry.yaml` and includes the reserved `usr_state` namespace automatically.

**Attach namespaced metadata** (namespacing required)

```bash
usr attach densegen/demo \
  --path src/dnadesign/usr/demo_material/demo_attachment_one.csv \
  --namespace quickstart --key sequence --key-col sequence --columns X_value

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
usr materialize densegen/demo
usr materialize densegen/demo --yes --snapshot-before
```

By default, overlays are **kept**. To remove or archive overlays after materialize:

```bash
usr materialize densegen/demo --drop-overlays
usr materialize densegen/demo --archive-overlays
```

**Inspect & validate**

```bash
usr ls                        # list datasets (pretty)
usr ls --format json          # JSON output (stable)
usr info densegen/demo        # rows, columns, namespaces
usr info densegen/demo --format json
usr head densegen/demo -n 5   # first N rows (pretty by default)
usr head densegen/demo -n 5 --columns id,sequence
usr head densegen/demo -n 5 --include-deleted
usr grep densegen/demo --pattern ATG --limit 10

usr schema densegen/demo               # Arrow schema (plain)
usr schema densegen/demo --tree        # pretty tree view
usr schema densegen/demo --format json
usr events tail densegen/demo --format json --follow
usr events tail /abs/path/to/datasets/densegen/demo --format json --n 5

usr validate densegen/demo             # checks schema, uniqueness, namespacing
usr validate densegen/demo --strict
usr validate densegen/demo --registry-mode frozen
usr validate densegen/demo --registry-mode either
usr validate /abs/path/to/datasets/densegen/demo --strict


```

**Maintenance (registry + overlays)**

```bash
usr maintenance registry-freeze densegen/demo
usr maintenance overlay-compact densegen/demo --namespace densegen
```

**Delete & restore (tombstones)**

```bash
usr delete densegen/demo --id e153ebc4...
usr delete densegen/demo --id-file /tmp/ids.txt --reason "bad input"
usr restore densegen/demo --id e153ebc4...

usr state set densegen/demo --id e153ebc4... --masked
usr state set densegen/demo --id e153ebc4... --qc-status pass --split train
usr state clear densegen/demo --id e153ebc4...
```

**Column-wise summary (types, null %, list stats)**
```bash
usr describe densegen/demo --sample 2048
```

**Fetch a single record by id (pretty table)**

```bash
usr get densegen/demo --id e153ebc4... --columns id,sequence,densegen__used_tfbs
```

**Export**

```bash
usr export densegen/demo --fmt csv   --out src/dnadesign/usr/demo_material/out.csv
usr export densegen/demo --fmt csv   --columns id,sequence --out src/dnadesign/usr/demo_material/out_small.csv
usr export densegen/demo --fmt jsonl --out src/dnadesign/usr/demo_material/out.jsonl
usr export densegen/demo --fmt csv --out src/dnadesign/usr/demo_material/out_with_deleted.csv --include-deleted

# or if you're in the cwd of records.parquet
usr export --fmt csv --out records.csv
```

**Snapshots**

```bash
usr snapshot densegen/demo   # writes records-YYYYMMDDThhmmssffffff.parquet under _snapshots/
```

---

## Interactive notebook (marimo)

There is a marimo notebook for interactive exploration (filters + summaries):

```bash
uv sync --locked --group notebooks
uv run marimo edit --sandbox --watch src/dnadesign/usr/notebooks/usr_explorer.py
```

Use the widgets to choose a dataset, sample size, and panel configuration.

### Path‑first tools (work on files or directories anywhere)

These commands accept a dataset name **or** a file/directory path. When a directory contains multiple Parquet files, USR presents an interactive picker.

```bash
usr head .                       # head of a Parquet in the current directory (picker if needed)
usr cols                         # list columns for ./records.parquet (or the file you pick)
usr cell --row 0 --col sequence  # print a single cell from ./records.parquet (or the file you pick)

# Explicit file path examples
usr head permuter/run42/records.parquet
usr cols ./some/dir --glob 'events*.parquet'
```

> When you run inside `src/dnadesign/usr/datasets/<namespace>/<dataset>` (or legacy `.../datasets/<dataset>`), commands default to that dataset.

---

## De‑duplication

Each `id` must map to exactly one sequence. De‑duplicate with an explicit key:

```bash
usr maintenance dedupe <dataset> --key sequence_ci --keep keep-first  # case-insensitive (dna_4 only)
usr maintenance dedupe <dataset> --key sequence --keep keep-last      # case-preserving
usr maintenance dedupe <dataset> --dry-run             # preview
usr densegen repair --dedupe keep-first
usr densegen repair --filter-single-tf  # remove rows that only contain one type of TF among their TFBSs
usr densegen repair --drop-id-seq-only  # remove rows that only contain 'id' and 'sequence'
```

`sequence_ci` is only valid for `dna_4` datasets; other keys preserve case.

---

## Merge datasets (USR ↔ USR)

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
usr maintenance merge \
  --dest 60bp_dual_promoter_cpxR_LexA \
  --src  60bp_dual_promoter_cpxR_LexA_v2 \
  --union-columns \
  --if-duplicate skip
```

---

## Remote sync (SSH)

Built-in SSH + rsync moves dataset folders and single files. `USR_REMOTES_PATH` is required.
Use this for HPC workflows where datasets are not Git-tracked.

```bash
export USR_REMOTES_PATH="$HOME/.config/dnadesign/usr-remotes.yaml"
usr remotes wizard \
  --preset bu-scc \
  --name bu-scc \
  --user <cluster-user> \
  --host scc1.bu.edu \
  --base-dir /project/<cluster-user>/densegen_runs/outputs/usr_datasets

usr remotes doctor --remote bu-scc

# Preview, then transfer
usr diff densegen/60bp_dual_promoter_cpxR_LexA bu-scc --verify auto
usr pull densegen/60bp_dual_promoter_cpxR_LexA bu-scc -y
usr push densegen/60bp_dual_promoter_cpxR_LexA bu-scc -y
```

**Dataset directory mode** supports explicit dataset paths outside `--root`:

```bash
usr diff /path/to/outputs/usr_datasets/densegen/demo_hpc bu-scc
usr pull /path/to/outputs/usr_datasets/densegen/demo_hpc bu-scc -y
usr push /path/to/outputs/usr_datasets/densegen/demo_hpc bu-scc -y
```

**FILE mode** lets you diff/pull/push arbitrary files by path:

```bash
usr diff permuter/run42/records.parquet bu-scc
usr pull permuter/run42/records.parquet bu-scc -y
```

See **docs/operations/sync.md** for full setup, storage-location guidance, and file-mode mapping options (`repo_root`, `local_repo_root`, `--repo-root`, `--remote-path`).

---

## Python API

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
| `created_at` | `timestamp[us, UTC]`     | Ingest time (UTC) |

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

USR enforces a strict registry for overlay namespaces. Create `registry.yaml` under your datasets root (default: `src/dnadesign/usr/datasets/registry.yaml`) and register each namespace before attaching overlays. **All dataset mutations (init/import/attach/snapshot/maintenance) require a registry.**

The registry must include the reserved `usr_state` namespace with the standardized columns listed above; USR will fail fast if it is missing or modified.

Register a namespace:

```bash
usr namespace register mock \
  --columns mock__score:float64,mock__vec:list<float64> \
  --owner "your-name" \
  --description "example derived metrics"
```

List or show registered namespaces:

```bash
usr namespace list
usr namespace show mock
```

Freeze the registry into a dataset (for historic compatibility):

```bash
usr maintenance registry-freeze densegen/demo
```

USR also auto-freezes the registry on the first dataset mutation when a registry is present. Freezing writes `_registry/registry.<hash>.yaml` and stamps `usr:registry_hash` into `records.parquet`. Use `usr validate --registry-mode frozen` when you want to validate against the frozen registry rather than the repo-wide current registry.

### Design notes & contracts

* **Immutability of essentials:** `id`, `bio_type`, `sequence`, `alphabet`, `length`, `source`, `created_at` are canonical and stable.
* **Namespacing:** All derived values must be namespaced `<namespace>__<field>`. Non‑namespaced derived columns are rejected.
* **Deterministic IDs:** `id = sha1(f"{bio_type}|{sequence_norm}".encode("utf-8"))` with case preserved; `bio_type` must not contain `|`.
* **Safety:** Writes are atomic; snapshots are kept in `_snapshots/`; operations append to `.events.log`.
* **Tombstones:** Deletions are logical (`usr__deleted`, `usr__deleted_at`, `usr__deleted_reason`) and live in the reserved `usr` namespace. `usr__deleted_at` is `timestamp[us, UTC]`. Deleted rows are hidden by default unless `include_deleted=True`. Use `usr materialize --drop-deleted` to physically remove them.
* **Record state:** The reserved `usr_state` namespace defines standardized state fields: `usr_state__masked` (bool), `usr_state__qc_status` (string), `usr_state__split` (string), `usr_state__supersedes` (string), `usr_state__lineage` (list<string>). Allowed `usr_state__qc_status`: `pass`, `fail`, `warn`, `unknown`. Allowed `usr_state__split`: `train`, `val`, `test`, `holdout`. Unset values are `null`. These are registry-governed like any other namespace.
* **Update policy:** Base records are append-only. Updates are overlays only; base rewrites are maintenance operations. In the library, use `with ds.maintenance(reason=...): ds.materialize(...)` (CLI handles this).
* **Registry:** Namespaces must be registered in `registry.yaml` before attaching or materializing overlays.
* **Registry hash:** `usr:registry_hash` is persisted in base and overlay metadata; overlay validation requires the hash to match the dataset's registry (current or frozen, depending on `--registry-mode`).
* **Overlay parts:** Overlays may be stored as append-only parts under `_derived/<namespace>/part-*.parquet`; compact with `usr maintenance overlay-compact`. **Compaction guidance:** compact at run end or when parts exceed ~200 files or the overlay exceeds ~1–2GB.

### Event log schema

Each line of `.events.log` is JSONL with:

* `event_version` (integer)
* `timestamp_utc` (RFC3339 UTC string)
* `action` (string)
* `dataset` (object with `name` and `root`)
* `args` (object, key-based secret redaction applied)
* `metrics` (object; empty object if not applicable)
* `artifacts` (object; empty object if not applicable)
* `maintenance` (object; empty object if not applicable)
* `fingerprint` (object with `rows`, `cols`, `size_bytes`, and optional `sha256` when `USR_EVENT_SHA256=1`)
* `registry_hash` (string or null)
* `actor` (object with `tool`, `run_id`, `host`, `pid`)
* `version` (USR package version)

Notify expects at minimum `event_version` and `action`. See: `../../../docs/notify/usr_events.md`.


---

@e-south
