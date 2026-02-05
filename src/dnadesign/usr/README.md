## USR — Universal Sequence Record

**USR** is a Parquet‑backed, single‑table store for biological sequence datasets inside the `dnadesign` monorepo.

- One canonical **base** table per dataset: `records.parquet` (atomic writes with snapshots)
- Derived overlays live in `_derived/` and can be merged into the base via `usr materialize`
- Immutable **essential** columns; derived columns are **namespaced**: `<tool>__<field>`
- Friendly CLI (`usr …`) with pretty (Rich) output, plus a minimal Python API
- Append‑only `.events.log` and human scratchpad `meta.md`
- Works from anywhere in your `dnadesign` repo: pass a dataset name *or* a file path. Many commands support interactive picking when run in a directory.

---

### Layout

````
usr/
├─ src/
├─ datasets/
│    ├─ <namespace>/
│    │    └─ <dataset_name>/
│    │         ├─ records.parquet     # canonical base table
│    │         ├─ _derived/           # derived overlays (namespace.parquet)
│    │         ├─ meta.md             # human notes + command snippets
│    │         ├─ .events.log         # append‑only JSONL event stream
│    │         └─ _snapshots/         # rolling copies of records.parquet
│    └─ <dataset_name>/               # legacy (still supported)
└─ demo_material/                      # example CSVs used in this README
````

**Dataset ids** are typically `namespace/dataset`. If you pass an unqualified name (e.g., `demo`), USR resolves it only when the match is unique; ambiguous names require the full `namespace/dataset` id.

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

> **Contract:** exactly one `records.parquet` per dataset directory (base table). Derived overlays live in `_derived/` and are merged via `usr materialize`.
> **Derived columns must be namespaced** as `<tool>__<field>` (e.g., `mock__score`, `infer__llr`).

---

### Install

Add a console script so you can type `usr`:

```toml
# pyproject.toml
[project.scripts]
usr = "dnadesign.usr.src.cli:main"   # Typer CLI (single entrypoint)
````

Editable install during development:

```bash
uv pip install -e .
```

---

## CLI quickstart (run from anywhere)

Demo inputs in this repo:

* Sequences: `usr/demo_material/demo_sequences.csv`
* Attachments: `usr/demo_material/demo_attachment_{one|two}.csv`
* OPAL labels (SFXI vec8): `usr/demo_material/demo_y_sfxi.csv` (includes `intensity_log2_offset_delta`)

**macOS note:** USR suppresses PyArrow `sysctlbyname` warnings by default. Set `USR_SHOW_PYARROW_SYSCTL=1` to re-enable. `USR_SUPPRESS_PYARROW_SYSCTL=1` is still supported.

**Demo dataset note:** `usr/datasets/demo` is tracked. If you want a scratch run, copy it first (or point `--root` to a scratch datasets folder) before running attach/materialize/snapshot.

**Subapps:** tool-specific utilities live under `usr maintenance`, `usr densegen`, `usr legacy`, and `usr dev` (dev commands are hidden unless `USR_SHOW_DEV_COMMANDS=1`).

**Create a dataset** (namespace is recommended)

```bash
usr init densegen/demo --source "readme quickstart" --notes "hello, world"
```

**Import sequences** (only essential USR columns are ingested; extra CSV columns are ignored)

```bash
usr import densegen/demo --from csv \
  --path usr/demo_material/demo_sequences.csv \
  --bio-type dna --alphabet dna_4
```

> Sequences must be non-empty. If you include `bio_type` or `alphabet` columns in your file, all rows must be filled; missing values are treated as errors.

**Attach namespaced metadata** (namespacing required)

```bash
usr attach densegen/demo \
  --path usr/demo_material/demo_attachment_one.csv \
  --namespace mock --key sequence --key-col sequence --columns X_value

usr attach densegen/demo \
  --path usr/demo_material/demo_attachment_two.csv \
  --namespace mock --key sequence --key-col sequence --columns y_label
```

Examples of resulting columns:

* `mock__X_value` → float (nullable)
* `mock__y_label` → list<float> (nullable)

> Re‑attaching the same columns requires `--allow-overwrite`.
> By default, unmatched ids/sequences raise an error; use `--allow-missing` to skip unmatched rows.
> JSON‑like strings are parsed by default; pass `--no-parse-json` to keep raw strings.
> Attachment files must have unique ids (or sequences); duplicates are rejected.
> For large parquet attachments, use `--backend duckdb` (parquet only) and pass `--no-parse-json`.
> **Registry is required:** register the namespace in `registry.yaml` before attaching.

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

usr validate densegen/demo             # checks schema, uniqueness, namespacing
usr validate densegen/demo --strict


```

**Delete & restore (tombstones)**

```bash
usr delete densegen/demo --id e153ebc4...
usr delete densegen/demo --id-file /tmp/ids.txt --reason "bad input"
usr restore densegen/demo --id e153ebc4...
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
usr export densegen/demo --fmt csv   --out usr/demo_material/out.csv
usr export densegen/demo --fmt csv   --columns id,sequence --out usr/demo_material/out_small.csv
usr export densegen/demo --fmt jsonl --out usr/demo_material/out.jsonl
usr export densegen/demo --fmt csv --out usr/demo_material/out_with_deleted.csv --include-deleted

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

> When you run inside `usr/datasets/<namespace>/<dataset>` (or legacy `usr/datasets/<dataset>`), commands default to that dataset.

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

Built‑in SSH + rsync moves whole dataset folders (and can also sync single files in FILE mode). See **SYNC.md** for key setup and details. `USR_REMOTES_PATH` is required (no repo-local fallback).

```bash
export USR_REMOTES_PATH="$HOME/.config/dnadesign/usr-remotes.yaml"
usr remotes add cluster --type ssh \
  --host scc1.bu.edu --user esouth \
  --base-dir /project/dunlop/esouth/dnadesign/src/dnadesign/usr/datasets

# Preview, then transfer
usr diff densegen/60bp_dual_promoter_cpxR_LexA --remote cluster --verify auto
usr pull densegen/60bp_dual_promoter_cpxR_LexA --remote cluster -y
usr push densegen/60bp_dual_promoter_cpxR_LexA --remote cluster -y
```

**FILE mode** lets you diff/pull/push arbitrary files by path:

```bash
usr diff permuter/run42/records.parquet --remote cluster
usr pull permuter/run42/records.parquet --remote cluster -y
```

See **SYNC.md** for `repo_root`, `local_repo_root`, `--repo-root`, and `--remote-path` mapping options.

---

## Python API

```python
from pathlib import Path
from dnadesign.usr import Dataset

root = Path(__file__).resolve().parent / "usr" / "datasets"

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

### Namespace registry (required)

USR enforces a strict registry for overlay namespaces. Create `registry.yaml` under your datasets root (default: `src/dnadesign/usr/datasets/registry.yaml`) and register each namespace before attaching overlays.

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

### Design notes & contracts

* **Immutability of essentials:** `id`, `bio_type`, `sequence`, `alphabet`, `length`, `source`, `created_at` are canonical and stable.
* **Namespacing:** All derived values must be namespaced `<namespace>__<field>`. Non‑namespaced derived columns are rejected.
* **Deterministic IDs:** `id = sha1(f"{bio_type}|{sequence_norm}".encode("utf-8"))` with case preserved; `bio_type` must not contain `|`.
* **Safety:** Writes are atomic; snapshots are kept in `_snapshots/`; operations append to `.events.log`.
* **Tombstones:** Deletions are logical (`usr__deleted`, `usr__deleted_at`, `usr__deleted_reason`) and live in the reserved `usr` namespace. `usr__deleted_at` is `timestamp[us, UTC]`. Deleted rows are hidden by default unless `include_deleted=True`. Use `usr materialize --drop-deleted` to physically remove them.
* **Update policy:** Base records are append-only. Updates are overlays only; base rewrites are maintenance operations. In the library, `Dataset.materialize(..., maintenance=True)` is required (CLI handles this).
* **Registry:** Namespaces must be registered in `registry.yaml` before attaching or materializing overlays.

### Event log schema

Each line of `.events.log` is JSONL with:

* `timestamp_utc` (RFC3339 UTC string)
* `action` (string)
* `dataset` (string)
* `args` (object, redacted of secrets)
* `fingerprint` (object with `rows`, `cols`, `size_bytes`, and optional `sha256` when `USR_EVENT_SHA256=1`)
* `version` (USR package version)


---

@e-south
