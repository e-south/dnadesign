## USR — Universal Sequence Record

**USR** is a Parquet‑backed, single‑table store for biological sequence datasets inside the `dnadesign` monorepo.

- One canonical table per dataset: `records.parquet` (atomic writes with snapshots)
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
│    └─ <dataset_name>/
│         ├─ records.parquet     # single source of truth
│         ├─ meta.md             # human notes + command snippets
│         ├─ .events.log         # append‑only JSONL event stream
│         └─ _snapshots/         # rolling copies of records.parquet
└─ template_demo/                # example CSVs used in this README
````

---

### Core schema

| column       | type               | notes                          |
|--------------|--------------------|--------------------------------|
| `id`         | string             | sha1(`bio_type` \| `sequence`) |
| `bio_type`   | string             | `"dna"` \| `"protein"`         |
| `sequence`   | string             | case‑preserving                |
| `alphabet`   | string             | e.g., `"dna_4"`                |
| `length`     | int32              | `len(sequence)`                |
| `source`     | string             | ingest provenance              |
| `created_at` | timestamp(us, UTC) | ingest time                    |

> **Contract:** exactly one `records.parquet` per dataset directory.
> **Derived columns must be namespaced** as `<tool>__<field>` (e.g., `mock__score`, `infer__llr`).

---

### Install

Add a console script so you can type `usr`:

```toml
# pyproject.toml
[project.scripts]
usr = "dnadesign.usr.src.cli:main"   # argparse CLI (full surface)
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

**Create a dataset**

```bash
usr init demo --source "readme quickstart" --notes "hello, world"
```

**Import sequences** (only essential USR columns are ingested; extra CSV columns are ignored)

```bash
usr import demo --from csv \
  --path usr/demo_material/demo_sequences.csv \
  --bio-type dna --alphabet dna_4
```

**Attach namespaced metadata** (namespacing required)

```bash
usr attach demo \
  --path usr/demo_material/demo_attachment_one.csv \
  --namespace mock --id-col sequence --columns X_value

usr attach demo \
  --path usr/demo_material/demo_attachment_two.csv \
  --namespace mock --id-col sequence --columns y_label
```

Examples of resulting columns:

* `mock__X_value` → float (nullable)
* `mock__y_label` → list<float> (nullable)

> Re‑attaching the same columns requires `--allow-overwrite`.

**Inspect & validate**

```bash
usr ls                        # list datasets (pretty)
usr info demo                 # rows, columns, namespaces
usr head demo -n 5            # first N rows (pretty by default)
usr grep demo --pattern ATG --limit 10

usr schema demo               # Arrow schema (plain)
usr schema demo --tree        # pretty tree view

usr validate demo             # checks schema, uniqueness, namespacing
usr validate demo --strict


```

**Column-wise summary (types, null %, list stats)**
```bash
usr describe demo --sample 2048
```

**Fetch a single record by id (pretty table)**

```bash
usr get demo --id e153ebc4... --columns id,sequence,densegen__used_tfbs
```

**Export**

```bash
usr export demo --fmt csv   --out usr/demo_material/out.csv
usr export demo --fmt jsonl --out usr/demo_material/out.jsonl

# or if you're in the cwd of records.parquet
usr export --fmt csv --out records.csv
```

**Snapshots**

```bash
usr snapshot demo   # writes records-YYYYMMDDThhmmss.parquet under _snapshots/
```

---

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

> When you run inside `usr/datasets/<dataset>`, commands default to that dataset (no need to pass the name).

---

## De‑duplication

Each `id` must map to exactly one sequence. Clean up case‑insensitive duplicates when necessary:

```bash
usr dedupe-sequences <dataset> --policy keep-first   # or keep-last, or ask (interactive)
usr dedupe-sequences <dataset> --dry-run             # preview
usr repair-densegen --dedupe keep-first
usr repair-densegen --dedupe ask --drop-missing-used-tfbs -y
usr repair-densegen --filter-single-tf  # remove rows that only contain one type of TF among their TFBSs
usr repair-densegen --drop-id-seq-only  # remove rows that only contain 'id' and 'sequence'
```

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
usr merge-datasets \
  --dest 60bp_dual_promoter_cpxR_LexA \
  --src  60bp_dual_promoter_cpxR_LexA_v2 \
  --union-columns \
  --if-duplicate skip
```

---

## Remote sync (SSH)

Built‑in SSH + rsync moves whole dataset folders (and can also sync single files in FILE mode). See **SYNC.md** for key setup and details.

```bash
usr remotes add cluster --type ssh \
  --host scc1.bu.edu --user esouth \
  --base-dir /project/dunlop/esouth/dnadesign/src/dnadesign/usr/datasets

# Preview, then transfer
usr diff 60bp_dual_promoter_cpxR_LexA --remote cluster
usr pull 60bp_dual_promoter_cpxR_LexA --remote cluster -y
usr push 60bp_dual_promoter_cpxR_LexA --remote cluster -y
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

ds = Dataset(root, "demo_py")
ds.init(source="python quickstart")

n = ds.import_rows([{
    "sequence": "ACGTACGTAC",
    "bio_type": "dna",
    "alphabet": "dna_4",
    "source": "unit-test",
}])

ds.attach_columns(
    Path("/tmp/attach.csv"),
    namespace="mock",
    id_col="id",
    columns=["score", "vec"],
)

print(ds.head(3))
```

---

### Design notes & contracts

* **Immutability of essentials:** `id`, `bio_type`, `sequence`, `alphabet`, `length`, `source`, `created_at` are canonical and stable.
* **Namespacing:** All derived values must be namespaced `<tool>__<field>`. Non‑namespaced derived columns are rejected (except a small set of backwards‑compatibility fields, when present).
* **Deterministic IDs:** `id = sha1(bio_type | normalized(sequence))` with case preserved.
* **Safety:** Writes are atomic; snapshots are kept in `_snapshots/`; operations append to `.events.log`.


---

@e-south
