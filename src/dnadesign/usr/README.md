## USR — Universal Sequence Record

**USR** is a Parquet‑backed, single‑table store for biological sequence datasets in `dnadesign`.

- One canonical table per dataset: `records.parquet` (atomic writes with snapshots)
- Immutable **essential** columns; derived columns must be namespaced: `<tool>__<field>`
- Friendly CLI (`usr …`) with pretty (Rich) output by default, and a small Python API
- Append‑only `.events.log` and human scratchpad `meta.md`

---

### Layout

```bash
usr/
├─ src/
├─ datasets/
│    └─ <dataset_name>/
│         ├─ records.parquet     # single source of truth
│         ├─ meta.md             # human notes + command snippets
│         ├─ .events.log         # append‑only JSONL event stream
│         └─ _snapshots/         # rolling copies of records.parquet
└─ template_demo/                # example CSVs used in this README
```

---

### Core schema

| column      | type               | notes                         |
|-------------|--------------------|-------------------------------|
| `id`        | string             | sha1(`bio_type` \| `sequence`)|
| `bio_type`  | string             | `"dna"` \| `"protein"`        |
| `sequence`  | string             | case‑preserving               |
| `alphabet`  | string             | e.g., `"dna_4"`               |
| `length`    | int32              | `len(sequence)`               |
| `source`    | string             | ingest provenance             |
| `created_at`| timestamp(us, UTC) | ingest time                   |

> **Contract:** one `records.parquet` per dataset directory.  
> **Derived columns must be namespaced** as `<tool>__<field>` (e.g., `opal__score`, `infer__llr`).

---

### Install

Add a console script so you can type `usr`:

```toml
# pyproject.toml
[project.scripts]
usr = "dnadesign.usr.src.cli:main"       # CLI (argparse; full surface)
````

Editable install during development:

```bash
uv pip install -e .
```

---

### Quickstart (CLI)

We’ll use the demo CSVs in this repo:

* Sequences: `usr/demo_material/demo_sequences.csv`
* Attachments: `usr/demo_material/demo_attachment_{one|two}.csv`
* OPAL demo labels: `usr/demo_material/demo_y_sfxi.csv`

#### Create a dataset

```bash
usr init demo --source "readme quickstart" --notes "hello, world"
```

#### Import sequences

Only the essential USR columns are ingested; any extra CSV columns are ignored.

```bash
usr import demo --from csv \
  --path usr/demo_material/demo_sequences.csv \
  --bio-type dna --alphabet dna_4
```

#### Attach namespaced metadata

Attach per‑sequence metadata under your own namespace (e.g. `mock`). **Namespacing is required**.

```bash
usr attach demo \
  --path usr/demo_material/demo_attachment_one.csv \
  --namespace mock --id-col sequence --columns X_value

usr attach demo \
  --path usr/demo_material/demo_attachment_two.csv \
  --namespace mock --id-col sequence --columns y_label
```

Resulting columns (examples):

* `mock__X_value` → float (nullable)
* `mock__y_label` → list<float> (nullable)

> Re‑attaching the same columns requires `--allow-overwrite`.

#### Inspect, grep, validate

```bash
usr ls                                  # list datasets (pretty)
usr head demo                           # first N rows (pretty by default)
usr info demo                           # rows, columns, discovered namespaces
usr grep demo --pattern "ATG" --limit 10
usr cols                                # -> defaults to ./records.parquet (or you pick)
usr head -n 5                           # -> pretty head of ./records.parquet
usr cell --row 0 --col sequence         # -> from ./records.parquet

usr schema demo           # Arrow schema (plain)
usr schema demo --tree    # tree view (pretty)

usr validate demo         # checks schema, uniqueness, namespacing
usr validate demo --strict
```

#### Export

```bash
usr export demo --fmt csv   --out usr/demo_material/out.csv
usr export demo --fmt jsonl --out usr/demo_material/out.jsonl
```

#### Snapshots

```bash
usr snapshot demo   # writes records-YYYYMMDDThhmmss.parquet under _snapshots/
```

> **Note:** If you run commands inside a dataset folder, they default to `.` (current directory). If multiple Parquet files are present, USR will list them and let you pick by number.

---

### De‑duplication

In USR, each `id` must map back to exactly one `sequence`. If this is ever not the case, you can cleanup:

```bash
# Drop duplicates that differ only by case. One row per duplicate group survives.
usr dedupe-sequences <dataset> --policy keep-first      # or keep-last, or ask
usr dedupe-sequences <dataset> --policy ask             # interactive picker per group
usr dedupe-sequences <dataset> --dry-run                # shows what would change
```

---

## Merge datasets (USR ↔ USR)

There are two policies to align schemas:

* `--require-same-columns` (strict; types and names must match exactly), or
* `--union-columns` (default; missing columns are filled with NULL).

Duplicate rows by `id` can be handled via:

* `--if-duplicate {error|skip|prefer-src|prefer-dest}` (default `skip`)

Overlapping columns with different types can be safely coerced to the destination’s type with:

* `--coerce-overlap to-dest` (default) or `--coerce-overlap none`

By default, rows having the **same letters** (ignoring case) on `(bio_type, sequence)` **are not merged** from the source; override with `--no-avoid-casefold-dups`.

Example:

```bash
usr merge-datasets \
  --dest 60bp_dual_promoter_cpxR_LexA \
  --src  60bp_dual_promoter_cpxR_LexA_v2 \
  --union-columns \
  --if-duplicate skip
```

---

## Remote sync (SSH)

Use built‑in SSH + rsync to move whole dataset folders.

```bash
usr remotes add cluster --type ssh \
  --host scc1.bu.edu --user esouth \
  --base-dir /project/dunlop/esouth/dnadesign/src/dnadesign/usr/datasets

usr diff 60bp_dual_promoter_cpxR_LexA --remote cluster
usr pull 60bp_dual_promoter_cpxR_LexA --remote cluster -y
usr push 60bp_dual_promoter_cpxR_LexA --remote cluster -y
```

See **SYNC.md** for SSH key setup.

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

@e-south