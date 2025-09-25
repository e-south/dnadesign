## USR — Universal Sequence Record

**USR** is the Parquet-backed home for biological sequence datasets in `dnadesign`. It gives you a single source of truth (`records.parquet`) with:

- immutable essential columns for sequences (id, bio_type, sequence, …)
- namespaced derived columns (`<tool>__<field>`) you can attach later
- atomic writes with snapshots + an append-only `.events.log`
- a friendly CLI (`usr …`) and a small Python API

---

### Layout at a glance

```text
usr/
├─ src/
├─ datasets/
│    └─ <dataset_name>/
│         ├─ records.parquet   # data lives here (single canonical table)
│         ├─ meta.md           # human-friendly scratch pad for notes/commands
│         ├─ .events.log       # append-only action history (JSON lines)
│         └─ _snapshots/       # rolling copies of records.parquet
└─ template_demo/              # example CSVs used in this README
````

---

### Core schema (USR)

| column      | type               | notes                  | 
| ----------- | ------------------ | ---------------------- | 
| id          | string             | sha1(`bio_type`)        | 
| bio\_type   | string             | `"dna"` \| `"protein"` | 
| sequence    | string             | case-preserving        | 
| alphabet    | string             | e.g. `"dna_4"`         | 
| length      | int32              | `len(sequence)`        | 
| source      | string             | ingest provenance      | 
| created\_at | timestamp(us, UTC) | ingest time            | 

> **Contract:** one `records.parquet` per dataset directory.
> **All derived columns must be namespaced** as `<tool>__<field>` (e.g. `opal__score`, `infer__llr`).

---

### Installation

Add a console script so you can type `usr` in a shell:

```toml
# pyproject.toml
[project.scripts]
usr = "dnadesign.usr.src.cli:main"
USR = "dnadesign.usr.src.cli:main"  # optional alias
```

Editable install during development:

```bash
pip install -e .
```

---

##  Usage demo

**Initialize a dataset, import new rows, attach columns, and inspect.**

We will use some ready-made demo CSVs:

* Sequences: `usr/demo_material/demo_sequences.csv`
* Attachments/annotations: `usr/demo_material/demo_attachments.csv`

#### Create a dataset

```bash
usr init toy --source "readme quickstart" --notes "hello, world"
```

#### Import sequences to an existing dataset

```bach
usr import toy --from csv \
  --path src/dnadesign/usr/demo_material/demo_sequences.csv \
  --bio-type dna --alphabet dna_4
```

#### Attach namespaced columns from the demo CSV

The demo attachments CSV includes per-sequence annotations. Attach them under a user-defined namespace (example: `mock`).
(Adjust `--columns` to the columns you want to bring in.)

```bash
usr attach toy \
  --path src/dnadesign/usr/demo_material/demo_attachments.csv \
  --namespace mock \
  --id-col sequence \
  --columns "tag,label" \
  --note "demo attach"
```

Resulting columns in `records.parquet` (examples):

* `mock__tag`        → string (nullable)
* `mock__label` → list<float> (nullable)

Unknown keys are ignored; missing values become `NULL`.
Re-attaching the same columns requires `--allow-overwrite`.

#### Inspect, grep, validate

```bash
usr ls                                    # list datasets under usr/datasets
usr head toy -n 5                         # peek at rows
usr info toy                              # rows, columns, discovered namespaces
usr grep toy --pattern "ATG" --limit 10   # regex over sequences
usr validate toy --strict                 # enforce namespacing & alphabet strictly
usr schema toy                            # dtypes per column
```

#### Export to CSV or JSONL

```bash
usr export toy --fmt csv   --out usr/demo_material/out.csv
usr export toy --fmt jsonl --out usr/demo_material/out.jsonl
```

#### Snapshots (roll a new copy into _snapshots/)
You can stash a current copy of `records.parquet` in a sibling `_snapshots/` directory.
```bash
usr snapshot toy
```

#### Merging datasets

There are **two** merge utilities. Choose the one that matches your source.

**1) USR ↔ USR dataset merge (folder → folder)**

Use when you have two USR datasets (each with a `records.parquet`) and you want to combine rows.

```bash
usr merge-datasets \
  --dest 60bp_dual_promoter_cpxR_LexA \
  --src  60bp_dual_promoter_cpxR_LexA_v2 \
  --union-columns \
  --if-duplicate skip \
  --dry-run
```

Options you may care about:

* `--require-same-columns` (strict) or `--union-columns`
* `--if-duplicate {error|skip|prefer-src|prefer-dest}` (default `skip`)
* `--columns <csv>` to restrict the schema (essentials are always included)
* `--dry-run` to preview, `-y/--yes` to skip prompts

Writes snapshots and logs to `.events.log`.

**2) Convert legacy `.pt` → new USR dataset**

Use this when your source is one or more archived PyTorch `.pt` files (each a `list[dict]`).

```bash
# 1) Convert: make a brand-new dataset from one or more .pt files
usr convert-legacy 60bp_dual_promoter_cpxR_LexA_from_archive \
  --paths usr/archived/densebatch_deg2tfbs_pipeline_tfbsfetcher_lexA_and_cpxR_n10000/densegenbatch_lexA_and_cpxR_n10000.pt
```

#### Attaching from Parquet / CSV / JSONL

* Input must include an `id` or `sequence` column.
* All other columns are attached (or the subset you pass with `--columns`).
* Namespacing is enforced; columns become `<namespace>__<name>` unless they are already namespaced.
* Strings that look like JSON arrays (e.g. `"[1.0, 2.0]"`) are parsed into lists.

---

## Python API

```python
from pathlib import Path
from dnadesign.usr import Dataset

root = Path(__file__).resolve().parent / "usr" / "datasets"

# 1) init
ds = Dataset(root, "toy_py")
ds.init(source="python quickstart")

# 2) import one row
n = ds.import_rows([{
    "sequence": "ACGTACGTAC",
    "bio_type": "dna",
    "alphabet": "dna_4",
    "source": "unit-test",
}])
print("imported", n)

# 3) attach a float and a vector
import pandas as pd, json, hashlib
rid = hashlib.sha1(b"dna|ACGTACGTAC").hexdigest()
attach_df = pd.DataFrame([{
    "id": rid,
    "score": 0.73,
    "vec": [0.1, 0.2, 0.3, 0.4],   # can also be a JSON string like "[0.1,0.2,0.3,0.4]"
}])
# write temp CSV (any of parquet/csv/jsonl is fine)
p = Path("/tmp/attach_py.csv"); attach_df.to_csv(p, index=False)
ds.attach_columns(p, namespace="mock", id_col="id", columns=["score", "vec"])
print(ds.head(3))
```

---

### Remote sync (cluster ↔ local)

Use the built-in SSH-backed sync to move whole dataset folders (not git-LFS):

```bash
# Configure once
usr remotes add cluster --type ssh \
  --host scc1.bu.edu --user esouth \
  --base-dir /project/dunlop/esouth/dnadesign/src/dnadesign/usr/datasets

# Pull from cluster → local (safe overwrite after showing a diff)
usr diff 60bp_dual_promoter_cpxR_LexA --remote cluster
usr pull 60bp_dual_promoter_cpxR_LexA --remote cluster -y

# Push local → cluster
usr push 60bp_dual_promoter_cpxR_LexA --remote cluster -y
```

See **SYNC.md** for SSH key setup and more details.

---

### Python API: validation & sanity checks

```python
from dnadesign.usr import Dataset
ds = Dataset(root, "demo_template")
print(ds.info())         # shows row count, column names, and discovered namespaces
ds.validate(strict=True) # enforce alphabet + namespacing
print(ds.head(5))
```

@e-south
