## USR — Universal Sequence Record

**USR** is the Parquet-backed home for biological sequence datasets in `dnadesign`.

### Layout

```text
usr/
├─ src/
├─ datasets/
│    └─ <dataset_name>/
│         ├─ records.parquet   # data lives here
│         ├─ .events.log       # action history (append-only)
│         └─ _snapshots/       # rolling copies of records.parquet
└─ template_demo/              # example CSVs used in this README
````

### Core schema (USR v1)

| column      | type               | notes                  |                   |
| ----------- | ------------------ | ---------------------- | ----------------- |
| id          | string             | sha1(\`bio\_type       | sequence\_norm\`) |
| bio_type   | string             | `"dna"` or `"protein"` |                   |
| sequence    | string             | case-preserving        |                   |
| alphabet    | string             | e.g. `"dna_4"`         |                   |
| length      | int32              | `len(sequence)`        |                   |
| source      | string             | ingest provenance      |                   |
| created\_at | timestamp(us, UTC) | ingest time            |                   |

> **Contract:** one `records.parquet` per dataset directory. All derived columns must be **namespaced** as `<tool>__<field>` (e.g., `opal__score`, `infer__llr`).

---

## Quickstart: Template demo

Goal: generate a mock dataset then attach new columns using USR's CLI commands.

### 0) (Optional) Set up the console script

```toml
# pyproject.toml
[project.scripts]
usr = "dnadesign.usr.src.cli:main"
USR = "dnadesign.usr.src.cli:main"
```

### 1) Make a mock dataset **from the template sequences**

This guarantees IDs match `attach_demo.csv`.

```bash
# creates ./usr/datasets/demo_template/records.parquet
usr make-mock demo_template \
  --from-csv template_demo/template_sequences.csv \
  --seed 7 \
  --namespace demo \
  --x-dim 512 --y-dim 8
```

What you get per row:

* essential columns: `id`, `bio_type`, `sequence`, `alphabet`, ...
* `demo__x_representation` → list<float32>\[512]
* `demo__label_vec8` → list<float32>\[8] 

### 2) Attach CSV annotations (namespaced)

`template_demo/attach_demo.csv` contains: `id`, `tag`, `score`.

```bash
usr attach demo_template \
  --path template_demo/attach_demo.csv \
  --namespace demo \
  --id-col id \
  --columns "tag,score" \
  --note "template demo attach"
```

Rules:

* `id` aligns rows; unknown `id`s are ignored; missing values → `NULL`.
* Columns become `demo__tag` and `demo__score` (unless already namespaced).
* Overwrites are blocked unless you pass `--allow-overwrite`.

### 3) Inspect & validate

```bash
usr head demo_template -n 5
usr info demo_template
usr validate demo_template --strict
```

---

## Reference: Common commands

```bash
# List datasets under the default datasets/ root
usr ls

# Show first 5 rows
usr head <dataset> -n 5

# Grep sequences
usr grep <dataset> --pattern "ATG" --limit 10

# Export to CSV/JSONL
usr export <dataset> --fmt csv  --out /tmp/out.csv
usr export <dataset> --fmt jsonl --out /tmp/out.jsonl

# Snapshots (rolls a copy into _snapshots/)
usr snapshot <dataset>
```

### Importing sequences explicitly (CSV/JSONL)

```bash
# Create dataset shell + import sequences
usr init my_dataset --source "template import"
usr import my_dataset --from csv \
  --path template_demo/template_sequences.csv \
  --bio-type dna --alphabet dna_4
```

---

## Remote sync (cluster ↔ local)

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

See **SYNC.md** for SSH key setup and options (`--primary-only`, `--skip-snapshots`, etc.).

---

### Python API example

```python
from pathlib import Path
from dnadesign.usr import Dataset

root = Path(__file__).resolve().parent / "usr" / "datasets"
ds = Dataset(root, "demo_template")

print(ds.info())
ds.validate(strict=True)
print(ds.head(5))
```

@e-south