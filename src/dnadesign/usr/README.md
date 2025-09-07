## USR — Universal Sequence Record

**USR** is the central, Parquet-backed store for biological sequence datasets used across the `dnadesign` repo.  

### Layout

```python
usr/
├─ src/
├─ datasets/
│    └─ <dataset_name>/
│         ├─ records.parquet
│         ├─ meta.yaml
│         └─ .snapshots/
└─ template_demo/        # example CSVs for the README walkthrough
````

### Schema (USR v1)

| column     | type               | notes                                     |
|------------|--------------------|-------------------------------------------|
| id         | string             | `sha1(bio_type|sequence_upper)`           |
| bio_type   | string             | `"dna"` or `"protein"`                    |
| sequence   | string             | normalized (e.g., DNA uppercased)         |
| alphabet   | string             | e.g. `"dna_4"`                            |
| length     | int32              | `len(sequence)`                           |
| source     | string             | ingest provenance                         |
| created_at | timestamp(us, UTC) | ingest time                               |

> **Contract:** one `records.parquet` per dataset directory. Everything else (derivatives, working files) lives elsewhere.

### Validation notes
- For `alphabet="dna_4"`, sequences must contain only `[ACGTacgt]` (case-insensitive check). Case is **not** coerced; lowercase may carry information and is preserved.

### Namespacing (core rule)

All **derived** columns must be prefixed `"<tool>__<field>"`, e.g.:

- `opal__score`, `opal__rank`
- `infer__llr`, `infer__model`
- `clustering__leiden`, `clustering__umap_x`

Essential columns** (table above) are immutable.

### CLI walkthrough with the included template

All commands assume you’re inside: `src/dnadesign/usr/`

#### 1) Create a dataset from the provided CSV
The repo ships a small CSV at `template_demo/template_sequences.csv`. We’ll import it into a new dataset called `mock_dataset`.

```bash
# create empty dataset directory and metadata
python -m dnadesign.usr init mock_dataset --source "template_demo import"

# import sequences (CSV has columns: sequence,bio_type,alphabet,source)
python -m dnadesign.usr import mock_dataset \
  --from csv \
  --path template_demo/template_sequences.csv \
  --bio-type dna \
  --alphabet dna_4
```

#### 2) Inspect, head, grep, validate

```bash
python -m dnadesign.usr ls
python -m dnadesign.usr info mock_dataset
python -m dnadesign.usr head mock_dataset -n 5
python -m dnadesign.usr grep mock_dataset --pattern "ATG" --limit 10
python -m dnadesign.usr validate mock_dataset --strict
```

#### 3) Attach a small, namespaced add-on

A pre-made attachment file is included at `template_demo/attach_demo.csv` with columns:

- `id` (keys that exist in the dataset),
- `tag` (string label),
- `score` (numeric).

Attach them under the demo namespace:
```bash
python -m dnadesign.usr attach mock_dataset \
  --path template_demo/attach_demo.csv \
  --namespace demo \
  --id-col id \
  --columns "tag,score" \
  --note "toy demo columns"

# verify
python -m dnadesign.usr head mock_dataset -n 5
python -m dnadesign.usr info mock_dataset
python -m dnadesign.usr validate mock_dataset --strict
```

### Attach behavior

* The input file must contain an **`id`** column; all other columns become `<namespace>__<name>` unless already prefixed.
* Overwrites are rejected unless `--allow-overwrite` is given.
* Values align by `id`. Unknown ids are ignored. Missing values become `NULL`.
* Strings that look like JSON arrays (e.g., `"[0.1, 0.2, 0.3]"`) are parsed—useful for generic vectors—without enforcing a dimension.

### Python API (equivalent)

```python
from pathlib import Path
from dnadesign.usr import Dataset

root = Path(__file__).resolve().parent / "usr"
ds = Dataset(root, "mock_dataset")

ds.init(source="template_demo import")
ds.import_csv(root / "template_demo" / "template_sequences.csv")
ds.attach(
    root / "template_demo" / "attach_demo.csv",
    namespace="demo",
    id_col="id",
    columns=["tag", "score"],
    note="toy demo columns",
)
ds.validate(strict=True)
print(ds.head(5))

```

### Optional console scripts (shortcuts)

If you add these to `pyproject.toml` and install in editable mode, you’ll get shell commands you can run from anywhere:

```toml
[project.scripts]
usr = "dnadesign.usr.src.cli:main"
USR = "dnadesign.usr.src.cli:main"
```

Both `usr` and `USR` point to the same CLI entrypoint. By default they operate on `src/dnadesign/usr/datasets/` (override with `--root`).

```bash
# list all datasets under the default root
usr ls

# peek at the first 5 rows of a dataset
USR head mock_dataset -n 5
```

### Add a **“Remote Sync”**

````md
### Remote Sync (cluster ↔ local)

Configure your HPC as a remote once:

```bash
usr remotes add cluster --type ssh \
  --host scc1.bu.edu --user esouth \
  --base-dir /project/dunlop/esouth/dnadesign/src/dnadesign/usr/datasets \
  --ssh-key-env USR_SSH_KEY
````

**Pull** results from the cluster:

```bash
# See a compact diff (sha, rows, cols). Confirm to overwrite local.
usr pull 60bp_dual_promoter_cpxR_Lex --from cluster
```

**Push** local changes up:

```bash
usr push 60bp_dual_promoter_cpxR_Lex --to cluster
```

Dry run, no changes:

```bash
usr diff 60bp_dual_promoter_cpxR_Lex --remote cluster
```

Power flags:
- `--yes` (no prompt), `--dry-run`
- `--primary-only` (just `records.parquet`)
- `--skip-snapshots` (omit `_snapshots/`)

### Extending USR

- **New tools:** write outputs as namespaced columns (e.g., `opal__*`, `infer__*`). Don’t modify essential columns.
- Vector data: store as JSON arrays in your file; attach will parse them (e.g., infer__emb).
- Snapshots as checkpoints: call snapshot after major stages so downstream analyses can reference exact versions.

@e-south