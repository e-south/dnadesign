## USR — Universal Sequence Record

**USR** is the central, Parquet-backed store for biological sequence datasets used across the `dnadesign` repo.  

### Layout

```python
usr/
├─ src/
├─ datasets/
│    └─ <dataset_name>/
│         ├─ records.parquet
│         └─ _snapshots/
└─ template_demo/        # example CSVs for the README walkthrough
````

### Schema (USR v1)

| column     | type               | notes                                     |
|------------|--------------------|-------------------------------------------|
| id         | string             | `sha1(bio_type|sequence_upper)`           |
| bio_type   | string             | `"dna"` or `"protein"`                    |
| sequence   | string             | case-insensitive                          |
| alphabet   | string             | e.g. `"dna_4"`                            |
| length     | int32              | `len(sequence)`                           |
| source     | string             | ingest provenance                         |
| created_at | timestamp(us, UTC) | ingest time                               |

> **Contract:** one `records.parquet` per dataset directory. Everything else (derivatives, working files) lives elsewhere.

### Namespacing (core rule)

All **secondary** columns must be prefixed `"<tool>__<field>"`, e.g.:

- `opal__score`, `opal__rank`
- `infer__llr`, `infer__model`
- `clustering__leiden`, `clustering__umap_x`

Essential columns (table above) are immutable.

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

If you add these to `pyproject.toml` and install `dnadesign` in editable mode, you’ll get shell commands you can run from anywhere:

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

### Remote Sync (cluster ↔ local)

`usr` includes a lightweight, SSH-backed sync for dataset folders you **don’t** want in git (e.g., large `records.parquet`, `_snapshots/`, `.events.log` files).


**Rule of thumb:** run the command **on the machine where you want the files to end up.** In practice, that means you usually run from your **laptop**.

- datasets live as folders (containing `records.parquet`, `_snapshots`, etc.).
- `usr pull <dataset> --from cluster` copies that folder down (safe overwrite).
- `usr push` sends your local folder back up.
- `usr` uses ssh under the hood, so authentication choices are:
  - Password prompts
  - SSH keys

For more information on how to set this up, see the sibling [**SYNC.md**](SYNC.md) documentation.

#### 1) Configure once

You can add a remote via CLI **or** by creating `usr/remotes.yaml`.

**CLI:**

```bash
# Define the cluster remote once
usr remotes add cluster --type ssh \
  --host scc1.bu.edu --user esouth \
  --base-dir /project/dunlop/esouth/dnadesign/src/dnadesign/usr/datasets

# (Optional) Inspect
usr remotes list
usr remotes show cluster
```

#### 2) When working locally

Bring results **down** from the cluster, then push updates **up** later.

```bash
# See a compact diff (sha/rows/cols) before any transfer
usr diff 60bp_dual_promoter_cpxR_Lex --remote cluster

# Pull cluster → local (prompt to confirm if different; add -y to skip prompt)
usr pull 60bp_dual_promoter_cpxR_Lex --from cluster

# Commit code changes as usual (datasets are not in git)

# Push local → cluster (same safety prompt)
usr push 60bp_dual_promoter_cpxR_Lex --to cluster
```

#### 3) From the **cluster** (only if needed)

If you’re logged into the cluster and want data to **end up on your laptop**, the simplest path is to **exit to your laptop and run `usr pull … --from cluster`**.

#### What gets transferred

By default: the **entire dataset folder**:

```
<dataset>/
  records.parquet      # primary
  meta.yaml
  .events.log
  _snapshots/
```

#### Safety

On every `pull`/`push`, `usr` shows a one-screen diff (sha/rows/cols) and will **ask before overwriting** unless you pass `-y`. If nothing changed, it prints “Already up to date.”







### Extending USR

- **New tools:** write outputs as namespaced columns (e.g., `opal__*`, `infer__*`). Don’t modify essential columns.
- Vector data: store as JSON arrays in your file; attach will parse them (e.g., infer__emb).
- Snapshots as checkpoints: call snapshot after major stages so downstream analyses can reference exact versions.

@e-south