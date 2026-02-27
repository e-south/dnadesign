![infer banner](assets/infer-banner.svg)

**infer** runs biological sequence models and writes results back to datasets.

See the [repository docs index](../../../docs/README.md) for workflow routes and operational references.

- **Adapters:** `evo2` (implemented), `esm2` (stub)
- **Ops (evo2):** `evo2.logits`, `evo2.embedding`, `evo2.log_likelihood`, `evo2.generate`
- **Ingest sources:** sequences | records | pt_file | usr
- **Outputs:** columnar dict; with USR attach, columns are:

```bash
infer__<model_id>__<job_id>__<out_id>
```

USR write-back contract:
- Infer writes USR outputs in chunk-sized attaches for resumable long runs.
- Resume scan requires a readable USR `records.parquet`; unreadable tables fail fast.

> Evo2 is from the Arc Institute: install [`evo2`](https://github.com/ArcInstitute/evo2) before use.

---

### Install

Add to `pyproject.toml` (already present in this repo):

```toml
[project.scripts]
infer = "dnadesign.infer.cli:app"
````

You can still run as a module:

```bash
python -m dnadesign.infer  # dispatches to the Typer app
```

---

### Command line presets

Presets are small YAML files shipped in `dnadesign.infer.presets`. They capture common jobs so you don’t have to remember flags.

List & inspect:

```bash
infer presets list
infer presets show evo2/extract_logits_ll
````

Run from a preset (extracting multiple outputs):

```bash
# From a USR dataset
infer run --preset evo2/extract_logits_ll --usr my_dataset --field sequence --device cuda:0 --precision bf16 --write-back

# Or ad-hoc (no YAML config), reading sequences from a file:
infer extract --preset evo2/extract_logits_ll --seq-file ./seqs.txt --device cpu --dry-run
```

> The `evo2/extract_logits_ll` preset collects:
>
> * `logits_mean` (mean‑pooled over dim=1)
> * `ll_mean` (reduction=mean)
> * `ll_sum` (reduction=sum)

### Python API

Run an in‑memory batch:

```python
from dnadesign.infer import run_extract

out = run_extract(
  ["ACGTACGT", "GTAC"],
  model_id="evo2_7b",
  outputs=[
    {"id": "logits_mean", "fn": "evo2.logits",
     "params": {"pool": {"method": "mean", "dim": 1}}, "format": "numpy"},
    {"id": "ll_mean", "fn": "evo2.log_likelihood",
     "params": {"method": "native", "reduction": "mean"}, "format": "float"},
  ],
  device="cuda:0", precision="bf16", alphabet="dna",
)
```

---

### CLI Quick Reference

#### `infer run` — run jobs from YAML

```bash
infer run --config ./config.yaml --job myjob
infer run --dry-run
infer run --device cuda:0 --precision bf16 --batch-size 128 --overwrite
```

#### `infer extract` — one output ad‑hoc (no YAML)

```bash
# From USR (attaches results if --write-back)
infer extract --model-id evo2_7b --device cuda:0 --precision bf16 \
  --fn evo2.log_likelihood --format float \
  --usr my_dataset --field sequence --write-back

# From sequences on stdin (file)
infer extract --model-id evo2_7b --fn evo2.logits --format list \
  --seq-file ./seqs.txt --pool-method mean --pool-dim 1
```

#### `infer generate` — prompt continuation

```bash
infer generate --model-id evo2_7b --device cuda:0 --precision bf16 \
  --prompt ACGTACGT --max-new-tokens 64 --temperature 0.8 --out gen.txt
```

#### `infer adapters` — introspection & cache

```bash
infer adapters list
infer adapters fns
infer adapters cache-clear
```

#### `infer validate` — config & USR checks

```bash
infer validate config --config ./config.yaml
infer validate usr --dataset my_dataset --field sequence
```

---

### Environment variables

* `DNADESIGN_INFER_BATCH` — fallback micro‑batch size
* `DNADESIGN_PROGRESS` — disable/enable progress (`0`/`1`)
* `DNADESIGN_USR_ROOT` — default USR datasets root
* `INFER_LOG_LEVEL` — default CLI log level
* `INFER_ALLOW_PICKLE` — allow `.pt` ingestion without `--i-know-this-is-pickle`
* `INFER_AUTO_DERATE_OOM` — `1` (default) to auto‑reduce batch on OOM

---

### Extending Presets

Add YAMLs under `dnadesign/infer/presets/your_ns/your_preset.yaml`:

```yaml
id: yourns/extract_example
kind: extract
model: { id: evo2_7b, precision: bf16, alphabet: dna }
outputs: [ ... ]
```

They will automatically show up in `infer presets list`.

---

## 🧱 Minimal structural tidy‑up

- **Presentation/UI** (console helpers) now clearly separated in `_console.py`.
- **Presets** live in `presets/` with a small **registry**; this is where we’ll grow preset coverage.
- The **core** (engine/config/adapters/ingest/writers) is left in place to avoid invasive churn, but the new structure makes it easy to split into `core/` and `ui/` in a future minor release without breaking imports. The CLI and Python API retain the same public signatures.

---

### Extending to new models

Add an adapter `dnadesign/infer/adapters/your_model.py` and register:

```python
from dnadesign.infer.registry import register_model, register_fn
from .your_model import YourAdapter

register_model("yourns_foo", YourAdapter)
register_fn("yourns.logits", "logits")
register_fn("yourns.embedding", "embedding")
register_fn("yourns.log_likelihood", "log_likelihood")
register_fn("yourns.generate", "generate")
```

Use `model.id="yourns_foo"` and `fn: yourns.*` in YAML/CLI.

---
