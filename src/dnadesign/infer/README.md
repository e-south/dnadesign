## infer

A small, **model-agnostic inference engine** for biological sequence models.

* **Adapters**: `evo2` (implemented), `esm2` (stub for later)
* **Ops**: logits • embeddings • native log-likelihood • generation
* **I/O**: columnar outputs; optional write-back for `.pt` list-of-dicts

> [Evo 2](https://github.com/ArcInstitute/evo2/tree/main) is provided by the Arc Institute. Install their package and prerequisites before using the `evo2` adapter.

---

### Install

```bash
# install this package (with evo2 extra)
pip install -e .[evo2]

# then install Evo 2 itself (see Evo2 README for CUDA/FlashAttention/TE setup)
pip install evo2
```

---

### Quick Usage

#### 1) Python (most flexible)

```python
from dnadesign.infer import run_extract, run_generate

# extract: logits mean + length-normalized score
out = run_extract(
  ["ACGTACGT", "GTAC"],
  model_id="evo2_7b",
  outputs=[
    {"id":"logits_mean","fn":"evo2.logits","params":{"pool":{"method":"mean","dim":1}},"format":"numpy"},
    {"id":"ll_mean","fn":"evo2.log_likelihood","params":{"method":"native","reduction":"mean"},"format":"float"},
  ],
  device="cuda:0", precision="bf16", alphabet="dna",
)

# generate: one continuation per prompt
gen = run_generate(
  ["ACGT"],
  model_id="evo2_7b",
  params={"max_new_tokens":128, "temperature":1.0, "top_k":4, "seed":7},
  device="cuda:0", precision="bf16", alphabet="dna",
)
```

#### 2) YAML + CLI (simple batch runs)

Place a `config.yaml` (example at top) next to your data, then run:

```bash
# uses ./config.yaml by default; otherwise looks for a sibling next to the module
dnadesign-infer

# or explicitly:
dnadesign-infer --config /path/to/config.yaml

# run a single job from the file:
dnadesign-infer --config config.yaml --job evo_keys
```

**Notes**

* The CLI currently supports only `ingest.source: pt_file`.
* It expects an input named `{job.id}.pt` sitting next to `config.yaml`.
* That `.pt` must be a `list[dict]` and each dict must contain `ingest.field` (e.g., `sequence`).

---

## What it returns

Everything is **columnar**: for N inputs you get length-N lists per output id.

* Example keys: `{"logits_mean": [...], "ll_mean": [...]}`
* Optional write-back (for `records`/`pt_file`) adds flat keys like
  `"<model_id>__<job_id>__<out_id>": value` into each record.

---

### Functions (Evo 2)

* `evo2.logits` — forward logits per token; optional pooling (`mean|sum|max`, `dim`)
* `evo2.embedding` — intermediate embeddings by layer name (e.g., `"blocks.28.mlp.l3"`); optional pooling
* `evo2.log_likelihood` — native `score_sequences`; `reduction: "sum" | "mean"`
* `evo2.generate` — `max_new_tokens` (+ `temperature`, `top_k`, `top_p`, `seed`) → `{"gen_seqs": List[str]}`

`format` can be `"tensor" | "numpy" | "list" | "float"` (use `"float"` only for scalar outputs).

---

### Practical tips

* Set compute via `model.device` (`cuda:0` or `cpu`) and `model.precision` (`fp32|fp16|bf16`).
  (Evo2 adapter uses CUDA autocast for fp16/bf16.)
* Mixed-length inputs are run one-by-one; equal lengths are batched automatically.
* For reproducible generation, pass `seed`.

---

### Extend it

Add an adapter under `dnadesign/infer/adapters/`, then register:

```python
from ..registry import register_fn, register_model
from .your_model import YourAdapter

register_model("your_model_id", YourAdapter)
register_fn("yourns.logits", "logits")
# ...
```

Now reference `model.id: your_model_id` and `fn: yourns.*` in code or YAML.

---

### Troubleshooting

* `Unknown model id` → register with `register_model(...)`.
* `Unknown function 'ns.name'` → register with `register_fn("ns.name","method")`.
* Alphabet errors → inputs are strict (`dna` = only A/C/G/T by default).
* CLI says only `pt_file` supported → use Python API for `sequences` / `records`.

---

