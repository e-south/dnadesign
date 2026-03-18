## Evo2 Promoter Feature Runbook

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-18

Use this runbook when you want the infer-owned part of the promoter feature flow without duplicating the USR-owned cross-tool workflow.

This page owns:

- Evo2 model lane choice inside `infer`
- feature-bundle config shape
- validate/dry-run/run commands
- OPAL-export handoff from infer outputs

This page does not own:

- multi-source promoter dataset assembly
- template realization with `construct`
- downstream OPAL round logic

Use the authoritative cross-tool workflow here:

- [Promoter characterization feature matrix](../../../usr/docs/operations/promoter-characterization-feature-matrix.md)

### Common path

1. Start from the packaged smoke workspace:
   - [promoter_evo2_smoke workspace](../../workspaces/promoter_evo2_smoke/README.md)
2. Validate the config:
   - `uv run infer validate config --config src/dnadesign/infer/workspaces/promoter_evo2_smoke/config.yaml`
3. Dry-run the jobs:
   - `uv run infer run --config src/dnadesign/infer/workspaces/promoter_evo2_smoke/config.yaml --dry-run`
4. Run the selected model lane once dependencies are installed:
   - `uv run infer run --config src/dnadesign/infer/workspaces/promoter_evo2_smoke/config.yaml`

### Config stance

- keep one infer config per model lane
- keep contexts explicit as separate jobs
- let `construct` supply template-backed resolved sequences and anchor coordinates
- let `feature_bundle` decide which feature groups and pooling modes are collected

Minimal example:

```yaml
model:
  id: evo2_7b
  device: cpu
  precision: fp32
  alphabet: dna

jobs:
  - id: anchor_only_7b_features
    operation: extract
    ingest:
      source: records
      path: inputs/anchor_only_promoters.jsonl
      field: sequence
    feature_bundle:
      intermediate_block: 26
      context:
        kind: anchor_only

  - id: template_1kb_7b_features
    operation: extract
    ingest:
      source: records
      path: inputs/template_1kb_promoters.jsonl
      field: sequence
    feature_bundle:
      intermediate_block: 26
      context:
        kind: template_1kb
```

### Choosing 7B vs 20B

- use `evo2_7b` for the first green path and local smoke runs
- switch to `evo2_20b` with a one-line config change: `model.id: evo2_20b`
- keep `intermediate_block: 26` unchanged unless repo-local benchmarks justify another lane

### What the bundle writes

For each job, the bundle writes:

- likelihood summaries
- `output_layer_mean`
- `intermediate_embedding`
- metadata columns including context ids, anchor spans, selector, schema version, and request digest

The persisted USR columns still follow the generic infer contract:

- `infer__<model_id>__<job_id>__<out_id>`

### OPAL handoff

When the downstream branch wants one flattened `X` matrix, do not explode vectors in storage. Export them explicitly:

```python
from dnadesign.infer import export_evo2_promoter_opal_matrix

payload = export_evo2_promoter_opal_matrix(
    row_ids=row_ids,
    columnar=columnar,
    model_id="evo2_7b",
    bundle={"context": {"kind": "template_1kb"}},
)
```

`payload["x"]` is the deterministic matrix and `payload["feature_names"]` is the deterministic flattened feature order.

### Failure modes

- `feature_bundle` with templated contexts fails if `construct__anchor_start` / `construct__anchor_end` are missing
- `pool.dim < 1` fails during config parsing
- `evo2_20b` on non-Hopper GPUs fails at validate/dry-run time
- stale bundle outputs are recomputed when `metadata__feature_request_digest` does not match the current request

For GPU environment setup and capacity gating, use:

- [SCC Evo2 GPU environment runbook](scc-evo2-gpu-uv-runbook.md)
