## Evo2 Sequence-Feature Runbook

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-18

Use this runbook when you want the infer-owned part of an Evo2 sequence-feature flow.
The examples use anchor-only and template-context DNA records because those are
the current promoter-study dogfood surfaces, but the contract is not
promoter-specific.

This page owns:

- Evo2 model lane choice inside `infer`
- feature-bundle config shape
- validate/dry-run/run commands
- OPAL-export handoff from infer outputs

This page does not own:

- multi-source study dataset assembly
- template realization with `construct`
- downstream OPAL round logic

Use the current cross-tool promoter-study workflow only when the task is about
that specific checked-in study:

- [Promoter characterization feature matrix](../../../usr/docs/operations/promoter-characterization-feature-matrix.md)

### Common path

1. Start from the packaged smoke workspace:
   - [evo2_feature_bundle_smoke workspace](../../workspaces/evo2_feature_bundle_smoke/README.md)
2. Validate the config:
   - `uv run infer validate config --config src/dnadesign/infer/workspaces/evo2_feature_bundle_smoke/config.yaml`
3. Dry-run the jobs:
   - `uv run infer run --config src/dnadesign/infer/workspaces/evo2_feature_bundle_smoke/config.yaml --dry-run`
4. Run the selected model lane once dependencies are installed:
   - `uv run infer run --config src/dnadesign/infer/workspaces/evo2_feature_bundle_smoke/config.yaml`

### Config stance

- keep one infer config per model lane
- keep contexts explicit as separate jobs
- let `construct` supply template-backed resolved sequences and anchor coordinates
- let `feature_bundle` decide which feature groups and pooling modes are collected

Minimal example:

```yaml
model: # Keep one model stanza per infer config.
  id: evo2_7b # Use the 7B lane for the first green path.
  device: cpu # Keep the smoke example CPU-safe.
  precision: fp32 # Use fp32 for the portable smoke path.
  alphabet: dna # Match Evo2's DNA alphabet contract.

jobs: # Keep contexts explicit as separate infer jobs.
  - id: anchor_only_7b_bundle # Collect anchor-only bundle outputs.
    operation: extract # Run the feature extraction surface.
    ingest: # Read direct records for the anchor-only lane.
      source: records # Load sequence records from a JSONL file.
      path: inputs/anchor_only_records.jsonl # Point at the anchor-only input plane.
      field: sequence # Read the sequence field from each input record.
    feature_bundle: # Let the bundle surface choose the default feature groups.
      intermediate_block: 26 # Use the stable config default; runtime resolves it model-aware.
      context: # Record the explicit context metadata for this job.
        kind: anchor_only # Mark this lane as the anchor-only context.

  - id: template_1kb_7b_bundle # Collect construct-expanded bundle outputs.
    operation: extract # Reuse the same extract operation for the templated lane.
    ingest: # Read direct records for the 1 kb templated context.
      source: records # Load sequence records from a JSONL file.
      path: inputs/template_1kb_records.jsonl # Point at the construct-expanded input plane.
      field: sequence # Read the sequence field from each input record.
    feature_bundle: # Keep feature collection consistent across contexts.
      intermediate_block: 26 # Keep the same config default; evo2_20b resolves it to block 23.
      context: # Record the explicit context metadata for this job.
        kind: template_1kb # Mark this lane as the default templated context.
```

### Choosing 7B vs 20B

- use `evo2_7b` for the first green path and local smoke runs
- switch to `evo2_20b` with a one-line config change: `model.id: evo2_20b`
- keep the project-default intermediate selector unchanged unless repo-local
  benchmarks justify another lane; the stable config default `26` resolves
  model-aware, so `evo2_20b` uses block 23 at runtime

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
from dnadesign.infer import export_evo2_sequence_opal_matrix

payload = export_evo2_sequence_opal_matrix(
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
- `evo2_20b` fails the capacity gate on GPUs below compute capability `9.0`
- stale bundle outputs are recomputed when `metadata__feature_request_digest` does not match the current request

For GPU environment setup and capacity gating, use:

- [SCC Evo2 GPU environment runbook](scc-evo2-gpu-uv-runbook.md)
