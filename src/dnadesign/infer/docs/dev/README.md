<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[Infer guide](../README.md)

# Develop and integrate Infer

Use `dnadesign.infer` for Python integrations. The [architecture map](../architecture/README.md) locates request parsing, execution, feature processing and storage internals.

## Validate a job from Python

After creating the workspace in the [quickstart](../getting-started/cli-quickstart.md), run this from that workspace in the same environment:

```python
from pathlib import Path
from dnadesign.infer import validate_infer_dry_run_contract

# Resolve configuration and inputs without loading model weights.
request = validate_infer_dry_run_contract(Path("config.yaml"))
print(request.model_id)
print(request.job_ids)
```

Expected output:

```text
evo2_7b
('workspace_local_records_extract',)
```

`validate_infer_config_contract` checks the configuration without resolving input records. Use `validate_infer_dry_run_contract` when you also want input preflight.

## Execute through the public API

`run_extract` and `run_generate` execute ad-hoc requests. `run_job` accepts a model and job configuration. Evo2 bundles use `run_evo2_sequence_features`; `export_evo2_sequence_opal_matrix` converts returned bundle values to a matrix with a declared feature order. These execution APIs require a configured model runtime; they are not dry runs.

See [feature schema and export](../reference/feature-schema.md) before integrating storage or downstream analysis. Keep consumers on the public APIs rather than importing `dnadesign.infer.src` modules.

## Check a change

From the repository root, use the development environment:

```bash
# Check documentation contracts and CLI request handling without model inference.
uv run --no-sync pytest src/dnadesign/infer/tests/docs src/dnadesign/infer/tests/cli
uv run --no-sync python -m dnadesign.devtools.docs.checks --repo-root .
uv run --no-sync python -m dnadesign.devtools.architecture.boundaries --repo-root .
```

Keep parsing, execution and write-back independently testable. Add model or storage tests for the behavior you change, and state whether validation used a real model runtime. The [repository change gate](../../../../../docs/repo-change-gate.md) describes broader checks.
