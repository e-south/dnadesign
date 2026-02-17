## DenseGen architecture guide

This page is maintainer-focused. It maps DenseGen runtime behavior to the key files you should start from when debugging or extending the system.

### Contents
This section lists the maintainer entry points in the order most debugging sessions should follow.

- [Where to start](#where-to-start)
- [CLI entrypoint](#cli-entrypoint)
- [Config loading and validation](#config-loading-and-validation)
- [Stage-A pipeline](#stage-a-pipeline)
- [Stage-B and solve pipeline](#stage-b-and-solve-pipeline)
- [Outputs and metadata](#outputs-and-metadata)
- [Analysis outputs (plots and notebooks)](#analysis-outputs-plots-and-notebooks)

### Where to start

If you are tracing runtime behavior end-to-end, start here:

- `src/dnadesign/densegen/src/core/pipeline/orchestrator.py`

This is the assembly and orchestration boundary for staged execution, resume policy, and run-level persistence.

### CLI entrypoint

Primary run entrypoint:

- `src/dnadesign/densegen/src/cli/run.py` calls into orchestrator.

Use CLI commands to reproduce issues before editing internals.

### Config loading and validation

Config is strict by design (unknown keys and removed keys must hard-fail).

Key files:

- `src/dnadesign/densegen/src/config/root.py` for load/validate/expand
- `src/dnadesign/densegen/src/config/generation.py` for plan resolution and template expansion
- `src/dnadesign/densegen/src/config/inputs.py` for Stage-A input contract
- `src/dnadesign/densegen/src/config/output.py` for sink routing and parity rules

If behavior is surprising, confirm the resolved config first (including expanded plans).

### Stage-A pipeline

Stage-A owns "input realization": building pools from binding sites, PWM artifacts, background mining, or USR-backed sources.

Key files:

- `src/dnadesign/densegen/src/core/pipeline/stage_a_pools.py` for pool build/load orchestration
- `src/dnadesign/densegen/src/core/stage_a/stage_a_pipeline.py` for PWM mining/selection logic
- `src/dnadesign/densegen/src/adapters/sources/factory.py` for source adapter construction

Stage-A artifacts are written under `outputs/pools/`, including a pool manifest.

### Stage-B and solve pipeline

Stage-B owns library construction and the solve loop owns quota fulfillment with retries and attempt accounting.

Key files:

- `src/dnadesign/densegen/src/core/pipeline/stage_b_library_builder.py` for plan-scoped library construction and feasibility checks
- `src/dnadesign/densegen/src/core/sampler.py` for Stage-B sampling behavior and weighting
- `src/dnadesign/densegen/src/core/pipeline/stage_b_runtime_runner.py` for the solve loop coordination
- `src/dnadesign/densegen/src/core/pipeline/stage_b_runtime_checks.py` for runtime constraint checks and postprocess validation
- `src/dnadesign/densegen/src/core/pipeline/stage_b_runtime_callbacks.py` for attempt accounting, rejection policies, and event emission

If you are debugging "why did this plan never fill quota," start by inspecting Stage-B feasibility and the runtime callbacks' rejection reasons.

### Outputs and metadata

Outputs are produced via configured sinks (local parquet and/or USR datasets). Sink creation is config-driven and strict.

Key files:

- `src/dnadesign/densegen/src/adapters/outputs/factory.py` for sink selection and construction
- `src/dnadesign/densegen/src/core/metadata.py` for run-level metadata assembly

### Analysis outputs (plots and notebooks)

Plots consume a configured source (`plots.source`) and written artifacts. Notebook generation uses a fixed render contract.

Key files:

- `src/dnadesign/densegen/src/cli/notebook.py` for notebook generation orchestration
- `src/dnadesign/densegen/src/integrations/baserender/notebook_contract.py` for BaseRender contract
