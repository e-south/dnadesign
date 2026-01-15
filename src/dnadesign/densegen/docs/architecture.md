# DenseGen Architecture

DenseGen is a **strict, staged pipeline**. Each stage consumes typed config and produces well-defined
artifacts, enabling decoupling and easy replacement of individual components.

## Pipeline (data flow)

```
YAML config
  -> config schema (validation + defaults)
  -> input ingestion (TF/TFBS or sequences)
  -> sampling (library / subsample)
  -> optimization (dense-arrays ILP)
  -> postprocess (gap fill, policies)
  -> outputs (USR + Parquet)
  -> plots (optional, derived from outputs)
```

## Modules and responsibilities

- `config/` — strict, versioned config schema; no fallback parsing.
- `core/canonical.py` — local canonical normalization + ID computation (aligned with USR).
- `adapters/sources/` — input source adapters (binding-site tables, PWM MEME/JASPAR/CSV, sequence libraries, USR). Paths resolved relative to config file.
- `core/sampler.py` — TF library construction and subsampling with explicit coverage policies.
- `adapters/optimizer/` — optimizer adapters (dense-arrays wrapper + strategy selection).
- `core/postprocess/` — gap fill and other sequence transforms.
- `core/pipeline.py` — CLI-agnostic orchestration and runtime guards (DI-ready).
- `cli.py` — thin CLI wrapper around the pipeline.
- `adapters/outputs/` — USR/Parquet sinks with canonical IDs + namespaced metadata.
- `adapters/outputs/record.py` — canonical OutputRecord builder shared by all sinks.
- `viz/plot_registry.py` — plot names + descriptions (no matplotlib import).
- `viz/plotting.py` — rendering from outputs.

## Architectural contracts (high‑level)

- **Strict config:** unknown keys, mixed quota/fraction plans, or missing required fields are errors.
- **Explicit policies:** sampling, solver, and GC fill behaviors are recorded in output metadata.
- **Canonical IDs:** Parquet and USR share the same deterministic ID computation.
- **Output schema:** `output.schema` defines `bio_type` and `alphabet` once for all sinks.
- **Run-scoped I/O:** config must live inside `densegen.run.root`; outputs/logs/plots are confined to the run root.
- **USR optional:** Parquet-only workflows must not import USR modules.
- **No hidden state:** RNG seeds are explicit; no global mutable caches outside runtime guards.
- **Valid motifs:** TFBS and sequence inputs must be A/C/G/T only.

## Extension points

- Add input types by implementing a new data source and wiring it into `adapters/sources/factory.py`.
- Add new plot types by registering them in `viz/plotting.py` (names are strict; unknowns error).
- Add new postprocess steps under `core/postprocess/` and wire them in the pipeline with explicit policy metadata.
