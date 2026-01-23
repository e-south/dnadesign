## DenseGen Architecture

DenseGen is a strict, staged pipeline. Each stage consumes typed config and produces well-defined artifacts, enabling decoupling and easy replacement of individual components.

### Contents
- [Pipeline (data flow)](#pipeline-data-flow) - high-level stage order.
- [Modules and responsibilities](#modules-and-responsibilities) - package map.
- [Architectural contracts (high-level)](#architectural-contracts-high-level) - invariants.
- [Extension points](#extension-points) - where to add new sources, plots, and steps.

---

### Pipeline (data flow)

```
YAML config
  -> config schema (validation + defaults)
  -> Stage‑A sampling (input ingestion + PWM mining → pools)
  -> Stage‑B sampling (library construction / resampling → libraries)
  -> optimization (dense-arrays ILP)
  -> postprocess (pad, policies)
  -> outputs (tables + manifests)
  -> plots (outputs/plots)
```

---

### Modules and responsibilities

- `config/` - strict, versioned config schema; no fallback parsing.
- `adapters/sources/` - Stage‑A input adapters (binding-site tables, PWM MEME/JASPAR/CSV,
  PWM artifacts, sequence libraries, USR sequences). Paths resolve relative to the config file.
- `core/artifacts/pool.py` + `core/artifacts/candidates.py` - Stage‑A pool + candidate artifacts.
- `core/sampler.py` - Stage‑B library sampling policies (coverage, balancing, penalties).
- `core/artifacts/library.py` - Stage‑B library artifacts.
- `adapters/optimizer/` - optimizer adapters (dense‑arrays wrapper + strategy selection).
- `core/pipeline.py` - CLI‑agnostic orchestration + runtime guards.
- `core/postprocess/` - pad and other sequence transforms.
- `core/metadata.py` + `core/metadata_schema.py` - output metadata derivation + validation.
- `adapters/outputs/` + `adapters/outputs/record.py` - Parquet/USR sinks + canonical IDs.
- `core/run_paths.py` + `core/run_manifest.py` + `core/run_state.py` + `core/runtime_policy.py` + `core/seeding.py`
  - run paths, manifests, checkpoints, runtime policy, and seeds.
- `core/reporting.py` + `viz/plot_registry.py` + `viz/plotting.py` - reports and plots.
- `cli.py` - thin CLI wrapper around the pipeline.

---

### Architectural contracts

- **Workspace‑first execution:** CLI resolves config from `./config.yaml` in CWD unless `-c` is provided.
- **No config fallbacks:** missing config exits immediately with an actionable error message.
- **Strict schema:** unknown keys, mixed quota/fraction plans, or missing required fields are errors.
- **Run‑scoped I/O:** outputs/tables/logs/plots/report must resolve inside `outputs/` under
  `densegen.run.root` (enforced).
- **Stage‑A invariants:** Stage‑A sampling is defined per input and produces pools (plus optional
  candidate artifacts). Stage‑A pools are cached per run.
- **Stage‑B invariants:** Stage‑B sampling constructs solver libraries from pools or artifacts;
  resampling happens only in Stage‑B. Library artifacts capture sampling metadata.
- **Explicit policies:** Stage‑A/Stage‑B sampling, solver settings, and pad policies are
  recorded in metadata.
- **Canonical IDs:** Parquet and USR share the same deterministic ID computation.
- **Output schema:** `output.schema` defines `bio_type` and `alphabet` once for all sinks.
- **Optional deps:** USR support is imported only when `output.targets` includes `usr`.
- **No hidden state:** RNG seeds are explicit; no global mutable caches outside runtime guards.
- **Valid motifs:** TFBS and sequence inputs must be A/C/G/T only.

---

### Extension points

- Add input types by implementing a new data source and wiring it into
  `adapters/sources/factory.py`.
- Add new plot types by registering them in `viz/plotting.py` (names are strict; unknowns error).
- Add new postprocess steps under `core/postprocess/` and wire them in the pipeline with
  explicit policy metadata.

---

@e-south
