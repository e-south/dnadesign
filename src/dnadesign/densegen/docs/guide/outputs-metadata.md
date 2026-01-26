## Outputs and metadata

DenseGen writes Parquet outputs with a shared, deterministic ID scheme. This guide focuses on
**what files exist**, **what they mean**, and **how to join them**. For schema‑level detail, see
`reference/outputs.md`.

### Contents
- [Outputs layout](#outputs-layout) - what files exist under `outputs/`.
- [What the artifacts mean](#what-the-artifacts-mean) - semantics by file.
- [Joining outputs](#joining-outputs) - stable join keys.
- [Metadata scheme](#metadata-scheme) - namespacing + categories.
- [Parquet vs USR encoding](#parquet-vs-usr-encoding) - storage differences.
- [Metadata registry](#metadata-registry) - canonical schema location.

---

### Outputs layout

Typical workspace output tree (Stage‑A + Stage‑B):

```
outputs/
  tables/
    dense_arrays.parquet
    attempts.parquet
    solutions.parquet
    composition.parquet
  pools/
  libraries/
  plots/
  report/
  meta/
  logs/
```

Optional targets (when enabled):
- `outputs/usr/` (USR sink)

---

### What the artifacts mean

- `outputs/tables/dense_arrays.parquet` — final sequences with `densegen__*` metadata (canonical dataset).
- `outputs/tables/attempts.parquet` — solver attempt audit log (success, duplicate, constraint failures).
- `outputs/tables/solutions.parquet` — accepted solutions keyed by `solution_id` + `attempt_id`.
- `outputs/tables/composition.parquet` — per‑TFBS placements for accepted solutions.
- `outputs/meta/run_manifest.json` — run‑level counts (Stage‑A pools, Stage‑B libraries, resamples, stalls).
- `outputs/meta/inputs_manifest.json` — resolved inputs and **Stage‑A sampling** settings.
- `outputs/meta/effective_config.json` — resolved config + derived seeds and caps.
- `outputs/meta/run_state.json` — checkpoint (resume guardrails).
- `outputs/meta/events.jsonl` — structured events (Stage‑A pool built, Stage‑B library built, stalls).
- `outputs/pools/` — **Stage‑A** pool artifacts:
  - `pool_manifest.json`
  - `<input>__pool.parquet`
  - `pool_manifest.json` captures Stage‑A sampling metadata when using FIMO
    (p‑value strata, retain depth, per‑regulator bin counts, and eligible p‑value histograms).
- `outputs/libraries/` — **Stage‑B** library artifacts:
  - `library_builds.parquet`
  - `library_members.parquet`
  - `library_manifest.json`
- `outputs/pools/candidates/` — **Stage‑A** candidate logs when `keep_all_candidates_debug: true`:
  - `candidates__<label>.parquet` (per‑input candidate rows)
  - `candidates.parquet` + `candidates_summary.parquet` + `candidates_manifest.json` (aggregates)
- `outputs/plots/` — plot images from `dense run` auto‑plotting or `dense plot`
  (format controlled by `plots.format`).
- `outputs/plots/plot_manifest.json` — plot inventory used by reports when `--plots include` is set.
- `outputs/report/` — audit report outputs:
  - `report.json`, `report.md`, `report.html`

---

### Joining outputs

Stable join paths (Stage‑B + outputs):

- `tables/dense_arrays.parquet.id` ↔ `tables/solutions.parquet.solution_id`
- `tables/solutions.parquet.attempt_id` ↔ `tables/attempts.parquet.attempt_id`
- `tables/solutions.parquet.solution_id` ↔ `tables/composition.parquet.solution_id`
- `tables/attempts.parquet.library_hash` / `library_index` ↔ `libraries/library_builds.parquet`
- `libraries/library_members.parquet` joins on `input_name`, `plan_name`, `library_index`

Stage‑A joins:

- `pools/<input>__pool.parquet` join on `tfbs_id` / `motif_id` where available.
- `pools/candidates/candidates.parquet` includes `input_name`, `motif_id`, and `run_id` for audits.

---

### Metadata scheme

All metadata keys are prefixed as `densegen__<key>`. Categories include:

- Provenance (`densegen__schema_version`, run identifiers, input info)
- Solver + policy (`densegen__solver_*`, `densegen__policy_*`)
- Stage‑B library sampling (`densegen__library_*`, `densegen__sampling_*`)
- Constraints + postprocess (`densegen__fixed_elements`, `densegen__pad_*`)
- Placement stats (`densegen__used_tfbs*`, `densegen__required_regulators*`)

`densegen__sampling_fraction` is the fraction of **unique** TFBS strings in the Stage‑B library
divided by the realized Stage‑A pool (`input_tfbs_count`).

---

### Parquet vs USR encoding

- Parquet stores list/dict metadata as native list/struct columns (no JSON encoding).
- USR stores list/dict metadata as JSON strings in attaches.

DenseGen fails fast if a Parquet dataset schema does not match the current registry.

---

### Metadata registry

DenseGen validates output metadata against a typed registry in
`src/dnadesign/densegen/src/core/metadata_schema.py`.

---

@e-south
