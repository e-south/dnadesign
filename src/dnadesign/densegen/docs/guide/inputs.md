## Inputs (Stage‑A)

This guide covers **Stage‑A ingestion and sampling** (`densegen.inputs[]`). Stage‑B sampling
(`densegen.generation.sampling` for Stage‑B sampling) is documented in the generation guide.

### Contents
- [Stage‑A PWM sampling config (common)](#stage-a-pwm-sampling-config-common) - shared Stage‑A sampling fields.
- [Binding site table](#binding-site-table-type-binding_sites) - explicit TF/TFBS pairs.
- [Sequence library](#sequence-library-type-sequence_library) - raw sequence seeds.
- [PWM MEME](#pwm-meme-type-pwm_meme) - sample from MEME PWMs.
- [PWM MEME set](#pwm-meme-set-type-pwm_meme_set) - merge multiple MEME files.
- [PWM JASPAR](#pwm-jaspar-type-pwm_jaspar) - sample from JASPAR PFMs.
- [PWM matrix CSV](#pwm-matrix-csv-type-pwm_matrix_csv) - sample from CSV matrices.
- [PWM artifact JSON](#pwm-artifact-json-type-pwm_artifact) - sample from artifact contracts.
- [PWM artifact set](#pwm-artifact-set-json-type-pwm_artifact_set) - merge multiple artifacts.
- [USR sequences](#usr-sequences-type-usr_sequences) - read sequences from USR.
- [Path resolution](#path-resolution) - how relative paths resolve.
- [Interaction with constraints](#interaction-with-constraints) - Stage‑A inputs + constraints.

---

### Stage‑A PWM sampling config (common)

Applies to `pwm_meme`, `pwm_meme_set`, `pwm_jaspar`, `pwm_matrix_csv`, `pwm_artifact`, and
`pwm_artifact_set`.

Required (always):
- `n_sites` (int > 0)

Required when `scoring_backend: densegen`:
- `scoring_backend: densegen` (default)
- exactly one of `score_threshold` or `score_percentile`

Required when `scoring_backend: fimo`:
- `scoring_backend: fimo`
- `pvalue_threshold` (float in (0, 1])

Optional (supported):
- `strategy`: `consensus | stochastic | background` (default `stochastic`)
- `oversample_factor` (int > 0; default `10`)
- `max_candidates` (densegen‑only; int > 0 when set)
- `max_seconds` (densegen‑only; float > 0 when set)
- `selection_policy` (fimo‑only): `random_uniform | top_n | stratified`
- `pvalue_bins` (fimo‑only): list of floats, strictly increasing, must end with `1.0`
- `mining` (fimo‑only):
  - `batch_size` (int > 0)
  - `max_batches` (optional int > 0)
  - `max_candidates` (optional int > 0; must be ≥ `n_sites`)
  - `max_seconds` (optional float > 0; default 60s)
  - `retain_bin_ids` (optional list of ints; unique, in‑range)
  - `log_every_batches` (int > 0)
- `bgfile` (fimo‑only): MEME background file
- `keep_all_candidates_debug` (bool): write candidate‑level Parquet under `outputs/pools/candidates/`
  (files named `candidates__<label>.parquet`) and aggregate to
  `outputs/pools/candidates/candidates.parquet` + `outputs/pools/candidates/candidates_summary.parquet`
- `include_matched_sequence` (fimo‑only)
- `length_policy`: `exact | range` (default `exact`)
- `length_range`: `[min, max]` (required when `length_policy: range`)
- `trim_window_length` (optional int > 0)
- `trim_window_strategy`: `max_info`

Strict validation behavior:
- Unknown keys are errors (extra fields are rejected).
- DenseGen backend requires exactly one of `score_threshold` or `score_percentile`.
- FIMO backend requires `pvalue_threshold`; `max_candidates`/`max_seconds` are **not** allowed.
- `consensus` requires `n_sites: 1`.

Minimal Stage‑A PWM example (DenseGen backend):

```yaml
inputs:
  - name: lexA
    type: pwm_meme
    path: inputs/lexA.txt
    sampling:  # Stage‑A sampling
      n_sites: 80
      score_percentile: 80
```

Minimal Stage‑A PWM example (FIMO backend):

```yaml
inputs:
  - name: lexA
    type: pwm_meme
    path: inputs/lexA.txt
    sampling:  # Stage‑A sampling
      scoring_backend: fimo
      pvalue_threshold: 1e-4
      n_sites: 80
```

---

### Binding site table (`type: binding_sites`)

Required fields:
- `name`, `type`, `path`

Supported optional fields:
- `format`: `csv | parquet | xlsx`
- `columns.regulator` (default: `tf`)
- `columns.sequence` (default: `tfbs`)
- `columns.site_id`, `columns.source`

Strict validation:
- Regulator + sequence must be non‑empty.
- Sequences must be A/C/G/T only.

Minimal example:

```yaml
inputs:
  - name: demo
    type: binding_sites
    path: inputs/binding_sites.csv
```

---

### Sequence library (`type: sequence_library`)

Required fields:
- `name`, `type`, `path`

Supported optional fields:
- `format`: `csv | parquet`
- `sequence_column` (default: `sequence`)

Strict validation:
- Sequence column must exist.
- Sequences must be non‑empty and A/C/G/T only.

Minimal example:

```yaml
inputs:
  - name: seeds
    type: sequence_library
    path: inputs/seed_sequences.csv
```

---

### PWM MEME (`type: pwm_meme`)

Required fields:
- `name`, `type`, `path`, Stage‑A `sampling` config

Supported optional fields:
- `motif_ids` (list of motif IDs to include)

Strict validation:
- `motif_ids` must be unique, non‑empty strings.
- Stage‑A sampling config is validated as above.

Minimal example:

```yaml
inputs:
  - name: lexA
    type: pwm_meme
    path: inputs/lexA.txt
    motif_ids: [lexA]
    sampling:  # Stage‑A sampling
      n_sites: 80
      score_percentile: 80
```

---

### PWM MEME set (`type: pwm_meme_set`)

Required fields:
- `name`, `type`, `paths` (non‑empty list), Stage‑A `sampling`

Supported optional fields:
- `motif_ids` (list of motif IDs to include)

Strict validation:
- `paths` must be unique, non‑empty.
- `motif_ids` must be unique, non‑empty.

Minimal example:

```yaml
inputs:
  - name: lexA_cpxR
    type: pwm_meme_set
    paths: [inputs/lexA.txt, inputs/cpxR.txt]
    sampling:  # Stage‑A sampling
      n_sites: 80
      score_percentile: 80
```

---

### PWM JASPAR (`type: pwm_jaspar`)

Required fields:
- `name`, `type`, `path`, Stage‑A `sampling`

Supported optional fields:
- `motif_ids` (list of motif IDs to include)

Strict validation:
- `motif_ids` must be unique, non‑empty.

Minimal example:

```yaml
inputs:
  - name: jaspar_demo
    type: pwm_jaspar
    path: inputs/motifs.jaspar
    sampling:  # Stage‑A sampling
      n_sites: 80
      score_percentile: 80
```

---

### PWM matrix CSV (`type: pwm_matrix_csv`)

Required fields:
- `name`, `type`, `path`, `motif_id`, Stage‑A `sampling`

Supported optional fields:
- `columns` (map column names to A/C/G/T)

Strict validation:
- `motif_id` must be a non‑empty string.

Minimal example:

```yaml
inputs:
  - name: pwm_csv
    type: pwm_matrix_csv
    path: inputs/lexA_matrix.csv
    motif_id: lexA
    sampling:  # Stage‑A sampling
      n_sites: 80
      score_percentile: 80
```

---

### PWM artifact JSON (`type: pwm_artifact`)

Required fields:
- `name`, `type`, `path`, Stage‑A `sampling`

Supported optional fields:
- none (aside from the Stage‑A sampling config)

Strict validation:
- Artifact must match the motif contract (see `reference/motif_artifacts.md`).

Minimal example:

```yaml
inputs:
  - name: lexA_artifact
    type: pwm_artifact
    path: inputs/motif_artifacts/lexA.json
    sampling:  # Stage‑A sampling
      n_sites: 80
      score_percentile: 80
```

---

### PWM artifact set (`type: pwm_artifact_set`)

Required fields:
- `name`, `type`, `paths` (non‑empty list), Stage‑A `sampling`

Supported optional fields:
- `overrides_by_motif_id` (per‑motif Stage‑A sampling overrides)

Strict validation:
- `paths` must be unique, non‑empty.
- `overrides_by_motif_id` keys must be unique, non‑empty strings.

Minimal example:

```yaml
inputs:
  - name: lexA_cpxR_artifacts
    type: pwm_artifact_set
    paths:
      - inputs/motif_artifacts/lexA.json
      - inputs/motif_artifacts/cpxR.json
    sampling:  # Stage‑A sampling
      n_sites: 80
      score_percentile: 80
```

---

### USR sequences (`type: usr_sequences`)

Required fields:
- `name`, `type`, `dataset`, `root`

Supported optional fields:
- `limit`

Strict validation:
- USR must be installed when this input is used.
- Sequences must be A/C/G/T only.

Minimal example:

```yaml
inputs:
  - name: usr_seqs
    type: usr_sequences
    dataset: my_dataset
    root: inputs/usr_datasets
```

---

### Path resolution

Input paths resolve relative to the config file directory. For `usr_sequences`, `root` is explicit
and must exist; there is no fallback search.

---

### Interaction with constraints

Stage‑A inputs must provide motifs needed by constraints (e.g., `side_biases` or
`promoter_constraints`). If a required motif is missing from the Stage‑A pool, DenseGen fails fast
during Stage‑B library construction.

---

@e-south
