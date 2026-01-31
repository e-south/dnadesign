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

Optional (supported):
- `strategy`: `consensus | stochastic | background` (default `stochastic`)
- `mining`:
  - `batch_size` (int > 0)
  - `budget`:
    - `mode`: `tier_target | fixed_candidates`
    - `target_tier_fraction` (required when `mode=tier_target`)
    - `candidates` (required when `mode=fixed_candidates`)
    - `max_candidates` (optional)
    - `max_seconds` (optional)
    - `min_candidates` (optional)
    - `growth_factor` (float > 1; default 1.25)
  - `log_every_batches` (int > 0)
- `fixed_candidates` is the recommended mining mode (direct, user-set budget).
  `tier_target` is advanced and may stop early at caps/time; shortfalls are recorded in the manifest.
- `bgfile`: MEME background file
- `keep_all_candidates_debug` (bool): write candidate‑level Parquet under `outputs/pools/candidates/`
  (files named `candidates__<label>.parquet`) and aggregate to
  `outputs/pools/candidates/candidates.parquet` + `outputs/pools/candidates/candidates_summary.parquet`
- `include_matched_sequence` (default true; must be true for PWM sampling; config validation rejects false)
- `tier_fractions` (optional list of three floats in (0, 1], non‑decreasing, sum ≤ 1.0; default
  `[0.001, 0.01, 0.09]`). Used for diagnostic tiers and as the cumulative rung ladder for MMR pool selection.
- `length`:
  - `policy`: `exact | range` (default `exact`)
  - `range`: `[min, max]` (required when `policy: range`)
- `trimming`:
  - `window_length` (optional int > 0)
  - `window_strategy`: `max_info`
- `uniqueness`:
  - `key`: `sequence | core` (default `core`)
- `selection`:
  - `policy`: `top_score | mmr` (default `top_score`)
  - `alpha` (float in (0, 1]; MMR score weight)
  - `pool` (required when `policy=mmr`)
    - `min_score_norm` (optional float in (0, 1]; recorded as a “within τ of max” reference in reports)
    - `max_candidates` (optional int > 0; cap the MMR pool to the top-by-score slice)
    - `relevance_norm` (optional: `percentile | minmax_raw_score`; default `minmax_raw_score`)
  - MMR pool selection uses the cumulative tier ladder derived from `sampling.tier_fractions`.
    It chooses the smallest rung that can supply `n_sites` (or the full list if none can).

Strict validation behavior:
- Unknown keys are errors (extra fields are rejected).
- `consensus` requires `n_sites: 1`.

Algorithm note:
Stage-A PWM mining and retention (FIMO score semantics, tier targeting math, MMR diversity, and core dedupe)
are documented in the canonical sampling guide:
- [Sampling (Stage-A + Stage-B)](sampling.md#stage-a-sampling)

Minimal Stage‑A PWM example (score‑based FIMO):

```yaml
inputs:
  - name: lexA
    type: pwm_meme
    path: inputs/lexA.txt
    sampling:  # Stage‑A sampling
      n_sites: 80
      mining:
        batch_size: 5000
        budget:
          mode: fixed_candidates
          candidates: 200000
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
Binding site inputs derive `tfbs_core` as the full binding‑site sequence for core‑level uniqueness checks.

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
      mining:
        batch_size: 5000
        budget:
          mode: fixed_candidates
          candidates: 200000
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
      mining:
        batch_size: 5000
        budget:
          mode: fixed_candidates
          candidates: 200000
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
      mining:
        batch_size: 5000
        budget:
          mode: fixed_candidates
          candidates: 200000
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
      mining:
        batch_size: 5000
        budget:
          mode: fixed_candidates
          candidates: 200000
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
      mining:
        batch_size: 5000
        budget:
          mode: fixed_candidates
          candidates: 200000
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
      mining:
        batch_size: 5000
        budget:
          mode: fixed_candidates
          candidates: 200000
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
