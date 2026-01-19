## Inputs

Inputs define the binding-site library that feeds planning and optimization. Each entry in
`densegen.inputs` is a named source (choose one input type per entry), and paths resolve relative
to the config file location.

PWM inputs perform **input sampling** (sampling sites from PWMs) via
`densegen.inputs[].sampling`. This is distinct from **library sampling**
(`densegen.generation.sampling`), which selects a solver library (`library_size`) from the
realized TFBS pool (`input_tfbs_count`). PWM sampling size is controlled only by the input
sampling config; solver library size is controlled only by `densegen.generation.sampling`.

Sample inputs live in:
- `src/dnadesign/densegen/workspaces/demo_meme_two_tf/inputs` (LexA + CpxR MEME files, copied from
  the Cruncher demo workspace for a self-contained example)
- For PWM artifacts, generate files with `cruncher catalog export-densegen` and place them
  under your run’s `inputs/` directory (see examples below).

Common end-to-end flow (Cruncher → DenseGen):
1) `cruncher fetch sites` (DAP-seq, RegulonDB, local MEME sites) and merge as needed.
2) `cruncher lock` → `cruncher parse` (optional) → `cruncher discover` (MEME/STREME) to generate PWMs.
3) `cruncher catalog export-densegen` to emit per-motif JSON artifacts.
4) (Optional) `cruncher catalog export-sites` to emit a binding-site table for DenseGen.
5) Point DenseGen `inputs` at those artifacts (or at MEME/JASPAR files directly).

### Contents
- [Binding site table](#binding-site-table-type-binding_sites) - explicit TF/TFBS pairs.
- [Sequence library](#sequence-library-type-sequence_library) - raw sequence seeds.
- [PWM MEME](#pwm-meme-type-pwm_meme) - sample from MEME PWMs.
- [PWM MEME set](#pwm-meme-set-type-pwm_meme_set) - merge multiple MEME files into one TF pool.
- [PWM JASPAR](#pwm-jaspar-type-pwm_jaspar) - sample from JASPAR PFMs.
- [PWM matrix CSV](#pwm-matrix-csv-type-pwm_matrix_csv) - sample from CSV matrices.
- [PWM artifact JSON](#pwm-artifact-json-type-pwm_artifact) - sample from contract-first motif artifacts.
- [PWM artifact set](#pwm-artifact-set-json-type-pwm_artifact_set) - combine multiple motif artifacts into one input.
- [USR sequences](#usr-sequences-type-usr_sequences) - read sequences from USR.
- [Path resolution](#path-resolution) - how relative paths are resolved.
- [Interaction with constraints](#interaction-with-constraints) - constraints that depend on inputs.

---

### Binding site table (`type: binding_sites`)

Use a CSV, Parquet, or XLSX table with regulator and binding-site sequences.

Required columns (override via `columns`):
- `regulator` (default column: `tf`)
- `sequence` (default column: `tfbs`)

Optional columns:
- `site_id`
- `source`

Strict rules:
- Regulator + sequence must be non-empty.
- Duplicate regulator/sequence pairs are allowed; use `generation.sampling.unique_binding_sites`
  to dedupe at sampling time (or keep duplicates as implicit weights).
- Sequences must be A/C/G/T only (DNA_4).

Example:

```yaml
inputs:
  - name: demo
    type: binding_sites
    path: inputs/binding_sites.xlsx
    format: xlsx
```

---

### Sequence library (`type: sequence_library`)

Use a CSV or Parquet table with a `sequence` column (override via `sequence_column`).

Strict rules:
- Sequences must be non-empty.
- Sequences must be A/C/G/T only.

Example:

```yaml
inputs:
  - name: seeds
    type: sequence_library
    path: inputs/seed_sequences.csv
    format: csv
```

---

### PWM MEME (`type: pwm_meme`)

Use a MEME-format PWM file and explicitly sample binding sites.

Required sampling fields:
- `strategy`: `consensus | stochastic | background`
- `n_sites`: number of binding sites to generate per motif
- `score_threshold` or `score_percentile` (exactly one)
- `oversample_factor`: oversampling multiplier for candidate generation
- `max_candidates` (optional): cap on candidate generation; helps bound long motifs
- `max_seconds` (optional): time limit for candidate generation (best-effort cap)

Notes:
- Sampling scores use PWM log-odds with the motif background (from MEME when available).
- `score_threshold` / `score_percentile` controls similarity to the PWM consensus
  (higher percentiles or thresholds yield stronger matches).
- `length_policy` defaults to `exact`. Use `length_policy: range` with `length_range: [min, max]`
  to sample variable lengths (min must be >= motif length).
- `trim_window_length` optionally trims the PWM to a max‑information window before sampling (useful
  for long motifs when you want shorter cores); `trim_window_strategy` currently supports `max_info`.
- `consensus` requires `n_sites: 1`.
- `background` selects low-scoring sequences from the PWM.

Example:

```yaml
inputs:
  - name: lexA_meme
    type: pwm_meme
    path: inputs/lexA.txt
    motif_ids: [lexA]
    sampling:
      strategy: stochastic
      n_sites: 80
      oversample_factor: 12
      max_candidates: 50000
      max_seconds: 5
      score_percentile: 80
      length_policy: range
      length_range: [22, 28]
```

---

### PWM MEME set (`type: pwm_meme_set`)

Use multiple MEME files and sample them into a single TF pool. This is the
recommended way to combine LexA + CpxR motifs for DenseGen so the solver stage
can see both TFs at once (rather than sampling two independent inputs).

Required sampling fields are identical to `pwm_meme`.

Example:

```yaml
inputs:
  - name: lexA_cpxR_meme
    type: pwm_meme_set
    paths:
      - inputs/lexA.txt
      - inputs/cpxR.txt
    motif_ids: [lexA, cpxR]
    sampling:
      strategy: stochastic
      n_sites: 80
      oversample_factor: 12
      max_candidates: 50000
      score_percentile: 80
      length_policy: range
      length_range: [22, 28]
```

---

### PWM JASPAR (`type: pwm_jaspar`)

Use a JASPAR PFM file and explicitly sample binding sites.

Required sampling fields are identical to `pwm_meme`.

Example:

```yaml
inputs:
  - name: jaspar_demo
    type: pwm_jaspar
    # Replace with your JASPAR PFM file.
    path: /path/to/motifs.jaspar
    motif_ids: [YourTF]
    sampling:
      strategy: background
      n_sites: 200
      oversample_factor: 5
      score_percentile: 10
```

---

### PWM matrix CSV (`type: pwm_matrix_csv`)

Use a CSV matrix with `A,C,G,T` columns (override via `columns`) and a single motif ID.

Required sampling fields are identical to `pwm_meme`.

Example:

```yaml
inputs:
  - name: matrix_demo
    type: pwm_matrix_csv
    # Replace with your CSV matrix file.
    path: /path/to/motif_matrix.csv
    motif_id: YourTF
    columns:
      A: A
      C: C
      G: G
      T: T
    sampling:
      strategy: stochastic
      n_sites: 200
      oversample_factor: 5
      score_threshold: -9.0
```

---

### PWM artifact JSON (`type: pwm_artifact`)

Use a per-motif JSON artifact that follows DenseGen's motif artifact contract
(see `docs/reference/motif_artifacts.md`). This is the most decoupled path:
DenseGen consumes explicit artifact files without parsing MEME/JASPAR directly.
Artifacts in the Cruncher demo are generated via `cruncher catalog export-densegen`
(implemented in `cruncher/src/app/motif_artifacts.py`).

Required sampling fields are identical to `pwm_meme`.
Log-odds in the artifact are used for scoring.

Example:

```yaml
inputs:
  - name: motif_artifact
    type: pwm_artifact
    path: inputs/motif_artifacts/lexA__demo_local_meme__lexA.json
    sampling:
      strategy: stochastic
      n_sites: 200
      oversample_factor: 5
      score_percentile: 90
      length_policy: exact
```

---

### PWM artifact set JSON (`type: pwm_artifact_set`)

Use multiple per-motif JSON artifacts (one file per motif) and sample each PWM
with the same policy. This keeps parsing decoupled while producing a single
combined input library.

Notes:
- Each artifact must declare a unique `motif_id` (duplicates are errors).
- `overrides_by_motif_id` lets you override PWM sampling per motif while keeping a shared default.

Example:

```yaml
inputs:
  - name: lexA_cpxR_artifacts
    type: pwm_artifact_set
    paths:
      - inputs/motif_artifacts/lexA__demo_local_meme__lexA.json
      - inputs/motif_artifacts/cpxR__demo_local_meme__cpxR.json
    sampling:
      strategy: stochastic
      n_sites: 80
      oversample_factor: 10
      score_percentile: 90
      length_policy: exact
    overrides_by_motif_id:
      cpxR:
        strategy: stochastic
        n_sites: 40
        oversample_factor: 10
        score_percentile: 85
        length_policy: exact
```

---

### USR sequences (`type: usr_sequences`)

Read sequences from a USR dataset.

Strict rules:
- `root` is required (no default lookup).
- Sequences must be non-empty and A/C/G/T only.

Example:

```yaml
inputs:
  - name: usr_seed
    type: usr_sequences
    dataset: my_dataset
    root: /path/to/usr/datasets
```

---

### Path resolution

All relative paths resolve against the config file directory.

---

### Interaction with constraints

- Promoter constraints are fixed motifs and enforced by the optimizer.
- `side_biases` motifs must exist in the sampled library (DenseGen fails fast if missing).
- `required_regulators` (per plan item) must appear in the sampled library and in each solution.

---

### Troubleshooting

- Required regulators cannot be satisfied: increase `densegen.generation.sampling.library_size`,
  reduce required regulators, or relax `cover_all_regulators`.
- `cover_all_regulators` impossible when TF count exceeds `library_size`: set
  `allow_incomplete_coverage: true` or increase `library_size`.
- PWM sampling produced too few unique sites: raise `oversample_factor`, lower the
  `score_threshold` / `score_percentile`, widen `length_range`, or raise `max_candidates` / `max_seconds`.

---

@e-south
