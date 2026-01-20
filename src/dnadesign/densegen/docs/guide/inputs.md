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
- `scoring_backend`: `densegen | fimo` (default: `densegen`)
- `score_threshold` or `score_percentile` (exactly one; densegen backend only)
- `pvalue_threshold` (float in (0, 1]; fimo backend only)
- `oversample_factor`: oversampling multiplier for candidate generation
- `max_candidates` (optional): cap on candidate generation; helps bound long motifs
- `max_seconds` (optional): time limit for candidate generation per batch (best-effort cap)
- `selection_policy`: `random_uniform | top_n | stratified` (default: `random_uniform`; fimo only)
- `pvalue_bins` (optional): list of p‑value bin edges (strictly increasing; must end with `1.0`)
- `pvalue_bin_ids` (deprecated; use `mining.retain_bin_ids`)
- `mining` (optional; fimo only): batch/time controls for mining with FIMO
  - `batch_size` (int > 0): candidates per batch
  - `max_batches` (optional int > 0): limit batches per motif
  - `max_seconds` (optional float > 0): limit total mining time per motif
  - `retain_bin_ids` (optional list of ints): keep only specific p‑value bins
  - `log_every_batches` (int > 0): log yield summaries every N batches
- `bgfile` (optional): MEME bfile-format background model for FIMO
- `keep_all_candidates_debug` (optional): write raw FIMO TSVs to `outputs/meta/fimo/` for inspection
- `include_matched_sequence` (optional): include `fimo_matched_sequence` column in the TFBS table

Notes:
- `densegen` scoring uses PWM log-odds with the motif background (from MEME when available).
- `fimo` scoring scans the entire emitted TFBS and uses a model-based p-value threshold.
  `pvalue_threshold` controls match strength (smaller values are stronger).
- `fimo` backend requires the `fimo` executable on PATH (run via pixi).
- If `bgfile` is omitted, FIMO uses the motif background (or uniform if none provided).
- `background` selects low-scoring sequences (<= threshold/percentile; or pvalue >= threshold for fimo).
- `selection_policy: stratified` uses fixed p‑value bins to balance strong/weak matches.
- Canonical p‑value bins (default): `[1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]`.
  Bin 0 is `(0, 1e-10]`, bin 1 is `(1e-10, 1e-8]`, etc.

#### FIMO p-values (beginner-friendly)
- A **p-value** is the probability that a random sequence (under the background model)
  would score **at least as well** as the observed match.
- Smaller p-values mean **stronger** motif matches; larger p-values mean **weaker** matches.
- As a rule of thumb: `1e-4` is a strong match, `1e-3` is moderate, `1e-2` is weak.
- DenseGen accepts a candidate if its **best hit** within the emitted TFBS passes the threshold.
- For `strategy: background`, DenseGen keeps **weak** matches where `pvalue >= pvalue_threshold`.
- If you set `mining.retain_bin_ids`, DenseGen only keeps candidates in those bins (useful for mining
  specific affinity ranges).
- FIMO adds per‑TFBS metadata columns: `fimo_score`, `fimo_pvalue`, `fimo_start`, `fimo_stop`,
  `fimo_strand`, `fimo_bin_id`, `fimo_bin_low`, `fimo_bin_high`, and (optionally)
  `fimo_matched_sequence` (the best‑hit window within the TFBS).
- `length_policy` defaults to `exact`. Use `length_policy: range` with `length_range: [min, max]`
  to sample variable lengths (min must be >= motif length).
- `trim_window_length` optionally trims the PWM to a max‑information window before sampling (useful
  for long motifs when you want shorter cores); `trim_window_strategy` currently supports `max_info`.
- `consensus` requires `n_sites: 1`.

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

FIMO-backed example:

```yaml
inputs:
  - name: lexA_meme
    type: pwm_meme
    path: inputs/lexA.txt
    motif_ids: [lexA]
    sampling:
      strategy: stochastic
      scoring_backend: fimo
      pvalue_threshold: 1e-4
      selection_policy: top_n
      n_sites: 80
      oversample_factor: 200
      max_candidates: 20000
      mining:
        batch_size: 5000
        max_batches: 4
        retain_bin_ids: [0, 1, 2, 3]
        log_every_batches: 1
```

#### Mining workflow (p‑value strata)
If you want to **mine** sequences across affinity strata, use `selection_policy: stratified` plus
canonical p‑value bins and the `mining` block. A typical workflow:

1) Oversample candidates (`oversample_factor`, `max_candidates`) and score with FIMO in batches
   (`mining.batch_size`).
2) Accept candidates using `pvalue_threshold` (global strength cutoff).
3) Use `mining.retain_bin_ids` to select one or more bins (e.g., moderate matches only).
4) Repeat runs (or increase `mining.max_batches` / `mining.max_seconds`) to accumulate a deduplicated
   reservoir of sequences per bin.
5) Use `dense summarize --library` to inspect which TFBS were offered vs used in Stage‑B sampling.

DenseGen reports per‑bin yield summaries (hits, accepted, selected) for retained bins only (or all
bins if `retain_bin_ids` is unset), so you can track how many candidates land in each stratum and
adjust thresholds or oversampling accordingly. With `selection_policy: stratified`, the selected‑bin
counts show how evenly the final pool spans strata.

#### Stdout UX for long runs
DenseGen supports three logging styles so long runs stay readable:

- `progress_style: stream` (default) logs per‑sequence updates; tune `progress_every` to reduce noise.
- `progress_style: summary` hides per‑sequence logs and only prints periodic leaderboard summaries.
- `progress_style: screen` clears and redraws a compact dashboard (progress, leaderboards, last sequence)
  at `progress_refresh_seconds`.

For iterative mining workflows, `screen` or `summary` modes are recommended to avoid log spam while still
seeing yield/leaderboard progress over time.

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
