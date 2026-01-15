# Inputs

DenseGen accepts four input types. Choose one per entry in `densegen.inputs`.
Sample input files live in:
- `src/dnadesign/densegen/runs/_campaigns/template/inputs` (binding sites)
- `src/dnadesign/densegen/runs/_campaigns/pwm_demo/inputs` (PWM demos)

## Binding site table (`type: binding_sites`)

Use a CSV or Parquet table with regulator and binding-site sequences.

Required columns (override via `columns`):
- `regulator` (default column: `tf`)
- `sequence` (default column: `tfbs`)

Optional columns:
- `site_id`
- `source`

Strict rules:
- Regulator + sequence must be non-empty.
- Duplicate regulator/sequence pairs are errors.
- Sequences must be A/C/G/T only (DNA_4).

Example:

```yaml
inputs:
  - name: demo
    type: binding_sites
    path: inputs/tf2tfbs_mapping_cpxR_LexA.csv
    format: csv
```

## Sequence library (`type: sequence_library`)

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

## PWM MEME (`type: pwm_meme`)

Use a MEME-format PWM file and explicitly sample binding sites.

Required sampling fields:
- `strategy`: `consensus | stochastic | background`
- `n_sites`: number of binding sites to generate per motif
- `score_threshold` **or** `score_percentile` (exactly one)
- `oversample_factor`: oversampling multiplier for candidate generation

Notes:
- `consensus` requires `n_sites: 1`.
- `background` selects **low-scoring** sequences from the PWM.

Example:

```yaml
inputs:
  - name: pwm_demo
    type: pwm_meme
    path: inputs/pwm_demo.meme
    motif_ids: [DemoTF]
    sampling:
      strategy: stochastic
      n_sites: 200
      oversample_factor: 5
      score_percentile: 90
```

## PWM JASPAR (`type: pwm_jaspar`)

Use a JASPAR PFM file and explicitly sample binding sites.

Required sampling fields are identical to `pwm_meme`.

Example:

```yaml
inputs:
  - name: pwm_demo
    type: pwm_jaspar
    path: inputs/pwm_demo.jaspar
    motif_ids: [DemoTF]
    sampling:
      strategy: background
      n_sites: 200
      oversample_factor: 5
      score_percentile: 10
```

## PWM matrix CSV (`type: pwm_matrix_csv`)

Use a CSV matrix with `A,C,G,T` columns (override via `columns`) and a single motif ID.

Required sampling fields are identical to `pwm_meme`.

Example:

```yaml
inputs:
  - name: pwm_demo
    type: pwm_matrix_csv
    path: inputs/pwm_matrix_demo.csv
    motif_id: DemoTF
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

## USR sequences (`type: usr_sequences`)

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

## Path resolution

All relative paths resolve against the **config file directory**.

## Interaction with constraints

- Promoter constraints are fixed motifs and are enforced by the optimizer.
- `side_biases` motifs must exist in the sampled library (DenseGen fails fast if missing).
- `required_regulators` (per plan item) must appear in the sampled library and in each solution.
