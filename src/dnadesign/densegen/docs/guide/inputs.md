## Inputs (Stage-A)

This guide explains what DenseGen accepts as Stage-A input and how to think about each type.

If you want exact YAML fields, use [../reference/config.md](../reference/config.md).

### Mental model

Stage-A turns raw input sources into a pool of candidate sites.

- PWM inputs: sample candidates, score with FIMO, deduplicate, retain `n_sites`
- Binding-site tables: ingest directly (no mining)
- Sequence libraries: ingest directly

Subprocess boundary:

1. Inputs are consumed only in Stage-A.
2. Stage-A writes pool parquet files under `outputs/pools/`.
3. Stage-B and solver consume those Stage-A pools; they do not re-read original input files.

### Input types

- **PWM-backed** (`pwm_*`, `pwm_artifact*`)
- **Binding sites** (`type: binding_sites`)
- **Sequence libraries** (`type: sequence_library`, `type: usr_sequences`)

PWM artifact contract:
- [../reference/motif_artifacts.md](../reference/motif_artifacts.md)

### Minimal YAML examples

PWM artifact set:

```yaml
inputs:
  - name: lexA_cpxR_artifacts
    type: pwm_artifact_set
    path: inputs/motif_artifacts
    sampling:
      strategy: stochastic
      n_sites: 500
      mining:
        budget:
          mode: fixed_candidates
          candidates: 5000000
      selection:
        policy: mmr
        alpha: 0.5
        pool:
          relevance_norm: minmax_raw_score
          min_score_norm: 0.75  # active lower bound on score_norm
```

Binding-site table:

```yaml
inputs:
  - name: curated_sites
    type: binding_sites
    path: inputs/densegen_sites.parquet
```

`selection.pool.min_score_norm` has no default. Set it when you want to enforce a
minimum score-normalized quality floor inside the MMR pool.

### Path resolution

Input paths resolve relative to `config.yaml` unless you pass absolute paths.

### PWM-specific highlights

- **Scoring**: FIMO log-odds, forward strand only (`--norc`)
- **Eligibility**: requires a FIMO hit and `best_hit_score > 0`
- **Deduplication**: controlled by `uniqueness.key` (`tfbs` vs `tfbs_core`)
- **Cross-regulator core guard**: for multi-motif PWM inputs, `uniqueness.cross_regulator_core_collisions`
  controls whether same-core collisions across regulators are warned, rejected, or allowed
- **Selection**: `top_score` or `mmr`
- **Score normalization**: `score_norm = best_hit_score / pwm_theoretical_max_score`

For deeper behavior details:
- [sampling guide](sampling.md)

---

@e-south
