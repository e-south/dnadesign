## Inputs (Stage-A)

DenseGen Stage-A builds TFBS pools from inputs. This guide stays high level; use the config
reference for exact schema details.

### Input types

- **PWM-backed** (`pwm_*`, `pwm_artifact*`)
  Stage-A samples candidate cores, scores them with FIMO, deduplicates, and retains `n_sites`.
- **Binding sites** (`type: binding_sites`)
  Stage-A ingests a table of sites; no mining.
- **Sequence libraries** (`type: sequence_library`, `type: usr_sequences`)
  Stage-A ingests sequences as provided (used as solver seeds in some workflows).

For PWM artifact contracts, see `reference/motif_artifacts.md`.

### Minimal examples

PWM artifacts:

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
          min_score_norm: 0.85  # report-only reference
```

Binding sites:

```yaml
inputs:
  - name: curated_sites
    type: binding_sites
    path: inputs/densegen_sites.parquet
```

`selection.pool.min_score_norm` has no default; set it explicitly if you want the report-only
"within tau of theoretical max" reference.

### Path resolution

All input paths resolve relative to the config file location unless you pass an absolute path.

### PWM Stage-A highlights

- **Scoring**: FIMO log-odds, forward strand only (`--norc`). Cores are treated as bricks
  and can be placed in either orientation later. If `sampling.bgfile` is set, DenseGen uses
  that background for theoretical max, `score_norm`, and MMR weights.
- **Eligibility**: candidate must have a FIMO hit and `best_hit_score > 0`.
- **Deduplication**: `uniqueness.key` controls whether uniqueness is by `tfbs` or `tfbs_core`.
- **Selection**: `top_score` or `mmr`. MMR uses a score normalization
  (`selection.pool.relevance_norm`) plus weighted-Hamming diversity on `tfbs_core`.
- **Score normalization**: `score_norm = best_hit_score / pwm_theoretical_max_score` is recorded
  for cross-TF comparability. `selection.pool.min_score_norm` is a report-only
  "within tau of theoretical max" reference.

For Stage-A sampling semantics and MMR details, see `guide/sampling.md`.
For the complete schema, see `reference/config.md`.

---

@e-south
