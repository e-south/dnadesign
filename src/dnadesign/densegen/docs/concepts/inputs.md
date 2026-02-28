## Input model

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-28
This concept page explains what DenseGen accepts as Stage-A inputs and how those inputs become retained candidate pools. Read it when you need to choose the right input type or diagnose Stage-A pool quality.

### How Stage-A builds input pools

1. Load or mine candidates from configured sources.
2. Apply scoring, filtering, deduplication, and selection rules.
3. Write retained pools to `outputs/pools/` for downstream stages.

### Input types

- `binding_sites`: direct ingestion of curated site tables.
- `sequence_library` and `usr_sequences`: direct ingestion of sequence collections.
- `pwm_artifact` and related PWM types: mined/scored candidate generation before retention.
- `background_pool`: mined background candidates for contextual placement.

### PWM input behavior

- Candidate mining effort is controlled by `sampling.mining.budget.candidates`.
- Retained pool size is controlled by `sampling.n_sites`.
- Selection policy is controlled by `sampling.selection.policy` (for example `top_score` or `mmr`).
- Deduplication is controlled by `sampling.uniqueness.key` and optional cross-regulator collision rules.

### Path resolution

- Relative paths are resolved against `config.yaml` location.
- `dense workspace init --copy-inputs` is the safest operator default because it decouples runs from mutable upstream files.
- Absolute paths are accepted but reduce workspace portability.

### Related schema and artifact docs

- Use **[config reference](../reference/config.md)** for exact schema fields.
- Use **[motif artifacts reference](../reference/motif_artifacts.md)** for PWM artifact JSON format.
- Use **[sampling concepts](sampling.md)** for Stage-A selection and Stage-B weighting details.
- Use **[Cruncher to DenseGen PWM handoff](../howto/cruncher_pwm_pipeline.md)** for end-to-end motif artifact handoff commands.
