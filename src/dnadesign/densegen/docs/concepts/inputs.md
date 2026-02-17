## Input concepts

This concept page explains what DenseGen accepts as Stage-A inputs and how those inputs become retained candidate pools. Read it when you need to choose the right input type or diagnose Stage-A pool quality.

### Stage-A mental model
This section summarizes the input-to-pool lifecycle that all input types feed.

1. Load or mine candidates from configured sources.
2. Apply scoring, filtering, deduplication, and selection rules.
3. Write retained pools to `outputs/pools/` for downstream stages.

### Input types
This section maps each supported input type to expected behavior.

- `binding_sites`: direct ingestion of curated site tables.
- `sequence_library` and `usr_sequences`: direct ingestion of sequence collections.
- `pwm_artifact` and related PWM types: mined/scored candidate generation before retention.
- `background_pool`: mined background candidates for contextual placement.

### PWM input behavior
This section highlights the specific semantics that typically affect quality and runtime.

- Candidate mining effort is controlled by `sampling.mining.budget.candidates`.
- Retained pool size is controlled by `sampling.n_sites`.
- Selection policy is controlled by `sampling.selection.policy` (for example `top_score` or `mmr`).
- Deduplication is controlled by `sampling.uniqueness.key` and optional cross-regulator collision rules.

### Path resolution
This section defines how DenseGen resolves input paths and why workspace-local copies are preferred.

- Relative paths are resolved against `config.yaml` location.
- `dense workspace init --copy-inputs` is the safest operator default because it decouples runs from mutable upstream files.
- Absolute paths are accepted but reduce workspace portability.

### Schema and artifact references
This section points to contract-grade docs for exact key semantics and formats.

- Use **[config reference](../reference/config.md)** for exact schema fields.
- Use **[motif artifacts reference](../reference/motif_artifacts.md)** for PWM artifact JSON format.
- Use **[sampling concepts](sampling.md)** for Stage-A selection and Stage-B weighting details.
