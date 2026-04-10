## <study-id>

Replace every placeholder before relying on this file for current-status checks.
If a branch of work is not active, mark it `n/a` explicitly instead of leaving a
placeholder behind.

- Last verified:
- Owner:
- Affiliated dataset registry: `datasets.yaml`
- USR root:
- DenseGen source row target:
- Current infer-bearing shared handoff datasets: `<dataset>`, `<dataset>` or `n/a`
- Canonical consolidated feature dataset: `<dataset>` or `n/a`
- Current consolidated feature-dataset row count:

Treat the DenseGen source target as an early-phase gate, not as the universal
headline forever. Once the study has advanced and the shared infer-bearing
handoff datasets already exceed that threshold, keep the target as historical
context and report the current phase from the live handoff plane.

### Source datasets

- DenseGen anchor shared dataset: `<dataset>` (source-growth plane; use `promoter-study-status` for live rows and target gap, last batch audit: `<path or runbook id>`)
- Wildtype or manual dataset: `<dataset>` (`<rows>` rows)
- Construct template seed dataset: `<dataset>` (`<rows>` rows) or `n/a`

### Shared infer-bearing handoff datasets

- Anchor-only handoff dataset: `<dataset>` or `n/a`
- Construct-expanded handoff dataset: `<dataset>` or `n/a`

### Planned consolidated outputs

- Canonical full-lane feature dataset: `<dataset>` or `n/a`
- Cluster results root: `<path>` or `n/a`
- OPAL config: `<path>` or `n/a`

### Infer matrix status

- `anchor_only`: `pending|dry-run-green|written`
  - config: `<path>`
  - outputs expected: `ll`, `output_layer_mean`, `intermediate_embedding`
- `anchor_plus_template`: `pending|dry-run-green|written`
  - config: `<path>`
  - outputs expected: `ll`, `output_layer_mean`, `intermediate_embedding`
- `full_lane_set`: `pending|dry-run-green|written`
  - config: `<path>`
  - outputs expected: `ll`, `output_layer_mean`, `intermediate_embedding`
  - model lanes: `evo2_7b`, `evo2_20b`

### Rollback and maintenance

- Infer reset: `uv run infer prune --usr <dataset> --usr-root <usr-root>`
- Infer namespace archive: `uv run usr maintenance overlay-remove <dataset> --namespace infer --mode archive`
- DenseGen overlay compaction: `uv run usr maintenance overlay-compact <dataset> --namespace densegen`

### Batch and notify

- DenseGen batch route: `<ops runbook path or qsub command>`
- Infer batch route: `<ops runbook path or qsub command>`
- Notify profile or config: `<path>`
- Watch command: `uv run notify usr-events watch --events <usr-root>/<feature-dataset>/.events.log --dry-run --no-advance-cursor-on-dry-run`

### Next actions

- `<action>`
- `<action>`
