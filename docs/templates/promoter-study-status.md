## <study-id>

Replace every placeholder before relying on this file for live-status answers.
If a branch of work is not active, mark it `n/a` explicitly instead of leaving a
placeholder behind.

- Last verified:
- Owner:
- Affiliated dataset registry: `datasets.yaml`
- USR root:
- Target row count:
- Current feature-dataset row count:

### Source datasets

- DenseGen anchor dataset: `<dataset>` (`<rows>` rows, last batch audit: `<path or runbook id>`)
- Wildtype or manual dataset: `<dataset>` (`<rows>` rows)
- Optional construct context dataset: `<dataset>` (`<rows>` rows) or `n/a`

### Canonical downstream datasets

- Anchor-only feature dataset: `<dataset>`
- Construct-expanded feature dataset: `<dataset>` or `n/a`
- Cluster results root: `<path>` or `n/a`
- OPAL config: `<path>` or `n/a`

### Infer matrix status

- `anchor_only`: `pending|dry-run-green|written`
  - config: `<path>`
  - outputs expected: `ll`, `output_layer_mean`, `intermediate_embedding`
- `template_plus_anchor`: `pending|dry-run-green|written`
  - config: `<path>`
  - outputs expected: `seq_mean`, `anchor_mean`
- `full_lane_set`: `pending|dry-run-green|written`
  - config: `<path>`
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
