## study_stress_ethanol_cipro_pdual10 workspace

This is the real study-owned Construct surface for `stress_ethanol_cipro_growth`.

It is not a demo. It assumes:

- the shared USR root is `src/dnadesign/usr/datasets`
- the merged anchor dataset is `promoter/stress_ethanol_cipro_anchor_set`
- the template dataset is `plasmids`
- the pDual-10 template record id is `c4f17db3c2dbc17c5cb32c5eec785ea4f091e51d`
- the single study project writes into `promoter/stress_ethanol_cipro_construct_contexts`
- the study anchor is always placed on the template forward strand at
  `3574..3666`, with upstream flank `CGCCAGCAACCGGGATCC` and downstream flank
  `GAATTCGCCAGCTGTCACCGGA`

Use this workspace after the study's merged anchor dataset exists.
The supporting study record lives under:

- `docs/studies/promoter/stress_ethanol_cipro_growth/pipeline.yaml`
- `docs/studies/promoter/stress_ethanol_cipro_growth/status.md`

Validate and preview before any write:

```bash
uv run construct workspace doctor \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10

uv run construct workspace validate-project \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 \
  --project forward_anchor_window \
  --runtime

uv run construct workspace run-project \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 \
  --project forward_anchor_window \
  --dry-run
```

Runtime validation prints the incumbent template interval plus the expected and
observed upstream/downstream flanks. The checked-in config also sets
`placement.guards.require_unique_forward_matches: true`, so repeated-kmer ambiguity fails fast
instead of being treated as a valid placement. The workspace registry also
tracks the expected config `job.id`, so a renamed or swapped study config fails
in `workspace doctor` before execution.

Materialize the study project only after the merged anchor dataset has been
validated strictly and the write target is still `output.on_conflict=error`.
