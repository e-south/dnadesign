## study_stress_ethanol_cipro_pdual10 workspace

This is the real study-owned Construct surface for `stress_ethanol_cipro_growth`.

It is not a demo. It assumes:

- the shared USR root is `src/dnadesign/usr/datasets`
- the merged anchor dataset is `usr_prom_eth_cip_anchor`
- the template dataset is `usr_pdual10_plasmid_template`
- the pDual-10 template record id is `c4f17db3c2dbc17c5cb32c5eec785ea4f091e51d`
- the single study project writes into `construct_prom_eth_cip_context`
- the study anchor is always placed on the template forward strand at
  `3574..3666`, with upstream flank `CGCCAGCAACCGGGATCC` and downstream flank
  `GAATTCGCCAGCTGTCACCGGA`

Use this workspace after the study's merged anchor dataset exists.
The checked-in project is refresh-safe by default: rerunning it against the
same shared context dataset skips already-present output ids and appends only
new Construct contexts.
The supporting study record lives under:

- `docs/studies/stress_ethanol_cipro_growth/pipeline.yaml`
- `docs/studies/stress_ethanol_cipro_growth/status.md`

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
validated strictly. The checked-in config now sets
`output.on_conflict=ignore`, so repeated refreshes preserve existing context
rows instead of failing on already-materialized output ids.
