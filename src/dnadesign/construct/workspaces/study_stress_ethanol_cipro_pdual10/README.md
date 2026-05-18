## study_stress_ethanol_cipro_pdual10 workspace

This is the real study-owned Construct surface for `stress_ethanol_cipro_growth`.

It is not a demo. It assumes:

- the shared USR root is `src/dnadesign/usr/datasets`
- the merged anchor dataset is `usr_prom_eth_cip_anchor`
- the template dataset is `usr_pdual10_plasmid_template`
- the pDual-10 template record id is `55e46be6e47613d42f38607d57b78496423418ee`
- the shared context project writes paired forward and reverse-complement rows into `construct_prom_eth_cip_context`
- the study anchor is always placed on the template forward strand at
  `3574..3666`, with upstream flank `CGCCAGCAACCGGGATCC` and downstream flank
  `GAATTCGCCAGCTGTCACCGGA`

Use this workspace after the study's merged anchor dataset exists.
The checked-in projects are refresh-safe by default: rerunning them against the
same shared datasets skips already-present output ids, appends missing Construct
contexts, and writes sequence-view rows for already-existing semantic variants.
The supporting study record lives under:

- `docs/studies/stress_ethanol_cipro_growth/operations/runtime/pipeline.yaml`
- `docs/studies/stress_ethanol_cipro_growth/record/status.md`

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
validated. The checked-in context config sets `output.on_conflict=ignore`, so
repeated refreshes preserve existing context rows while completing missing
forward or reverse-complement semantic views.
