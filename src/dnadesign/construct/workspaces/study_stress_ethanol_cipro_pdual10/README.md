## study_stress_ethanol_cipro_pdual10 workspace

This is the real study-owned Construct surface for `stress_ethanol_cipro_growth`.

It is not a demo. It assumes:

- the shared USR root is `src/dnadesign/usr/datasets`
- the merged anchor dataset is `promoter/stress_ethanol_cipro_anchor_set`
- the template dataset is `plasmids`
- the pDual-10 template record id is `c4f17db3c2dbc17c5cb32c5eec785ea4f091e51d`
- both projects write into `promoter/stress_ethanol_cipro_construct_contexts`

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
  --project slot_a_window \
  --runtime

uv run construct workspace validate-project \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 \
  --project slot_b_window \
  --runtime

uv run construct workspace run-project \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 \
  --project slot_a_window \
  --dry-run

uv run construct workspace run-project \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 \
  --project slot_b_window \
  --dry-run
```

Materialize the two study projects only after the merged anchor dataset has been
validated strictly and the write target is still `output.on_conflict=error`.
