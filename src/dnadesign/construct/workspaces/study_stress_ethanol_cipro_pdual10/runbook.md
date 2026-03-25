## study_stress_ethanol_cipro_pdual10 Runbook

Use this runbook after `densegen/study_stress_ethanol_cipro` and
`mg1655_promoters` have been merged into the study-owned anchor set.

### 1) Bootstrap the merged anchor dataset without mutating the source datasets

```bash
# Initialize the shared merged-anchor handoff dataset.
uv run usr --root src/dnadesign/usr/datasets init \
  promoter/stress_ethanol_cipro_anchor_set \
  --source stress_ethanol_cipro_growth \
  --notes "Merged anchor set for Construct and Infer"

# Merge the curated wildtype anchors without mutating their source dataset.
uv run usr --root src/dnadesign/usr/datasets maintenance merge \
  --dest promoter/stress_ethanol_cipro_anchor_set \
  --src mg1655_promoters \
  --union-columns \
  --if-duplicate error \
  --carry-namespace usr_label

# Merge the DenseGen study output into the same shared anchor handoff.
uv run usr --root src/dnadesign/usr/datasets maintenance merge \
  --dest promoter/stress_ethanol_cipro_anchor_set \
  --src densegen/study_stress_ethanol_cipro \
  --union-columns \
  --if-duplicate error \
  --carry-namespace usr_label

# Validate the merged handoff dataset before Construct reads it.
uv run usr --root src/dnadesign/usr/datasets validate \
  promoter/stress_ethanol_cipro_anchor_set \
  --strict
```

### 2) Validate the Construct workspace against the real study inputs

```bash
# Confirm the checked-in workspace registry and project inventory are consistent.
uv run construct workspace doctor \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10

# Resolve the real template plus input dataset and print the placement contract.
uv run construct workspace validate-project \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 \
  --project forward_anchor_window \
  --runtime
```

### 3) Preview the shared downstream writes

```bash
# Plan the Construct write without mutating the shared downstream dataset.
uv run construct workspace run-project \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 \
  --project forward_anchor_window \
  --dry-run
```

### 4) Materialize the pDual-10 context dataset

```bash
# Materialize the shared Construct context dataset once the dry run is green.
uv run construct workspace run-project \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 \
  --project forward_anchor_window

# Validate the resulting shared Construct context dataset strictly.
uv run usr --root src/dnadesign/usr/datasets validate \
  promoter/stress_ethanol_cipro_construct_contexts \
  --strict
```

The checked-in Construct config enforces one study-owned placement contract:

- anchor orientation must stay `forward`
- the pDual-10 replace interval is `3574..3666`
- the forward-strand upstream flank must be `CGCCAGCAACCGGGATCC`
- the forward-strand downstream flank must be `GAATTCGCCAGCTGTCACCGGA`
- `placement.guards.require_unique_forward_matches: true` rejects repeated-kmer ambiguity

### 5) Continue into Infer and Notify

Hand off to:

- `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/`
- `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_only_7b_batch_with_notify.yaml`
- `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_plus_template_7b_batch_with_notify.yaml`
