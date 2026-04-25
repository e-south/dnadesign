## study_stress_ethanol_cipro_pdual10 Runbook

Use this runbook after `densegen_prom_eth_cip_source` grows or when the
study-owned shared anchor/context datasets need to be refreshed without losing
existing Construct outputs.

### 1) Refresh the merged anchor dataset without mutating the source datasets

```bash
# Initialize the shared merged-anchor handoff dataset only once.
uv run usr --root src/dnadesign/usr/datasets init \
  usr_prom_eth_cip_anchor \
  --source stress_ethanol_cipro_growth \
  --notes "Merged anchor set for Construct and Infer"

# Preview the DenseGen delta before mutating the shared anchor dataset.
uv run usr --root src/dnadesign/usr/datasets maintenance merge \
  --dest usr_prom_eth_cip_anchor \
  --src densegen_prom_eth_cip_source \
  --union-columns \
  --if-duplicate error \
  --dry-run

# Merge the curated wildtype anchors without mutating their source dataset.
# This is idempotent for the current study handoff and adds 0 rows on refresh.
uv run usr --root src/dnadesign/usr/datasets maintenance merge \
  --dest usr_prom_eth_cip_anchor \
  --src usr_mg1655_promoter_controls \
  --union-columns \
  --if-duplicate error

# Merge the DenseGen study output into the same shared anchor handoff.
uv run usr --root src/dnadesign/usr/datasets maintenance merge \
  --dest usr_prom_eth_cip_anchor \
  --src densegen_prom_eth_cip_source \
  --union-columns \
  --if-duplicate error

# Validate the merged handoff dataset before Construct reads it.
uv run usr --root src/dnadesign/usr/datasets validate \
  usr_prom_eth_cip_anchor \
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
# Plan the Construct refresh without mutating the shared downstream dataset.
# The checked-in config now uses output.on_conflict=ignore, so existing
# contexts are skipped and only new upstream anchors are planned for write.
uv run construct workspace run-project \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 \
  --project forward_anchor_window \
  --dry-run
```

### 4) Materialize the pDual-10 context dataset

```bash
# Materialize the shared Construct context dataset once the dry run is green.
# Existing output ids are preserved; only new Construct contexts are appended.
uv run construct workspace run-project \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 \
  --project forward_anchor_window

# Validate the resulting shared Construct context dataset strictly.
uv run usr --root src/dnadesign/usr/datasets validate \
  construct_prom_eth_cip_context \
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
