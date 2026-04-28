## study_stress_ethanol_cipro_pdual10 Runbook

Use this runbook after `densegen_prom_eth_cip_source` grows or when the
study-owned shared anchor/context datasets need to be refreshed without losing
existing Construct outputs.

### 1) Refresh reference core60 and paired reference contexts

Native promoter references remain source-length biological inserts in
`usr_promoter_references`. Construct derives the separate 60 bp analysis-only
core rows from explicit `sigma70_minus35` and `sigma70_minus10` annotations; the
job fails before writes if either site is missing, ambiguous, or imprecise.
Inputs longer than 60 bp are truncated around the midpoint between those two
sigma-site annotations, with retained/clipped/lost feature metadata recorded in
the `derived` overlay. Inputs shorter than 60 bp are expanded only by replacing
the pDual-10 anchor cassette interval (`3574..3666`) with the short promoter
before extracting the 60 bp analysis window. Native reference rows are not
overwritten by either operation.

```bash
# Derive 60 bp reference cores. Under-length references are expanded by
# replacing the pDual-10 anchor cassette interval before extracting the core.
uv run construct workspace run-project \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 \
  --project reference_core60 \
  --dry-run

# Materialize the reference core60 rows after the dry-run reports the expected
# 48 records and zero unexpected collisions.
uv run construct workspace run-project \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 \
  --project reference_core60

# Materialize paired forward and whole-output reverse-complement contexts for
# the reference core60 rows. The construct overlay stores emitted-orientation
# anchor bounds for downstream anchor_mean pooling; reverse-complement bounds
# are already transformed by Construct and must not be transformed again by Infer.
uv run construct workspace run-project \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 \
  --project reference_core60_contexts \
  --dry-run

# Materialize the paired reference contexts after the dry-run reports 96 planned
# rows, one forward and one reverse-complement context per reference core.
uv run construct workspace run-project \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 \
  --project reference_core60_contexts
```

### 2) Refresh the merged anchor dataset without mutating the source datasets

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

# Merge the curated promoter references without mutating their source dataset.
# The current source has some sequence-equivalent incumbent rows in the handoff,
# so use skip plus explicit namespace carry for the rows that survive the merge.
uv run usr --root src/dnadesign/usr/datasets maintenance merge \
  --dest usr_prom_eth_cip_anchor \
  --src usr_promoter_references \
  --union-columns \
  --if-duplicate skip \
  --carry-namespace usr_label \
  --carry-namespace construct_seed \
  --carry-namespace derived \
  --carry-namespace seq_annot \
  --carry-namespace promoter_standard

# Merge reference core60 analysis views after the native references. These are
# distinct 60 bp analysis-only rows and should not collide with native inserts or
# DenseGen rows. A duplicate here is a contract signal that needs review.
uv run usr --root src/dnadesign/usr/datasets maintenance merge \
  --dest usr_prom_eth_cip_anchor \
  --src construct_prom_eth_cip_reference_core60 \
  --union-columns \
  --if-duplicate error \
  --carry-namespace construct \
  --carry-namespace derived \
  --carry-namespace usr_label

# Merge the Reader-backed SFXI pDual-10 DenseGen source cohort into the same
# shared anchor handoff. Use duplicate-error here; duplicate hits should be
# inspected as evidence over existing rows rather than silently skipped.
uv run usr --root src/dnadesign/usr/datasets maintenance merge \
  --dest usr_prom_eth_cip_anchor \
  --src usr_sfxi_pdual10_densegen_promoters \
  --union-columns \
  --if-duplicate error \
  --carry-namespace densegen \
  --carry-namespace usr_label

# Validate the merged handoff dataset before Construct reads it. Use
# namespace-current when old generated Infer overlay parts are present; those
# feature parts are not owned by this anchor assembly step.
uv run usr --root src/dnadesign/usr/datasets validate \
  usr_prom_eth_cip_anchor \
  --registry-mode namespace-current

# Materialize one construct-ready construct_insert sequence view per merged
# anchor row. This does not duplicate native or designed 60 bp rows as
# analysis_window; true derived core rows remain authoritative in
# construct_prom_eth_cip_reference_core60 and are only marked analysis_only in
# this merged handoff.
uv run python -m dnadesign.usr.scripts.materialize_promoter_anchor_sequence_views \
  --dataset usr_prom_eth_cip_anchor \
  --write
```

### 3) Validate the Construct workspace against the real study inputs

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

### 4) Preview the shared downstream writes

```bash
# Plan the Construct refresh without mutating the shared downstream dataset.
# The checked-in config emits paired forward and whole-output reverse-complement
# contexts. It uses output.on_conflict=ignore, so existing base rows are skipped
# while missing semantic variants and sequence-view rows are completed.
uv run construct workspace run-project \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 \
  --project forward_anchor_window \
  --dry-run
```

### 5) Materialize the pDual-10 context dataset

```bash
# Materialize the shared Construct context dataset once the dry run is green.
# Existing output ids are preserved; missing forward and reverse-complement variants are appended and
# planned sequence views are written even when the base row already existed.
uv run construct workspace run-project \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 \
  --project forward_anchor_window

# Project DenseGen metadata from the merged anchor onto context rows by anchor id.
# This keeps archive-backed SFXI DenseGen annotations visible after context
# realization while leaving promoter-reference rows with null DenseGen fields.
uv run usr --root src/dnadesign/usr/datasets maintenance overlay-project \
  --src usr_prom_eth_cip_anchor \
  --dest construct_prom_eth_cip_context \
  --namespace densegen \
  --src-join id \
  --dest-join construct__anchor_id \
  --allow-missing

# Validate the resulting shared Construct context dataset. Use namespace-current
# when old generated Infer overlay parts are present; construct and carried
# source namespaces are validated against the active namespace contracts.
uv run usr --root src/dnadesign/usr/datasets validate \
  construct_prom_eth_cip_context \
  --registry-mode namespace-current
```

The checked-in Construct config enforces one study-owned placement contract:

- anchor orientation must stay `forward`
- the pDual-10 replace interval is `3574..3666`
- the forward-strand upstream flank must be `CGCCAGCAACCGGGATCC`
- the forward-strand downstream flank must be `GAATTCGCCAGCTGTCACCGGA`
- `placement.guards.require_unique_forward_matches: true` rejects repeated-kmer ambiguity

### 6) Continue into Infer and Notify

Hand off to:

- `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/`
- `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_sequence_views_anchor_construct_insert_7b_batch_with_notify.yaml`
- `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_sequence_views_context_forward_anchor_mean_7b_batch_with_notify.yaml`
- `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_sequence_views_context_reverse_complement_anchor_mean_7b_batch_with_notify.yaml`
