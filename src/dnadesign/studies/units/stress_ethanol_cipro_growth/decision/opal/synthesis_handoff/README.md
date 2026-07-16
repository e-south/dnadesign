# Stress OPAL Synthesis Handoff

**Owner:** stress_ethanol_cipro_growth study
**Lifecycle:** round-0 source available; synthesis authorization pending
**Last verified:** 2026-07-15

Study-owned conversion from an OPAL logical selection batch to physical
synthesis artifacts.

## Ownership

OPAL owns model fitting, candidate predictions, selection views, and the
deduplicated logical `selection_batch`. This package owns cloning transforms,
order aliases, manifests, vendor projections, and physical lifecycle records.
Candidate `id` and promoter `sequence` remain unchanged across the boundary.

The measured-round contract is one campaign, one run, and one logical batch.
Target membership is carried by `selection_memberships`; it is not inferred
from campaign names.

## Sources

### Pre-assay batch zero

`study_batch0_selector` preserves the digest-pinned pre-assay 18-row seed. Its
SFXI source-run slugs map explicitly to declared selection-view IDs:

- `secg_ethanol_rf_sfxi_topn` -> `ethanol`
- `secg_cipro_rf_sfxi_topn` -> `ciprofloxacin`
- `secg_and_rf_sfxi_topn` -> `and`

Those slugs are source-provenance labels, not executable OPAL configurations.
Their artifacts and batch-zero exports are stored under the study-owned
`workbench/source_evidence/opal_sfxi_round0/` root.

### Measured OPAL rounds

`opal_selection_batch` loads one v3 campaign run. It loads every declared view
through OPAL's public `load_selection_set` contract, loads the deduplicated
batch, verifies every membership ID/rank/score, and verifies sequence parity
against the campaign records table.

No campaign list, per-campaign run map, implicit view, or duplicate-filling
policy is supported.

## Manifest

Every measured-round row includes campaign, round, run, JSON
`selection_view_ids`, structured `selection_memberships`, canonical candidate
identity, cloning spans, hashes, and physical validation. A candidate selected
by multiple views appears once with every membership.

## Lifecycle Record

The authority is
`docs/studies/stress_ethanol_cipro_growth/record/synthesis_handoffs.yaml`.

The first physical assay batch selected by the RMF campaign has assay index 1
and was chosen by the model fitted at OPAL round 0. Those axes are related but
not interchangeable: `assay_batch_index` counts physical assay batches, while
`model_as_of_round` identifies the label snapshot and model that made the
selection.

If synthesis is authorized, its reviewed handoff record uses this mapping:

```yaml
- handoff_id: stress-opal-assay-b1-r0-rmf-v1
  lifecycle_status: generated_pending_acceptance
  source_authority: opal_selection_batch
  selection_epoch: opal_model_round
  assay_batch_index: 1
  model_as_of_round: 0
  campaign_slug: secg_rmf_greedy
  run_id: r0-2026-07-16T01:32:16+00:00
  strategy_id: stress_promoter_insert:v1
  expected_selection_views:
    - {selection_view_id: ethanol, expected_rows: 6}
    - {selection_view_id: ciprofloxacin, expected_rows: 6}
    - {selection_view_id: and, expected_rows: 6}
  expected_artifact:
    campaign_slug: secg_rmf_greedy
    expected_rows: 18
    manifest_path: <generated-manifest-path>
    vendor_workbook_path: <generated-workbook-path>
    genbank_dir_path: <generated-genbank-directory>
    genbank_feature_table_path: <generated-feature-table-path>
```

The record fails if campaign, run, unique batch count, or per-view membership
counts drift.

## Commands

Inspect the unified selections:

```bash
uv run opal selection-set show \
  -c src/dnadesign/opal/campaigns/secg_rmf_greedy/configs/campaign.yaml \
  --view ethanol --round latest --json

uv run opal selection-batch show \
  -c src/dnadesign/opal/campaigns/secg_rmf_greedy/configs/campaign.yaml \
  --round latest --json
```

Build a draft from one explicit run:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff \
  --source opal-round --round <as_of_round> --run-id <run_id> --write --json
```

Production generation starts from a checked-in lifecycle record:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff \
  --handoff-id <measured_round_handoff_id> --write --json
```

`--source selected-csv` is fixture/debug input and requires explicit JSON
`selection_memberships`. It is not a production authority.

## Physical Validation

The declared strategy wraps an uppercase 60 nt promoter core with lowercase
15 nt flanks. BamHI is allowed only in the left flank and EcoRI only in the
right flank. The assembled insert is rescanned before files are written.

Generated outputs contain a vendor-neutral manifest, Azenta/GeneWiz workbook,
one GenBank file per insert, and a DenseGen feature-table sidecar. DenseGen
TFBS coordinates use `offset`; sigma-70 fixed elements use `offset_raw`.
Neither is a fallback for the other.

Generating files does not authorize synthesis. Acceptance remains an explicit
study-record lifecycle change.
