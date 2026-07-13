---
id: stress-ethanol-cipro-growth-opal-candidate-table
title: OPAL candidate table
owner: dnadesign-maintainers
status: active
last_verified: 2026-07-12
audience:
  - operator
  - agent
---

## OPAL Candidate Table Context

Use one shared USR `opal_candidate_feature_table` for the unified OPAL
campaign. Its materialized universe starts with the dense generated promoter
subset from the LatentDNA view and includes measured-reader batch0 additions
that already exist in the same selected LatentDNA view.

The unified campaign uses `data.location.kind: usr`; the checked-in dataset ID
and sidecar paths are authoritative. It does not depend on a campaign-local
copy of `records.parquet`.

Materialized rows:

- 157160 generated promoter candidates.
- 23 measured pDual-10 SFXI reference rows with reader vec8 labels.
- 2 measured pDual-10 control promoter rows: `pDual-10-spyp` and
  `pDual-10-sulAp`.

Native and unrelated reference rows remain review context unless they are
explicitly added to the materialization contract.

The table carries `opal_candidate__source_class`,
`opal_candidate__design_family`, and `opal_candidate__sfxi_ref__collection_id`
so selection filters remain auditable without joining back to LatentDNA.
DenseGen plan/run/hash provenance is normalized from
`usr_prom_eth_cip_anchor/_derived/densegen.parquet` during table
materialization. The same sidecar supplies `densegen__used_tfbs_detail` and
`densegen__required_regulators`; every DenseGen-backed candidate must carry
both fields. The nested TFBS schema is checked against the BaseRender
`densegen_tfbs` adapter contract, so selected sequence rendering does not rely
on a later metadata join. Missing, duplicate, null, or empty TFBS sidecar rows
are hard failures. `opal_candidate__source_class == densegen` is equivalent to
complete DenseGen identity and TFBS metadata; either direction of mismatch is a
hard failure. The two pDual-10 control rows are not DenseGen records, so their
DenseGen fields are null by contract.

Coverage after the 2026-07-12 materialization:

- 157183 DenseGen-backed rows with renderable TFBS metadata.
- 2 explicit non-DenseGen control exemptions.
- 0 partially populated DenseGen metadata rows.

The 2026-07-12 materialization repaired 79505 null
`densegen__used_tfbs_detail` values by reading the authoritative sidecar.
Candidate-table validation rejects that state.

BaseRender uses its public configured adapter surface:

```python
from dnadesign.baserender import render_sequence_panel_image, sequence_panel_config_for_adapter

config = sequence_panel_config_for_adapter("densegen_tfbs")
panel = render_sequence_panel_image(candidate_row, config=config)
```

### Campaign Boundary

- Keep X, Y, labels, model fitting, predictions, and round history in one OPAL
  campaign. Keep target masks and selectors in named selection views.
- Treat observed assay labels as study-level truth, not selection-view truth.
  The SFXI source runs used one shared pool; the RMF campaign will
  ingest one typed response-window sidecar after promotion.
- OPAL run/explain and `ingest-y` read the unified RMF labels through
  `labels.source.kind: usr_sidecar` at
  `_opal/response_window_observed_labels.parquet`.
- The stress config sets `writeback.prediction_records: ledger_only`, so runs keep
  predictions, scores, selections, and batch ledgers campaign-local instead of
  writing prediction history into the shared `records.parquet`.
- Shared `ingest-y` rejects unknown IDs unless explicitly dropped; it does not
  create candidate rows.
- Measured controls are valid observed-label rows but are excluded from
  synthesis-candidate eligibility by exact ID before restriction-site scanning.
- Selected DenseGen sequences render directly from candidate-table
  `densegen__used_tfbs_detail`; null annotations are an integrity failure, not
  an unannotated display state.
- Do not mint one USR dataset per campaign unless a campaign needs a distinct
  candidate universe, a distinct `X`, or a distinct data-validation contract.
- LatentDNA owns the completed evidence for selecting `X`; the study owns
  materializing the table; OPAL owns round state, scoring, active selection,
  and ledgers. RegulonDB/native appendix views are review context, not OPAL
  readiness inputs.

### Target Label Contract

- Source of truth: one append-only response-window label store keyed by stable
  candidate `id`, observed round or batch, Reader artifact identity, reduction
  contract, and eight-component Y schema. The promoted path is
  `usr_prom_eth_cip_opal_candidates/_opal/response_window_observed_labels.parquet`.
- Consumer contract: one campaign fits the shared eight-output phenotype model.
  Ethanol, ciprofloxacin, and AND masks are named selection views over the same
  predictions.
- Write contract: the campaign writes one prediction ledger, view-indexed
  objective scores and selections, and one deduplicated selection batch. It
  does not rewrite the candidate universe or duplicate assay labels.
- Metric boundary: `_opal/observed_labels.parquet` and the SFXI source ledgers
  remain immutable round-0 evidence; they are not RMF inputs.
- Materialization contract: notebook and review summaries are regenerated from
  the label sidecar and campaign ledger. They are derived artifacts, not assay
  truth.
