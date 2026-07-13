## OPAL Decision Surfaces

**Owner:** stress_ethanol_cipro_growth study
**Last verified:** 2026-07-13

This directory contains study-owned OPAL surfaces. OPAL core remains generic;
study-specific candidate universes, DenseGen-label probes, and biological
guardrails live here.

- `batch0/`: pre-assay candidate-table materialization and provenance review.
- `densegen_axis_probe/`: DenseGen construction-label OPAL probes, including
  the strict TFBS learnability workflow.
- `reader_promoter_evidence/`: verifies Reader publication bundles and writes
  display-only manifests for the `secg_rmf_greedy` campaign.
- `response_metastudy/`: read-only metric, label, and predictor review over the
  digest-pinned SFXI source ledgers plus Reader's response-window bundle. It
  records the evidence and risks behind the inactive RMF campaign.
- `synthesis_handoff/`: study-owned physical synthesis handoff for selected
  OPAL promoters, including cloning-strategy transforms, vendor-neutral
  manifests, and vendor export adapters. It wraps the checked-in batch0
  selector for pre-assay ordering and the unified campaign's selection views
  and logical batch for measured rounds. Physical order batches are invoked
  through checked-in synthesis-handoff IDs.

Promoter alias, candidate, sequence, and BaseRender routing is owned by the
study-level `promoter_candidate_bindings/` package. OPAL adapters consume that
identity contract; they do not redefine it or add model-feature fields to it.
