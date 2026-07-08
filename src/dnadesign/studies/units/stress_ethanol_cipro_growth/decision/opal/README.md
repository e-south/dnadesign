## OPAL Decision Surfaces

This directory contains study-owned OPAL surfaces. OPAL core remains generic;
study-specific candidate universes, DenseGen-label probes, and biological
guardrails live here.

- `batch0/`: pre-assay candidate-table materialization and provenance review.
- `densegen_axis_probe/`: DenseGen construction-label OPAL probes, including
  the strict TFBS learnability workflow.
- `measured_reader_vec8/`: measured reader SFXI `vec8` staging for round-0 OPAL
  label inputs, including exact candidate/sequence/X validation and campaign
  input CSV writes.
- `synthesis_handoff/`: study-owned physical synthesis handoff for selected
  OPAL promoters, including cloning-strategy transforms, vendor-neutral
  manifests, and vendor export adapters. It wraps the checked-in batch0
  selector for pre-assay ordering and OPAL `selection-set` records for measured
  rounds; physical order batches should be invoked through checked-in
  synthesis handoff ids.
