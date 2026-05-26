# Study Notes

This directory is the study-owned note surface for
`stress_ethanol_cipro_growth`.

Use it for study-specific interpretation that is too detailed for the top-level
record plane but still worth checking into the repo.

## Scope

- `docs/studies/stress_ethanol_cipro_growth/record/status.md` stays factual:
  current phase, datasets, row counts, downstream posture, and concise next
  actions.
- `docs/studies/stress_ethanol_cipro_growth/routes/README.md` stays a one-hop route
  map for DenseGen, Construct, Infer, LatentDNA, Cluster, and OPAL.
- `docs/studies/stress_ethanol_cipro_growth/operations/runtime/command-groups/pipeline.yaml` stays structural:
  study-owned runtime bindings, execution surfaces, and downstream entry
  points.
- `src/dnadesign/studies/units/stress_ethanol_cipro_growth/deliverables/`
  stays deliverable-scoped, with role-based lanes for gates, primary review,
  native-reference overlays, and appendix material.
- `docs/studies/stress_ethanol_cipro_growth/audits/` stays audit-scoped:
  contract, quality, and sync audits.

This `notes/` surface is where longer interpretation should go.

## What Belongs Here

- commentary on how to read the current LatentDNA views as a study package
- method notes comparing the study to external work such as Goodfire or
  EVOLVEpro
- ranked or tiered candidate-`X` reasoning that goes beyond one plot
- future-step notes for assay-era extensions, geodesic pilots, or new view
  families
- scoped critiques that should persist without bloating the record-plane docs

## Guardrails

- Keep notes study-specific and clearly dated or topic-scoped.
- Do not turn `record/status.md` or `routes/README.md` into narrative interpretation docs.
- Do not restate every deliverable; link to them and add only the extra
  synthesis that needs a home.
- Keep unsupported claims explicit. If assay data does not exist, say so.
- Treat external-method notes as commentary, not as authority over the checked
  study record.

## Current Notes

- Triage: [2026-04-19 LatentDNA pre-assay](triage/2026-04-19-latentdna-preassay.md)
- Audits: [2026-05-09 bidirectional context-anchor mean confidence](audits/2026-05-09-bidirectional-context-anchor-mean-confidence.md)
- Audits: [2026-05-09 candidate-view language prose](audits/2026-05-09-view-language-prose.md)
- Audits: [2026-05-10 native reference processing and ontology](audits/2026-05-10-native-reference-processing-and-ontology.md)
- Rationale: [2026-05-10 candidate-X story surfaces](rationale/2026-05-10-candidate-x-story-surfaces.md)
- Handoffs: [2026-05-15 OPAL batch0 rationale](handoffs/2026-05-15-opal-batch0-rationale.md)
