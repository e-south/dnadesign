## Eco1 RT Repack Vocabulary

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-19

| Term | Meaning |
| --- | --- |
| `thread` | Planned reusable tool for fixed-backbone protein sequence design intent. |
| `eco1_rt_v1` | First Eco1 RT fixed-backbone profile for this study. |
| `BackboneBundle` | Generic structure/reference bundle for fixed-backbone design. |
| `ResidueMap` | Generic residue-numbering map across structure, protein, CDS, and design positions. |
| `ConservationProfile` | Generic per-position conservation evidence mapped to the residue map. |
| `ContactProfile` | Generic per-position structure-contact evidence mapped to retained context atoms. |
| `RtAnnotationTracks` | Study-owned target-position motif annotations that can be rendered by generic `aligner.msa.visualization` sidecars without controlling designability. |
| `MsaExemplarRows` | Study-owned explicit aligned FASTA row selections used to render local motif windows; these rows ground visualization only and are not the denominator for plurality. |
| `MsaPanelSpec` | Study-owned display contract for selected-row overview and plurality/gap histogram panels; this controls figure sidecars, not conservation scoring. |
| `ResidueMaskSet` | Generic mutable/fixed/protected/unresolved mask contract. |
| `ThreadPlan` | Generic fixed-backbone sampling plan with backend, seeds, temperature, and fixed-position policy. |
| `BackendRequestManifest` | Declared backend request hash and execution intent produced by `thread` before any model process runs. |
| `BackendResultManifest` | Declared backend result hash and run metadata ingested by `thread` from direct files or an execution provider such as `infer`. |
| `ThreadSample` | Raw backend output sequence with backend provenance and scores. |
| `ThreadCandidate` | Deduplicated fixed-backbone design candidate with provenance. |
| `FoldCheckReport` | Computational structural QA report for candidate sequences. |
| `AssemblyFeasibilityReport` | Computational report deciding full-gene, bounded-window, sparse recombination, or reject posture. |
| `CandidateHandoff` | Selected candidates plus upstream hashes and downstream target. |
| `RtLnrnaCandidateAcceptance` | Downstream accept/reject record for RT-only candidate handoffs before any construct subject exists. |
| `mask_source` | A named reason that fixes or protects a residue. |
| `window_haplotype` | A bounded coding-window sequence segment derived from one accepted full-protein parent. |
| `nearest_parent_candidate` | The accepted full-protein candidate closest to a recombined sequence. |

### Naming Rules

- Use Eco1 only for profile and study identity.
- Use generic object names for reusable `thread` contracts.
- Use `thread_` prefixes only when a table or schema would otherwise collide
  with another tool's candidate vocabulary.
- Do not use `permuter__var_id` for MPNN samples unless a later import
  contract routes them through Permuter.
- Do not use RT-lnRNA construct-subject ids before the downstream study promotes
  candidates into paired constructs.
- Use `*_profile` for reusable evidence inputs, `*_set` for composed masks,
  `*_report` for evaluation outputs, and `*_handoff` for downstream bundles.
- Use `*_manifest` for backend request/result metadata that points at execution
  artifacts without owning model-process execution.
- Use `*_acceptance` for downstream accept/reject records that do not create
  new domain subjects by implication.
- Use `*_annotation_tracks` for visualization annotations over an already
  selected target coordinate space; these tracks are not evidence profiles or
  mask sources by themselves. Label placement and border/fill styling are
  display grammar only.
- Use `*_exemplar_rows` for explicit row selections in visualization sidecars;
  never infer representative biological rows from FASTA order.
- Use `*_panel_spec` for display-only visualization settings such as selected
  row limits, high-gap trim declarations, and enabled panel types; never use it
  as a conservation denominator or mask source.
