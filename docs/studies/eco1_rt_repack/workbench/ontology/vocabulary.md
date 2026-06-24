## Eco1 RT Repack Vocabulary

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-23

| Term | Meaning |
| --- | --- |
| `thread` | Planned reusable tool for fixed-backbone protein sequence design intent. |
| `eco1_rt_v1` | First Eco1 RT fixed-backbone profile for this study. |
| `BackboneBundle` | Generic structure/reference bundle for fixed-backbone design. |
| `ResidueMap` | Generic residue-numbering map across structure, protein, CDS, and design positions. |
| `ConservationProfile` | Generic per-position conservation evidence mapped to the residue map. |
| `ContactProfile` | Generic per-position structure-contact evidence mapped to retained context atoms. |
| `StructurePreprocessingManifest` | Study-owned runtime manifest that records raw 7V9U dimer context, selected ec86kit protomer-1 model, retained protein/DNA/RNA chains, excluded paired-protomer context, explicit dimerization non-objective, and upstream hashes before geometry is computed. |
| `ContactGeometryProfile` | Study-owned per-position atom-class geometry evidence from the selected 7V9U protomer context. It records side-chain, backbone, DNA/RNA split, contact-density, and retained-chain-count measurements; it is evidence for mask policy, not a mask by itself. |
| `ContactRiskProfile` | Study-owned evidence review that joins contact, contact-geometry, conservation, manual-mask, and Wang candidate-prior evidence. This profile is not a mask by itself. |
| `Clade9Plurality25DirectContact5aPolicy` | Selected simple Eco1 mask policy. It protects NAxxH/YADD/VTG, Wang/Ec86 direct substrate-contact priors, positions evolutionarily conserved at >=25% WT plurality in the Ec86 clade 9 MSA, and mapped residues within 5 A of retained DNA/RNA. |
| `NonFixedMissingBackbone` | Terminal Eco1 positions 1, 2, and 312-320. They are not protected by `eco1_rt_clade9_plurality25_direct_contact5a_v1`, but fixed-backbone ProteinMPNN cannot mutate them directly until coordinates are supplied or handled separately. |
| `RtAnnotationTracks` | Study-owned target-position motif annotations that can be rendered by generic `aligner.msa.visualization` sidecars without controlling designability. |
| `MsaExemplarRows` | Study-owned explicit aligned FASTA row selections used to render local motif windows; these rows ground visualization only and are not the denominator for plurality. |
| `MsaPanelSpec` | Study-owned display contract for selected-row overview and plurality/gap histogram panels; this controls figure sidecars, not conservation scoring. |
| `ManualMaskAuthority` | Study-owned mask ontology and generated runtime artifact. It records audited motif anchors, RT1-RT7 annotation/review spans, and Wang/Ec86 substrate-contact priors. RT1-RT7 spans do not blanket hard-fix residues under `eco1_rt_clade9_plurality25_direct_contact5a_v1`. |
| `MaskRowAlgebra` | Study-local executable contract for composing protected, non-fixed mapped, and non-fixed missing-backbone rows under `eco1_rt_clade9_plurality25_direct_contact5a_v1`. Implemented under `operations/masking/`, not inside a runtime writer. |
| `EvidenceReviewArtifacts` | Contact-risk and surface-accessibility artifacts that can help explain structure context. They do not protect or release residues under `eco1_rt_clade9_plurality25_direct_contact5a_v1`. |
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
- Use `*_geometry_profile` for measured structural geometry that informs later
  policy, distinct from `*_risk_profile` audit summaries and distinct from
  `*_set` mask decisions.
- Use `*_preprocessing_manifest` for raw-source-to-selected-model provenance
  before derived structural measurements are accepted.
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
- Use `*_mask_authority` for study-owned records that are allowed to fix
  residues directly. Do not use annotation-track or panel-spec names for
  designability decisions.
- Use `candidate_prior_*` or `*_candidate_priors` for structural residues that
  should inform a later mask review but are not allowed to set
  `manual_mask=true`.
- Use `*_risk_profile` for evidence-review artifacts that classify upstream evidence and
  missing measurements; never use a risk profile name as a direct mask or
  sampling-plan authority.
