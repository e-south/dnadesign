---
doc_id: study-eco1-rt-repack-fixed-backbone-method
surface: study-context
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-06-19
---

## Fixed-Backbone Method

This study adapts an AI-guided RT redesign pattern to Eco1 RT for downstream
sponging workflows. The method is computational and contract-first:

1. Choose one structure authority and chain policy.
2. Map every designable residue into a canonical Eco1 RT numbering system.
3. Compose conservative fixed/mutable masks from structure contacts,
   conservation, catalytic policy, and unresolved-residue policy.
4. Generate fixed-backbone sequence samples with a declared MPNN backend.
5. Deduplicate and rank candidates.
6. Validate structural fidelity with declared fold-check metrics.
7. Emit a candidate handoff only when every upstream artifact is present and
   hash-linked.

The motivating source method is Tao et al., Nature Biotechnology 2026,
DOI `10.1038/s41587-026-03149-6`. This study uses the computational pattern,
not the prime-editing objective.

### Method Posture

Treat ProteinMPNN/LigandMPNN output as fold-compatible sequence proposals, not
as proof of improved stability or function. A candidate becomes useful only
after it passes mask audit, deduplication, structural QA, and downstream
promotion checks.

The conservative Eco1 pass asks a narrow question:

```text
Can distal, nonprotected Eco1 RT scaffold positions be repacked while preserving
the mapped catalytic and nucleic-acid-recognition machine?
```

The first pass should not jointly redesign the RT and lnRNA/pretroDNA substrate.
Use a constant downstream substrate context until RT-only candidate behavior is
understood.

### Stage Contracts

| Stage | Input contract | Output contract | Owner |
| --- | --- | --- | --- |
| Structure authority | Selected PDB/mmCIF, chain policy, reference sequence hash. | `BackboneBundle`, `ResidueMap`. | study then `thread` |
| Evidence profiles | Residue map plus MSA/contact source declarations. | `ConservationProfile`, `ContactProfile`. | study policy, `thread` normalization |
| Mask algebra | Evidence profiles plus manual study masks. | `ResidueMaskSet`. | `thread` mechanics, study policy |
| Sampling | Mask set plus backend request. | `ThreadPlan`, `ThreadSample` rows. | `thread` contracts; `infer` optional execution provider |
| Candidate selection | Sample rows plus ranking policy. | `ThreadCandidate` table. | `thread` mechanics, study ranking |
| Fold QA | Candidate table plus fold runtime declaration. | `FoldCheckReport`. | `thread` normalization; `infer` optional execution provider |
| Synthesis feasibility | Accepted full-sequence candidates. | `AssemblyFeasibilityReport`. | `thread` mutation-window QA plus study policy |
| Downstream handoff | Accepted candidates and hashes. | `CandidateHandoff`. | `thread` bundle, study selection policy, then RT-lnRNA acceptance |

Every stage accepts only the previous stage's declared artifact, never an
ad-hoc reconstruction from filenames or transient notebook state.

### Study Boundary

Eco1 profile choices, catalytic protection, structural-source selection, and
candidate-batch policy stay study-owned. Generic mechanics should graduate to
`thread` only after the tracer bullet proves the contract.

### Execution Boundary

Do not implement a generic `thread` package until the tracer bullet has a real
need for executable validation. The first code slice should be a small contract
validator or artifact builder, not a model-running framework.

Use `implementation-roadmap.md` for the exact implementation slice order. That
page is the current owner of code-home, input/output, and negative-path
decisions for the transition from scaffold to executable contracts.
