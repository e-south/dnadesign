---
doc_id: study-eco1-rt-repack-synthesis-feasibility-policy
surface: study-context
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-06-19
---

## Synthesis Feasibility Policy

This page records computational feasibility only. It is not a wet-lab assembly
protocol and not a Construct placement contract.

### Tiers

| Tier | Use when | Gate |
| --- | --- | --- |
| Full-gene candidate | Mutations are dispersed, epistatic context matters, or no bounded window is validated. | Candidate is validated as the full protein sequence. |
| Bounded-window candidate | Mutations collapse into one or two explicit windows with stable flanks and preserved haplotypes. | Every proposed full sequence is represented, not an unvalidated Cartesian recombination. |
| Sparse recombination panel | Windows are structurally independent and the enumerated panel is bounded. | Every enumerated full sequence passes cheap QA and a representative fold-check panel. |

### Assembly Feasibility Report

The report is a computational planning artifact. It must include:

- `candidate_id`
- `full_sequence_hash`
- `mutation_count`
- `mutation_windows`
- `window_ids`
- `window_haplotype_ids`
- `nearest_parent_candidate_id`
- `distance_to_nearest_parent`
- `structural_coupling_flags`
- `recommended_synthesis_tier`
- `rejection_reason`

Window ids are derived from accepted full-sequence candidates. They are not a
license to generate an unlimited Cartesian library.

### No-Go Signals

- Variable residues are distributed across many distant windows.
- A recombined pooled candidate was never sampled or structurally checked as a
  full sequence.
- A window overlaps protected catalytic/contact residues.
- Flanks or guards are not unique enough for unambiguous computational
  placement.
- Downstream RT-lnRNA construct promotion cannot preserve the RT CDS sequence
  authority.
- Candidate windows contact each other in the selected structure and the
  recombination plan breaks parent haplotypes.
- Candidate ids cannot be traced back to full-protein MPNN samples and
  fold-check rows.

### Owner Boundary

`thread` may assess mutation windows, nearest-parent distance, parent
haplotypes, and structural coupling for accepted full-protein candidates. The
decision to pay for Eco1 full-gene or bounded-window candidates is study
policy. Generic sequence realization, named slots, flanks, scars, and
placement/window feasibility belong to Construct when they become reusable.

### Handoff Rule

The first RT-lnRNA downstream handoff should carry RT-only candidate sequences
and their provenance. It should not imply that a construct subject exists until
the downstream study explicitly binds the RT to an lnRNA/TF-sponging context.
