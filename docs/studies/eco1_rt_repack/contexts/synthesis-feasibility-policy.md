---
doc_id: study-eco1-rt-repack-synthesis-feasibility-policy
surface: study-context
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-06-30
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
- `sequence_hash`
- `parent_sequence_id`
- `parent_sequence_hash`
- `mutation_count_total`
- `mutation_count_mutable_region`
- `mutation_count_protected_region`
- `protected_mutation_violation_count`
- `protected_mutation_violations_json`
- `mutation_windows_json`
- `max_mutation_window_length`
- `max_mutation_window_mutation_count`
- `mutation_window_density_max`
- `nearest_parent_id`
- `nearest_parent_distance_aa`
- `nearest_parent_distance_fraction`
- `parent_haplotype_id`
- `parent_haplotype_distance_aa`
- `synthesis_tier`
- `synthesis_blockers_json`
- `codon_policy_id`
- `sequence_complexity_flags_json`
- `feasibility_status`
- `feasibility_reason`
- `feasibility_policy_id`
- `input_candidate_table_hash`
- `input_mask_policy_hash`
- `input_foldcheck_report_hash`
- `created_at_utc`
- `created_by`

Allowed `synthesis_tier` values are `easy`, `standard`, `difficult`,
`blocked`, and `unknown`. Allowed `feasibility_status` values are `feasible`,
`review`, `blocked`, and `missing_inputs`.

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

The reviewer-facing sequence handoff should stay flat: selected candidate id,
selection slot, protein sequence, sequence hash, feasibility state, and explicit
DNA-design status belong in `candidate_handoff_sequences.csv`. The CSV is a
protein-sequence table. E. coli codon design, DNA restriction-site screening,
and any hosted sequence-optimization workflow are separate downstream steps and
must not be implied by the protein-only handoff.
