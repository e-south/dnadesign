---
doc_id: study-eco1-rt-repack-selection-hardening-dev-spec
surface: study-context
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-07-15
status: active-selection-contract
primary_audience:
  - future-agents
  - dnadesign-maintainers
  - study-reviewers
---

## Selection Method Contract

### Premise

This study compares WT Eco1 RT with complete ProteinMPNN-designed sequences
that either repack distal scaffold positions, redesign a non-acidifying,
MSA-supported peripheral nucleic-acid-facing shell, or do both, while keeping
declared catalytic, direct-contact, Wang thumb-track, and mapped residues
255-311 fixed and requiring preserved predicted local backbone geometry.

The selected rows are protein hypotheses. The method does not establish
activity, affinity, processivity, strand displacement, or safety.

### Required Inputs

Selection requires:

1. complete ProteinMPNN sequences with one v3 policy id and matching policy
   manifest hash;
2. accepted ColabFold models with pLDDT and reference-alignment metrics;
3. local C-alpha RMSD by named region after one global mapped fit;
4. canonical mutation positions and exact substitutions;
5. peripheral charge events and regional MSA support;
6. exact F10/R13 substitutions and the Wang R13A evidence-match annotation.

ESMC and SAE are optional model checks. They are not required inputs and do not
select rows.

### Visible Selection Flow

The notebook and manifest show four stages:

1. **Accepted sequences**: complete, provenance-linked ProteinMPNN outputs.
2. **Constraint and local geometry pass**: rows retaining the generation
   invariants and local C-alpha RMSD at or below 2.5 A in every non-distal
   review region.
3. **Design groups**: assign passing rows to distal, peripheral, or combined
   groups by their complete generation policy. This is experimental-design
   assignment, not a quality gate.
4. **Selected panel**: retain two distal, three peripheral, and three combined
   sequences. Within each group, use mutated-position and exact-substitution
   distance, followed by the declared late evidence.

Protected-position checks that remove no rows are recorded as invariants in the
candidate table. The chemistry and support stage remains visible because it
validates the declared generation contract.

### Local Geometry Rule

Each candidate is aligned once to the mapped 7V9U-backed reference with a
Kabsch-style C-alpha fit. Regional RMSD is then measured without fitting each
region separately.

One declared 2.5 A cutoff applies to all non-distal regions:

- catalytic YADD context;
- retron X/NAxxH context;
- retron Y/VTG context;
- Wang thumb-contact track;
- mapped C-terminal primer-RNA recognition context, residues `255-311`;
- peripheral retained DNA/RNA shell.

Distal RMSD is reported for review but does not gate selection. The 2.5 A value
is a study-declared review cutoff, not a literature-derived functional boundary.

### Mutation-Set Selection

Within each policy, the first pair is selected by exhaustive comparison of
all eligible pairs. The first criterion is mutated-position Jaccard distance;
the second is exact-substitution Jaccard distance. The peripheral and combined
third rows then maximize their minimum distance from the first pair in
the same policy.

Later tie-breaks are, in order:

1. fewer peripheral basic losses;
2. fewer peripheral Pro/Gly gains;
3. higher regional MSA support;
4. lower C-terminal and Wang-track RMSD within the gate;
5. fold metrics;
6. sequence hash.

The method is deterministic. It does not claim globally optimal distance or
independent mechanisms. Report shared positions and substitutions within each
policy because cross-policy distances are inflated by different open-position
sets.

The first pair is globally maximal within its group. A three-row group is built
by adding the candidate farthest from that pair; it is not a global search over
all possible triples. In the materialized panel, position distance determines
all identities and exact-substitution distance resolves one distal tie. The
remaining evidence fields are available tie-breakers but did not choose a row.

Policy allocation defines the experimental comparison: two distal, three
peripheral, and three combined rows in one selected panel. These sequence
instances are not biological replicates, and the policy groups are not quality
tiers.

### Charge Interpretation

Peripheral alphabets prevent new D/E residues upstream. Selection reports
basic gains, basic losses, acidic gains, Pro/Gly gains, and net charge change.
These values describe the designed contrast. They do not predict binding or
activity, and the selector does not require a positive-charge quota.

The v3 global no-cysteine rule forces an open WT cysteine such as C233 to
change. C233 is outside the fixed set. Its recurrence is reported as generation
bias and mutation-set overlap, not as evidence of function.

All selected peripheral and combined rows change C233 and G254; five change
K230. The two distal rows do not. These positions belong to the designable
`230-254` boundary, not the fixed `255-311` set. Review figures must show that
boundary separately.

### RT-msDNA Assembly Scope

Wang reports that two Ec86 RT-msDNA protomers form a homodimer through
reciprocal contacts between RT alpha-1 residues F10/R13 and msDNA in the other
protomer. R13A disrupted the interaction while retaining msDNA and the tested
antiphage phenotype. The paper did not test other F10 or R13 substitutions as
general monomerization mutations.

The candidate structures contain one RT chain and do not evaluate oligomeric
state. No selected sequence contains R13A. Exact F10/R13 states and whether a
sequence matches the tested R13A substitution are review fields, not selection
criteria. A monomerization panel would require R13A to be fixed upstream in
every complete generation policy and the resulting sequences to be folded and
reviewed as new candidates.

### Minimal Review Bundle

Core evidence:

- four-stage selection flow;
- candidate local-RMSD distributions with the 2.5 A cutoff;
- selected local RMSD by region;
- selected substitutions across Eco1 RT;
- regional mutation burden, including the designable `230-254` boundary, and peripheral charge events;
- regional MSA support;
- panel-wide and within-policy mutation-position and exact-substitution
  Jaccard distance;
- selected complete sequences and policy provenance.

Threshold sensitivity, sequence-distance-only views, ESMC, SAE, and broad
generation summaries are context. They must not appear as extra selectors.

### Literature Roles

| Source | Role | Limit |
| --- | --- | --- |
| Wang et al. 2022, DOI `10.1038/s41564-022-01197-7`; RCSB `7V9U` | Ec86 RT-msDNA/RNA geometry, direct contacts, electropositive-surface context, and the F10/R13 cross-protomer interface. | R13A disrupted the two-protomer interaction while retaining msDNA and the tested antiphage phenotype. This does not establish the assembly state or function of other F10/R13 substitutions. |
| Inouye et al. 1999, DOI `10.1074/jbc.274.44.31236` | C-terminal 91-residue primer-template recognition context. | Does not define a structural cutoff. |
| Inouye et al. 2004, DOI `10.1074/jbc.M408462200` | `255-320` primer-recognition RNA-binding fragment. | Does not make C-terminal redesign routine. |
| Tao et al. 2026, DOI `10.1038/s41587-026-03149-6` | Constraint-first RT redesign, ProteinMPNN generation, and structural filtering. | Does not validate Eco1 shells, cutoffs, or activity claims. |
| Kabsch 1976/1978, DOI `10.1107/S0567739476001873` and `10.1107/S0567739478001680` | Global rigid-body superposition method. | Does not validate the 2.5 A cutoff. |

### Current State

The full v3 generation, 1007-row candidate pool, ColabFold report, fold review,
selection tables, plots, and eight-row selected panel are materialized. The
panel contains two distal, three peripheral, and three combined sequences.
Current counts and candidate ids remain artifact-owned.
