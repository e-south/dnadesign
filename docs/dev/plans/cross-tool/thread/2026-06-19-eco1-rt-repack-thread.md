---
doc_id: dev-thread-eco1-rt-repack-candidate-review
surface: cross-tool-dev-spec
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-07-11
status: implemented
primary_slice: generation-policy-v3
---

## Eco1 RT Repack Generation And Review

### Study Premise

This study asks whether complete ProteinMPNN-designed Eco1/Ec86 RT sequences
can keep declared catalytic, direct-contact, Wang thumb-track, and mapped
residues 255-311 fixed, preserve local C-alpha backbone geometry, and support
three testable sequence interventions: distal scaffold repacking, acid-free
MSA-supported redesign of the peripheral nucleic-acid-facing shell, or both.

The study does not claim improved activity, affinity, processivity, strand
displacement, or safety.

### Materialized Result

Generation policy v3, ProteinMPNN sampling, ColabFold folding, local-structure
review, panel selection, and the review notebook are materialized under:

```text
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v3/
```

The materialized path is `1007 -> 738 -> 335 distal + 226 peripheral + 177
combined -> 6 core + 2 expansion`: complete sequences, local geometry retained,
policy-defined comparison pools, and deterministic within-policy selection.
Policy assignment defines experimental contrasts; it is not a quality filter.

### Generation Contract

Each ProteinMPNN candidate is a complete sequence generated under one policy.
Mutations from separate policies are never combined.

| Policy | Open positions | Requested sequences |
| --- | --- | ---: |
| `distal_scaffold_repack_v1` | 25 distal positions | 336 |
| `near_dna_rna_acid_free_v1` | 59 peripheral positions | 336 |
| `combined_near_acid_free_plus_distal_v1` | 59 peripheral and 25 distal positions | 336 |

Every policy fixes:

- NAxxH, YADD, and VTG motif contexts;
- retained DNA/RNA contacts at or below 5 A;
- Wang thumb-track positions;
- mapped residues 255-311;
- declared conserved or core positions.

RT1-RT7 intervals are annotation labels, not protection rules. Residues 230-254
remain designable where a policy opens them and stay visible in mutation and
local-structure review.

Peripheral positions are more than 5 A and at most 10 A from retained DNA/RNA.
Their ProteinMPNN `omit_AA_jsonl` entries allow MSA-observed alternatives while
excluding new D/E, P, and G. V3 also uses global `--omit_AAs C`. C233 is open in
the peripheral policies, so its WT cysteine is omitted and it must change. That
recurrence is generation bias, not a protected-position violation or functional
signal.

The six selected peripheral or combined rows change C233 and G254, and five of
those six change K230. These positions are reported as a designable `230-254`
boundary, separate from the fixed `255-311` context. The two distal rows do not
change these positions.

Every generated row carries one policy id, policy version, and policy-manifest
hash. Downstream stages reject missing or mismatched policy provenance.

### Structural Review

Each ColabFold model is aligned once to the 7V9U-backed reference over mapped
C-alpha atoms. Regional C-alpha RMSD is then measured without fitting regions
again.

A sequence passes the local-geometry rule when every named non-distal review
region is at or below the declared 2.5 A cutoff. Distal RMSD, fold confidence,
and global RMSD remain review fields. The cutoff is a study review rule, not a
literature-derived functional boundary.

### Panel Selection

The visible flow separates filtering from comparison-panel construction:

```text
complete ProteinMPNN sequences
-> local-geometry pass
-> distal, peripheral, and combined comparison pools
-> two rows per policy in the core panel
-> one additional peripheral and one additional combined row in the expansion
```

Within each policy, the core pair maximizes mutated-position Jaccard distance,
then exact-substitution Jaccard distance. The expansion row maximizes its
minimum distance from that policy's core pair. Charge-event counts, regional
MSA support, local RMSD, fold metrics, and sequence hash are later tie-breaks.
The fixed `2 + 2 + 2` composition is the declared experimental comparison
design, not evidence that the policies are equivalent quality tiers. The
expansion strengthens the two nucleic-acid-facing interventions without adding
a new category.

Panel-wide distance is reported only as context because policies expose
different positions to ProteinMPNN. Within-policy distance is the selection
metric. The selected set is mutation-set-diverse or partially nonredundant; it
is not described as orthogonal.

Protected-position, direct-contact, Wang-track, acidic-gain, and proximal MSA
checks remove no v3 rows because generation already enforces those conditions.
They remain audited invariants rather than displayed funnel stages.

### Evidence Roles

- Wang and 7V9U support retained nucleic-acid geometry, direct-contact
  protection, and cautious review of the electropositive surface.
- Inouye 1999 supports caution across the broader C-terminal primer-template
  recognition context. Inouye 2004 supports fixing the mapped portion of the
  255-320 primer-RNA recognition fragment.
- Tao supports constraint-first fixed-backbone RT generation followed by
  structural and experimental screening. It does not establish the Eco1
  distance shell, local RMSD cutoff, or functional outcome.
- ProteinMPNN proposes sequences and ColabFold supplies predicted structures.
  Neither validates function.
- ESMC and SAE are optional model annotations and do not select panel rows.

### Review Contract

The selection-readiness manifest links the candidate trace, selected panel,
complete RT protein sequences, plots, and notebook. The quantitative flow plot
uses candidate-count-proportional ribbons and names exclusions at every
transition. Every static visual has a title, description, alt text, and an
interpretation limit. Dropdowns expose only existing, nonempty artifacts.

The canonical execution commands are maintained in
`docs/studies/eco1_rt_repack/operations/runtime/command-groups/README.md` and
`pipeline.yaml`.

### Acceptance Criteria

- Three v3 policy request bundles total 1008 requested sequences.
- Every sample belongs to one policy and carries the v3 policy hash.
- Fixed positions and residue-specific alphabets validate against generated
  sequences.
- No selected row changes a protected, direct-contact, Wang-track, or mapped
  255-311 position.
- The core panel contains two distal, two peripheral, and two combined rows.
- The expansion adds one peripheral and one combined row.
- Within-policy selection uses mutated-position distance before exact-
  substitution distance and does not rank by total mutation count or net
  charge.
- The notebook states the premise and evidence limits directly.
- Plot roles distinguish row selection, review evidence, and optional model
  context.

### Verification

```bash
uv run pytest -q src/dnadesign/thread/tests/adapters/proteinmpnn
uv run pytest -q src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/generation_policies
uv run pytest -q src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness
uv run pytest -q src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables
uv run marimo check src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/review_deliverables/notebooks/eco1_review_deliverables.py
uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .
uv run python -m dnadesign.devtools.docs.checks --repo-root .
git diff --check
```
