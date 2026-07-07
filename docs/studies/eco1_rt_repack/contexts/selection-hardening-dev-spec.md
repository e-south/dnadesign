---
doc_id: study-eco1-rt-repack-selection-hardening-dev-spec
surface: study-context
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-07-07
status: planning
primary_audience:
  - future-agents
  - dnadesign-maintainers
  - study-reviewers
---

## Selection Hardening Dev Spec

This spec defines the next hardening pass for the Eco1 RT repack selection
notebook and panel contract. It is a study-owned development spec, not a
generated artifact and not a candidate-handoff record.

### Intent

Harden the six-candidate Eco1 RT protein review panel around the smallest
defensible premise:

```text
Preserve catalytic and branch-recognition machinery.
Preserve direct retained DNA/RNA contacts.
Review peripheral near-DNA/RNA and thumb-contact changes cautiously.
Treat distal redesign as scaffold/fold context.
Keep ESMC and SAE as annotations, not selectors.
```

The study may use strand-displacement and processivity as motivation for later
assays. It must not claim that the current panel proves either phenotype.

### Literature Roles

Use the cited literature in narrow roles:

| Source | Role in this study | Boundary |
| --- | --- | --- |
| Tao et al. 2026, Nature Biotechnology, DOI `10.1038/s41587-026-03149-6` | Precedent for constraint-first RT redesign with ProteinMPNN, protected functional regions, and fold-model filtering with pLDDT/RMSD. | Supports the redesign-and-filter pattern; does not provide an Eco1 strand-displacement predictor. |
| Wang et al. 2022, Nature Microbiology, DOI `10.1038/s41564-022-01197-7` | Structural authority for Ec86 RT, retained DNA/RNA, electropositive substrate-facing surface, X/NAxxH and Y/VTG context, and thumb/contact annotations. | Supports region definitions and contact caution; does not prove which Eco1 substitutions improve processivity. |
| Inouye et al. 1999, Journal of Biological Chemistry, DOI `10.1074/jbc.274.44.31236` | Primary evidence that RT-Ec86 recognizes specific primer-template RNA structures and that the C-terminal 91 residues are important for Ec86 primer-template recognition. | Supports thumb/C-terminal specificity caution; does not define a safe redesign threshold. |
| Inouye et al. 2004, Journal of Biological Chemistry, DOI `10.1074/jbc.M408462200` | Primary evidence that the RT-Ec86 255-320 C-terminal fragment binds primer-recognition RNA. | Supports tracking the C-terminal/thumb domain as a specificity context; does not make thumb mutation a conservative default. |
| Lim and Maas 1989, Cell, DOI `10.1016/0092-8674(89)90693-4` | Primary retron/msDNA evidence for a covalently linked branched DNA-RNA compound in E. coli B. | Background for retron branch-linked product biology; not an Eco1 RT redesign rule. |
| Inouye and Inouye 1992, Journal of Bacteriology, DOI `10.1128/jb.174.8.2419-2424.1992` | Primary review-style source for retrons and multicopy single-stranded DNA. | Background for retron biology; not a candidate-selection metric. |
| Inouye and Inouye 1991, Annual Review of Microbiology, DOI `10.1146/annurev.mi.45.100191.001115` | Review source for msDNA and bacterial reverse transcriptase context. | Background source only; not a mask authority. |
| RCSB PDB `7V9U` | Coordinate source for the Ec86 RT-msDNA-RNA structure and chain identity. | Coordinate authority only; not a design score. |
| Kabsch-style RMSD superposition literature | Precedent for optimal rigid-body superposition before comparing C-alpha displacements. | Supports the geometry calculation; threshold values still need study-specific justification. |
| Enzyme-design and motif-scaffolding literature | General precedent for protecting functionally important sites and substrate/catalytic geometry. | Supports local-site caution; does not define Eco1 RT-specific thresholds. |

Useful source links:

- Tao et al.: <https://www.nature.com/articles/s41587-026-03149-6>
- Wang et al.: <https://www.nature.com/articles/s41564-022-01197-7>
- Inouye et al. 1999: <https://doi.org/10.1074/jbc.274.44.31236>
- Inouye et al. 2004: <https://doi.org/10.1074/jbc.M408462200>
- Lim and Maas 1989: <https://doi.org/10.1016/0092-8674(89)90693-4>
- Inouye and Inouye 1992: <https://doi.org/10.1128/jb.174.8.2419-2424.1992>
- Inouye and Inouye 1991: <https://doi.org/10.1146/annurev.mi.45.100191.001115>
- RCSB 7V9U: <https://www.rcsb.org/structure/7V9U>
- RMSD superposition background: <https://link.springer.com/article/10.1186/1471-2105-11-363>
- Functionally important site design context: <https://proceedings.mlr.press/v235/song24k.html>

### Plain Region Vocabulary

Use plain names in notebooks, captions, legends, and reviewer-facing docs.

| Current term | Preferred user-facing term | Meaning |
| --- | --- | --- |
| near retained DNA/RNA annulus | near retained DNA/RNA region | Mapped residues `>5 A` and `<=10 A` from retained DNA/RNA after excluding protected motifs, direct-contact residues, and Wang thumb-contact-track positions. |
| substrate-proximal annulus | near retained DNA/RNA region | Older shorthand for the same distance-defined region. |
| thumb-contact track | Wang thumb-contact track | Explicit Wang/Ec86 positions `238,239,240,249,257,261,264,298`. |
| C-terminal primer-RNA recognition region | C-terminal primer-RNA recognition region | Eco1/Ec86 C-terminal context motivated by RT-Ec86 primer-RNA recognition studies, especially mapped residues `255-311` in the 7V9U-backed fixed-backbone scope; canonical `312-320` are missing backbone in this structure. |
| branch-recognition machinery | retron motif and initiation context | NAxxH/X, VTG/Y, YADD/catalytic context, and retained nucleic-acid-contact geometry. If this phrase is used, the exact residue basis must be shown. |

Do not use "annulus" in new user-facing prose unless a figure also shows the
distance-shell definition. Existing machine ids can be migrated in a deliberate
artifact-regeneration slice; do not add aliases or compatibility shims just to
hide old labels.

### Required Region Semantics

The next implementation pass must keep these review axes separate:

| Axis | Required treatment |
| --- | --- |
| Catalytic/retron motif contexts | Hard preservation gate. Use explicit residue windows and local RMSD gates. |
| Direct retained DNA/RNA contacts | Hard preservation gate. Nonzero edits are ineligible for the main panel. |
| Wang thumb-contact track | Separate metric and plot column. Do not merge into the near retained DNA/RNA region. |
| Near retained DNA/RNA region | Primary peripheral review window. Evaluate mutation count, MSA support, charge changes, and chemistry warnings. |
| Distal scaffold | Fold/scaffold context. Do not describe distal edits as direct processivity tuning. |

The thumb-contact track is relevant enough to track even when the selected six
have zero thumb-track substitutions. A zero count is a result, not a failure:
the notebook should state that the current selected panel does not directly test
thumb-track substitution effects.

The C-terminal/thumb domain is also a cognate-RNA recognition context, not just
a generic processivity clamp. Inouye et al. 1999 supports treating the broader
RT-Ec86 C-terminal region as important for primer-template recognition, and
Inouye et al. 2004 supports tracking the 255-320 C-terminal fragment as a
primer-recognition RNA-binding context. In the current fixed-backbone scope,
only mapped residues `255-311` can be directly sampled; canonical residues
`312-320` are missing backbone and remain outside direct ProteinMPNN redesign.

### Local RMSD Method

Local RMSD must remain a structure-preservation screen, not a functional
prediction.

The gate method should be:

1. Parse reference and candidate C-alpha coordinates by canonical Eco1 position.
2. Fit each candidate to the Ec86/7V9U-backed reference once over shared mapped
   RT C-alpha positions using a Kabsch-style rigid-body superposition.
3. After that global mapped fit, compute regional residual C-alpha RMSD,
   mean displacement, and max displacement for each declared region.
4. Gate on regional residuals against declared thresholds.
5. Record the threshold policy id, threshold values, residue list, coordinate
   scope, source basis ids, and pass/fail reason for every candidate x region
   row.

Do not fit each local region independently for the selection gate. Per-region
fitting can make a shifted local patch look preserved by hiding the local shift
relative to the RT scaffold. If a local-fit diagnostic is ever added, label it
as diagnostic-only and keep it out of the selection gate.

### Local RMSD Threshold Policy

Thresholds are allowed only when the notebook shows enough distribution context
to make them auditable.

Required threshold evidence:

- candidate-pool local RMSD distributions by region;
- selected-row positions over those distributions;
- threshold lines and pass/fail counts by region;
- threshold sensitivity under at least one tighter and one looser setting;
- explicit residue lists and source basis for each region;
- failure reasons for every threshold-exceeded row.

Acceptance rule:

```text
All candidates must have local-structure rows for every declared region.
Missing local-structure metrics are missing inputs.
Threshold-exceeded rows are ineligible for the main six-row panel.
Selected rows must pass every declared local RMSD threshold.
```

Do not treat "metric available" as preservation. Availability is a precondition;
the threshold result is the gate.

### Chemistry Warning Smell

High near-retained-DNA/RNA chemistry-warning counts are a review smell. They do
not automatically invalidate a candidate unless the warning touches a hard-gate
region, but they must be explained by within-class comparison.

The next selection notebook should answer:

```text
Did the selected row have fewer chemistry warnings than close alternatives?
Were warnings mostly distal or near retained DNA/RNA?
Were acidic gains and basic losses separated from neutral/polar substitutions?
Did MSA support justify accepting any risky peripheral substitution?
```

If selected candidates still carry many warnings, add a per-class "why this row"
strip before adding more plots. The strip should show the selected row and the
next 3-5 eligible within-class alternatives across the actual lexicographic
fields:

```text
MSA support
unobserved substitutions
near retained DNA/RNA chemistry warnings
near retained DNA/RNA mutation burden
Wang thumb-contact-track mutation count
local RMSD max and key regional RMSDs
global fold metrics
sequence distance from already selected rows
```

Do not collapse these fields into a single composite score.

### Design Class Non-Contrivance Check

Design classes should read as deliberate mask-policy contrasts, not arbitrary
slots.

Add or keep one view that shows, for every design class:

- the conservation profile and threshold;
- the contact-distance rule;
- fixed and mutable residue counts;
- mutable residues in the near retained DNA/RNA region;
- mutable Wang thumb-contact-track residues;
- expected design intent in one plain sentence;
- selected candidate id and whether the selected row actually exercises the
  intended regional contrast.

If a design class does not create a distinct review hypothesis after gates and
selection, mark it as a removal or merge candidate in the next planning pass.
The contract can remain one representative per declared class, but the classes
must earn their place as design-policy contrasts.

#### Current Mask Orthogonality Audit

The current six classes are useful protection-policy contrasts, but they are not
six independent biological surfaces. Four classes are nested clade-9 contact
shells: 6 A, 8 A, and 10 A only remove mutable residues from the 5 A baseline.
The 50% clade-9 class adds 16 mutable mapped positions to the 5 A baseline,
while the II-A3/`42_1` 50% class adds 13 and loses 18 relative to the baseline.

| Design class | Mutable mapped residues | Mutable near retained DNA/RNA residues | Mutable Wang thumb-track residues | Mutable C-terminal 255-311 residues | Main contrast |
| --- | ---: | ---: | ---: | ---: | --- |
| `clade9_p25_5a` | 123 | 82 | 0 | 26 | Baseline permissive contact shell. |
| `clade9_p25_6a` | 103 | 65 | 0 | 21 | Nested 5 A subset with a modestly larger protected shell. |
| `clade9_p25_8a` | 51 | 18 | 0 | 10 | Stronger retained-DNA/RNA protection. |
| `clade9_p25_10a` | 32 | 0 | 0 | 3 | Conservative sentinel; mostly distal scaffold. |
| `clade9_p50_5a` | 139 | 89 | 0 | 29 | Less restrictive clade-9 conservation threshold. |
| `iia3_42_1_p50_5a` | 118 | 73 | 0 | 23 | Different conservation denominator with a similar 5 A contact shell. |

Selected-row behavior follows the same pattern. The selected six have zero
declared Wang thumb-contact-track substitutions. Their selected near retained
DNA/RNA mutation counts are `60`, `45`, `13`, `0`, `60`, and `51`; selected
C-terminal 255-311 mutation counts are `14`, `15`, `7`, `2`, `14`, and `15`.
The current panel therefore preserves the declared thumb track while sampling
near retained DNA/RNA and C-terminal/thumb-domain-adjacent residues to different
degrees.

Interpretation: one representative per class is reasonable if the claim is
"mask-policy sensitivity panel." It is weaker if read as six orthogonal
mechanistic hypotheses. A future thumb-focused or strand-displacement-motivated
panel should either add a deliberate thumb-adjacent/C-terminal class or merge
nested radius classes that do not add a distinct review question.

### Plot Set To Harden

Keep the dropdown-based progressive disclosure. Harden the plot content and
order inside the selection section.

Core views:

1. Local RMSD stratification by region.
   - Purpose: threshold sanity check.
   - Must show thresholds, selected rows, and failure counts.

2. Local structure by region for the selected six.
   - Purpose: selected-row structural preservation.
   - Must show threshold values or normalized RMSD-to-threshold values.

3. Selected substitutions across RT.
   - Purpose: show where mutations are and what chemistry they change.
   - Must keep motif/contact/thumb-track tracks visible.

4. Regional mutation burden and chemistry balance.
   - Purpose: show near retained DNA/RNA, thumb-track, C-terminal
     primer-RNA-recognition, and distal burden without implying function.
   - Must keep thumb-contact-track and C-terminal counts separate and show zero
     thumb-track edits explicitly when that is the selected-panel state.

5. Region-wise MSA support.
   - Purpose: separate unsupported distal edits from unsupported
     near retained DNA/RNA or C-terminal-context edits.
   - This is review evidence, not a composite selection score.

Support views:

- six-sequence distance heatmap for nonredundancy;
- design-class gate counts for opening context;
- ESMC LLR and SAE feature heatmaps under model/method checks only.

Avoid:

- SAE/ESMC ranking views in the core selector story;
- a global composite "processivity plausibility" score;
- a full 576-row mutation heatmap;
- repeated handoff-boundary panels that restate the same non-DNA/non-construct
  fact without a new contract check.

### Implementation Status And Remaining Work

Implemented:

- user-facing labels use "near retained DNA/RNA region" rather than annulus
  except where a legacy machine id is shown;
- local RMSD gating uses one global mapped C-alpha fit, then regional residuals;
- tests cover the global-fit regional-residual contract;
- threshold-sensitivity data and plots are materialized;
- local-region tables and manifests carry explicit residue positions,
  residue-source text, source-basis ids, thresholds, and status fields;
- region-wise MSA support is materialized by catalytic/direct-contact,
  near retained DNA/RNA, thumb-contact-track, C-terminal primer-RNA recognition,
  and distal scaffold contexts;
- direct Wang thumb-contact-track substitutions are not ordinary-panel
  eligible;
- selection readiness and review deliverables are regenerated through
  materializers.

Remaining optional work:

- Add design-class contrast validation.
   - Fail or warn when a declared design class has no eligible rows.
   - Report whether the selected row exercises the class's intended regional
     contrast.
   - Record removal/merge candidates for design classes that only add slots.

- Add side-chain/contact-graph or patch-continuity metrics only after there is a
  clear geometry contract and a non-decorative use in selection or review.
- Consider a deliberate future thumb-adjacent/C-terminal design class if the
  study needs mechanistic thumb-region contrast rather than mask-policy
  sensitivity.

### Reviewer-Facing Language

Use:

> The selected candidates preserve declared catalytic/contact regions and pass
> local backbone-shift thresholds after one global mapped C-alpha fit. They
> sample different mutation burdens and chemistry in the near retained DNA/RNA
> region and distal scaffold. The current selected six do not mutate the
> declared Wang thumb-contact track.

Avoid:

> These candidates improve strand displacement.

Avoid:

> These candidates optimize processivity features.

Avoid:

> The near-DNA/RNA region is a processivity proxy.

Prefer:

> The near retained DNA/RNA region is a distance-defined review window. It is
> useful for choosing protein hypotheses, not for proving a phenotype.

### Done Criteria

The next hardening slice is ready when:

- local RMSD thresholds are visibly enforced and visually auditable;
- every local-structure region lists explicit residue positions and source
  basis;
- the notebook uses "near retained DNA/RNA region" in user-facing prose;
- thumb-contact-track metrics are separate from near-DNA/RNA region metrics;
- design classes are explained as mask-policy contrasts;
- ESMC and SAE remain visible only as model/method annotations;
- docs, manifest rows, captions, and tests agree on the same claim boundary.
