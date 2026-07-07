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
| RCSB PDB `7V9U` | Coordinate source for the Ec86 RT-msDNA-RNA structure and chain identity. | Coordinate authority only; not a design score. |
| Kabsch-style RMSD superposition literature | Precedent for optimal rigid-body superposition before comparing C-alpha displacements. | Supports the geometry calculation; threshold values still need study-specific justification. |
| Enzyme-design and motif-scaffolding literature | General precedent for protecting functionally important sites and substrate/catalytic geometry. | Supports local-site caution; does not define Eco1 RT-specific thresholds. |

Useful source links:

- Tao et al.: <https://www.nature.com/articles/s41587-026-03149-6>
- Wang et al.: <https://www.nature.com/articles/s41564-022-01197-7>
- RCSB 7V9U: <https://www.rcsb.org/structure/7V9U>
- RMSD superposition background: <https://link.springer.com/article/10.1186/1471-2105-11-363>
- Functionally important site design context: <https://proceedings.mlr.press/v235/song24k.html>

### Plain Region Vocabulary

Use plain names in notebooks, captions, legends, and reviewer-facing docs.

| Current term | Preferred user-facing term | Meaning |
| --- | --- | --- |
| near retained DNA/RNA annulus | near retained DNA/RNA region | Mapped residues `>5 A` and `<=10 A` from retained DNA/RNA after excluding protected motifs, direct-contact residues, and Wang thumb-contact-track positions. |
| substrate-proximal annulus | near retained DNA/RNA region | Same region, only when the exact distance rule is shown nearby. |
| thumb-contact track | Wang thumb-contact track | Explicit Wang/Ec86 positions `238,239,240,249,257,261,264,298`. |
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

3. Per-class "why selected" strip.
   - Purpose: show the lexicographic comparison that chose the representative.
   - Must show close alternatives and direction of each field.

4. Selected substitutions across RT.
   - Purpose: show where mutations are and what chemistry they change.
   - Must keep motif/contact/thumb-track tracks visible.

5. Regional mutation burden and chemistry balance.
   - Purpose: show near retained DNA/RNA, thumb-track, and distal burden without
     implying function.
   - Must keep thumb-contact-track count separate and show zero explicitly.

6. Region-wise MSA support.
   - Purpose: separate unsupported distal edits from unsupported
     substrate-proximal edits.
   - This is the highest-value missing evidence view after local RMSD gates.

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

### Implementation Plan For Future Agents

1. Rename user-facing labels.
   - Replace "annulus" in titles, legends, captions, and notebook prose with
     "near retained DNA/RNA region".
   - Keep machine-id migration as a deliberate code-and-artifact slice, not an
     alias layer.

2. Verify local RMSD gate behavior.
   - Confirm the code globally fits mapped C-alpha atoms once, then computes
     regional residuals.
   - Add a test that would fail if a per-region fit were used for gating.
   - Add threshold sensitivity data and a plot if not already present.

3. Add residue provenance to every local region.
   - Persist explicit residue lists, residue-source text, and source basis ids
     into metric tables and manifests.
   - Ensure notebook captions can recover exact residue definitions without
     reading source code.

4. Add within-class selection comparison.
   - Materialize a compact top-alternative table or plot input.
   - Show the selected row against the next 3-5 eligible alternatives per
     design class.
   - Use the exact selector order; do not invent a new score for display.

5. Add region-wise MSA support.
   - Compute observed fraction and unobserved substitutions by region.
   - Treat unsupported direct-contact/catalytic edits as hard failures through
     existing gates.
   - Treat unsupported near retained DNA/RNA edits as review risk.

6. Add design-class contrast validation.
   - Fail or warn when a declared design class has no eligible rows.
   - Report whether the selected row exercises the class's intended regional
     contrast.
   - Record removal/merge candidates for design classes that only add slots.

7. Regenerate through materializers.
   - Do not hand-edit `outputs/`.
   - Regenerate selection readiness and review deliverables.
   - Run targeted selection/readiness tests, marimo check, docs checks, and the
     architecture boundary check.

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

> The near retained DNA/RNA region is a substrate-proximal review window. It is
> useful for choosing protein hypotheses, not for proving a phenotype.

### Done Criteria

The next hardening slice is ready when:

- local RMSD thresholds are visibly enforced and visually auditable;
- every local-structure region lists explicit residue positions and source
  basis;
- the notebook uses "near retained DNA/RNA region" in user-facing prose;
- thumb-contact-track metrics are separate from near-DNA/RNA region metrics;
- selected rows are justified against close within-class alternatives;
- design classes are explained as mask-policy contrasts;
- ESMC and SAE remain visible only as model/method annotations;
- docs, manifest rows, captions, and tests agree on the same claim boundary.
