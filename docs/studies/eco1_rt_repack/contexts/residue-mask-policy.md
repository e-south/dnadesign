---
doc_id: study-eco1-rt-repack-residue-mask-policy
surface: study-context
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-07-07
---

## Residue Mask Policy

The current mask rule is `eco1_rt_clade9_plurality25_direct_contact5a_v1`.
Contact-risk review artifacts do not protect or release residues.

The rule is deliberately small:

```text
protected =
  NAxxH / YADD / VTG
  OR Wang/Ec86 direct substrate-contact prior
  OR Eco1 amino acid is evolutionarily conserved at >=25% WT plurality in the Ec86 clade 9 MSA
  OR mapped residue is within 5 A of retained DNA/RNA

non_fixed = NOT protected
```

Eco1's retained msDNA/msrRNA wraps broadly around the RT, so broad distance
cutoffs such as 15-20 A overprotect the whole enzyme. This rule uses
direct contact instead: only mapped residues within 5 A of retained DNA/RNA are
protected by distance.

Terminal residues `1`, `2`, and `312-320` are present in the Eco1 sequence but
missing from the selected fixed-backbone structure. They are not protected by
the mask policy. They should be reported as `non_fixed_missing_backbone`, which
means they are unprotected but cannot be mutated by fixed-backbone ProteinMPNN
until coordinates are supplied or handled separately.

### Protected Sources

| Source | Policy |
| --- | --- |
| NAxxH `105-109` | protected |
| YADD `195-198` | protected |
| VTG `243-245` | protected |
| Wang/Ec86 direct substrate-contact priors | protected |
| >=25% WT plurality in the Ec86 clade 9 MSA | protected |
| mapped residue within 5 A of retained DNA/RNA | protected |
| terminal missing-backbone residues `1`, `2`, `312-320` | `non_fixed_missing_backbone`; not protected, not directly fixed-backbone ProteinMPNN mutable until coordinates exist |

RT1-RT7 intervals remain annotation and review labels. They do not blanket
hard-fix residues under this rule.

### Current Counts

Applying the current rule gives this row-level classification:

| Class | Count |
| --- | ---: |
| `non_fixed` mapped residues | 123 |
| `non_fixed_missing_backbone` terminal residues | 11 |
| evolutionarily conserved at >=25% WT plurality in the Ec86 clade 9 MSA | 106 |
| within 5 A retained DNA/RNA | 120 |
| NAxxH / YADD / VTG | 12 |
| Wang/Ec86 direct substrate-contact priors | 8 |

Total unprotected positions: `134`. Directly fixed-backbone ProteinMPNN mutable
positions from the current 7V9U backbone: `123`.

### Evidence Roles

The method rationale is intentionally plain:

- Tao supplies the fixed-backbone RT redesign method prior and homolog-MSA
  WT-plurality conservation rule.
- Mestre supplies the Ec86 clade/type source ontology for homolog panels.
- Wang supplies the Eco1/Ec86 cryo-EM structure and specific RT-msDNA/msrRNA
  substrate-contact priors.
- Simon supplies RT-region and motif annotation grammar.
- Inouye et al. 1999 and Inouye et al. 2004 supply Ec86 C-terminal/thumb
  primer-RNA recognition context. They justify keeping thumb and C-terminal
  review axes visible, but they do not turn thumb-domain mutation into a
  conservative default.

Evidence-review artifacts explain the structure context but are not mask inputs.
WT ESMC masked-marginal entropy and substitution LLRs are also review-only
model check evidence under this policy; they do not protect or release
residues unless a future mask policy explicitly promotes them.

### Design-Class Expansion

The 5 A class remains the baseline. Additional classes are now separate
ProteinMPNN request surfaces, not replacements for the baseline and not
candidate-handoff decisions. Each class has its own `mask_set.yaml`,
`thread_plan.yaml`, and `proteinmpnn_request/request_manifest.yaml` under:

```text
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/design_classes/
```

The expansion classes are:

| Class | Purpose | Mutable mapped positions |
| --- | --- | ---: |
| `eco1_rt_clade9_plurality25_contact5a_v1` | Existing baseline: clade 9 25% WT plurality and 5 A retained DNA/RNA contact | 123 |
| `eco1_rt_clade9_plurality25_contact6a_v1` | Modest contact-shell sensitivity | 103 |
| `eco1_rt_clade9_plurality25_contact8a_v1` | Stronger contact-shell sensitivity | 51 |
| `eco1_rt_clade9_plurality25_contact10a_v1` | Conservative sentinel class with a small mutable surface | 32 |
| `eco1_rt_clade9_plurality50_contact5a_v1` | Less restrictive clade 9 conservation threshold with the 5 A contact rule | 139 |
| `eco1_rt_iia3_cluster42_1_plurality50_contact5a_v1` | Closer II-A3/`42_1` family conservation denominator with the 5 A contact rule | 118 |

The current design classes are mostly protection-stringency contrasts. The
6 A, 8 A, and 10 A classes are nested subsets of the 5 A clade-9 baseline; they
do not add new mutable positions. Neither the baseline nor any expanded class
leaves declared Wang thumb-contact-track positions mutable. The classes still
sample C-terminal/thumb-domain-adjacent mapped residues, especially under the
5 A contact-shell classes, but the panel should not be described as a direct
thumb-track tuning experiment.

The candidate pool is nonredundant by `sequence_hash`. If the same sequence is
produced in more than one class, the pool keeps one row and records the duplicate
class ids and source candidate ids. The baseline has priority so the original 96
candidate identities stay stable.

Downstream fold-check, ESMC SAE, and ESMC LLR feature tables should be generated
from the nonredundant class pool after the new ProteinMPNN candidate tables are
present. The expanded fold-check request intentionally fails if only baseline
candidates are available, because a baseline-only expanded FASTA would look
complete while containing no new designs.

### Implementation Contract

Study-local mask row algebra lives in
`src/dnadesign/studies/units/eco1_rt_repack/operations/masking/`. The
`mask_set` materializer should write one row-level artifact for this rule.
Wang/Ec86 direct-contact priors must come from the study authority surface, not
a hard-coded list inside the materializer.

Expected `mask_set.yaml` row shape:

```text
canonical_position
wt_aa
design_position
has_backbone_coordinates
motif_protected
wang_ec86_direct_contact_prior
wt_plurality_frequency
wt_plurality_aa
min_distance_to_retained_dna_rna_angstrom
protected
non_fixed
non_fixed_missing_backbone
protection_reasons
```

### Fail-Fast Rules

- Missing `mask_set.yaml` blocks thread execution.
- Every protected residue must name at least one protection source.
- `non_fixed` must equal `NOT protected`.
- Residues `1`, `2`, and `312-320` must be
  `non_fixed_missing_backbone`, not protected.
- Missing-backbone residues must not be emitted as directly fixed-backbone
  ProteinMPNN mutable positions until coordinates exist.
- No RT1-RT7 interval may blanket hard-fix residues.
- Wang/Ec86 direct-contact priors must be explicit study-owned records before
  they protect residues.
