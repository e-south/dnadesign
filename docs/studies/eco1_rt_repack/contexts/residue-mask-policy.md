---
doc_id: study-eco1-rt-repack-residue-mask-policy
surface: study-context
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-06-23
---

## Residue Mask Policy

The current mask rule is `eco1_rt_clade9_plurality25_direct_contact5a_v1`.
Earlier SASA and contact-density checks remain evidence reviews only; they do
not protect or release residues.

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

Evidence-review artifacts explain the structure context but are not mask inputs.
The rule does not search across SASA, contact-density, contact-class, or
conservation-threshold variants.
The next sampling plan should use the materialized `mask_set.yaml` under
`eco1_rt_clade9_plurality25_direct_contact5a_v1`.

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
