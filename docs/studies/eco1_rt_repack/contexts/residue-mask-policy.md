---
doc_id: study-eco1-rt-repack-residue-mask-policy
surface: study-context
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-06-19
---

## Residue Mask Policy

The conservative mask tier protects function before diversity. The first
`mask_set.yaml` should be the union of all fixed/protected sources and only
then expose residues to MPNN sampling.

```text
final_fixed = manual_mask
  OR contact_mask
  OR conservation_mask
  OR unresolved_residue_mask
  OR cysteine_control_mask

designable = mapped_residue AND NOT final_fixed
```

The mask set should never infer that a residue is designable merely because one
source is missing. Missing evidence fails the conservative tier unless the
profile explicitly waives that source.

### Mask Sources

| Source | Owner | Initial policy |
| --- | --- | --- |
| Catalytic residues | study | fixed until manually reopened; proposed anchors include YADD, RT1-RT7, retron X, and retron Y motifs after numbering audit |
| Substrate/contact residues | `thread` structure evidence plus study thresholds | fixed when within declared contact threshold; conservative first pass uses heavy-atom distance to retained nucleic acid context |
| MSA conservation | planned evidence contract | fixed above declared conservation threshold when the Eco1 residue is the plurality amino acid |
| Unresolved structure residues | `thread` residue map | fixed or excluded |
| Interface/ligand context | study and `thread` contact profile | fixed unless explicitly waived |
| Cysteine control | study profile | no new cysteine in conservative tier; existing cysteine policy must be explicit |

### Conservative Defaults

These are starting policy values, not residue-map evidence:

| Setting | First-pass value | Notes |
| --- | --- | --- |
| Contact threshold | `20 A` | Use all retained nucleic-acid atoms. Add `18 A` and `15 A` only as later relaxed tiers. |
| Conservation threshold | `0.25` | Apply to broad retron-RT and Eco1-like MSAs separately; fixed if either profile passes. |
| MSA gap policy | non-gap denominator | Positions with insufficient non-gap support fail until the profile declares a minimum. |
| Manual motif policy | fixed | Exact residue ids remain provisional until structure and sequence authority are selected. |
| Effector-interface policy | not automatically fixed | Preserve only if also contact/conserved/manual, or if the study declares an effector-retention objective. |

### Residue Mask Table

The runtime authority should be a typed table. A review export may use CSV with
the same columns:

```text
canonical_position
wt_aa
structure_chain_id
structure_residue_id
design_position
domain_or_region
distance_to_retained_nucleic_acid_angstrom
contact_mask
broad_msa_non_gap_count
broad_msa_wt_frequency
broad_msa_plurality_aa
eco1_like_msa_non_gap_count
eco1_like_msa_wt_frequency
eco1_like_msa_plurality_aa
conservation_mask
manual_mask
manual_mask_reason
unresolved_residue_mask
cysteine_control_mask
final_fixed
proteinmpnn_designable
```

### Fail-Fast Rules

- A mutable residue must have one canonical residue-map row.
- A fixed residue must record at least one source reason.
- Contact and conservation masks must carry thresholds and source hashes.
- Conflicting mask inputs fail until manually resolved in the profile fixture.
- Empty mutable masks fail; all-mutable masks fail for the conservative tier.
- A manual Eco1 residue id is advisory until it resolves through the residue map.
- No mask file may mix canonical Eco1 numbering and PDB residue ids without
  explicit columns for each.
