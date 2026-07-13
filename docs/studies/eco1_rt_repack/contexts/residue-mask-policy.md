---
doc_id: study-eco1-rt-repack-residue-mask-policy
surface: study-context
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-07-11
---

## Generation Policy

The active ProteinMPNN input is a complete generation policy, not a set of
mutations to combine later. Each policy declares fixed positions, open
positions, and one allowed amino-acid alphabet for every open residue.

### Shared Fixed Positions

All active policies fix:

- Eco1 motif contexts `99-115`, `189-204`, and `237-251`;
- mapped residues at or below 5 A from retained DNA/RNA;
- Wang thumb-track positions `238`, `239`, `240`, `249`, `257`, `261`, `264`,
  and `298`;
- mapped residues `255-311`, covering the mapped part of the `255-320`
  primer-recognition RNA-binding fragment;
- positions in the declared Ec86 clade 9 conserved/core mask.

Residues `1`, `2`, and `312-320` lack mapped 7V9U backbone coordinates and are
not fixed-backbone design positions. RT1-RT7 intervals are annotation labels,
not protection rules.

### Open Sets

The peripheral shell contains mapped, unprotected positions more than 5 A and
at or below 10 A from retained DNA/RNA.

| Policy | Open positions | Requested sequences |
| --- | --- | ---: |
| `distal_scaffold_repack_v1` | 25 distal scaffold positions | 336 |
| `near_dna_rna_acid_free_v1` | 59 peripheral positions | 336 |
| `combined_near_acid_free_plus_distal_v1` | 59 peripheral plus 25 distal positions | 336 |

The combined policy asks ProteinMPNN to design distal and peripheral positions
jointly. The policies produce complete sequences, not mutation bins. Policy
identity is provenance and does not rank candidate quality. The downstream
selection contract retains two distal, three peripheral, and three combined
sequences as explicit experimental contrasts.

### Amino-Acid Alphabets

Peripheral residues are represented in ProteinMPNN's public `omit_AA_jsonl`
input. The v3 requests also use global `--omit_AAs C`.

- Distal positions use the broad standard alphabet without cysteine.
- Peripheral positions allow MSA-observed alternatives without new D/E or new
  P/G substitutions.
- An open WT cysteine is omitted. C233 is therefore forced to change in the
  proximal policies.

The peripheral rule prevents clear acidic regressions. It does not assert that
positive charge improves binding or function.

### Evidence Roles

- Wang and 7V9U define retained DNA/RNA geometry, direct contacts, and the
  electropositive-surface prior.
- Inouye 1999 supports the C-terminal 91-residue primer-template recognition
  context; Inouye 2004 supports the `255-320` RNA-binding fragment.
- Mestre supplies the Ec86 clade source set used for observed alternatives and
  conservation.
- Tao supports the constraint-first fixed-backbone generation and structural
  review pattern, not the Eco1 distance shells or functional claims.
- Simon supplies motif and region annotation grammar.

ESMC, SAE, contact-risk plots, and RT1-RT7 tracks are review context. They do
not change fixed or open positions.

### Materialized Contract

`generation_policy_manifest.yaml`, `generation_policy_positions.parquet`, and
`generation_policy_alphabets.parquet` are materialized under
`outputs/thread/generation_policies_v3/`. Each ProteinMPNN request carries the
policy id, policy version, policy-manifest hash, fixed positions, open
positions, global omissions, and any residue-specific omission sidecar required
by the policy.

### Fail-Fast Rules

- A candidate must be one complete ProteinMPNN output from one policy.
- Missing or mismatched v3 policy hashes block downstream ingestion.
- Every protected position must name at least one reason.
- Fixed and open positions must be disjoint.
- Every peripheral open position must have one residue-level alphabet.
- Global cysteine omission and any forced open-WT-cysteine change must remain
  explicit.
- New cysteine and peripheral D/E, P, or G must be absent from allowed
  alternatives.
- Missing-backbone positions must not appear in ProteinMPNN open positions.
