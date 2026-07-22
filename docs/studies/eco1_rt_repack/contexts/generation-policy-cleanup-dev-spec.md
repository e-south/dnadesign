---
doc_id: study-eco1-rt-repack-generation-policy-cleanup-dev-spec
surface: study-context
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-07-11
status: implemented
primary_audience:
  - future-agents
  - dnadesign-maintainers
  - study-reviewers
depends_on:
  - docs/studies/eco1_rt_repack/contexts/residue-mask-policy.md
  - docs/studies/eco1_rt_repack/contexts/selection-hardening-dev-spec.md
  - docs/studies/eco1_rt_repack/workbench/ontology/manual-mask-authority.yaml
---

## Generation Policy Contract

### Objective

Generate complete fixed-backbone Eco1/Ec86 RT sequences under explicit fixed,
open, and residue-alphabet rules. A candidate belongs to one generation policy.
Mutations from separate ProteinMPNN outputs are never combined.

### Version

```text
generation_policy_version: 3
output_root: outputs/thread/generation_policies_v3/
requested_total: 1008
```

Every downstream row carries `policy_id`, `policy_version`, and
`policy_manifest_hash`. Missing or mismatched provenance is an error.

### Shared Fixed Set

Every policy fixes:

- Eco1 motif contexts `99-115`, `189-204`, and `237-251`;
- mapped residues at or below `5 A` from retained DNA/RNA;
- Wang thumb-track positions `238`, `239`, `240`, `249`, `257`, `261`, `264`,
  and `298`;
- mapped residues `255-311`;
- declared Ec86 clade 9 conserved/core positions.

Residues without mapped `7V9U` backbone coordinates are not designable.

### Policies

| Policy id | Open set | Requested |
| --- | --- | ---: |
| `distal_scaffold_repack_v1` | 25 distal positions | 336 |
| `near_dna_rna_acid_free_v1` | 59 peripheral positions | 336 |
| `combined_near_acid_free_plus_distal_v1` | 59 peripheral plus 25 distal positions | 336 |

Peripheral positions are more than `5 A` and at or below `10 A` from retained
DNA/RNA, outside the shared fixed set. The combined policy opens peripheral and
distal positions in one request so ProteinMPNN designs the complete sequence
jointly.

### Residue Alphabets

- Distal positions use the broad ProteinMPNN alphabet with global cysteine
  omission.
- Peripheral positions allow MSA-observed alternatives while excluding new
  D/E and new P/G substitutions.
- The v3 global `--omit_AAs C` rule applies to every open position.
- Residue-specific peripheral omissions use ProteinMPNN's public
  `--omit_AA_jsonl` input.

At open WT-Cys position C233, the global rule omits C and therefore forces a
substitution. C233 is not in the fixed set. This behavior is valid v3
provenance, but it creates shared panel overlap and must be disclosed.

The alphabets constrain sampling. They do not predict activity, binding, or
strand displacement.

### ProteinMPNN Request Contract

Each policy request contains the official public sidecars needed by the active
policy:

```text
request_manifest.yaml
chain_a_backbone.pdb
parsed_pdbs.jsonl
assigned_chains.jsonl
fixed_positions.jsonl
omit_AA.jsonl  # peripheral and combined policies
```

Request manifests declare seeds `101`, `202`, and `303`; temperatures `0.1`
and `0.3`; `num_seq_per_target: 56`; and `336` expected samples. The generic
runner resolves request paths before changing into the official ProteinMPNN
checkout and validates the manifest before external execution.

### Materialized Flow

1. Materialize policy, position, alphabet, and request manifests.
2. Run ProteinMPNN separately for each complete policy.
3. Ingest samples with one-policy provenance.
4. Deduplicate exact sequences while retaining source-policy ids.
5. Fold WT and `1007` unique candidate sequences with ColabFold.
6. Normalize fold outputs and compute regional RMSD after one global mapped
   C-alpha fit.
7. Apply the selection method in `selection-hardening-dev-spec.md`.
8. Regenerate the review manifest, plots, notebook, and protein sequence export.

ESMC and SAE are optional model checks and are not required for this flow.

### Required Checks

- fixed and open sets are disjoint;
- residues `255-311` are fixed;
- the near and distal open sets are disjoint;
- the combined open set equals their union;
- new D/E substitutions are omitted at peripheral positions;
- global cysteine omission and forced open-WT-cysteine changes are explicit;
- stale policy versions and hashes fail;
- generated samples obey fixed-position and declared alphabet contracts.

### Current State

The full v3 generation, candidate pool, ColabFold report, fold review,
selection tables, plots, and eight-row selected panel are materialized. Current
counts and selected candidate ids are recorded in `../record/status.md`.
