---
doc_id: eco1-rt-repack-command-groups
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-07-15
status: active
---

## Eco1 RT Repack Commands

This page is the executable route for the current Eco1 RT repack study. Each
command writes one declared artifact boundary. `pipeline.yaml` is the
machine-readable command registry.

### Premise

This study asks whether complete ProteinMPNN-designed Eco1/Ec86 RT sequences
can keep declared catalytic, direct-contact, Wang thumb-track, and mapped
residues 255-311 fixed, preserve local C-alpha backbone geometry, and introduce
MSA-observed, non-acidifying substitutions in the declared peripheral
nucleic-acid-facing shell for a diversity-seeking experimental panel.

The workflow does not predict activity, affinity, processivity, strand
displacement, or safety.

### Current Boundary

The active root is:

```text
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v3/
```

The full v3 path is materialized: 1007 unique ProteinMPNN candidates, ColabFold
models, fold review, 978 strong-class rows, 732 rows that also pass local
geometry, three policy comparison pools, one eight-row selected panel, review
plots, canonical selected proteins, and exact full-CDS Twist handoff sequences
for all eight rows. Every downstream stage must match the v3 policy version and
manifest hash.

### 1. Materialize Policies And Requests

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies \
  --repo-root . \
  all
```

This writes three complete policy requests, 336 sequences per policy and 1008
in total:

- `distal_scaffold_repack_v1`
- `near_dna_rna_acid_free_v1`
- `combined_near_acid_free_plus_distal_v1`

Peripheral residues have an `omit_AA_jsonl` rule. Peripheral alternatives are
MSA-observed and introduce no new D/E, P, or G. V3 also uses global
`--omit_AAs C`; an open WT cysteine such as C233 is therefore forced to change.

### 2. Run ProteinMPNN

The SCC job maps one complete policy to each array task. It does not compose
mutations across policies.

Smoke one policy:

```bash
qsub -t 1 \
  -v DNADESIGN_REPO=<scc_dnadesign_repo>,PROTEINMPNN_ROOT=<scc_dnadesign_repo>/.var/tools/proteinmpnn \
  docs/bu-scc/jobs/eco1-proteinmpnn-generation-policy.qsub
```

After the smoke output passes request, fixed-position, and alphabet checks, run
all policies:

```bash
qsub -t 1-3 \
  -v DNADESIGN_REPO=<scc_dnadesign_repo>,PROTEINMPNN_ROOT=<scc_dnadesign_repo>/.var/tools/proteinmpnn \
  docs/bu-scc/jobs/eco1-proteinmpnn-generation-policy.qsub
```

Each task writes one `sample_table.parquet` and one
`candidate_table.parquet`. Generated rows belong to exactly one policy.

### 3. Aggregate Candidates

After all policy outputs are local:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies \
  --repo-root . \
  candidate-pool
```

The aggregate table deduplicates complete protein sequences and retains all
source-policy provenance. It never merges mutation sets.

### 4. Materialize Fold Inputs

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies \
  --repo-root . \
  foldcheck-request
```

For a six-sequence ColabFold smoke on SCC:

```bash
qsub \
  -v DNADESIGN_REPO=<scc_dnadesign_repo>,\
FOLDCHECK_REQUEST_MANIFEST=src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v3/foldcheck_request/foldcheck_request_manifest.yaml,\
FOLDCHECK_SEQUENCE_LIMIT=6,\
COLABFOLD_BATCH=/projectnb/dunlop/esouth/tools/localcolabfold/.pixi/envs/default/bin/colabfold_batch,\
COLABFOLD_EXTRA_ARGS='--num-models 1' \
  docs/bu-scc/jobs/eco1-colabfold-foldcheck.qsub
```

For the full pool, use the same request manifest with an explicit run root and
`FOLDCHECK_SEQUENCE_LIMIT=all`, or submit bounded array shards with
`FOLDCHECK_SHARD_SIZE`. Do not write into an existing output directory unless
the run is explicitly resumable and validated.

### 5. Normalize Fold Evidence

Normalize completed ColabFold outputs through the public study materializer,
then build fold review under the v3 root. The normalized candidate rows must
carry the v3 policy hash before selection runs.

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_report \
  --repo-root . \
  --output-root src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v3 \
  --colabfold-output-root <v3_colabfold_output_root> \
  --runtime-version <colabfold_version>

uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review \
  --repo-root . \
  --output-root src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v3
```

The active structural screen first requires the strong class: mean pLDDT at
least 91.5 and same-run candidate-to-WT C-alpha RMSD at most 1.25 A. Every
named non-distal local region must then be at or below the declared 2.5 A
cutoff after one global mapped C-alpha fit. Distal RMSD is reported only.

### 6. Select The Panel

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness \
  --repo-root . \
  --output-root src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v3
```

The visible flow is complete candidates, strong-class models, local-geometry
pass, assignment to the distal, peripheral, and combined policy pools, and the
eight-row selected panel.
The first pair in each policy maximizes
mutated-position Jaccard distance before exact-substitution distance. Each third
peripheral or combined row maximizes minimum distance from the corresponding
pair. Charge counts, MSA support, local RMSD, fold metrics, and sequence hash
are used only if earlier criteria tie; they did not determine the selected ids.
Exact F10/R13 substitutions and a Wang R13A evidence-match field are reported;
they do not filter or rank candidates. The fold review does not establish the
RT-msDNA oligomeric state.

### 7. Build The Twist Handoff

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.twist_handoff \
  --repo-root .
```

This writes eight exact 963-bp CDS designs, a vendor CSV, FASTA, one annotated
GenBank file per sequence, and a hash-linked manifest. Unchanged residues retain
the authoritative WT codon; changed residues use the highest-frequency codon
in the packaged E. coli table. The manifest records E. coli K-12 MG1655 as the
assay host and cites the exact Kazusa E. coli K-12 frequency record used by the
table. This is substitution-only minimal recoding, not whole-gene codon
optimization. Vendor codon optimization is disabled. The sequence bundle
excludes internal BsaI and BsmBI sites. Assembly flanks, junctions, and the live
vendor complexity result remain separate order decisions.

### 8. Build The Review Notebook

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables \
  --repo-root . \
  --selection-root src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v3/selection
```

Movie rendering is explicit because the three targets have different costs:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables \
  --repo-root . \
  --selection-root src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v3/selection \
  --render-communication-movie protected-evidence \
  --render-communication-movie proposal-backbones \
  --render-communication-movie selected-electrostatics
```

`protected-evidence` gives each protected or open residue category one full
turn on the retained complex. `proposal-backbones` cycles all 738 models
retained by the declared local-geometry review over one centered cryo-EM RT
reference. Distal, peripheral, and combined chapters open and close one model
at a time. Each frame reports exact full-length WT sequence identity and
substitution count, with evenly distributed candidate dwell time. These values
describe sequence change rather than predicted function. Candidate sticks show
all modeled side chains rather than mutation-only highlights. The movie is a
local-geometry visualization, so it includes six good-class peripheral models
that are not eligible for final selection. All 1,007 ColabFold model files were
parsed; the active funnel excludes 29 good-class rows before applying the local
review.
`selected-electrostatics` gives the reference and each selected model one full
turn with an opaque protein surface and a fixed unit-bearing Coulombic scale.
Each target starts 180 degrees from the approved interactive pose, completes
full turns, is hash-tracked, and is reusable. Default materialization does not
launch ChimeraX and omits absent movies from notebook choices.

Validate the generated notebook:

```bash
uv run marimo check \
  src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/review_deliverables/notebooks/eco1_review_deliverables.py
```

The notebook reads declared manifests and precomputed tables. It does not
recompute selection. Static visuals and py3Dmol views must include plain
descriptions and interpretation limits. ESMC and SAE remain optional model
context and are not selection evidence.

### Source Roles

- Wang/7V9U supports the retained nucleic-acid geometry, direct-contact
  protection, and cautious review of the electropositive surface.
- Inouye 2004 supports fixing mapped residues 255-311 in the primer-recognition
  RNA-binding fragment. Inouye 1999 supports caution when reviewing the broader
  C-terminal context; it does not make residues 230-254 part of the v3 fixed set.
- Tao supports constraint-first fixed-backbone RT redesign followed by
  structural and experimental screening. It does not establish Eco1 activity
  or the 2.5 A cutoff.
- Kabsch supports the global rigid-body fit, not a functional RMSD boundary.
- ProteinMPNN is a sequence proposal backend. ColabFold provides predicted
  structure evidence. Neither validates function.

### Verification

```bash
uv run pytest -q src/dnadesign/thread/tests/adapters/proteinmpnn
uv run pytest -q src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/generation_policies
uv run pytest -q src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness
uv run pytest -q src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables
uv run pytest -q src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/twist_handoff
uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .
uv run python -m dnadesign.devtools.docs.checks --repo-root .
git diff --check
```
