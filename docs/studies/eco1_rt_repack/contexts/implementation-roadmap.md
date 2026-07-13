---
doc_id: study-eco1-rt-repack-implementation-roadmap
surface: study-context
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-07-11
status: active-runtime-sequence
primary_audience:
  - dnadesign-maintainers
  - study-reviewers
  - runtime-agents
depends_on:
  - docs/studies/eco1_rt_repack/contexts/generation-policy-cleanup-dev-spec.md
  - docs/studies/eco1_rt_repack/contexts/selection-hardening-dev-spec.md
  - docs/studies/eco1_rt_repack/record/status.md
---

## Runtime Sequence

This page defines the active Eco1 RT repack execution order. The study produces
complete fixed-backbone protein sequence hypotheses for later testing. It does
not predict activity, processivity, strand displacement, or assay performance.

### Owner Boundaries

| Surface | Responsibility |
| --- | --- |
| `eco1_rt_repack` | Eco1 residue authority, generation policies, protected regions, chemistry and local-structure contracts, selection, and review prose. |
| `dnadesign.thread` | Generic ProteinMPNN request/result normalization, candidate identity, fold-report contracts, and neutral protein handoff fields. |
| `docs/bu-scc` | SCC scheduler templates and runtime operations. |
| downstream RT-lnRNA study | Explicit acceptance of an RT-only protein handoff before construct design. |

Study-specific biology must remain in `eco1_rt_repack`. Generic backend and
artifact mechanics must use public `dnadesign.thread` APIs.

### Active Data Flow

1. **Structure and residue authority**
   - Map Eco1 numbering to the 7V9U-backed Ec86 RT scaffold and retained DNA/RNA.
   - Materialize residue, contact-geometry, conservation, and manual authority records.

2. **Generation-policy materialization**
   - Build the shared protected set.
   - Materialize distal, peripheral, and combined peripheral-plus-distal policies.
   - Emit fixed positions, global omissions, and residue-specific peripheral alphabet sidecars.

3. **ProteinMPNN generation**
   - Run each complete policy independently.
   - Keep one policy provenance record per raw sample.
   - Do not combine mutations from separate generated sequences.

4. **Candidate pool**
   - Normalize and deduplicate complete protein sequences.
   - Require policy id, version, and manifest hash on every candidate.
   - The materialized v3 pool contains 1007 unique complete sequences.

5. **Fold and local-structure evidence**
   - Normalize one ColabFold model per candidate and the WT control.
   - Fit mapped C-alpha coordinates once, then calculate residual RMSD for each named region.
   - Apply one declared 2.5 A cutoff to all non-distal review regions; retain distal RMSD as review context.

6. **Selection**
   - Keep accepted fold models that pass the local-geometry contract.
   - Validate the declared peripheral chemistry and proximal MSA support.
   - Report R13 and other alpha-1 substitutions without filtering or ranking by them.
   - Assign selection-contract rows to distal, peripheral, and combined policy pools without treating policy as quality.
   - Select the most mutation-set-dissimilar pair within each pool.
   - Add one peripheral and one combined row by maximum minimum distance from the corresponding pair.
   - Retain all eight rows: two distal, three peripheral, and three combined.
   - Use chemistry, MSA support, local RMSD, fold metrics, and sequence hash as later tie-breaks.

7. **Review bundle**
   - Regenerate the selection manifest, plots, selected structure browser, and marimo notebook.
   - Keep ESMC/SAE in the model-check lane; they do not select candidates.

8. **Protein handoff boundary**
   - Export RT-only protein sequences and hashes.
   - Do not claim DNA, codon, restriction-site, construct, or assay readiness until the owning downstream stage materializes those records.

The materialized v3 flow is `1007` accepted sequences, `738` local-geometry
and generation-contract pass rows, policy pools of `335` distal, `226`
peripheral, and `177` combined rows, followed by one eight-row selected panel.
These counts come from candidate,
local-geometry, and selection tables rather than request totals.

### Canonical Commands

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies --repo-root . all
qsub -t 1-3 docs/bu-scc/jobs/eco1-proteinmpnn-generation-policy.qsub
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness --repo-root .
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables --repo-root .
uv run marimo check src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/review_deliverables/notebooks/eco1_review_deliverables.py
```

ProteinMPNN and ColabFold execution use the checked-in SCC jobs under
`docs/bu-scc/jobs/`. Their request and result manifests are the runtime
authority; shell transcripts are not study evidence.

### Fail-Fast Conditions

Stop the active path when:

- policy id, version, or manifest hash is missing or mismatched;
- a protected, direct-contact, Wang-track, or mapped `255-311` position is open;
- a peripheral request lacks its residue-specific alphabet sidecar or fails to
  declare the v3 global cysteine omission;
- a candidate sequence is assembled from mutations drawn from separate outputs;
- fold or named local-structure evidence is missing;
- an acidic near-region gain or unsupported proximal substitution enters the selected scope;
- a notebook dropdown points to a missing artifact;
- ESMC/SAE annotations are used as acceptance evidence;
- an RT-only sequence is presented as construct-ready.

### Evidence Surfaces

- Current counts and selected-panel interpretation: `../record/status.md`
- Generation-policy definitions: `generation-policy-cleanup-dev-spec.md`
- Selection method and literature boundaries: `selection-hardening-dev-spec.md`
- Runtime command groups: `../operations/runtime/command-groups/pipeline.yaml`
- Machine-readable artifact registry: `../record/datasets.yaml`
