---
doc_id: study-eco1-rt-repack-fold-validation-policy
surface: study-context
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-07-11
status: active-v3-structural-review
---

## Fold And Local-Geometry Review

### Purpose

ColabFold supplies predicted structures for complete ProteinMPNN sequences. The
review asks whether each predicted RT retains the declared backbone geometry. It
does not measure activity, affinity, processivity, strand displacement, or
safety.

### Current V3 Run

The v3 request contains WT Ec86 RT and `1007` deduplicated candidate sequences
as complete `320`-residue proteins. ColabFold produced one ranked model per
sequence with `--num-models 1`. The normalized report contains `1008` rows: WT
plus every candidate.

Canonical artifacts are under
`outputs/thread/generation_policies_v3/`:

- `foldcheck_request/input_sequences.fasta`;
- `foldcheck_request/foldcheck_request_manifest.yaml`;
- `foldcheck_report.parquet`;
- `foldcheck_review/foldcheck_candidate_ranking.parquet`;
- `foldcheck_review/foldcheck_full_structure_set.yaml`;
- `selection/local_structure_region_metrics.parquet`.

Raw ColabFold outputs remain on BU SCC project storage. The checked-in workflow
normalizes model and score files; it does not copy the raw run tree into the
repository.

### Structural Method

1. Require one WT model generated with the same runtime settings as the
   candidates.
2. Normalize model provenance, pLDDT, PAE fields when available, and model-file
   paths.
3. Fit each candidate model once to the 7V9U-backed reference over mapped
   C-alpha atoms.
4. Measure regional C-alpha RMSD without fitting each region again.
5. Keep candidates at or below the declared `2.5 A` cutoff in every non-distal
   review region.

The non-distal regions are:

- catalytic YADD context;
- retron X/NAxxH context;
- retron Y/VTG context;
- Wang thumb-contact track;
- mapped residues `255-311` in the C-terminal primer-RNA recognition context;
- the peripheral retained DNA/RNA shell.

Distal RMSD is reported but does not filter rows. The `2.5 A` cutoff is a
study-declared review rule, not a literature-derived functional boundary.

### Metric Semantics

| Field | Meaning | Use |
| --- | --- | --- |
| `plddt` | ColabFold confidence summary. | Review context and late selection tie-break. |
| `pae_summary` | Predicted alignment-error summary when available. | Review domain-orientation uncertainty. |
| `wt_runtime_ca_rmsd` | Candidate-to-WT RMSD within the same ColabFold run. | Detect same-runtime structural outliers. |
| `cryoem_mapped_ca_rmsd` | Candidate-to-7V9U-backed reference RMSD over mapped C-alpha atoms. | Review overall scaffold similarity. |
| regional C-alpha RMSD | Residual RMSD after one global mapped fit. | Apply the declared local-geometry rule. |
| `review_class` | Fold-review label derived from model metrics. | Audit context; it is not a separate visible funnel stage when it removes no rows. |

No model-derived field is an activity score.

### Fail-Fast Rules

- Require WT in the same fold request.
- Require matching candidate ids and sequence hashes across request, report,
  candidate pool, and model files.
- Require one accepted model row for every candidate before selection.
- Require reference-structure and runtime provenance.
- Reject missing or mismatched model files; do not substitute fixture metrics.
- Reject missing regional measurements for any non-distal review region.
- Record runtime failures as explicit rows rather than silent omissions.

### Ownership

- `eco1_rt_repack` owns the reference structure, named regions, cutoff, and
  interpretation.
- `docs/bu-scc/jobs/` owns scheduler and LocalColabFold execution details.
- `dnadesign.thread.adapters.colabfold` owns generic ColabFold output parsing.
- `dnadesign.thread.foldcheck` owns backend-neutral request and report fields.

Optional ESMC, SAE, and Atlas artifacts are model checks. They do not replace
ColabFold, filter panel rows, or validate function.

### Literature Boundary

- Wang et al. 2022 and RCSB `7V9U` define the Ec86 RT-msDNA/RNA structural
  context and support direct-contact review.
- Inouye et al. 1999 supports caution across the C-terminal 91-residue
  primer-template recognition context. Inouye et al. 2004 identifies the
  `255-320` primer-recognition RNA-binding fragment.
- Tao et al. 2026 supports constraint-first ProteinMPNN redesign followed by
  predicted-structure review. It does not validate the Eco1 shell or cutoff.
- Kabsch 1976/1978 supports the rigid-body superposition method, not the `2.5 A`
  decision boundary.

### Validation

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review --repo-root .
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness --repo-root .
uv run pytest -q src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness
```
