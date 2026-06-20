---
doc_id: study-eco1-rt-repack-fold-validation-policy
surface: study-context
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-06-19
---

## Fold Validation Policy

Fold validation is a computational QA gate, not proof of improved stability or
function. A candidate can advance only when the fold-check report states which
runtime produced the prediction, which starting structure was compared, and
which thresholds were applied.

### Required Fields

- `candidate_id`
- `runtime_kind`
- `runtime_version`
- `input_sequence_hash`
- `reference_structure_id`
- `wt_baseline_artifact_id`
- `runtime_parameters_hash`
- `threshold_id`
- `threshold_values`
- `plddt`
- `pae_summary`
- `backbone_rmsd_to_reference`
- `protected_contact_retention`
- `status`
- `rejection_reason`
- `missing_metric_reason`

### Metric Semantics

Fold validation compares each full-length candidate against the selected Eco1
RT structural authority. Wild type must be evaluated with the same runtime
settings before variant thresholds are interpreted.

| Metric | Meaning | Conservative use |
| --- | --- | --- |
| `plddt` | Global confidence summary from the fold runtime. | Reject substantial degradation from WT baseline or threshold-free acceptance. |
| `core_plddt` | Confidence over mapped RT core/palm/fingers/thumb regions. | Reject if the catalytic core is low confidence. |
| `pae_summary` | Domain-orientation uncertainty summary. | Reject high uncertainty between palm, fingers, and thumb. |
| `backbone_rmsd_to_reference` | C-alpha or declared backbone RMSD after alignment to the reference. | Reject first-pass candidates with major structure drift. |
| `protected_region_rmsd` | RMSD over protected catalytic/contact regions. | Reject protected-region movement beyond the declared threshold. |
| `protected_contact_retention` | Whether modeled protected contacts remain geometrically plausible after superposition. | Reject loss of retained nucleic-acid/contact geometry. |

Thresholds belong in the fold-check report or profile fixture. This page names
required semantics; it does not bless universal numeric cutoffs. A fold-check
row cannot be accepted from raw metric values alone; it must point to the WT
baseline artifact, runtime parameters, and threshold policy used for the
decision.

### Fail-Fast Rules

- No accepted fold-check row without a candidate-table row.
- No accepted fold-check row without reference structure provenance.
- No hidden fallback from missing real metrics to fixture metrics.
- No pooled-window handoff from candidates that lack fold-check coverage for
  the actual full sequence being proposed.
- A fixture fold-check row must be labeled `fixture` and cannot satisfy a
  materialized candidate handoff.
- Runtime failures are rows with `status: rejected` or `status: errored`, not
  silent omissions.
