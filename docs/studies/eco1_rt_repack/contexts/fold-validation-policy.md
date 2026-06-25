---
doc_id: study-eco1-rt-repack-fold-validation-policy
surface: study-context
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-06-25
---

## Fold Validation Policy

Fold validation is a computational QA gate, not proof of improved stability or
function. A candidate can advance only when the fold-check report states which
runtime produced the prediction, which starting structure was compared, and
which thresholds were applied.

The first Eco1 implementation uses a ColabFold-planned request as the practical
AlphaFold-family CLI path for BU SCC. This is not a claim that the exact
DeepMind AlphaFold2 distribution has been run. ColabFold is the first backend
because it is batch-FASTA friendly, SCC-suitable, and close enough to the
AlphaFold-family fold-check role used after ProteinMPNN design in Tao-style
work. The contract stays backend-neutral so later AlphaFold2, AlphaFold3,
Boltz, or other fold runtimes can write the same normalized report fields.

The materialized fold-check request is:

- `src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/foldcheck_request/input_sequences.fasta`
- `src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/foldcheck_request/foldcheck_request_manifest.yaml`
- `docs/bu-scc/jobs/eco1-colabfold-foldcheck.qsub` for SCC smoke/full
  ColabFold execution from that manifest.

Completed ColabFold output directories are normalized by
`src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_report/`,
which calls the generic `dnadesign.thread.adapters.colabfold` parser and writes
the compact `foldcheck_report.parquet` artifact. This parser does not run a
fold model and does not copy raw SCC output trees into the repository.

The first SCC smoke run, job `6224446`, folded the WT baseline plus five
accepted ProteinMPNN candidates. Its raw output remains on SCC project storage
under `/project/dunlop/esouth/foldcheck/eco1_rt/eco1_colabfold_foldcheck.6224446/`.
The compact local report has been regenerated from that raw output for the
current request hash. It contains six `accepted` rows and 91 explicit `errored`
rows for candidates outside the smoke subset. That smoke output is runtime-path
evidence, not a full candidate screen.

The FASTA contains one WT baseline plus accepted ProteinMPNN candidates as
full 320-aa canonical Eco1 sequences. Terminal positions without 7V9U backbone
coordinates are retained as WT residues in the fold-check sequence; they were
not directly mutated by fixed-backbone ProteinMPNN.

### Runtime Ownership

Fold validation is split across three owners. Eco1 selects the sequences,
reference structure, and thresholds. BU SCC job templates own scheduler and
device execution details such as storage roots, queue resources, and the
`colabfold_batch` command. `dnadesign.thread.adapters.colabfold` owns generic
ColabFold output parsing, and `dnadesign.thread.foldcheck` owns the normalized
request/report fields so ColabFold, AlphaFold-family, or later fold runtimes can
write the same compact artifact without importing Eco1 biology.

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

- No fold-check run without a WT baseline sequence in the same request.
- No materialized fold-check report can pass validation unless the WT baseline
  row is `accepted`; candidate RMSD and degradation checks depend on that
  baseline.
- No accepted fold-check row without a candidate-table row.
- No fold-check row can carry an `input_sequence_hash` that disagrees with the
  current fold-check request manifest for the same candidate id.
- No accepted fold-check row without reference structure provenance.
- No accepted fold-check row without runtime kind, runtime version, runtime
  parameter hash, threshold id, and threshold values.
- No hidden fallback from missing real metrics to fixture metrics.
- No pooled-window handoff from candidates that lack fold-check coverage for
  the actual full sequence being proposed.
- A fixture fold-check row must be labeled `fixture` and cannot satisfy a
  materialized candidate handoff.
- Runtime failures are rows with `status: rejected` or `status: errored`, not
  silent omissions.
- Smoke subsets must leave candidates outside the subset as explicit
  `errored` rows, so a partial fold run cannot masquerade as complete
  fold-check coverage.
- Candidate handoff selection must require `accepted` fold-check rows for the
  selected candidate ids. Presence as an `errored` smoke-row is not acceptance.

### BU SCC Storage Posture

Heavy fold-model outputs should be created on BU SCC project storage, not in
the laptop checkout and not in git. The first SCC preflight is:

```bash
df -h /project/dunlop/esouth /projectnb/dunlop/esouth /scratch/$USER
```

Use `/project/dunlop/esouth` for the repo, environment, and compact run
outputs when live capacity allows. Use `/projectnb/dunlop/esouth` only after a
fresh capacity check. USR sync should move compact manifests and normalized
reports by default; raw ColabFold output directories should stay on SCC unless
a later handoff explicitly selects small structure files for transfer.

Future smoke runs should use `COLABFOLD_EXTRA_ARGS='--num-models 1'` when the
goal is runtime-path validation. Full candidate screens should declare model
count explicitly before submission.
