## OPAL DenseGen Axis Probe v0

`opal_densegen_axis_probe_v0` is a minimal study-owned OPAL probe for
`stress_ethanol_cipro_growth`. It tests whether the current 8192-D
LatentDNA/Evo2 X surface lets the three existing RF + SFXI + top-n stress
campaigns recover DenseGen part-derived stress-axis grammar.

This is now a historical K6/single-seed probe. The current planned benchmark is
`densegen_motif_qa_k12_s3_v1`, documented in
`densegen-motif-qa-k12-s3-v1.md`.

The positive oracle is a binary SFXI-compatible vec8 from
`densegen__used_tfbs_detail`: LexA defines cipro, CpxR/BaeR define ethanol, and
both axes define dual/AND. `densegen__plan`,
`densegen__required_regulators`, `sigma35_variant`, and
`densegen__sampling_library_hash` are audit/split fields, not primary label
sources. A distribution-preserving permuted null is a paired diagnostic control.

### Boundary

- `densegen__used_tfbs_detail`
- `densegen__plan` for audit and sigma35 suffix parsing
- `densegen__required_regulators` for audit
- `densegen__sampling_library_hash` for collapse audit
- pad and GC metadata for future audit only

- `latentdna__*`
- `infer__*`
- OPAL predictions or selections
- UMAP coordinates or cluster labels
- archive SFXI labels
- observed assay labels
- previous synthetic labels

Synthetic labels are scratch-only. They must not be written to the shared
`usr_prom_eth_cip_opal_candidates/_opal/observed_labels.parquet` sidecar.

### Vec8 Meaning

The v0 oracle emits binary vec8 labels in the same state order used by the
existing SFXI campaigns: baseline/no-stress, ethanol, ciprofloxacin, and
ethanol-plus-ciprofloxacin. The first four channels are a binary logic proxy.
The last four channels are a binary effect proxy equal to the logic proxy,
because no measured phenotypic effect magnitudes exist pre-assay.

These labels are not real growth phenotypes and are not logic-fidelity values.
`sfxi_v1` computes setpoint fidelity and effect scoring from predicted/observed
vec8 during OPAL evaluation.

### Axis Mapping

| Axis class | Condition | logic4 | effect4 |
| --- | --- | --- | --- |
| `background_only` | no LexA and no CpxR/BaeR | `[0,0,0,0]` | `[0,0,0,0]` |
| `ethanol_only` | CpxR/BaeR and no LexA | `[0,1,0,1]` | `[0,1,0,1]` |
| `cipro_only` | LexA and no CpxR/BaeR | `[0,0,1,1]` | `[0,0,1,1]` |
| `dual_axis_and` | LexA and CpxR/BaeR | `[0,0,0,1]` | `[0,0,0,1]` |

`dual_axis_and` is intentionally `[0,0,0,1]` rather than `[0,1,1,1]` so the AND
campaign distinguishes dual-condition specificity from single-axis OR behavior.

### Run Matrix

The v0 matrix is scoped but uses real OPAL scoring inside each split: train from
observed labels, score the full unlabeled split pool, select greedy top K, ingest
those labels, and repeat. There is no durable-probe candidate cap.

- Oracles: `densegen_part_axis_vec8_v0`, `permuted_densegen_part_axis_vec8_v0`
- Campaigns: cipro, ethanol, AND
- Splits: `random_id`, `leave_sigma35_variant`
- Initial labels: 6, stratified across axis classes
- Selection K: 6 labels per OPAL round
- Seed: 7

The full matrix is 12 runs. Gates allow narrower execution: `source`,
`cipro-random`, `random-all`, `leave-sigma35`, and `all`. `source` validates
DenseGen source, candidate-table X schema/dimension, and label generation only.

### Historical Dogfood Evidence

Latest inspected K6 scratch run before removal:

```text
.var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/20260523T220954Z_seed7_initial6_k6
```

That run was mechanically complete: 12 campaigns finished, 144 prediction ledgers
exist, 12 campaign reviews exist, 12 plot manifests exist, 1,782 configured
OPAL PNG plots are referenced, and the review audit found 0 missing, zero-byte,
bad, or undersized media references. The aggregate review manifest was generated
at `2026-05-24T18:53:46+00:00`.

The K6 scientific decision used the now-retired hard `null_lift > 1.25` STOP
threshold. That threshold is no longer the v1 QA criterion. The retained
interpretation is narrower: K6 validated end-to-end OPAL mechanics and showed
recoverable synthetic DenseGen signal, but it did not support a biological
conclusion or assay-readiness claim. Synthetic labels remain scratch-only and no
shared observed-label sidecar is written.

### CLI

Dry-run is the default:

```bash
uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe run \
  --initial-labels 12 \
  --selection-k 12 \
  --seed 7 \
  --splits random_id,leave_sigma35_variant
```

`--apply` is required to create scratch artifacts or invoke OPAL:

```bash
uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe run \
  --initial-labels 12 --selection-k 12 --seed 7 --rounds 12 \
  --splits random_id,leave_sigma35_variant --score-batch-size 512 --apply
```

`--rounds` controls a scratch active-learning loop: round 0 uses the planned
training split, and later rounds label the previous OPAL selections from the
study-owned DenseGen oracle or permuted null before rerunning OPAL. Synthetic
labels still never enter the shared observed-label sidecar.

Metrics are written for the final scored round and every available round in
`reports/round_metrics.csv` and `reports/round_metrics.jsonl`. Reviews cover
`precision@K`, prevalence, lift, binomial tail p-values, null lift diagnostics,
round dynamics, and paired positive-vs-null trajectory AUC.

The probe inherits OPAL's `safety.max_x_matrix_gib` guard and uses
`writeback.prediction_records: ledger_only`. OPAL validates the 8192-D X
contract, loads records without X, streams score batches, and fails when one
train plus score batch exceeds the budget. Lower `--score-batch-size` before
raising `--max-x-matrix-gib`. Scratch split records are written in filtered
Parquet batches.

By default, apply-mode run roots must live under:

```text
.var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/
```

Use `--allow-custom-run-root` only for an explicit external scratch location so
generated parquet stays out of source, docs, tests, and shared USR truth paths.

For cheap dogfooding without scoring the full candidate pool, stop after scratch
config validation. This still runs OPAL validation and the full candidate-table
X-column scan:

```bash
uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe run \
  --gate cipro-random --stop-after validate --apply
```

Audit a materialized run root:

```bash
uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe status \
  --run-root .var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/<run_id>
```

Split JSON is compact by contract. Full train/eval IDs are stored as split-level
Parquet files so status and review artifacts do not balloon with 157k candidate
IDs.

### Decision

- `PASS_CIPRO_RANDOM_GATE`: the cipro/random positive-vs-null slice safely
  demonstrates non-null DenseGen grammar recovery for that scoped gate only.
- `PASS_RANDOM_ALL_GATE`, `PASS_LEAVE_SIGMA35_GATE`, and
  `PASS_FULL_MATRIX_GATE`: progressively broader scored gates. Narrow gates do
  not imply the broader gates.
- `PASS_SCOPED_GATE`: a valid pass for a custom scored subset whose coverage
  does not match one of the named gates.
- `DEBUG`: OPAL prediction metrics exist, but signal is weak or inconsistent.
- `STOP`: leakage, path contamination, null learnability, or unclear decision
  value.
- `PENDING`: source/materialization/validation gates passed, but no OPAL run
  metrics exist yet; this is not a research decision.

Review artifacts are layered: scratch OPAL campaigns write reusable campaign
review artifacts under `outputs/review/`; the probe writes only the
study-specific aggregate benchmark layer under `reports/`; configured OPAL
campaign plots are refreshed separately from the aggregate report and are
first-class `opal.plot_artifact.v1` artifacts. The scratch campaign `plots.yaml`
uses the same suitable registered OPAL primitives as the stress bundles,
including feature-importance, scalar/vector round history, and SFXI diagnostics.
Per-round browsing is driven by OPAL `round_variants` manifests, not
report-layer file scraping. For final plot review, run
  `uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe plot --run-root <run> --round all --json`,
  then rerun the report with `--plots --json`.
Study-specific aggregate plots remain in `reports/` unless promoted to OPAL
through the registered plot API (`PlotMeta`, `PlotContext`, media/tidy output,
and manifests). This prevents drift between probe-only reports and notebooks.
- `uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe report --run-root <run>`
  rebuilds the review layer over an existing run root.
- `uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe progress --run-root <run>`
  summarizes scratch OPAL round logs without digging through campaign
  directories.

Missing prediction ledgers after a scored OPAL stage are execution contract
failures, not research decisions. The probe is valuable only if it changes a
concrete OPAL readiness or assay-design decision.
