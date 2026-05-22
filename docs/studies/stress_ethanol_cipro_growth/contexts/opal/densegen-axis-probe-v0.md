## OPAL DenseGen Axis Probe v0

`opal_densegen_axis_probe_v0` is a minimal study-owned OPAL probe for
`stress_ethanol_cipro_growth`. It tests whether the current 8192-D
LatentDNA/Evo2 X surface lets the three existing RF + SFXI + top-n stress
campaigns recover DenseGen part-derived stress-axis grammar.

The positive oracle is a binary SFXI-compatible vec8 generated from
`densegen__used_tfbs_detail` regulator composition: LexA defines the cipro
axis, CpxR/BaeR define the ethanol axis, and both axes define the dual/AND
class. `densegen__plan`, `densegen__required_regulators`, `sigma35_variant`,
and `densegen__sampling_library_hash` are audit/split fields, not primary label
sources. A distribution-preserving permuted null must fail.

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

The v0 matrix is intentionally scoped, but it should use real OPAL scoring
semantics inside each split: OPAL trains from observed labels, scores the full
unlabeled split pool, selects the greedy top K, ingests those labels, and repeats
round over round. There is no candidate cap in the durable probe contract.

- Oracles: `densegen_part_axis_vec8_v0`,
  `permuted_densegen_part_axis_vec8_v0`
- Campaigns: cipro, ethanol, AND
- Splits: `random_id`, `leave_sigma35_variant`
- Initial labels: 6, stratified across axis classes
- Selection K: 6 labels per OPAL round
- Seed: 7

The full matrix is 12 runs. Gates allow narrower execution:

- `source`: DenseGen oracle source validation, candidate-table X schema/dimension
  validation, and label generation only
- `cipro-random`: positive/null cipro random split
- `random-all`: all three campaigns under random split
- `leave-sigma35`: all three campaigns under held-out sigma35
- `all`: full matrix

### CLI

Dry-run is the default:

```bash
uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe run \
  --initial-labels 6 \
  --selection-k 6 \
  --seed 7 \
  --splits random_id,leave_sigma35_variant
```

`--apply` is required to create scratch artifacts or invoke OPAL:

```bash
uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe run \
  --initial-labels 6 --selection-k 6 --seed 7 --rounds 12 \
  --splits random_id,leave_sigma35_variant --score-batch-size 512 --apply
```

`--rounds` controls a scratch active-learning loop: round 0 uses the planned
training split, and later rounds label the previous OPAL selections from the
study-owned DenseGen oracle or permuted null before rerunning OPAL. Synthetic
labels still never enter the shared observed-label sidecar.

Metrics are written for the final scored round and every available round in
`reports/round_metrics.csv` and `reports/round_metrics.jsonl`. Reviews cover
`precision@K`, prevalence, lift, binomial tail p-values, and null lift.

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

Review artifacts are layered:

- Each scratch OPAL campaign writes reusable OPAL campaign review artifacts under
  `outputs/review/`.
- The probe writes only the study-specific aggregate benchmark layer under
  `reports/`.
- Configured OPAL campaign plots are refreshed separately from the aggregate
  report. For final plot review, run
  `uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe plot --run-root <run> --round all --json`,
  then rerun the report with `--plots --json`.
- `uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe report --run-root <run>`
  rebuilds the review layer over an existing run root.
- `uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe progress --run-root <run>`
  summarizes scratch OPAL round logs without digging through campaign
  directories.

Missing prediction ledgers after a scored OPAL stage are execution contract
failures, not research decisions. The probe is valuable only if it changes a
concrete OPAL readiness or assay-design decision.
