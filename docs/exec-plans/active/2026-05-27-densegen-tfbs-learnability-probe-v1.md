## Exec plan: DenseGen TFBS learnability probe v1

**Status:** active
**Owner:** Shockwing / Codex implementation handoff
**Created:** 2026-05-27
**Last updated:** 2026-05-30
**Authority:** implementation tracker for the study-owned v1 production probe

### Purpose / Big Picture

Track implementation of the stress-study DenseGen TFBS learnability probe v1.
The controlling spec is
[`densegen-tfbs-learnability-probe-v1.md`](../../studies/stress_ethanol_cipro_growth/contexts/opal/densegen-tfbs-learnability-probe-v1.md).

The production probe asks whether OPAL can enrich for DenseGen variable TFBS
construction labels from the current OPAL X representation. DenseGen parsing,
TF family names, slot contracts, sigma-core controls, label/null manifests, and
study-specific captions stay in the study package. OPAL core remains
campaign-agnostic.

### Progress

- [x] (2026-05-27 00:00Z) Add strict v1 parser for
  `densegen__used_tfbs_detail`.
- [x] (2026-05-27 00:00Z) Enforce three TFBS entries, two fixed elements,
  60 bp sequence length,
  `offset_raw` slot ordering, coordinate bounds, sigma35/sigma10 role mapping,
  spacer relation, and unknown-regulator failures.
- [x] (2026-05-27 00:00Z) Add positive-oracle label table builder for count,
  presence,
  count-fraction, and slot-family labels.
- [x] (2026-05-27 00:00Z) Add v1 label-family manifest surface without making
  historical
  `densegen_plan_logic4` a production v1 family.
- [x] (2026-05-27 00:00Z) Run a read-only live Stage A parser/label-rate check
  on the current candidate table and DenseGen sidecar.
- [x] (2026-05-30 00:00Z) Run Stage A on the live candidate/sidecar snapshot.
- [x] (2026-05-30 00:00Z) Implement matched null construction and null
  viability reports.
- [x] (2026-05-30 00:00Z) Add compact null-label artifact writer with
  replay hashes for Parquet/Zstandard label tables and viability JSON.
- [x] (2026-05-30 00:00Z) Add v1 scalar expected-label active target specs for
  sentinel/config materialization without plan-vector similarity.
- [x] (2026-05-30 00:00Z) Add Stage A materialization API/CLI for positive
  labels, sentinel nulls, source-file hashes, pairing manifest, and Stage A
  summary manifest.
- [x] (2026-05-30 00:00Z) Generate and validate Stage B sentinel OPAL
  configs for seed 7, random_id, positive and matched-null roles.
- [x] (2026-05-30 00:00Z) Add retention preflight estimator and Stage A
  retention estimate manifest for sentinel and full-matrix modes.
- [ ] (2026-05-27 00:00Z) Add true-label enrichment and sigma-core balance
  review surfaces.

### Surprises & Discoveries

- Observation: The existing package still needs the historical
  `densegen_plan_logic4` and K12/S3 surfaces for prior evidence and tests.
- Evidence: `label_families.py`, `active_targets.py`, and existing probe tests
  still exercise historical suite contracts.

### Decision Log

- Decision: Implement v1 parsing and positive-label construction in new
  semantic modules rather than rewriting `axis_oracle.py`.
- Rationale: This preserves historical evidence while giving the production v1
  probe strict names, fail-fast contracts, and no legacy target aliases.
- Date/Author: 2026-05-27 / Codex

- Decision: Implement v1 null construction in `tfbs_nulls.py` plus a separate
  `tfbs_null_artifacts.py` writer instead of extending the historical
  `nulls.py` helper.
- Rationale: The old helper is tied to global plan-logic permutations; v1 needs
  explicit family-content and slot-geometry matched nulls, viability statuses,
  and artifact hashes without changing historical K12/S3 behavior.
- Date/Author: 2026-05-30 / Codex

- Decision: Implement scalar expected-label target declarations in
  `tfbs_active_targets.py` instead of expanding the legacy
  campaign-key-oriented `active_targets.py`.
- Rationale: The v1 target ontology is literal label-name driven, while the
  legacy surface is still needed for historical plan-logic and TF-count suite
  tests.
- Date/Author: 2026-05-30 / Codex

- Decision: Implement Stage A materialization in `tfbs_stage_a.py` with
  retention estimates isolated in `tfbs_retention.py` and source/pairing/stage
  manifest construction isolated in `tfbs_stage_a_manifests.py`.
- Rationale: Stage A is a study-owned label/null/preflight boundary. Splitting
  retention and manifest builders keeps DenseGen semantics out of OPAL core and
  avoids turning the CLI path into a file monolith.
- Date/Author: 2026-05-30 / Codex

- Decision: Implement Stage B sentinel config generation in
  `tfbs_stage_b_configs.py` instead of extending the historical scratch
  planner.
- Rationale: Stage B v1 uses literal TFBS label-name targets and
  positive/null pairing manifests. A separate generator keeps DenseGen oracle
  semantics, split/seed contracts, initial label inputs, actual config hashes,
  and validation reports in the study package while OPAL consumes only generic
  `vector_from_table_v1` and `vector_channel_v1` YAML.
- Date/Author: 2026-05-30 / Codex

### Outcomes & Retrospective

Pending. Close or move this tracker only after Stage A/B/C status and artifact
locations are recorded.

### Context and Orientation

Study-owned implementation paths:

- `src/dnadesign/studies/units/stress_ethanol_cipro_growth/opal_densegen_axis_probe/`
- `src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/opal_densegen_axis_probe/`
- `docs/studies/stress_ethanol_cipro_growth/contexts/opal/densegen-tfbs-learnability-probe-v1.md`

Boundaries:

- DenseGen parsing, TF family names, sigma-core controls, label/null manifests,
  and study-specific captions stay in the study package.
- OPAL core remains campaign-agnostic.
- The v1 active objective is predicted expected scalar label value.
- Historical plan-logic targets remain evidence surfaces, not v1 production
  active labels.

### Plan of Work

1. Land the v1 parser and positive oracle label table.
2. Run Stage A on the live source snapshot and record label-rate evidence.
3. Add matched nulls and null viability reports.
4. Materialize sentinel OPAL configs only after Stage A gates pass.
5. Add retention estimates/manifests before running larger matrices.
6. Add report surfaces for raw true-label enrichment, positive/null deltas, and
   sigma-core balance diagnostics.

### Concrete Steps

1. Add v1 row parser and coordinate/sigma-core contracts under
   `opal_densegen_axis_probe/tfbs_contracts.py`.
2. Add positive-oracle label/manifests under
   `opal_densegen_axis_probe/tfbs_oracle.py`.
3. Add v1 label-family registry surface in
   `opal_densegen_axis_probe/label_families.py`.
4. Add focused parser and label tests under
   `tests/opal_densegen_axis_probe/test_tfbs_learnability_*.py`.
5. Add live Stage A CLI/materialization path and record generated artifact
   locations here.
6. Add matched-null construction and null artifact writers.
7. Add sentinel config generation in a separate follow-up slice.

### Stage Gates

Stage A: complete for the live seed-7 snapshot. Parser, positive-label,
matched-null, null-viability, null-artifact, scalar target, Stage A
materialization, source-file hash, pairing-manifest, and retention-estimate
contracts exist. Live materialization produced 157,160 active rows, 23 expected
sidecar-only rows, five sentinel null tables with `PASS` viability reports, and
a `PASS` retention estimate. The generated Stage A run root is
`.var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/densegen_tfbs_learnability_stage_a_seed7_20260530`.

Stage B: exact-budget sentinel execution complete for the seed-7 sentinel pass.
The generator produced 10 OPAL configs for five sentinel labels across positive
and matched-null roles, with `random_id` split, 24 rounds, `selection_k=6`,
`initial_label_count=6`, `tie_handling=ordinal`,
`selection_budget_mode=exact_top_k`, manifest-backed positive/null pairs,
deterministic round-0 label input files, candidate scope, source records
symlink, actual config hashes, and OPAL validation reports. All 10 generated
configs passed `opal validate`, all 10 campaigns executed with status `PASS`,
and audit checks confirmed exactly 6 selected rows per round with 144 sidecar
labels per campaign.

Stage C: not started. The full matrix must wait for Stage A and Stage B gates.

### Validation and Acceptance

Run and record the exact commands and outcomes:

- `uv run pytest -q src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/opal_densegen_axis_probe`
- `uv run ruff check src/dnadesign/studies/units/stress_ethanol_cipro_growth/opal_densegen_axis_probe src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/opal_densegen_axis_probe`
- `uv run ruff format --check src/dnadesign/studies/units/stress_ethanol_cipro_growth/opal_densegen_axis_probe src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/opal_densegen_axis_probe`
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .`
- `uv run python -m dnadesign.devtools.docs.checks`
- `git diff --check`

Recent targeted validation:

- `uv run pytest -q src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/opal_densegen_axis_probe/test_tfbs_stage_b_configs.py`
  passed with 3 tests.
- `uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe tfbs-stage-b-configs --stage-a-run-root .var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/densegen_tfbs_learnability_stage_a_seed7_20260530 --replace-out-dir --json`
  passed and reported `campaign_count=10`, `validation_status=PASS`.
- `uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe tfbs-stage-a --run-root .var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/densegen_tfbs_learnability_stage_a_seed7_20260530 --replace-run-root --json`
  passed after replacing stale Stage B campaign artifacts under the run root.
- `uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe tfbs-stage-b-configs --stage-a-run-root .var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/densegen_tfbs_learnability_stage_a_seed7_20260530 --json`
  passed with 10 configs, 24 rounds, `selection_k=6`, `initial_label_count=6`,
  `tie_handling=ordinal`, and `selection_budget_mode=exact_top_k`.
- `uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe tfbs-stage-b-run --config-manifest .var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/densegen_tfbs_learnability_stage_a_seed7_20260530/stage_b_sentinel_configs/manifests/stage_b_sentinel_config_manifest.json --json`
  passed with `campaign_count=10` and 24 rounds.
- Stage B artifact audit passed: every `selection_top_k.csv` contained exactly
  6 rows, every sidecar contained 144 labels, every round contributed exactly 6
  labels, and all 10 campaigns wrote `outputs/retention_manifest.json`.
- Targeted validation passed:
  `uv run pytest -q src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/opal_densegen_axis_probe/test_tfbs_stage_a_materialization.py src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/opal_densegen_axis_probe/test_tfbs_stage_b_configs.py src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/opal_densegen_axis_probe/test_tfbs_stage_b_execution.py`
  reported 11 tests passed.
- `uv run ruff check ...` and `uv run ruff format --check ...` passed for the
  touched Stage B implementation and tests.
- `uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .`
  passed.

### Known Blockers And Deviations

- Full Stage C production review remains gated on review/reporting surfaces and
  seed-replicate summaries.
- Runtime retention enforcement exists for the generated Stage B campaign
  outputs and writes retention manifests, but broader report surfaces are still
  pending.
- Stage B exact-budget execution is complete; gate review/reporting remains
  pending.
- The in-memory oracle build still uses dataframe hashes before materialization;
  the Stage A command replaces them with actual source-file hashes in written
  manifests.

### Artifact Locations

Planned Stage A local artifacts:

- `.var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/<run_id>/labels/densegen_tfbs_learnability_positive_v1.parquet`
- `.var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/<run_id>/manifests/row_universe_manifest.json`
- `.var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/<run_id>/manifests/label_manifest.json`
- `.var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/<run_id>/manifests/source_hash_manifest.json`
- `.var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/<run_id>/manifests/retention_estimate.json`
- `.var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/<run_id>/manifests/pairing_manifest.json`
- `.var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/<run_id>/manifests/tfbs_stage_a_manifest.json`
- `.var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/<run_id>/labels/densegen_tfbs_learnability_*_null_v1__<label>__seed<seed>.parquet`
- `.var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/<run_id>/manifests/densegen_tfbs_learnability_*_null_v1__<label>__seed<seed>.null_viability_report.json`

Live Stage A run root:

- `.var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/densegen_tfbs_learnability_stage_a_seed7_20260530`

Live Stage A evidence:

- `active_row_count`: 157,160
- `sidecar_only_id_count`: 23
- source file hashes recorded in `row_universe_manifest.json` and
  `source_hash_manifest.json`
- sentinel null tables: 5
- null viability reports: all `PASS`
- retention estimate: `PASS`, full-matrix planned campaigns 144, sentinel
  initial planned campaigns 10, `max_expected_total_bytes` 4,440,175,872

Live Stage B sentinel config root:

- `.var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/densegen_tfbs_learnability_stage_a_seed7_20260530/stage_b_sentinel_configs`

Live Stage B sentinel evidence:

- config manifest:
  `.var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/densegen_tfbs_learnability_stage_a_seed7_20260530/stage_b_sentinel_configs/manifests/stage_b_sentinel_config_manifest.json`
- execution manifest:
  `.var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/densegen_tfbs_learnability_stage_a_seed7_20260530/stage_b_sentinel_configs/manifests/stage_b_sentinel_execution_manifest.json`
- campaign count: 10
- pair count: 5
- rounds: 24
- `selection_k`: 6
- `initial_label_count`: 6
- `selection_tie_handling`: `ordinal`
- `selection_budget_mode`: `exact_top_k`
- sentinel labels:
  `lexA_present`, `cpxR_or_baeR_present`, `lexA_count_fraction`,
  `lexA_in_slot0`, `cpxR_or_baeR_in_slot2`
- validation status: `PASS` for all 10 generated OPAL configs
- execution status: `PASS` for all 10 campaigns
- exact-budget audit: 6 selected rows per round and 144 labels per sidecar for
  every campaign
- exact-budget positive/null trajectory snapshot:
  `lexA_count_fraction` remains the cleanest signal
  (`final_positive_minus_null_lift=3.13`, `AUC delta=2.60`);
  `lexA_present` and `cpxR_or_baeR_present` remain learnable under the fixed
  budget (`final delta=1.02` each);
  `lexA_in_slot0` separates better than in the stale tie-expanded run
  (`final delta=1.63`);
  `cpxR_or_baeR_in_slot2` remains weak/ambiguous because final null lift exceeds
  positive lift even though AUC delta is slightly positive.

### Links

- Spec: [DenseGen TFBS learnability probe v1](../../studies/stress_ethanol_cipro_growth/contexts/opal/densegen-tfbs-learnability-probe-v1.md)
