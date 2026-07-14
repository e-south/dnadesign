## Exec plan: Stress OPAL synthesis handoff

**Status:** active
**Owner:** Shockwing / Codex implementation handoff
**Created:** 2026-06-17
**Last updated:** 2026-06-17
**Authority:** implementation tracker for study-owned physical synthesis handoff

### Purpose / Big Picture

Track implementation of a study-owned OPAL synthesis handoff for the stress,
ethanol, and ciprofloxacin promoter campaigns. The controlling dev spec is
[`synthesis-handoff.md`](../../studies/stress_ethanol_cipro_growth/contexts/opal/synthesis-handoff.md).

This work matters because OPAL-selected promoters must be physically ordered
for assay. The software boundary needs to preserve candidate/model semantics
while producing vendor-ready records with deterministic aliases, lowercase
cloning flanks, uppercase promoter cores, and readback validation.

### Progress

- [x] (2026-06-17 10:34Z) Complete read-only IA audit of OPAL, study, campaign,
  and sibling cloning workbook surfaces.
- [x] (2026-06-17 10:34Z) Persist the improved synthesis handoff dev spec and
  execution tracker before implementation.
- [x] (2026-06-17 10:34Z) Add RED tests for manifest, strategy, alias, and
  workbook readback contracts.
- [x] (2026-06-17 10:39Z) Implement the first study-owned synthesis handoff
  package slice.
- [x] (2026-06-17 10:39Z) Add CLI, route docs, package README, and validation
  evidence.
- [x] (2026-06-17 11:16Z) Extend from fixture selected rows to the checked-in
  batch-0 pre-assay selector and campaign-scoped synthesis output layout.
- [x] (2026-06-17 12:05Z) Extend from batch-0 pre-assay selected rows to
  measured OPAL-ledger selected rows with run/round disambiguation.
- [x] (2026-06-17 12:05Z) Document the repeated ingest/run/verify/synthesis
  lifecycle harness and the anti-drift boundary with the in-silico probe.
- [x] (2026-06-17 16:30Z) Add the OPAL public selection-set reader/export
  surface and route synthesis handoff plus DenseGen probe selection readers
  through shared OPAL selection-artifact semantics.
- [x] (2026-06-17 16:30Z) Add explicit handoff lifecycle fields
  `selection_epoch`, `assay_batch_index`, and `model_as_of_round`.
- [x] (2026-06-17 16:30Z) Add checked-in synthesis handoff lifecycle registry
  scaffold under the stress study record plane.
- [x] (2026-06-17 17:05Z) Make the lifecycle registry executable through
  `--handoff-id`, including source resolution, campaign row-count checks,
  exact artifact-path reporting, SHA-256 inspection, and workbook readback
  status.
- [x] (2026-06-17 18:10Z) Extend lifecycle records for measured OPAL rounds so
  `--handoff-id` carries per-campaign OPAL `run_id` values and stamps the
  physical `assay_batch_index` into the synthesis manifest.
- [x] (2026-06-17 19:05Z) Refine the pending batch-0 pre-assay selector in
  place to a BaeR-forward acquisition prior with actual parsed TFBS regulator
  requirements, explicit CpxR comparator slots, f/e strong sigma-35 slots, d/c
  exploratory slots, and 16-19 bp spacer bounds.

### Surprises & Discoveries

- Observation: Existing OPAL candidates are clean promoter cores, not
  synthesis-order constructs.
- Evidence: Candidate-table validation reported 157,160 rows, unique IDs, 60 nt
  uppercase DNA sequences, and `x_dim=8192`.

- Observation: Sibling cloning workbooks use human-friendly order
  aliases and lowercase flanks around uppercase promoter cores.
- Evidence: `../cloning/genewiz_orders/opal-sfxi-stress-campaigns/*.xlsx`
  examples use `Sequence Name` and `Sequence` columns, with names like
  `ES-promoter-1`.

- Observation: The repo docs gate is currently blocked by unrelated stale
  metadata outside the synthesis-handoff scope.
- Evidence: `uv run python -m dnadesign.devtools.docs.checks` fails only on
  `src/dnadesign/cluster/docs/reference/verification.md` and
  `src/dnadesign/construct/docs/reference/template-contexts.md` stale
  `Last verified` dates.

### Decision Log

- Decision: Put synthesis handoff under the stress-study OPAL decision package,
  not generic OPAL core.
- Rationale: OPAL should remain responsible for active-learning runtime
  semantics; cloning strategy and vendor export are study-owned physical assay
  logistics.
- Date/Author: 2026-06-17 / Codex

- Decision: Use a vendor-neutral manifest as the canonical output and make
  Azenta/GeneWiz a renderer.
- Rationale: This avoids overfitting the ontology to one vendor workbook while
  preserving the concrete order format currently needed.
- Date/Author: 2026-06-17 / Codex

- Decision: Keep canonical candidate `id` and human `synthesis_name` separate.
- Rationale: Vendor-facing names must be stable and human-readable, but assay
  labels and OPAL joins need canonical candidate identity.
- Date/Author: 2026-06-17 / Codex

- Decision: Batch-0 generated synthesis files live under each OPAL campaign's
  `outputs/synthesis_handoff/<batch_id>/` directory.
- Rationale: Operators naturally start from the campaign they are ordering, and
  the matching manifest beside the workbook preserves canonical ID/provenance
  checks without introducing a second checked-in vendor-artifact tree.
- Date/Author: 2026-06-17 / Codex

- Decision: Measured-round synthesis handoff reads OPAL ledger-selected rows,
  not DenseGen probe selection helpers and not only `selection_top_k.csv`.
- Rationale: The ledger resolves `run_id` and `as_of_round`, while
  `opal verify-outputs` remains the OPAL-side artifact consistency check.
- Date/Author: 2026-06-17 / Codex

- Decision: Use one stable synthesis handoff CLI with source modes
  `batch0` and `opal-round`.
- Rationale: Batch zero and later physical rounds differ in source authority,
  but they should produce the same manifest and vendor workbook artifacts.
- Date/Author: 2026-06-17 / Codex

- Decision: OPAL exposes selected rows through `opal selection-set`, and study
  handoff code consumes that public contract rather than parsing OPAL ledger
  parquet directly.
- Rationale: The same selected-row semantics can now be dogfooded by physical
  synthesis handoff and in-silico probe diagnostics without creating a second
  selection ontology.
- Date/Author: 2026-06-17 / Codex

- Decision: Batch zero is represented as `selection_epoch=pre_assay_seed`,
  `assay_batch_index=0`, and `model_as_of_round=null`.
- Rationale: This makes the pre-assay physical seed order distinct from future
  measured OPAL model rounds while keeping it in the same synthesis lifecycle.
- Date/Author: 2026-06-17 / Codex

- Decision: Human operators should use `--handoff-id` for checked-in lifecycle
  records; raw `--source` modes remain implementation/source-authority controls.
- Rationale: The handoff id is the stable record-plane reference that answers
  "which synthesis batch is this?" and lets the CLI validate expected campaign
  rows and artifact paths before order files are used.
- Date/Author: 2026-06-17 / Codex

- Decision: Measured-round handoff records may declare campaign-scoped
  `expected_campaigns[].run_id` values, and record-driven CLI execution passes
  those into OPAL `selection-set`.
- Rationale: OPAL reruns can differ per campaign. The physical handoff record
  must preserve the exact run selected for ordering instead of relying on
  operator shell flags or latest-run inference.
- Date/Author: 2026-06-17 / Codex

- Decision: Refine the existing pending batch-0 handoff instead of creating a
  second batch-zero variant or compatibility path.
- Rationale: The first batch-zero handoff was still
  `generated_pending_acceptance`; carrying multiple batch-zero sources would
  create avoidable operator drift before any order was accepted.
- Date/Author: 2026-06-17 / Codex

- Decision: Treat the BaeR literature prior as an acquisition prior enforced by
  actual parsed DenseGen TFBS regulators, not just regulator-composition
  metadata.
- Rationale: The refined order should test BaeR-bearing promoter designs while
  retaining explicit CpxR comparators; metadata-only motif labels are too weak
  for physical ordering.
- Date/Author: 2026-06-17 / Codex

### Outcomes & Retrospective

Pending. Close or move this tracker only after the first package slice, docs,
and validation evidence are recorded.

### Context and Orientation

Study-owned implementation paths:

- `src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff/`
- `src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff/`
- `docs/studies/stress_ethanol_cipro_growth/contexts/opal/synthesis-handoff.md`
- `docs/studies/stress_ethanol_cipro_growth/routes/decision/opal/README.md`

Relevant existing surfaces:

- Candidate table:
  `src/dnadesign/usr/datasets/usr_prom_eth_cip_opal_candidates/records.parquet`
- Batch-0 selector/provenance package:
  `src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/batch0/`
- OPAL campaign configs:
  `src/dnadesign/opal/campaigns/stress_eth_cip_*_rf_sfxi_topn/configs/campaign.yaml`
- Sibling order examples:
  `../cloning/genewiz_orders/opal-sfxi-stress-campaigns/`

Boundaries:

- Do not mutate candidate records to add cloning flanks.
- Do not add vendor-specific order semantics to OPAL core.
- Do not use vendor aliases as canonical IDs.
- Generated manifests/workbooks belong under `outputs/**` and require review
  before commit.

### Plan of Work

1. Land the dev spec and active execution tracker.
2. Add contract tests for strategy validation, manifest construction, alias
   uniqueness, and workbook round trip.
3. Implement the smallest vendor-neutral package that satisfies those tests.
4. Add CLI and docs once the package contract exists.
5. Extend from fixture selected rows to the checked-in batch-0 selector and
   campaign-scoped output writer.
6. Extend from batch-0 selected rows to live OPAL ledger-selected rows in a
   measured-round source mode.

### Concrete Steps

1. Add this execution tracker and the controlling study spec.
2. Update OPAL study route docs so the handoff surface is discoverable.
3. Add tests under
   `src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff/`.
4. Add contracts and strategy transform modules under
   `src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff/`.
5. Add Azenta/GeneWiz workbook render/readback support.
6. Add `__main__.py` and CLI help with dry-run default behavior.
7. Run targeted tests, ruff, architecture boundaries, docs checks, and
   `git diff --check`.

### Validation and Acceptance

Run and record the exact commands and outcomes:

- `uv run pytest -q src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff`
- `uv run ruff check src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff`
- `uv run ruff format --check src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff`
- `uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff --help`
- `uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .`
- `uv run python -m dnadesign.devtools.docs.checks`
- `git diff --check`

Recent targeted validation:

- RED: `uv run pytest -q src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff`
  failed during collection with
  `ModuleNotFoundError: ...decision.opal.synthesis_handoff`, confirming the
  package contract was not implemented yet.
- GREEN: `uv run pytest -q src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff`
  passed with 8 tests.
- `uv run ruff check src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff`
  passed.
- `uv run ruff format --check src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff`
  passed.
- `uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff --json`
  passed and reported `strategy_id=sfxi_promoter_insert:v1`.
- `uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.batch0.select --config src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/batch0/sampling.yaml`
  passed and reported 18 selected rows across the three campaigns.
- `uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff --source batch0 --json`
  passed and reported 18 validated rows, 6 per campaign, with default
  campaign-local output directories.
- `uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff --source batch0 --write --json`
  passed and wrote per-campaign `synthesis_manifest.csv` and
  `azenta_gene_synthesis.xlsx` files with workbook readback status `pass`.
- RED: `uv run pytest -q src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff`
  failed during collection with
  `ImportError: cannot import name 'selected_candidates_from_opal_round_campaigns'`,
  confirming the measured-round source contract was not implemented yet.
- GREEN: `uv run pytest -q src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff`
  passed with 17 tests after adding the OPAL-ledger source, CLI mode, and
  missing-ledger clean-failure regression.
- RED: `uv run pytest -q src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff/test_synthesis_handoff_contracts.py::test_cli_writes_opal_round_handoff_from_campaign_ledgers`
  failed with `invalid choice: 'opal-round'`, confirming the CLI had no
  measured-round source mode.
- GREEN: the same CLI test passed after wiring `--source opal-round`.
- RED: `uv run pytest -q src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff/test_synthesis_handoff_contracts.py::test_cli_opal_round_missing_ledger_exits_without_traceback`
  failed with an escaping `ValueError`, confirming missing measured-round
  ledgers produced a traceback.
- GREEN: the same missing-ledger CLI test passed after routing validation
  failures through argparse; live dogfood of `--source opal-round --round 0
  --json` exits 2 with a concise missing `outputs/ledger/runs.parquet` error.
- `uv run ruff check src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff`
  passed.
- `uv run ruff format --check src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff`
  passed.
- `uv run ruff check .` passed.
- `uv run ruff format --check .` passed.
- `uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .`
  passed after replacing OPAL internal imports with OPAL public `load_config`
  plus documented campaign output paths.
- `uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff --help`
  passed and shows `--source {selected-csv,batch0,opal-round}`.
- `uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff --source batch0 --write --json`
  passed after the measured-round changes and wrote three ignored campaign
  output folders with readback status `pass`.
- RED: `uv run pytest -q src/dnadesign/opal/tests/cli/test_cli_selection_set.py`
  failed during collection with
  `ImportError: cannot import name 'load_selection_set'`, confirming OPAL had
  no public selected-row contract.
- GREEN: `uv run pytest -q src/dnadesign/opal/tests/cli/test_cli_selection_set.py`
  passed with 4 tests after adding `load_selection_set` and
  `opal selection-set show/export`.
- GREEN: `uv run pytest -q src/dnadesign/opal/tests/cli/test_cli_selection_set.py src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff/test_synthesis_handoff_contracts.py src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe/test_execution_artifacts.py`
  passed with 28 tests after routing synthesis handoff and DenseGen probe
  selection readers through shared OPAL selection-artifact semantics.
- `uv run opal selection-set --help` passed and shows `show` and `export`.
- `uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff --source batch0 --json`
  passed, reporting 18 rows, 6 per campaign, and the three default
  campaign-local output directories.
- The three SFXI source runs have no campaign config and are stored under the
  stress study's `workbench/source_evidence/opal_sfxi_round0/` root. Attempting
  to use one as an executable OPAL campaign exits with a structured
  `opal.cli_error.v1` missing-ledger payload, which
  is expected in the current pre-assay state.
- `uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff --source opal-round --round 0 --json`
  exits 2 with a concise missing
  `outputs/ledger/runs.parquet` error and no traceback, which is expected
  until measured labels and OPAL campaign ledgers exist.
- `uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .`
  passed after the OPAL selection-set public API was added.
- `uv run python -m dnadesign.devtools.docs.checks` still fails only on
  unrelated stale metadata in `src/dnadesign/cluster/docs/reference/verification.md`
  and `src/dnadesign/construct/docs/reference/template-contexts.md`.
- Scoped `uv run ruff check ...`, scoped `uv run ruff format --check ...`,
  and `git diff --check` passed for this implementation surface. Full
  repo-wide `ruff check .` / `ruff format --check .` remain blocked by
  unrelated DenseGen Stage B formatting dirt outside this synthesis handoff
  scope.
- `wc -l` over the three batch0 manifests reported `7` lines each: header plus
  6 campaign rows.
- `git status --short --ignored -- src/dnadesign/opal/campaigns/stress_eth_cip_*_rf_sfxi_topn/outputs/synthesis_handoff`
  reported the campaign `outputs/` folders as ignored (`!!`).
- Campaign-scoped generated output folders are ignored by Git:
  `src/dnadesign/opal/campaigns/stress_eth_cip_*_rf_sfxi_topn/outputs/`.
- `uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .`
  passed.
- `git diff --check -- <touched synthesis-handoff docs/code/tests>` passed.
- `uv run python -m dnadesign.devtools.docs.checks` failed on unrelated stale
  metadata in cluster and construct reference docs; no synthesis-handoff doc
  errors were reported.
- RED: `uv run pytest -q src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff/test_synthesis_handoff_contracts.py::test_cli_resolves_batch0_from_checked_in_handoff_record src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff/test_synthesis_handoff_contracts.py::test_cli_handoff_record_rejects_campaign_count_drift`
  failed because the CLI did not recognize `--handoff-id` or `--record-yaml`.
- GREEN: the same two tests passed after adding the lifecycle record reader,
  source resolution, manifest-record validation, and artifact-status reporting.
- `uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff --handoff-id stress-opal-batch0-sfxi-v1 --json`
  passed and reported 18 rows, 6 per campaign, manifest validation `pass`,
  three expected campaign artifacts, and three workbook readbacks `pass` for
  currently generated ignored output files.

### Known Blockers And Deviations

- Live campaign handoff from OPAL ledgers is implemented and fixture-tested,
  but real stress campaigns do not yet have measured assay labels or OPAL
  ledger runs to dogfood against.
- Production alias policy is not yet accepted for post-batch-0 rounds. Batch
  zero currently uses deterministic `SECG-B0-<campaign>-NN`; measured OPAL
  rounds use `SECG-R<round>-<campaign>-NN` and fail on duplicate aliases.
- Production cloning strategy naming and flanks still need maintainer
  confirmation before live ordering.
- Full docs gate remains blocked by unrelated stale verification dates in
  cluster and construct docs. This implementation did not update those files.

### Links

- Proposal/spec:
  [OPAL synthesis handoff](../../studies/stress_ethanol_cipro_growth/contexts/opal/synthesis-handoff.md)
- Route:
  [OPAL route detail](../../studies/stress_ethanol_cipro_growth/routes/decision/opal/README.md)
