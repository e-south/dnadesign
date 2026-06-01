# OPAL DenseGen Plan-Logic Probe

This package owns two study-local OPAL probes that use DenseGen construction
metadata as synthetic labels. The root package is only the entrypoint layer
(`cli.py` and `__main__.py`). The older plan-logic probe implementation is split
across semantic subpackages; the `tfbs/` subpackage owns the strict TFBS
learnability probe, where each positive campaign is paired with a matched null
so OPAL learnability can be reviewed against a scrambling control rather than
inferred from predicted scores alone.

The organizing rule is ownership, not historical arrival order: study-specific
DenseGen semantics stay here, while OPAL core remains campaign-agnostic.

- `core/`: shared constants, run specs, artifact layouts, path policy, and
  source-column contracts.
- `plan_logic/`: DenseGen metadata parsing, four-channel plan-logic labels,
  active-target wiring, label-family registry, and legacy plan-logic nulls.
- `runtime/`: run-matrix planning, scratch USR materialization, OPAL round
  execution, run-root fingerprinting, and sweep guards.
- `evaluation/`: prediction ledger checks, prediction scoring, round dynamics,
  trajectory metrics, and decision rendering.
- `reporting/`: status/progress surfaces, plot generation, suite manifests,
  suite notebooks, seed-replicate summaries, and review documents.
- `cli.py` and `__main__.py`: command-line entrypoints only.

## TFBS Learnability Subpackage

`tfbs/` answers a narrower question than the root plan-logic probe: can OPAL
learn literal DenseGen TFBS construction labels from the sequence feature
surface without label leakage, and does that learning exceed a matched null?

- `schema.py`: literal label ontology: `count`, `presence`,
  `count_fraction`, and `slot_family_presence`.
- `contracts.py`: strict row parser, final-coordinate slot contract, and
  passive sigma-core validation.
- `oracle.py`, `manifests.py`: positive-label construction and replay
  manifests.
- `nulls/`: matched-null construction split into contracts, exchangeability
  strata, validators, report/provenance generation, and public builders.
- `null_artifacts.py`: null artifact writing.
- `active_targets.py`: scalar expected-label targets for generic OPAL
  `vector_from_table_v1` and `vector_channel_v1` use.
- `retention.py`: preflight retention estimates for sentinel and full-matrix
  campaign footprints.
- `stage_a/materialization.py`, `stage_a/manifests.py`: Stage A label/null
  materialization, source fingerprints, pairings, and retention estimates.
- `stage_b/configs/`: Stage B campaign-set config generation split by
  contract dataclasses, fail-fast validation, and artifact materialization.
- `stage_b/layout.py`: Stage B filesystem ontology.
- `stage_b/io.py`: fail-fast filesystem and parquet/JSON contracts.
- `stage_b/commands.py`: OPAL validation and ingest command contracts.
- `stage_b/payloads.py`: OPAL YAML payload builders.
- `stage_b/seed.py`, `stage_b/semantics.py`: seed policy and campaign
  identity semantics.
- `stage_b/execution.py`, `stage_b/prune.py`: campaign execution and scoped
  artifact cleanup.
- `stage_b/review/`, `stage_b/claims.py`, `stage_b/review_plots.py`,
  `stage_b/slot_diagnostics/`, `stage_b/slot_plots.py`,
  `stage_b/notebook_visuals/`: realized-label review, claim gates, and
  registry-backed notebook-facing visual registration. The `review/` package
  keeps artifact readers, trajectory frames, summary payloads, and
  materialization separate.

Run with:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.densegen_axis_probe run --gate source
```

Materialize the v1 TFBS learnability Stage A label/null/preflight artifacts with:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.densegen_axis_probe tfbs-stage-a \
  --run-id densegen_tfbs_learnability_stage_a_seed7 \
  --json
```

This writes the positive label table, sentinel matched-null label tables, null
viability reports, row-universe/source/label manifests, `pairing_manifest.json`,
`retention_estimate.json`, and `tfbs_stage_a_manifest.json`. It does not run OPAL
campaigns.

The default suite is `densegen_motif_qa_k12_s3_v1`: K12, initial labels 12,
seeds 7/17/29, 12 planned rounds, and active `densegen_plan_logic4` plus
`tf_family_count` campaign matrices. A single `run` invocation executes one
seed; repeat with `--seed 17` and `--seed 29` after the seed-7 burn-in is clean.

Use `--rounds N` to run a synthetic multi-round OPAL loop in scratch space.
Round 0 ingests the planned train IDs. Later rounds ingest labels for the
previous round's OPAL-selected candidates, using the study-owned DenseGen oracle
or permuted null for that scratch run only.

Applied runs write `probe_plan.json` at the run root and refuse to reuse a
nonempty root with a missing or mismatched plan. Use a new `--run-id` for normal
reruns; `--replace-run-root` intentionally deletes and rebuilds the scratch root.
Dry-run JSON reports `planned_plan_path` and `writes_artifacts: false`; it does
not claim that `probe_plan.json` already exists.
`progress --json` is compact by default; add `--full` to include the nested OPAL
campaign progress payloads.
`status --json` exits nonzero for materialized or partially scored roots; `ok`
means scored metrics, a decision, and expected final-round coverage exist for
the run plan.
Refresh configured OPAL plot indexes with `plot --run-root <run> --round all
--json`, then rerun `report --run-root <run> --plots --json` for the final
artifact review.
After seeds 7/17/29 are complete, write the cross-seed completion manifest with:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.densegen_axis_probe suite \
  --run-root .var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/densegen_motif_qa_k12_s3_v1_seed7_all_r12 \
  --run-root .var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/densegen_motif_qa_k12_s3_v1_seed17_all_r12 \
  --run-root .var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/densegen_motif_qa_k12_s3_v1_seed29_all_r12 \
  --out-dir .var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/_suite_reviews/densegen_motif_qa_k12_s3_v1_all_r12 \
  --json
```
