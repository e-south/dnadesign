# OPAL DenseGen Axis Probe

This package owns the study-local scratch probe that checks whether the current OPAL X surface can recover DenseGen part-derived stress-axis grammar. It is organized by contract boundary rather than as a root-level study script.

- `axis_oracle.py`: DenseGen metadata parsing and vec8 label construction.
- `artifacts.py`: run specs, run-root audit DTOs, and artifact-path layout semantics.
- `label_families.py`: study-owned active/passive synthetic label-family registry.
- `suite_manifest.py`: K12/S3 suite contract and manifest payload.
- `nulls.py`: null-label provenance for quality-ok global permutations.
- `trajectory_metrics.py`: positive-vs-null lift trajectory QA metrics.
- `suite_review.py`: three-seed completion verifier and aggregate summary.
- `plan.py`: run-matrix planning, OPAL stage command construction, and scratch-path policy.
- `plan_fingerprint.py`: `probe_plan.json` hashing and run-root reuse guards.
- `scratch.py`: candidate-table reads, scratch USR cloning, campaign config materialization, and subprocess execution.
- `decision.py`: split metadata, prediction evaluation, decision policy, and decision-report rendering.
- `status.py`: materialized run-root audits.
- `cli.py` and `__main__.py`: command-line entrypoints.

Run with:

```bash
uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe run --gate source
```

The default suite is `densegen_motif_qa_k12_s3_v1`: K12, initial labels 12,
seeds 7/17/29, and 12 planned rounds. A single `run` invocation executes one
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
uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe suite \
  --run-root .var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/densegen_motif_qa_k12_s3_v1_seed7_all_r12 \
  --run-root .var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/densegen_motif_qa_k12_s3_v1_seed17_all_r12 \
  --run-root .var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/densegen_motif_qa_k12_s3_v1_seed29_all_r12 \
  --out-dir .var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/_suite_reviews/densegen_motif_qa_k12_s3_v1_all_r12 \
  --json
```
