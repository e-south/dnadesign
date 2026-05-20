# OPAL DenseGen Axis Probe

This package owns the study-local scratch probe that checks whether the current OPAL X surface can recover DenseGen part-derived stress-axis grammar. It is organized by contract boundary rather than as a root-level study script.

- `axis_oracle.py`: DenseGen metadata parsing and vec8 oracle construction.
- `artifacts.py`: run specs, run-root audit DTOs, and artifact-path layout semantics.
- `plan.py`: run-matrix planning, OPAL stage command construction, and scratch-path policy.
- `scratch.py`: candidate-table reads, scratch USR cloning, campaign config materialization, and subprocess execution.
- `decision.py`: split metadata, prediction evaluation, decision policy, and decision-report rendering.
- `status.py`: materialized run-root audits.
- `cli.py` and `__main__.py`: command-line entrypoints.

Run with:

```bash
uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe run --gate source
```

Use `--rounds N` to run a synthetic multi-round OPAL loop in scratch space.
Round 0 ingests the planned train IDs. Later rounds ingest labels for the
previous round's OPAL-selected candidates, using the study-owned DenseGen oracle
or permuted null for that scratch run only.
