<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

# Test OPAL workflows

Use the demo matrix to check the complete selection loop in isolated campaign
copies. For a first run with your own data, follow the
[campaign guide](../workflows/campaign-round.md).

## Run the demo matrix

```bash
uv run opal demo-matrix --rounds 0 --fail-fast
```

For each maintained demo, this resets the temporary campaign, initializes and
validates it, imports synthetic measurements, fits the predictor, selects
candidates and checks the saved selections against the recorded predictions.
It does not exercise every inspection, prediction or plotting command.

Use `--rounds 0,1` to check successive rounds using the synthetic label fixture.
Add `--keep` to inspect the temporary campaigns afterward. Success ends with
`overall_ok: True`; a failed demo produces a nonzero exit status.

## Choose checks for a change

The [test layout](../../tests/README.md) identifies the suites that own CLI,
configuration, storage, model, objective and selection behavior. Start with the
suite covering the behavior you changed. Use the
[repository change gate](../../../../../docs/repo-change-gate.md) for the matching
formatting, architecture, documentation and routing checks.

The [architecture guide](../concepts/architecture.md) maps configuration to the
runtime stages when a change crosses those boundaries.
