<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[SeqLedger guide](../../README.md) / [Operations](../README.md) / [Sync](README.md)

# Sync verification drills

Use these contributor checks after changing transfer, overlay or registry behavior. They exercise isolated fixtures; normal dataset users can follow the [transfer quickstart](quickstart.md).

## Local sync audit drill

From the repository root, run the drill against temporary local and simulated remote datasets. It tests `diff`, `pull` and `push` across `_derived`, `_auxiliary` and `_registry` changes without an SSH account.

```bash
# Save the audit in the ignored build directory.
mkdir -p build/seqledger-checks
# Choose one location for the reports.
ARTIFACT_ROOT="$PWD/build/seqledger-checks"
# Run the isolated transfer checks.
uv run seqledger-sync-audit-drill   --report-json "$ARTIFACT_ROOT/sync-audit.json"
```

Add `--work-dir "$ARTIFACT_ROOT/sync-drill"` to retain the fixture roots for inspection. The report records each comparison and transfer check; a nonzero exit requires investigation before accepting the change.

## Dataset and schema checks

```bash
# Exercise iterative batches, invalid schemas and drill behavior.
uv run pytest -q   src/dnadesign/seqledger/tests/sync/test_sync_iterative_batch_flow.py   src/dnadesign/seqledger/tests/sync/test_sync_schema_adversarial.py   src/dnadesign/seqledger/tests/test_usr_sync_audit_drill_script.py
```

These checks cover overlay continuity, malformed writes and sync audit outcomes. Default dataset transfer verification includes primary hashes, sidecar inventory and `_derived`/`_auxiliary` content hashes. A test should not disable those checks merely to make a failed transfer pass.

## Combined harness cycle

The harness combines mock data, schema tests and optional sync auditing:

```bash
# Run the cycle with reports and the local sync drill enabled.
SEQLEDGER_HARNESS_RUN_SYNC_AUDIT_DRILL=1 SEQLEDGER_HARNESS_REPORT_PATH="$ARTIFACT_ROOT/harness-report.json" SEQLEDGER_HARNESS_SYNC_AUDIT_REPORT_PATH="$ARTIFACT_ROOT/sync-drill-report.json"   bash src/dnadesign/seqledger/scripts/run_seqledger_harness_cycle.sh
```

For command routing and ownership, see the [architecture guide](../../architecture-introspection.md). For live connection failures, use [sync troubleshooting](troubleshooting.md); do not run destructive fault-injection drills against working datasets.
