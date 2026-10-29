<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[SeqLedger guide](../../README.md)

# Read sync audit output

Use `--audit-json-out` to capture comparison and transfer results for scripts or run records. Configure the local root and remote profile using the [sync setup](setup.md) first. These examples assume an activated SeqLedger environment; `jq` is used only to inspect JSON.

## Goal

- Keep pull/push decisions low friction while preserving strict dataset fidelity.
- Emit JSON artifacts that can be consumed by scripts, notebooks, and chained tool flows.
- Choose `pull` for remote-to-local transfer or `push` for local-to-remote transfer.

## Contract

- Dataset sync defaults to `--verify hash` plus strict sidecar and `_derived`/`_auxiliary` content-hash checks.
- `--audit-json-out` writes a JSON artifact with `usr_output_version` and a stable `data` payload.
- If fidelity must be reduced intentionally, use `--no-verify-sidecars` or `--no-verify-derived-hashes` explicitly.

## Quick loop

```bash
# Set dataset id used by pull/push calls.
DATASET_ID="my_dataset"
# Set configured remote profile name.
REMOTE="cluster"
# Set local directory for audit JSON artifacts.
AUDIT_DIR="${PWD}/.seqledger-sync-audit"
# Create directory for audit JSON outputs.
mkdir -p "$AUDIT_DIR"

# 1) Preview before transfer and persist machine-readable diff state.
seqledger diff "$DATASET_ID" "$REMOTE" \
  --audit-json-out "$AUDIT_DIR/diff-before-transfer.json"

# 2) Pull remote -> local and capture machine-readable audit.
seqledger pull "$DATASET_ID" "$REMOTE" \
  --audit-json-out "$AUDIT_DIR/pull.json"

```

For local-to-remote transfer, substitute `push` and save its audit under a corresponding filename. Do not run both directions automatically.

## Decision checks from JSON

```bash
# Show transfer state and high-level change indicators.
jq -r '.data | [.action, .transfer_state, .primary.changed, ._derived.changed, ._auxiliary.changed] | @tsv' \
  "$AUDIT_DIR/pull.json"

# Show contract version and verify profile.
jq -r '[.usr_output_version, .data.verify.primary, .data.verify.sidecars, .data.verify.content_hashes] | @tsv' \
  "$AUDIT_DIR/pull.json"

# Show concrete file deltas for sidecar decisions.
jq -r '.data | {derived_local_only: ._derived.local_only, derived_remote_only: ._derived.remote_only, aux_local_only: ._auxiliary.local_only, aux_remote_only: ._auxiliary.remote_only}' \
  "$AUDIT_DIR/pull.json"
```

Interpretation:
- `transfer_state=NO-OP` with no changed sections means this invocation found nothing to transfer.
- `_derived.changed=true` or `_auxiliary.changed=true` indicates a difference in non-primary data. Read it together with the action and transfer state to determine whether data were transferred.
- `_derived.local_only` / `_derived.remote_only` and `_auxiliary.local_only` / `_auxiliary.remote_only` list the exact file deltas to review.
- `verify.content_hashes=on` confirms strict `_derived` and `_auxiliary` hash parity checks were active.

## Chained workflow usage

Use these artifacts to gate chained execution:

1. Check the transfer command's exit status before consuming its audit.
2. Parse `.data.transfer_state` and the changed sections.
3. Run downstream steps when the expected data are available.
4. Keep the audit with the run outputs. Push later only if local work should be transferred back.

## Failure handling

- If a transfer fails, inspect the error and compare the copies before retrying. Pull stages before promotion; push verifies after transfer.
- If fidelity errors occur, inspect sync audit fields (`Primary`, `.events.log`, `_snapshots`, `_derived`, `_auxiliary`) before retrying.
- If remote prerequisites fail, run `seqledger remotes doctor --remote <name>` and resolve toolchain/lock issues first.
