# USR Sync Audit Loop

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


Use this runbook when you need machine-readable transfer decisions during iterative sync between HPC and local clones.

## Goal

- Keep pull/push decisions low friction while preserving strict dataset fidelity.
- Emit JSON artifacts that can be consumed by scripts, notebooks, and chained tool flows.
- Use the same command loop for either endpoint (run on the machine where you want files to land).

## Contract

- Dataset sync defaults to `--verify hash` plus strict sidecar and `_derived`/`_auxiliary` content-hash checks.
- `--audit-json-out` writes a JSON artifact with `usr_output_version` and a stable `data` payload.
- If fidelity must be reduced intentionally, use `--no-verify-sidecars` or `--no-verify-derived-hashes` explicitly.

## Quick loop

```bash
# Set dataset id used by pull/push calls.
DATASET_ID="densegen/my_dataset"
# Set configured remote profile name.
REMOTE="bu-scc"
# Set local directory for audit JSON artifacts.
AUDIT_DIR="${PWD}/.usr-sync-audit"
# Create directory for audit JSON outputs.
mkdir -p "$AUDIT_DIR"

# 1) Preview before transfer and persist machine-readable diff state.
uv run usr diff "$DATASET_ID" "$REMOTE" \
  --audit-json-out "$AUDIT_DIR/diff-before-transfer.json"

# 2) Pull remote -> local and capture machine-readable audit.
uv run usr pull "$DATASET_ID" "$REMOTE" -y \
  --audit-json-out "$AUDIT_DIR/pull.json"

# 3) Optional local updates (notebook, infer write-back, overlay attach, etc.).
# 4) Push local -> remote and capture machine-readable audit.
uv run usr push "$DATASET_ID" "$REMOTE" -y \
  --audit-json-out "$AUDIT_DIR/push.json"
```

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
- `transfer_state=NO-OP` with no changed sections means repeated sync calls are safe.
- `_derived.changed=true` or `_auxiliary.changed=true` means non-primary data moved and should be reviewed.
- `_derived.local_only` / `_derived.remote_only` and `_auxiliary.local_only` / `_auxiliary.remote_only` list the exact file deltas to review.
- `verify.content_hashes=on` confirms strict `_derived` and `_auxiliary` hash parity checks were active.

## Chained workflow usage

Use these artifacts to gate chained execution:

1. Run `pull` with `--audit-json-out`.
2. Parse `.data.transfer_state` and changed sections.
3. Run downstream steps only when expected changes are present.
4. Run `push` with `--audit-json-out`.
5. Archive audit JSON files with run metadata for reproducible operator traces.

## Failure handling

- If transfer fails, rerun the same command; pull uses staged promotion and push verifies post-transfer.
- If fidelity errors occur, inspect sync audit fields (`Primary`, `.events.log`, `_snapshots`, `_derived`, `_auxiliary`) before retrying.
- If remote prerequisites fail, run `uv run usr remotes doctor --remote <name>` and resolve toolchain/lock issues first.
