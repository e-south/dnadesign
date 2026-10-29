<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[SeqLedger guide](../../README.md) / [Sync](README.md)

# Preview and transfer a dataset

Run these commands in an activated SeqLedger environment after [configuring an SSH remote](setup.md). Replace the paths and names with an existing local root, a dataset ID and a remote profile from your private configuration.

```bash
# Select the local data root and remote configuration for this shell.
export DNADESIGN_SEQLEDGER_ROOT="/absolute/path/to/seqledger_datasets"
# Select the private connection settings.
export SEQLEDGER_REMOTES_PATH="$HOME/.config/dnadesign/seqledger-remotes.yaml"
# Name the dataset to compare.
DATASET="example_sequences"
# Choose a profile from the remote configuration.
REMOTE="cluster"
```

Without an explicit root, SeqLedger uses `./seqledger_datasets`. Dataset IDs resolve beneath the local root and the remote profile's `base_dir`.

## 1. Compare the copies

```bash
# Preview differences without transferring data.
seqledger diff "$DATASET" "$REMOTE"
```

Review the primary table, event log, overlays and auxiliary-file differences. If both copies have independently changed, reconcile their histories before transferring; sync does not merge divergent annotation histories automatically.

## 2. Choose one direction

To bring remote results onto this machine:

```bash
# Preview a pull, then transfer remote state to the local root.
seqledger pull "$DATASET" "$REMOTE" --dry-run
# Apply the remote-to-local transfer.
seqledger pull "$DATASET" "$REMOTE"
```

To send local results to the remote machine instead:

```bash
# Preview a push, then transfer local state to the remote root.
seqledger push "$DATASET" "$REMOTE" --dry-run
# Apply the local-to-remote transfer.
seqledger push "$DATASET" "$REMOTE"
```

The commands prompt when confirmation is required. Add `-y` only when the transfer direction and overwrite decision are already known. A pull can create a missing local dataset from an existing remote one.

## 3. Check the result

```bash
# Confirm that the copies agree and save the comparison for a script or record.
seqledger diff "$DATASET" "$REMOTE" --audit-json-out sync-audit.json
```

A full transfer checks the primary hash, sidecar inventory, and `_derived` and `_auxiliary` content hashes. The final comparison should report no remaining differences. See [audit output](audit-loop.md) for the JSON fields.

## Transfer options

- `--strict-bootstrap-id` requires the explicit canonical dataset ID when creating a local copy.
- `--audit-json-out <path>` saves a machine-readable audit alongside console output.
- `--primary-only` and `--skip-snapshots` transfer less data and require disabling full sidecar verification. Use them only when a partial copy is intended.
- `--verify`, `--no-verify-sidecars` and `--no-verify-derived-hashes` change the checks performed. See [target modes and verification](modes.md) before relaxing them.

For a failed transfer, follow [troubleshooting](troubleshooting.md). For repeated HPC batches, use the [HPC workflow](hpc-agent-flow.md).
