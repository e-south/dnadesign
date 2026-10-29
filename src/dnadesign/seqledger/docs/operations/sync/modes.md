<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[SeqLedger guide](../../README.md) / [Sync](README.md)

# Target modes and verification

Dataset-ID mode is the usual choice: `seqledger pull example_sequences cluster` resolves the dataset beneath the local root and the remote profile's `base_dir`. Use an explicit directory or single-file target when that mapping does not fit.

## Dataset directory mode

An absolute dataset directory can be outside the selected local root:

```bash
# Compare an explicit dataset directory, then pull after reviewing the diff.
seqledger diff /data/seqledger_datasets/example_sequences cluster
# Copy remote dataset state after reviewing the comparison.
seqledger pull /data/seqledger_datasets/example_sequences cluster
```

Use `push` for the reverse direction. The [quickstart](quickstart.md) explains the common preview and verification steps.

## File mode

File mode transfers one file rather than the complete dataset. It cannot establish parity for omitted overlays, views or event history.

A remote profile can map repository-relative files using `repo_root` and `local_repo_root`:

```yaml
# Optional repository mappings supplement the dataset root.
remotes:
  cluster:
    type: ssh
    host: compute.example.org
    user: your-user
    base_dir: /project/your-group/seqledger_datasets
    repo_root: /project/your-group/analysis
    local_repo_root: /absolute/path/to/analysis
```

```bash
# Compare and retrieve one file relative to the configured repository roots.
seqledger diff outputs/annotated_sequences.parquet cluster
# Retrieve the selected file.
seqledger pull outputs/annotated_sequences.parquet cluster

# Supply a local mapping explicitly when local_repo_root is not configured.
seqledger pull outputs/annotated_sequences.parquet cluster   --repo-root /absolute/path/to/analysis

# Specify the remote file directly when its path cannot be inferred.
seqledger pull outputs/annotated_sequences.parquet cluster   --remote-path /project/your-group/results/annotated_sequences.parquet
```

`DNADESIGN_REPO_ROOT` can supply the local mapping when `--repo-root` is absent. Dataset sidecar checks do not apply to file mode.

## Verification choices

The primary verification mode defaults to `--verify hash`. `--verify auto` tries SHA-256, then file size, then Parquet row/column checks; fallback emits a warning. Size and table-shape checks do not establish identical file contents.

Full dataset transfers additionally verify sidecar inventories and content hashes for `_derived` and `_auxiliary` files. `--no-verify-derived-hashes` skips those content hashes while retaining inventory checks. `--no-verify-sidecars` disables strict sidecar parity and is required for partial transfers using `--primary-only` or `--skip-snapshots`.

## Transfer consistency

- Transfers require an existing source `records.parquet` and fail on SSH authentication or transport errors.
- Full transfers lock dataset and event state. Runtime lock and lease files are excluded from transfer inventories.
- Push requires remote event history to be an exact prefix of local history; pull requires remote history to extend the local prefix. Independent branches are rejected rather than silently merged.
- Transfer actions do not append host-local events to the replicated `.events.log`. Successful full transfers leave the copies byte-identical.
- `--dry-run` does not create remote directories or locks. A real push repeats the history check under transaction locks.
- Pull stages its payload, rejects symlinks and unsupported entry types, and verifies before promotion. A failed staged verification does not replace the local primary.
- Transfers skip unchanged datasets and do not preserve remote owner, group or permission bits.

Use one transfer coordinator per dataset. Reciprocal transfers wait only until the lock timeout. After a transient failure, inspect the audit and compare again before retrying. See [troubleshooting](troubleshooting.md) and [audit output](audit-loop.md).
