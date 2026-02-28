# USR sync target modes

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


Use this page when dataset-id sync is not enough and you need explicit path mapping.

## Dataset directory mode

Use this when you have an explicit dataset directory path outside `--root`.

```bash
# Diff dataset directory path outside --root.
uv run usr diff /path/to/src/dnadesign/usr/workspaces/<workspace>/outputs/usr_datasets/densegen/demo_hpc bu-scc

# Pull dataset directory by explicit path.
uv run usr pull /path/to/src/dnadesign/usr/workspaces/<workspace>/outputs/usr_datasets/densegen/demo_hpc bu-scc -y

# Push dataset directory by explicit path.
uv run usr push /path/to/src/dnadesign/usr/workspaces/<workspace>/outputs/usr_datasets/densegen/demo_hpc bu-scc -y
```

## File mode

Use this when syncing a single file.

Remote config example with repo mapping:

```yaml
remotes:
  bu-scc:
    type: ssh                                                                                          # Sets `type` for this example configuration.
    host: scc1.bu.edu                                                                                  # Sets `host` for this example configuration.
    user: <user>                                                                                       # Sets `user` for this example configuration.
    base_dir: /project/<user>/dnadesign/src/dnadesign/usr/workspaces/<workspace>/outputs/usr_datasets  # Sets `base_dir` for this example configuration.
    repo_root: /path/to/remote/dnadesign                                                               # Sets `repo_root` for this example configuration.
    local_repo_root: /path/to/local/dnadesign                                                          # Sets `local_repo_root` for this example configuration.
```

Examples:

```bash
# Diff one file by repo-relative path.
uv run usr diff permuter/run42/records.parquet bu-scc

# Pull one file by repo-relative path.
uv run usr pull permuter/run42/records.parquet bu-scc -y

# If local repo root is not configured.
uv run usr pull permuter/run42/records.parquet bu-scc \
  --repo-root /path/to/local/dnadesign -y

# If remote path mapping cannot be inferred.
uv run usr pull permuter/run42/records.parquet bu-scc \
  --remote-path /path/to/remote/dnadesign/src/dnadesign/permuter/run42/records.parquet -y
```

Verification in `--verify auto` uses this order:

1. SHA-256 (if available on both hosts)
2. File size
3. For Parquet files, row/column checks (`pyarrow` required on remote)

If auto mode falls back, USR prints a warning.

Environment variable for file mode:

- `DNADESIGN_REPO_ROOT` can provide local repo root when not passed with `--repo-root`.

BU transfer-heavy option:

- Use host `scc-globus.bu.edu` for transfer-focused workflows.
- BU also supports download-node jobs via `qsub -l download`.

## Next

- Baseline operator loop: [sync-quickstart.md](sync-quickstart.md)
- Failure diagnostics: [sync-troubleshooting.md](sync-troubleshooting.md)
