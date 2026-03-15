# USR HPC Sync Flow

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-14


Use this runbook when a dataset is produced incrementally on BU SCC (or similar HPC) and local analysis must stay in sync without moving data through git.

Default sync contract:
- Dataset sync defaults to `--verify hash` plus strict sidecar and `_derived`/`_auxiliary` content-hash fidelity checks.
- Dataset sync preserves contents and sidecars, not cross-host owner/group/permission bits.
- Use `--no-verify-derived-hashes` only when an operator intentionally trades content-hash fidelity for speed.

## Scope

- Source of truth for datasets: USR dataset roots (not git-tracked files).
- Transfer path: `uv run usr diff/pull/push` over SSH remotes.
- Target workflows: `densegen` and `infer` batch outputs feeding shared USR datasets.

## Preflight

1. Confirm remote profile health.

```bash
# Validate remote reachability, transfer tools, and lock capability.
uv run usr remotes doctor --remote bu-scc
```

2. Confirm dataset roots are correct on both hosts.

```bash
# Local root example (canonical repo-local datasets root)
LOCAL_USR_ROOT="src/dnadesign/usr/datasets"
echo "$LOCAL_USR_ROOT"
# Remote base_dir is shown by:
# Print the configured remote profile and dataset base path.
uv run usr remotes show bu-scc
```

`usr --root src/dnadesign/usr ...` is also accepted and normalized automatically, but this runbook uses the canonical datasets root to keep path ownership explicit.

If BU SCC auth works in your shell only when SSH BatchMode is disabled, set `batch_mode: false` in the configured remote profile before running `diff` / `pull` / `push`.

3. Confirm the dataset id you intend to sync is explicit and canonical.

```bash
# Set the canonical dataset id used for sync calls.
DATASET_ID="my_dataset"
```

4. Preview drift before transfer.

```bash
# Compare local and remote dataset state before transfer decisions.
uv run usr --root "$LOCAL_USR_ROOT" diff "$DATASET_ID" bu-scc
```

Short form when your active shell already points at the intended USR root:

```bash
# Compare local and remote dataset state using the default root.
uv run usr diff "$DATASET_ID" bu-scc
```

## Bootstrap from either side

Use this section when only one side currently has dataset contents.

### HPC has dataset, local does not

```bash
# Preview remote-first dataset state.
uv run usr diff "$DATASET_ID" bu-scc
# Bootstrap local dataset contents from HPC.
uv run usr pull "$DATASET_ID" bu-scc -y
```

### Local has dataset, HPC does not

```bash
# Preview local-first dataset state.
uv run usr diff "$DATASET_ID" bu-scc
# Bootstrap HPC dataset contents from local.
uv run usr push "$DATASET_ID" bu-scc -y
```

## Run loop (HPC side writes, local side reads)

1. Submit or continue batch jobs on HPC that append to the dataset.
2. After each batch increment, pull to local and inspect.

```bash
# Pull remote updates into local with strict primary + sidecar verification.
uv run usr --root "$LOCAL_USR_ROOT" pull "$DATASET_ID" bu-scc -y
```

Short form:

```bash
# Pull remote updates using the default root.
uv run usr pull "$DATASET_ID" bu-scc -y
```

3. Open notebook/analysis locally against the local USR root.

## Verify loop

1. Re-run `diff` to confirm no pending remote deltas after pull.

```bash
# Confirm there are no remaining remote updates after pull.
uv run usr --root "$LOCAL_USR_ROOT" diff "$DATASET_ID" bu-scc
```

2. Check post-action audit output from `pull/push`:
- transfer state (`TRANSFERRED` or `NO-OP`)
- primary verification mode
- sidecar strict mode (`strict` vs `off`)
- meta/events/snapshot indicators
- `_auxiliary` indicator for non-core file inventory drift

3. If strict fidelity is required, keep default checks enabled and avoid `--no-verify-sidecars`, `--no-verify-derived-hashes`, `--primary-only`, and `--skip-snapshots`.

## Local annotations back to HPC

When local workflows add overlays (for example `infer` annotations) and should be available on HPC:

```bash
# Preview divergence before pushing local overlays/annotations.
uv run usr --root "$LOCAL_USR_ROOT" diff "$DATASET_ID" bu-scc
# Push local dataset updates back to the remote root with strict checks.
uv run usr --root "$LOCAL_USR_ROOT" push "$DATASET_ID" bu-scc -y
```

Short form:

```bash
# Push local updates using the default root.
uv run usr push "$DATASET_ID" bu-scc -y
```

Then verify on HPC by pulling or diffing from that side.

## Failure handling

- If transfer fails mid-run: rerun the same `pull`/`push` command. Pull stages and verifies before promotion.
- If lock acquisition fails: resolve remote lock holder contention, then retry.
- If the remote shell emits benign startup noise before the lock marker, USR now ignores that noise and continues waiting for the real lock marker.
- If strict sidecar verification fails: inspect `meta.md`, `.events.log`, and `_snapshots` drift before retrying.
- If doctor reports missing `flock`: install `flock` (util-linux) on remote host.
- If a local filesystem rejects remote permission metadata, use the standard `usr pull` / `usr push` path; rsync metadata replay is already disabled in that path.

## Checklist

1. `remotes doctor`
2. `diff`
3. `pull` or `push`
4. read sync audit summary
5. `diff` again for no-op confirmation
