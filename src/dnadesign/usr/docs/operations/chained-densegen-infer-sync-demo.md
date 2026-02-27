# Chained DenseGen and Infer Sync Demo

Use this runbook for the full asynchronous loop where DenseGen writes on HPC and Infer writes back overlays locally, with USR sync as the transfer contract.

Default sync contract:
- Dataset sync defaults to `--verify hash` plus strict sidecar and `_derived`/`_auxiliary` content-hash fidelity checks.
- Use `--no-verify-derived-hashes` only when an operator intentionally trades content-hash fidelity for speed.

## Scope

- Dataset source of truth: USR dataset roots, not git.
- Transfer boundary: `uv run usr diff/pull/push` over SSH remotes.
- Chained tools: DenseGen batch writes plus Infer write-back overlays.

## Quick path

```bash
# Compare local dataset state against HPC before any transfer.
uv run usr --root "$LOCAL_USR_ROOT" diff "$DATASET_ID" bu-scc
# Pull HPC updates locally with strict sidecar checks.
uv run usr --root "$LOCAL_USR_ROOT" pull "$DATASET_ID" bu-scc -y
# Push local overlay updates back to HPC with strict sidecar checks.
uv run usr --root "$LOCAL_USR_ROOT" push "$DATASET_ID" bu-scc -y
```

## Full chained loop

### 1) One-time preflight

```bash
# Validate remote connectivity, transfer prerequisites, and lock support.
uv run usr remotes doctor --remote bu-scc
# Show remote base_dir and profile wiring used by sync calls.
uv run usr remotes show bu-scc
# Set the namespace-qualified dataset id used by both hosts.
DATASET_ID="densegen/my_dataset"
```

### 2) HPC side DenseGen batch increment

Submit the scheduler template using workspace config rooted at your HPC clone.

```bash
# Submit DenseGen CPU batch job to append USR output on HPC.
qsub -P <project> \
  -v DENSEGEN_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml \
  docs/bu-scc/jobs/densegen-cpu.qsub
```

For resumed quota extension runs:

```bash
# Resume previous run and extend quota with explicit run args.
qsub -P <project> \
  -v DENSEGEN_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml,DENSEGEN_RUN_ARGS='--resume --extend-quota 8 --no-plot' \
  docs/bu-scc/jobs/densegen-cpu.qsub
```

### 3) Local pull and analysis loop

```bash
# Compare local against HPC before pulling.
uv run usr --root "$LOCAL_USR_ROOT" diff "$DATASET_ID" bu-scc
# Pull HPC updates into local dataset for notebooks/analysis.
uv run usr --root "$LOCAL_USR_ROOT" pull "$DATASET_ID" bu-scc -y
# Confirm no remaining drift after pull.
uv run usr --root "$LOCAL_USR_ROOT" diff "$DATASET_ID" bu-scc
```

Optional local dataset view checks:

```bash
# Inspect latest records from the synchronized dataset.
uv run usr --root "$LOCAL_USR_ROOT" head "$DATASET_ID" -n 5
# Inspect the event stream tail for operator context.
uv run usr --root "$LOCAL_USR_ROOT" events tail "$DATASET_ID" -n 10
```

### 4) Local Infer write-back and push to HPC

```bash
# Write Infer outputs back to the same USR dataset namespace.
uv run infer run --preset evo2/extract_logits_ll --usr "$DATASET_ID" --field sequence --device cpu --write-back
# Preview local-vs-remote drift after write-back.
uv run usr --root "$LOCAL_USR_ROOT" diff "$DATASET_ID" bu-scc
# Push local infer overlays and sidecars back to HPC.
uv run usr --root "$LOCAL_USR_ROOT" push "$DATASET_ID" bu-scc -y
```

### 5) HPC side pull (optional rebalance)

Run this on HPC when local-first updates should become the new remote baseline before the next batch phase.

```bash
# Pull the latest dataset state into the HPC workspace copy.
uv run usr --root "$HPC_USR_ROOT" pull "$DATASET_ID" bu-scc -y
```

## Audit interpretation

Every pull/push prints an audit summary. Use it for low-friction decisions:

- `Primary changed`: base table content changed by verify mode.
- `meta.md changed`: metadata notes changed.
- `.events.log local/remote`: event stream drift context.
- `_snapshots changed`: snapshot inventory drift.
- `_derived changed`: overlay-file inventory drift.
- `_auxiliary changed`: non-core file inventory drift (for example `_artifacts`, `_registry`).

Recommended operator rule:

1. If `_derived` or `_auxiliary` is `changed`, run transfer even when `.events.log` delta is small.
2. If strict fidelity is required, keep default checks enabled and avoid `--no-verify-sidecars`, `--no-verify-derived-hashes`, `--primary-only`, and `--skip-snapshots`.
3. Re-run `diff` after transfer; expected result is `up-to-date`.

## Failure drills

```bash
# Re-check remote toolchain and lock support after failures.
uv run usr remotes doctor --remote bu-scc
# Re-run pull with explicit verification mode when auto-mode is insufficient.
uv run usr --root "$LOCAL_USR_ROOT" pull "$DATASET_ID" bu-scc -y
```

If transfer fails mid-stream, rerun the same command. Pull stages payloads before promotion and push verifies post-transfer primary/sidecar contracts.
