# USR Sync Fidelity Drills (Adversarial)

Use this runbook to pressure test strict sync fidelity for iterative batch workflows (`densegen`, `infer`, and sibling tools) across local and HPC clones.

## Scope

- Validate overlay fidelity when transfers are interrupted or partial.
- Validate schema contract enforcement for overlay write paths.
- Keep transfer decisions low-friction using `diff` and sync audit output.
- Exercise the default dataset sync contract (`hash` primary verification plus strict sidecar and `_derived`/`_auxiliary` content-hash checks).
- Use `--verify-derived-hashes` when you want scripts to state content-hash enforcement explicitly (same behavior as default).
- If required for emergency throughput, explicitly opt out of content hashes with `--no-verify-derived-hashes`.
- Include `_auxiliary` inventory checks so non-core files remain in lockstep.

## Preconditions

```bash
# Remote profile health must be clean before drills.
uv run usr remotes doctor --remote bu-scc
# Use a namespaced dataset id.
DATASET_ID="densegen/my_dataset"
```

## Drill 1: Pull must fail when `_derived` payload is missing

Goal: ensure strict pull rejects partial payloads that omit overlays.

```bash
# Baseline preview.
uv run usr --root "$LOCAL_USR_ROOT" diff "$DATASET_ID" bu-scc --verify parquet
# Strict pull must validate sidecars plus overlay inventory.
uv run usr --root "$LOCAL_USR_ROOT" pull "$DATASET_ID" bu-scc -y --verify parquet --verify-sidecars
```

Expected contract:
- If remote and staged payload disagree on `_derived` inventory, pull fails with `post-pull-sidecars`.
- Local `records.parquet` is not promoted from staged payload on failure.

Operator decision:
- Re-run after fixing remote payload integrity; do not disable `--verify-sidecars`.
- If you must trade fidelity for emergency transfer speed, use `--no-verify-sidecars` or `--no-verify-derived-hashes` explicitly and record that decision.

## Drill 2: Push must fail when remote misses local overlays

Goal: ensure strict push rejects remote post-transfer state that drops overlays.

```bash
# Local write-back from sibling tool (example: infer).
uv run infer run --preset evo2/extract_logits_ll --usr "$DATASET_ID" --field sequence --device cpu --write-back
# Preview drift before transfer.
uv run usr --root "$LOCAL_USR_ROOT" diff "$DATASET_ID" bu-scc --verify parquet
# Strict push requires post-transfer sidecar and overlay parity.
uv run usr --root "$LOCAL_USR_ROOT" push "$DATASET_ID" bu-scc -y --verify parquet --verify-sidecars
```

Expected contract:
- Push fails with `post-push-sidecars` when remote `_derived` inventory does not match local.
- Primary verification still runs; sidecar/overlay mismatch remains fatal.

Operator decision:
- Treat this as transfer integrity failure, not a benign warning.

## Drill 3: Overlay schema attack surface

Goal: confirm schema enforcement blocks invalid overlay writes before sync.

```bash
# Example failure path: namespace not registered or invalid column shape.
uv run usr --root "$LOCAL_USR_ROOT" attach "$DATASET_ID" \
  --path /tmp/bad_overlay.csv \
  --namespace infer --key sequence --key-col sequence --columns bad_field
```

Expected contract:
- Attach fails fast with actionable schema/registry errors.
- Invalid overlays are not written into `_derived`.

## Chained command loop

Use this loop in automation or operator sessions:

1. `uv run usr remotes doctor --remote bu-scc`
2. `uv run usr diff "$DATASET_ID" bu-scc --verify parquet`
3. `uv run usr pull "$DATASET_ID" bu-scc -y --verify parquet --verify-sidecars`
4. Run local notebook or sibling tool write-back.
5. `uv run usr diff "$DATASET_ID" bu-scc --verify parquet`
6. `uv run usr push "$DATASET_ID" bu-scc -y --verify parquet --verify-sidecars`
7. Re-run `diff` and require `up-to-date`.

Use sync audit lines (`Primary`, `.events.log`, `_snapshots`, `_derived`, `_auxiliary`) as the final transfer decision summary.
