# USR sync quickstart

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


Use this page for the minimum reliable loop: preflight diff, transfer, verify.

## Quick path

1. One-time setup: [sync-setup.md](sync-setup.md).
2. Every run: `diff` -> `pull`/`push`.
3. For DenseGen runs, sync datasets under workspace `outputs/usr_datasets`.

## Minimum command loop

```bash
# Preview local-vs-remote diff.
uv run usr diff densegen/my_dataset bu-scc
# Pull remote state into local dataset path.
uv run usr pull densegen/my_dataset bu-scc -y
```

Default sync contract:

- Dataset sync defaults to `--verify hash` plus strict sidecar and `_derived`/`_auxiliary` content-hash fidelity checks.
- Use `--no-verify-sidecars` only when you intentionally trade fidelity for speed.
- Use `--no-verify-derived-hashes` only when you intentionally keep sidecar inventory checks but skip content-hash parity.
- Use `--verify auto|size|parquet` only when hash verification is intentionally unavailable.

## Daily sync workflow

Rule: run command on the machine where files should end up.

```bash
# Preview local-vs-remote differences.
uv run usr diff densegen/my_dataset bu-scc

# Pull remote -> local.
uv run usr pull densegen/my_dataset bu-scc -y

# Push local -> remote.
uv run usr push densegen/my_dataset bu-scc -y
```

Useful flags:

- `-y/--yes`: non-interactive overwrite confirmation
- `--primary-only`: transfer only `records.parquet`
- `--skip-snapshots`: exclude `_snapshots/`
- `--dry-run`: preview only
- `--verify {hash,auto,size,parquet}`: primary verification mode (default `hash`)
- `--verify-sidecars` and `--no-verify-sidecars`
- `--verify-derived-hashes` and `--no-verify-derived-hashes`
- `--audit-json-out <path>`: machine-readable sync audit JSON
- `--strict-bootstrap-id`: require `<namespace>/<dataset>` for bootstrap pulls

## Iterative batch loop (HPC clone -> local clone)

Use when datasets are produced on HPC and too large for git transfer.

```bash
# Local dataset may not exist yet; this is supported for namespaced ids.
uv run usr diff densegen/my_dataset bu-scc
# Materialize remote dataset locally.
uv run usr pull densegen/my_dataset bu-scc -y
```

```bash
# Pull latest remote state before local analysis/notebook work.
uv run usr diff densegen/my_dataset bu-scc
# Apply remote updates to local workspace copy.
uv run usr pull densegen/my_dataset bu-scc -y
```

```bash
# Preview whether local overlays diverged from remote.
uv run usr diff densegen/my_dataset bu-scc
# Push local changes back to the HPC dataset root.
uv run usr push densegen/my_dataset bu-scc -y
```

Safety guardrails:

- `usr pull` fails fast when remote `records.parquet` is missing.
- `usr push` fails fast when local `records.parquet` is missing.
- Dataset transfers acquire the shared remote dataset lock (`.usr.lock`) to avoid cross-host write races.
- `usr pull` and `usr push` skip transfer when no changes are detected.
- Pull transfers stage into a temporary directory and only promote after verification.
- Staged pull payloads reject symlink and unsupported entry types before promotion.
- Strict sidecar and `_derived`/`_auxiliary` content-hash fidelity checks are enabled by default.
- `--verify-sidecars` requires full dataset transfer and is incompatible with `--primary-only` / `--skip-snapshots`.
- Re-run `usr pull`/`usr push` after transient failure; post-transfer verification is always enforced.
- Every pull/push prints a post-action sync audit summary.

Optional strict bootstrap mode:

```bash
# Enforce namespace-qualified ids for bootstrap pulls.
uv run usr pull demo_dataset bu-scc -y --strict-bootstrap-id
```

```bash
# Enable strict bootstrap id mode for current shell session.
export USR_SYNC_STRICT_BOOTSTRAP_ID=1
```

## Next

- Setup and SSH/key management: [sync-setup.md](sync-setup.md)
- Dataset vs file mode details: [sync-modes.md](sync-modes.md)
- Failure diagnostics: [sync-troubleshooting.md](sync-troubleshooting.md)
