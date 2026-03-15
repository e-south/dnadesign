# USR sync quickstart

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-14


Use this page for the minimum reliable loop: preflight diff, transfer, verify.

## Quick path

1. One-time setup: [sync-setup.md](sync-setup.md).
2. Every run: `diff` -> `pull`/`push`.
3. The canonical repo-local datasets root is `src/dnadesign/usr/datasets`; `usr --root src/dnadesign/usr ...` is accepted and normalized to that canonical datasets root.

## Minimum command loop

```bash
# Preview local-vs-remote diff.
uv run usr diff my_dataset bu-scc
# Pull remote state into local dataset path.
uv run usr pull my_dataset bu-scc -y
```

Default sync contract:

- Dataset sync defaults to `--verify hash` plus strict sidecar and `_derived`/`_auxiliary` content-hash fidelity checks.
- Dataset sync preserves dataset contents and sidecars, not cross-host owner/group/permission bits.
- Use `--no-verify-sidecars` only when you intentionally trade fidelity for speed.
- Use `--no-verify-derived-hashes` only when you intentionally keep sidecar inventory checks but skip content-hash parity.
- Use `--verify auto|size|parquet` only when hash verification is intentionally unavailable.

## Daily sync workflow

Rule: run command on the machine where files should end up.

```bash
# Preview local-vs-remote differences.
uv run usr diff my_dataset bu-scc

# Pull remote -> local.
uv run usr pull my_dataset bu-scc -y

# Push local -> remote.
uv run usr push my_dataset bu-scc -y
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
- `--strict-bootstrap-id`: require an explicit canonical dataset id for bootstrap pulls and disable local name guessing

## Iterative batch loop (HPC clone -> local clone)

Use when datasets are produced on HPC and too large for git transfer.

```bash
# Local dataset may not exist yet; explicit flat or namespace-qualified ids both work.
uv run usr diff my_dataset bu-scc
# Materialize remote dataset locally.
uv run usr pull my_dataset bu-scc -y
```

```bash
# Pull latest remote state before local analysis/notebook work.
uv run usr diff my_dataset bu-scc
# Apply remote updates to local workspace copy.
uv run usr pull my_dataset bu-scc -y
```

```bash
# Preview whether local overlays diverged from remote.
uv run usr diff my_dataset bu-scc
# Push local changes back to the HPC dataset root.
uv run usr push my_dataset bu-scc -y
```

Safety guardrails:

- `usr pull` fails fast when remote `records.parquet` is missing.
- `usr push` fails fast when local `records.parquet` is missing.
- Dataset transfers acquire the shared remote dataset lock (`.usr.lock`) to avoid cross-host write races.
- The remote lock handshake tolerates benign shell noise before the lock marker, so SCC environment chatter does not break normal `usr pull` / `usr push`.
- `usr pull` and `usr push` skip transfer when no changes are detected.
- Pull transfers stage into a temporary directory and only promote after verification.
- Staged pull payloads reject symlink and unsupported entry types before promotion.
- Rsync intentionally avoids replaying remote owner/group/permission metadata on the destination so SCC pulls stay portable across local filesystems.
- Strict sidecar and `_derived`/`_auxiliary` content-hash fidelity checks are enabled by default.
- `--verify-sidecars` requires full dataset transfer and is incompatible with `--primary-only` / `--skip-snapshots`.
- Re-run `usr pull`/`usr push` after transient failure; post-transfer verification is always enforced.
- Every pull/push prints a post-action sync audit summary.

Optional strict bootstrap mode:

```bash
# Require an explicit canonical dataset id and disable local name guessing.
uv run usr pull mg1655_promoters bu-scc -y --strict-bootstrap-id
```

```bash
# Enable strict bootstrap id mode for current shell session.
export USR_SYNC_STRICT_BOOTSTRAP_ID=1
```

## Next

- Setup and SSH/key management: [sync-setup.md](sync-setup.md)
- Dataset vs file mode details: [sync-modes.md](sync-modes.md)
- Failure diagnostics: [sync-troubleshooting.md](sync-troubleshooting.md)
