## USR Sync over SSH

This page covers syncing USR datasets between local and remote hosts (for example, BU SCC).
For USR concepts and CLI basics, see [../../README.md](../../README.md).
Commands below use `uv run usr ...` to match this monorepo workflow.

### Contents
- [Quick path](#quick-path)
- [Advanced path](#advanced-path)
- [Failure diagnosis](#failure-diagnosis)
- [Runbook links](#runbook-links)
- [0) HPC to local pattern (datasets are not in git)](#0-hpc-to-local-pattern-datasets-are-not-in-git)
- [1) Prepare SSH keys (one-time)](#1-prepare-ssh-keys-one-time)
- [2) Configure a USR remote](#2-configure-a-usr-remote)
- [3) Daily sync workflow](#3-daily-sync-workflow)
- [4) Dataset directory mode + file mode](#4-dataset-directory-mode-file-mode)
- [5) Key rotation hygiene](#5-key-rotation-hygiene)

---

### Quick path

Use this when you need the minimum reliable operator loop:

1. One-time setup: sections [1](#1-prepare-ssh-keys-one-time) and [2](#2-configure-a-usr-remote).
2. Every run: section [3](#3-daily-sync-workflow) (`diff` -> `pull`/`push`).
3. For DenseGen runs: sync datasets under workspace `outputs/usr_datasets` using the default dataset contract.

Minimum command loop:

```bash
# Preview local-vs-remote diff.
uv run usr diff densegen/my_dataset bu-scc
# Pull remote state into local dataset path.
uv run usr pull densegen/my_dataset bu-scc -y
```

Default sync contract:
- Dataset sync defaults to `--verify hash` plus strict sidecar fidelity checks.
- Use `--no-verify-sidecars` only when you intentionally trade fidelity for speed.
- Use `--verify-derived-hashes` for strict `_derived` file-content hash parity (high assurance, slower).
- Use `--verify auto|size|parquet` only when hash verification is intentionally unavailable.

### Advanced path

Use this when basic dataset-id sync is not enough:

- Dataset directory targets outside `--root`: section [4](#4-dataset-directory-mode-file-mode).
- File-mode path mapping (`repo_root`, `local_repo_root`, `--remote-path`): section [4](#4-dataset-directory-mode-file-mode).
- Transfer-heavy BU workflows and key rotation hygiene: sections [4](#4-dataset-directory-mode-file-mode) and [5](#5-key-rotation-hygiene).

### Failure diagnosis

Use this sequence when sync commands fail or verification blocks a transfer:

1. Check remote wiring: `uv run usr remotes doctor --remote <name>`.
2. Re-run with explicit verification mode: `--verify hash|size|parquet`.
3. If `verify=auto` prints fallback warnings, resolve missing remote capabilities rather than ignoring warnings.
4. For dataset directory mode, ensure target contains `records.parquet` and has a discoverable `registry.yaml` ancestor.
5. For file mode, confirm remote path mapping with `--remote-path` or `remote.repo_root` plus local repo root mapping.
6. If doctor reports `Remote flock is unavailable`, install `flock` (util-linux) on the remote host.

Common failure signatures:

- `verify=hash requires remote sha256`: remote host lacks hash utility in PATH.
- `verify=parquet requires remote parquet row/col stats`: remote host lacks `pyarrow`.
- `Remote flock is unavailable`: remote host cannot provide cross-host dataset lock for sync transfers.
- `Dataset directory path is outside --root and no registry.yaml ancestor was found`: pass correct `--root` or use dataset id.

### Runbook links

Use the standalone runbooks for operator loops:

- [hpc-agent-sync-flow.md](hpc-agent-sync-flow.md)
- [chained-densegen-infer-sync-demo.md](chained-densegen-infer-sync-demo.md)

---

### 0) HPC to local pattern (datasets are not in git)

Recommended storage layout:
- Local datasets root outside the repo, for example `~/data/usr_datasets/`
- SCC datasets root in project/scratch storage, for example `/project/$USER/dnadesign/src/dnadesign/usr/workspaces/<workspace>/outputs/usr_datasets`

Notes:
- Scratch may have retention/purge policies; use project storage for long-lived datasets.
- Keep code in git, keep datasets in USR roots, and sync with `uv run usr diff/pull/push`.

---

### 1) Prepare SSH keys (one-time)

Check whether you already have an Ed25519 key:

```bash
# Check if an Ed25519 key already exists.
ls -l ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub 2>/dev/null || echo "no Ed25519 key yet"
```

Generate one if needed:

```bash
# Generate a new Ed25519 key.
ssh-keygen -t ed25519 -C "<you>@<host>" -f ~/.ssh/id_ed25519

# Lock private key permissions.
chmod 600 ~/.ssh/id_ed25519
```

Install your public key on the remote host:

```bash
# Copy public key to remote host.
ssh-copy-id -i ~/.ssh/id_ed25519.pub <user>@<host>
```

macOS keychain convenience:

```bash
# Start agent and add key to macOS keychain.
eval "$(ssh-agent -s)"
# Add the SSH key to macOS keychain-backed ssh-agent.
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
```

Optional `~/.ssh/config` entry:

```text
Host <alias>
  HostName <host>
  User <user>
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  AddKeysToAgent yes
  UseKeychain yes
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 10m
```

---

### 2) Configure a USR remote

`USR_REMOTES_PATH` is required.

```bash
# Set remote config path for USR CLI commands.
export USR_REMOTES_PATH="$HOME/.config/dnadesign/usr-remotes.yaml"

# Create remote profile.
uv run usr remotes wizard \
  --preset bu-scc \
  --name bu-scc \
  --user <user> \
  --host scc1.bu.edu \
  --base-dir /project/<user>/dnadesign/src/dnadesign/usr/workspaces/<workspace>/outputs/usr_datasets

# Validate remote profile wiring.
uv run usr remotes doctor --remote bu-scc
```

Inspect remote config:

```bash
# List configured remotes.
uv run usr remotes list

# Show one remote in detail.
uv run usr remotes show bu-scc
```

File-based config example:

```yaml
# $USR_REMOTES_PATH
remotes:
  bu-scc:
    type: ssh                                                                                          # Sets `type` for this example configuration.
    host: scc1.bu.edu                                                                                  # Sets `host` for this example configuration.
    user: <user>                                                                                       # Sets `user` for this example configuration.
    base_dir: /project/<user>/dnadesign/src/dnadesign/usr/workspaces/<workspace>/outputs/usr_datasets  # Sets `base_dir` for this example configuration.
    # Optional explicit key via environment variable:
    # ssh_key_env: USR_SSH_KEY
```

If using `ssh_key_env`:

```bash
# Export environment variables consumed by later commands.
export USR_SSH_KEY="$HOME/.ssh/id_ed25519"
```

---

### 3) Daily sync workflow

Rule: run the command on the machine where you want files to end up.

Preview:

```bash
# Preview local-vs-remote differences.
uv run usr diff densegen/my_dataset bu-scc
```

Pull remote -> local:

```bash
# Pull remote dataset into local machine.
uv run usr pull densegen/my_dataset bu-scc -y
```

Push local -> remote:

```bash
# Push local dataset to remote machine.
uv run usr push densegen/my_dataset bu-scc -y
```

Useful flags:

- `-y/--yes`: non-interactive overwrite confirmation
- `--primary-only`: transfer only `records.parquet`
- `--skip-snapshots`: exclude `_snapshots/`
- `--dry-run`: preview only
- `--verify {hash,auto,size,parquet}`: primary verification mode (default: `hash`)
- `--verify-sidecars`: explicitly enable strict sidecar fidelity checks
- `--no-verify-sidecars`: disable strict sidecar checks for dataset sync
- `--verify-derived-hashes`: verify `_derived` file-content hashes in addition to sidecar inventory
- `--strict-bootstrap-id`: require `<namespace>/<dataset>` for bootstrap pulls when local dataset is missing

---

### 3a) Iterative batch loop (HPC clone -> local clone)

Use this when datasets are produced on HPC and are too large to move through git.

One-time bootstrap on local:

```bash
# Local dataset may not exist yet; this is supported for namespaced ids.
uv run usr diff densegen/my_dataset bu-scc
# Materialize remote dataset locally.
uv run usr pull densegen/my_dataset bu-scc -y
```

After each batch increment on HPC:

```bash
# Pull latest remote state before local analysis/notebook work.
uv run usr diff densegen/my_dataset bu-scc
# Apply remote updates to local workspace copy.
uv run usr pull densegen/my_dataset bu-scc -y
```

If you add local overlays/annotations and want them back on HPC:

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
- `usr pull` and `usr push` skip transfer when no changes are detected, so repeated loop calls are safe.
- Pull transfers stage into a temporary directory and only promote after verification.
- Staged pull payloads reject symlink and unsupported entry types before promotion.
- `--verify-sidecars` enforces exact sidecar parity and fails fast on mismatch.
- Strict sidecar fidelity checks are enabled by default for dataset sync.
- `--verify-sidecars` requires full dataset transfer and is incompatible with `--primary-only` / `--skip-snapshots`.
- Re-run `usr pull`/`usr push` after transient transfer failure; post-transfer verification is always enforced.
- Every pull/push prints a post-action sync audit summary for fast operator decisions.
- Sync audit summaries include `Primary`, `.events.log`, `_snapshots`, `_derived`, and `_auxiliary`.

Optional strict bootstrap mode:

```bash
# Enforce namespace-qualified ids for bootstrap pulls.
uv run usr pull demo_dataset bu-scc -y --strict-bootstrap-id
```

Environment equivalent:

```bash
# Enable strict bootstrap id mode for current shell session.
export USR_SYNC_STRICT_BOOTSTRAP_ID=1
```

---

### 4) Dataset directory mode + file mode

Use dataset directory mode when you have an explicit dataset path outside `--root`:

```bash
# Diff dataset directory path outside --root.
uv run usr diff /path/to/src/dnadesign/usr/workspaces/<workspace>/outputs/usr_datasets/densegen/demo_hpc bu-scc

# Pull dataset directory by explicit path.
uv run usr pull /path/to/src/dnadesign/usr/workspaces/<workspace>/outputs/usr_datasets/densegen/demo_hpc bu-scc -y

# Push dataset directory by explicit path.
uv run usr push /path/to/src/dnadesign/usr/workspaces/<workspace>/outputs/usr_datasets/densegen/demo_hpc bu-scc -y
```

Use file mode when syncing a single file.

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

# If local repo root is not configured
uv run usr pull permuter/run42/records.parquet bu-scc \
  --repo-root /path/to/local/dnadesign -y

# If remote path mapping cannot be inferred
uv run usr pull permuter/run42/records.parquet bu-scc \
  --remote-path /path/to/remote/dnadesign/src/dnadesign/permuter/run42/records.parquet -y
```

Verification in `--verify auto` uses this order:

1. SHA-256 (if available on both hosts)
2. file size
3. for Parquet files, row/column checks (`pyarrow` required on remote)

If auto mode falls back, USR prints a warning.

Environment variable for file mode:

- `DNADESIGN_REPO_ROOT` can provide local repo root when not passed with `--repo-root`.

BU transfer-heavy option:
- Use host `scc-globus.bu.edu` for transfer-focused workflows.
- BU also supports download-node jobs via `qsub -l download`.

---

### 5) Key rotation hygiene

```bash
# Generate replacement key.
ssh-keygen -t ed25519 -C "<you>@<host>" -f ~/.ssh/id_ed25519_new

# Install replacement public key on remote.
ssh-copy-id -i ~/.ssh/id_ed25519_new.pub <user>@<host>
```

Then update `~/.ssh/config` and remove old keys when ready.

Keep permissions strict:

```bash
# Keep key permissions strict.
chmod 600 ~/.ssh/id_*
# Lock down ~/.ssh directory permissions.
chmod 700 ~/.ssh
```

---

@e-south
