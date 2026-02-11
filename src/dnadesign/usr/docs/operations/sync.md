## USR Sync over SSH

This page covers syncing USR datasets between local and remote hosts (for example, BU SCC).
For USR concepts and CLI basics, see [../../README.md](../../README.md).
Commands below use `uv run usr ...` to match this monorepo workflow.

### Contents
- [0) HPC to local pattern (datasets are not in git)](#0-hpc-to-local-pattern-datasets-are-not-in-git)
- [1) Prepare SSH keys (one-time)](#1-prepare-ssh-keys-one-time)
- [2) Configure a USR remote](#2-configure-a-usr-remote)
- [3) Daily sync workflow](#3-daily-sync-workflow)
- [4) Dataset directory mode + file mode](#4-dataset-directory-mode-file-mode)
- [5) Key rotation hygiene](#5-key-rotation-hygiene)

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
    type: ssh
    host: scc1.bu.edu
    user: <user>
    base_dir: /project/<user>/dnadesign/src/dnadesign/usr/workspaces/<workspace>/outputs/usr_datasets
    # Optional explicit key via environment variable:
    # ssh_key_env: USR_SSH_KEY
```

If using `ssh_key_env`:

```bash
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
- `--verify {auto,hash,size,parquet}`: verification mode

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
    type: ssh
    host: scc1.bu.edu
    user: <user>
    base_dir: /project/<user>/dnadesign/src/dnadesign/usr/workspaces/<workspace>/outputs/usr_datasets
    repo_root: /path/to/remote/dnadesign
    local_repo_root: /path/to/local/dnadesign
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
chmod 700 ~/.ssh
```

---

@e-south
