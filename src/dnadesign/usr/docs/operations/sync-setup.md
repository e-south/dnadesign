# USR sync setup

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


Use this page for one-time setup and periodic key hygiene.

## HPC to local pattern (datasets are not in git)

Recommended storage layout:

- Local datasets root outside repo, for example `~/data/usr_datasets/`
- SCC datasets root in project/scratch storage, for example `/project/$USER/dnadesign/src/dnadesign/usr/workspaces/<workspace>/outputs/usr_datasets`

Notes:

- Scratch may have retention/purge policies; use project storage for long-lived datasets.
- Keep code in git, keep datasets in USR roots, and sync with `uv run usr diff/pull/push`.

## Prepare SSH keys (one-time)

```bash
# Check if an Ed25519 key already exists.
ls -l ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub 2>/dev/null || echo "no Ed25519 key yet"
```

```bash
# Generate a new Ed25519 key.
ssh-keygen -t ed25519 -C "<you>@<host>" -f ~/.ssh/id_ed25519

# Lock private key permissions.
chmod 600 ~/.ssh/id_ed25519
```

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

## Configure a USR remote

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

## Key rotation hygiene

```bash
# Generate replacement key.
ssh-keygen -t ed25519 -C "<you>@<host>" -f ~/.ssh/id_ed25519_new

# Install replacement public key on remote.
ssh-copy-id -i ~/.ssh/id_ed25519_new.pub <user>@<host>
```

Then update `~/.ssh/config` and remove old keys when ready.

```bash
# Keep key permissions strict.
chmod 600 ~/.ssh/id_*
# Lock down ~/.ssh directory permissions.
chmod 700 ~/.ssh
```

## Next

- Minimum transfer loop: [sync-quickstart.md](sync-quickstart.md)
- Target-mode details: [sync-modes.md](sync-modes.md)
