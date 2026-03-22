# USR sync setup

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-14


Use this page for one-time setup and periodic key hygiene.

## HPC to local pattern (datasets are not in git)

Recommended storage layout:

- Shared repo-local datasets should live under `src/dnadesign/usr/datasets`.
- External dataset roots are still allowed for ad-hoc sync or mirror workflows, for example `~/data/usr_datasets/` or another explicit operator-owned location.
- SCC dataset roots should stay in project storage for long-lived runs, for example `/project/$USER/dnadesign/src/dnadesign/usr/datasets`.

Notes:

- Scratch may have retention/purge policies; use project storage for long-lived datasets.
- Keep curated tool configs and logs in their tool workspaces.
- Treat a tool workspace USR sink as explicit in the study record, whether it
  writes to a workspace-local export root or directly to the shared root.
- Keep shared USR datasets under the package USR root unless the study record
  makes another shared root explicit.
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

Prefer `--remotes-config <path>` for each `usr` command so the chosen remotes
file is explicit in the command line. `USR_REMOTES_PATH` is the fallback for a
shell session that will reuse the same remotes file repeatedly.

```bash
# Create remote profile in an explicit remotes file.
uv run usr --remotes-config "$HOME/.config/dnadesign/usr-remotes.yaml" remotes wizard \
  --preset bu-scc \
  --name bu-scc \
  --user <user> \
  --host scc1.bu.edu \
  --base-dir /project/<user>/dnadesign/src/dnadesign/usr/datasets

# Validate remote profile wiring.
uv run usr --remotes-config "$HOME/.config/dnadesign/usr-remotes.yaml" remotes doctor --remote bu-scc
```

Shell-session fallback:

```bash
# Reuse one remotes file across many commands in the current shell.
export USR_REMOTES_PATH="$HOME/.config/dnadesign/usr-remotes.yaml"
```

Inspect remote config:

```bash
# List configured remotes.
uv run usr remotes list

# Show one remote in detail.
uv run usr remotes show bu-scc

# Show whether a reusable SSH control socket is already live.
uv run usr remotes status --remote bu-scc
```

File-based config example:

```yaml
# $USR_REMOTES_PATH
remotes:
  bu-scc:
    type: ssh                                                                                          # Sets `type` for this example configuration.
    host: scc1.bu.edu                                                                                  # Sets `host` for this example configuration.
    user: <user>                                                                                       # Sets `user` for this example configuration.
    base_dir: /project/<user>/dnadesign/src/dnadesign/usr/datasets                                     # Sets `base_dir` for this example configuration.
    batch_mode: true                                                                                   # Set false when SCC auth works only without BatchMode=yes.
    # Optional explicit key via environment variable:
    # ssh_key_env: USR_SSH_KEY
```

If BU SCC auth fails with `Permission denied (keyboard-interactive,hostbased)` under strict batch mode, re-save the remote with `--no-batch-mode` or set `batch_mode: false` in the remotes YAML passed through `--remotes-config` or `USR_REMOTES_PATH`.

If SCC still requires Duo or other keyboard-interactive follow-up after publickey auth, establish `ssh scc1` or `ssh scc1.bu.edu` once in a terminal first so the SSH ControlMaster socket is already live before running `usr remotes doctor`, `usr diff`, `usr pull`, or `usr push`. This matters because the sync lock handshake and other preflight probes run through piped SSH helpers that cannot complete a fresh keyboard-interactive prompt on their own.

Repo-native bootstrap path:

```bash
# Establish or reuse the SSH control socket used by later sync commands.
uv run usr remotes warm-auth --remote bu-scc
```

USR sync preserves dataset contents and sidecars across hosts, but it intentionally does not preserve remote owner/group/permission metadata on the destination.

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
