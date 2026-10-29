<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[SeqLedger guide](../../README.md) / [Sync](README.md)

# Set up an SSH remote

SeqLedger sync uses an SSH remote profile to map a local dataset ID to a directory on another machine. Keep datasets outside the installed package and keep connection settings in a private, untracked file.

## 1. Confirm SSH access

Use an SSH account and key accepted by the remote host. Confirm that ordinary SSH works before configuring SeqLedger:

```bash
# Replace the host and user with your remote account.
ssh your-user@your-host
```

If your host requires interactive authentication, complete it in a terminal. SeqLedger's transfer probes cannot answer an interactive login prompt. The [HPC workflow](hpc-agent-flow.md) covers reusable SSH sessions for that case.

## 2. Create a private configuration

From a DNADesign checkout, copy the example into a new private file:

```bash
# Create a private configuration directory and copy only if the file is absent.
mkdir -p "$HOME/.config/dnadesign"
# Name the private configuration file.
REMOTE_CONFIG="$HOME/.config/dnadesign/seqledger-remotes.yaml"
# Preserve an existing configuration.
test ! -e "$REMOTE_CONFIG" && install -m 600   src/dnadesign/seqledger/remotes.example.yaml "$REMOTE_CONFIG"
```

Edit that file to match your host. A minimal profile is:

```yaml
# A profile maps dataset IDs beneath base_dir on the remote machine.
remotes:
  cluster:
    type: ssh
    host: compute.example.org
    user: your-user
    base_dir: /project/your-group/seqledger_datasets
```

The profile name `cluster` is the positional remote argument in sync commands. `base_dir` is a data root, not the DNADesign checkout. Select storage with a retention policy appropriate for the data.

To use a particular key, set `ssh_key_env` to the name of an environment variable containing its path. Do not put private key contents in the configuration.

## 3. Select and check the profile

```bash
# Reuse this private configuration in the current shell.
export SEQLEDGER_REMOTES_PATH="$HOME/.config/dnadesign/seqledger-remotes.yaml"

# Inspect settings, then check the remote connection.
seqledger remotes list
# Inspect the selected profile.
seqledger remotes show cluster
# Check authentication and transfer prerequisites.
seqledger remotes doctor --remote cluster
```

For a single command, pass `--remotes-config /path/to/file.yaml` before the subcommand instead. An explicit flag takes precedence over `SEQLEDGER_REMOTES_PATH`.

If authentication requires a reusable SSH session, establish it before the check:

```bash
# Authenticate in the terminal and reuse the SSH control socket.
seqledger remotes warm-auth --remote cluster
```

The default batch mode does not prompt for credentials. If your site's SSH authentication requires it, configure `batch_mode: false` and establish the interactive session first. Use [troubleshooting](troubleshooting.md) if the connection still fails.

Continue with the [transfer quickstart](quickstart.md) to select a local root, compare copies and choose a transfer direction.
