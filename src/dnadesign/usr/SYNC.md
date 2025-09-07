## Setting up USR sync and SSH guide (SCC)

*A workflow for moving USR datasets between your local computer and the BU SCC. For USR basics (schema, commands), see the sibling [**`README.md`**](README.md). This doc covers SSH keys, remote config, and everyday sync.*

### 1) Local computer: set up SSH keys (reuse or create)

**Check for an existing key:**

```bash
ls -l ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub 2>/dev/null || echo "no Ed25519 key yet"
```

* If keys exist: skip generation and go to **Install your key on SCC**.
* If not, generate:

```bash
ssh-keygen -t ed25519 -C "esouth@scc1" -f ~/.ssh/id_ed25519
chmod 600 ~/.ssh/id_ed25519
```

**Install your key on SCC (adds to `authorized_keys`):**

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub esouth@scc1.bu.edu
```

You’ll be prompted for your SCC password once. This step is what lets future ssh/rsync use your key + Duo/MFA, no password.

**macOS (recommended): remember your key in the agent + keychain**

```bash
eval "$(ssh-agent -s)"
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
```

**Add a handy SSH alias** in `~/.ssh/config`:

```
Host scc1 scc1.bu.edu
  HostName scc1.bu.edu
  User esouth
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  AddKeysToAgent yes
  UseKeychain yes
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 10m
```

```bash
chmod 600 ~/.ssh/config
ssh scc1   # should log you in (with Duo/MFA as required)
exit
```

> **Password-only fallback:** If you ever can’t use keys, you can still run `usr pull/push`; ssh/rsync will prompt for a password.

### 2) Repo: configure the USR remote

Run **from your laptop**, inside the repo root. Change to where your datasets live.

```bash
usr remotes add cluster --type ssh \
  --host scc1.bu.edu --user esouth \
  --base-dir /project/dunlop/esouth/dnadesign/src/dnadesign/usr/datasets
```

Check:

```bash
usr remotes list
usr remotes show cluster
# Expect: ssh_key : (ssh-agent or default key)
```

### Alternative: file-based config

You can also commit `usr/remotes.yaml`:

```yaml
remotes:
  cluster:
    type: ssh
    host: scc1.bu.edu
    user: esouth
    base_dir: /project/dunlop/esouth/dnadesign/src/dnadesign/usr/datasets
    # ssh_key_env: USR_SSH_KEY   # optional; set to ~/.ssh/id_ed25519 if you prefer explicit key
```

If you do use `ssh_key_env`, export it before running `usr`:

```bash
export USR_SSH_KEY="$HOME/.ssh/id_ed25519"
```

### 3) Everyday sync workflow

> **Rule of thumb:** Run the command on the machine where you want the files to end up.

#### Pull cluster → local

```bash
usr diff my_usr_dataset --remote cluster   # preview sha/rows/cols and snapshots
usr pull my_usr_dataset --from cluster -y  # copies dataset folder down
```

#### Push local → cluster

```bash
usr push my_usr_dataset --to cluster -y
```

#### Useful flags

* `-y/--yes` non-interactive overwrite confirmation
* `--primary-only` transfer only `records.parquet`
* `--skip-snapshots` exclude snapshot files
* `--dry-run` show what would copy without copying

#### What actually transfers

A dataset is a **folder**:

```
<dataset>/
  records.parquet      # primary table
  meta.yaml
  .events.log
  _snapshots/          # rolling parquet checkpoints
```

### 5) Key rotation & safety

* Generate a **new** key:

  ```bash
  ssh-keygen -t ed25519 -C "esouth@scc1" -f ~/.ssh/id_ed25519_new
  ssh-copy-id -i ~/.ssh/id_ed25519_new.pub esouth@scc1.bu.edu
  ```

  Update your `~/.ssh/config` to point `IdentityFile ~/.ssh/id_ed25519_new`, then remove the old key later.

* Keep private keys `chmod 600`, `~/.ssh` as `chmod 700`.
