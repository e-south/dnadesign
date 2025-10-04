## USR sync & SSH (SCC)

This is a guide for moving USR datasets between your laptop and the BU SCC. For USR concepts and CLI basics, see the sibling **README.md**. This page focuses on SSH keys, remote configuration, and day‑to‑day sync.

---

### 1) Prepare SSH keys (once)

**Check if you already have an Ed25519 key:**
```bash
ls -l ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub 2>/dev/null || echo "no Ed25519 key yet"
````

**Generate a key if needed:**

```bash
ssh-keygen -t ed25519 -C "esouth@scc1" -f ~/.ssh/id_ed25519
chmod 600 ~/.ssh/id_ed25519
```

**Install your public key on SCC (adds to `authorized_keys`):**

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub esouth@scc1.bu.edu
```

You’ll enter your SCC password once. After this, ssh/rsync will use your key + Duo/MFA.

**macOS: keep the key loaded**

```bash
eval "$(ssh-agent -s)"
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
```

**Optional convenience alias** in `~/.ssh/config`:

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
ssh scc1   # should log in (with Duo/MFA)
exit
```

> **Fallback:** If keys aren’t available, `usr pull/push` will still prompt for your password.

---

### 2) Configure a USR remote

Run from your laptop in the repo where your dataset lives.

**CLI-based config (writes `~/.config/usr/remotes.yaml` if none exists):**

```bash
usr remotes add cluster --type ssh \
  --host scc1.bu.edu --user esouth \
  --base-dir /project/dunlop/esouth/dnadesign/src/dnadesign/usr/datasets
```

**Inspect config:**

```bash
usr remotes list
usr remotes show cluster
# Expected: ssh user/host and base_dir shown; ssh_key : (ssh-agent or default key)
```

**File-based config (alternative):**

```yaml
# usr/remotes.yaml
remotes:
  cluster:
    type: ssh
    host: scc1.bu.edu
    user: esouth
    base_dir: /project/dunlop/esouth/dnadesign/src/dnadesign/usr/datasets
    # Optional explicit key via environment variable:
    # ssh_key_env: USR_SSH_KEY
```

If you set `ssh_key_env`, export it before running `usr`:

```bash
export USR_SSH_KEY="$HOME/.ssh/id_ed25519"
```

---

### 3) Everyday sync

> **Rule of thumb:** Run the command *on the machine where you want the files to end up.*

**Preview differences:**

```bash
usr diff my_usr_dataset --remote cluster
# or: usr status my_usr_dataset --remote cluster
```

**Pull cluster → local:**

```bash
usr pull my_usr_dataset --from cluster -y
```

**Push local → cluster:**

```bash
usr push my_usr_dataset --to cluster -y
```

**Useful flags**

* `-y/--yes` confirm overwrites non‑interactively
* `--primary-only` transfers only `records.parquet`
* `--skip-snapshots` excludes `_snapshots/`
* `--dry-run` shows what would copy without copying

---

### 4) Path‑first sync (FILE mode)

Use FILE mode when you want to sync a single file outside a canonical USR dataset directory (e.g., anywhere in your monorepo).

1. Make sure the remote knows both the datasets base and your repo root:

```yaml
# usr/remotes.yaml
remotes:
  cluster:
    type: ssh
    host: scc1.bu.edu
    user: esouth
    base_dir: /project/dunlop/esouth/dnadesign/src/dnadesign/usr/datasets   # dataset mode
    repo_root: /project/dunlop/esouth/dnadesign                              # FILE mode (remote side)
    local_repo_root: /Users/Shockwing/Dropbox/projects/phd/dnadesign         # optional (local fallback)
```

2. From anywhere in your local repo, operate on a **file path**:

```bash
# Show diff for a file path
usr diff permuter/run42/records.parquet --remote cluster

# Pull that file cluster → local
usr pull permuter/run42/records.parquet --remote cluster -y

# If your local root isn’t configured, pass it explicitly:
usr pull permuter/run42/records.parquet --remote cluster \
  --repo-root "$HOME/.../dnadesign" -y

# If automatic mapping can’t be derived, give the remote path explicitly:
usr pull permuter/run42/records.parquet --remote cluster \
  --remote-path /project/dunlop/esouth/dnadesign/src/dnadesign/permuter/run42/records.parquet -y
```

3. Dataset‑folder convenience:

```bash
cd usr/datasets/my_usr_dataset
usr diff --remote cluster
usr pull --remote cluster -y
```

**Verification behavior (both modes)**
Transfers are verified automatically. Preference order:

* **SHA‑256** (if available on both ends)
* file **size**
* for Parquet files (and if `pyarrow` is available), **rows/cols**
  If a check can’t be performed, you’ll get a clear error.

> Environment variable for FILE mode:
> `DNADESIGN_REPO_ROOT` can supply your local repo root if not passed via `--repo-root` and not present in `remotes.yaml`.

---

### 5) Key rotation & hygiene

* Generate a fresh key and install it:

  ```bash
  ssh-keygen -t ed25519 -C "esouth@scc1" -f ~/.ssh/id_ed25519_new
  ssh-copy-id -i ~/.ssh/id_ed25519_new.pub esouth@scc1.bu.edu
  ```

  Update `~/.ssh/config` to use the new key, then remove the old one later.

* Keep permissions tight: `chmod 600 ~/.ssh/id_*`, `chmod 700 ~/.ssh`.

---

@ e-south