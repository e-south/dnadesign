"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/remotes.py

Remote endpoint command handlers for USR CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shlex
import shutil

from ..config import SSHRemoteConfig, get_remote, load_all, save_remote
from ..remote import SSHRemote

_BU_SCC_PRESET = "bu-scc"
_BU_SCC_LOGIN_HOST = "scc1.bu.edu"
_BU_SCC_TRANSFER_HOST = "scc-globus.bu.edu"


def _render_ssh_config_snippet(*, alias: str, host: str, user: str) -> str:
    return "\n".join(
        [
            f"Host {alias}",
            f"  HostName {host}",
            f"  User {user}",
            "  IdentitiesOnly yes",
            "  AddKeysToAgent yes",
            "  ServerAliveInterval 60",
            "  ServerAliveCountMax 2",
            "  # If Duo prompts fail in your client, try:",
            "  # PasswordAuthentication no",
        ]
    )


def cmd_remotes_list(args) -> None:
    remotes = load_all()
    if not remotes:
        print("(no remotes configured)")
        return
    for name, cfg in remotes.items():
        print(f"{name:20s} ssh {cfg.user}@{cfg.host}  base_dir={cfg.base_dir}")


def cmd_remotes_show(args) -> None:
    cfg = get_remote(args.name)
    print(f"name     : {cfg.name}")
    print("type     : ssh")
    print(f"ssh      : {cfg.user}@{cfg.host}")
    print(f"base_dir : {cfg.base_dir}")
    print(f"ssh_key  : {cfg.ssh_key_env or '(ssh-agent or default key)'}")


def cmd_remotes_add(args) -> None:
    if args.type != "ssh":
        raise SystemExit("Only --type ssh is supported.")
    cfg = SSHRemoteConfig(
        name=args.name,
        host=args.host,
        user=args.user,
        base_dir=args.base_dir,
        ssh_key_env=args.ssh_key_env,
    )
    path = save_remote(cfg)
    print(f"Saved remote '{cfg.name}' to {path}")


def cmd_remotes_wizard(args) -> None:
    preset = str(args.preset).strip().lower()
    if preset != _BU_SCC_PRESET:
        raise SystemExit(f"Unsupported preset '{args.preset}'. Supported presets: {_BU_SCC_PRESET}.")
    host = (
        args.host
        if str(getattr(args, "host", "")).strip()
        else (_BU_SCC_TRANSFER_HOST if bool(getattr(args, "transfer_node", False)) else _BU_SCC_LOGIN_HOST)
    )
    cfg = SSHRemoteConfig(
        name=args.name,
        host=host,
        user=args.user,
        base_dir=args.base_dir,
        ssh_key_env=args.ssh_key_env,
    )
    path = save_remote(cfg)
    print(f"Saved remote '{cfg.name}' to {path}")
    print("\nSSH config snippet (copy into ~/.ssh/config):")
    print(_render_ssh_config_snippet(alias=cfg.name, host=cfg.host, user=cfg.user))


def cmd_remotes_doctor(args) -> None:
    cfg = get_remote(args.remote)

    if shutil.which("ssh") is None:
        raise SystemExit("ssh not found on local PATH.")
    if shutil.which("rsync") is None:
        raise SystemExit("rsync not found on local PATH.")

    remote = SSHRemote(cfg)
    rc, _out, err = remote._ssh_run("echo USR_REMOTE_OK", check=False)
    if rc != 0:
        detail = err.strip() or "unknown ssh error"
        raise SystemExit(f"SSH connectivity check failed for {cfg.ssh_target}: {detail}")

    rc, _out, _err = remote._ssh_run("command -v rsync >/dev/null 2>&1", check=False)
    if rc != 0:
        raise SystemExit(f"Remote rsync is unavailable on {cfg.ssh_target}.")

    if bool(getattr(args, "check_base_dir", True)):
        base_dir = shlex.quote(cfg.base_dir)
        rc, _out, _err = remote._ssh_run(f"test -d {base_dir}", check=False)
        if rc != 0:
            raise SystemExit(f"Remote base_dir does not exist: {cfg.base_dir}")

    print(f"Remote: {cfg.name}")
    print(f"SSH: {cfg.ssh_target} (ok)")
    print("Remote rsync: ok")
    if bool(getattr(args, "check_base_dir", True)):
        print(f"base_dir: {cfg.base_dir} (ok)")
    print("Doctor checks passed.")
