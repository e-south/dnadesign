"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli/commands/remotes/__init__.py

Remote endpoint command handlers for USR CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import shlex
import shutil

from ....sync.remote.config import SSHRemoteConfig, get_remote, load_all, locate_config, save_remote
from ....sync.remote.remote import SSHControlSessionStatus, SSHRemote

_BU_SCC_PRESET = "bu-scc"
_BU_SCC_LOGIN_HOST = "scc1.bu.edu"
_BU_SCC_TRANSFER_HOST = "scc-globus.bu.edu"
_REMOTE_PROCESS_START_PROBE = 'test -n "$(LC_ALL=C TZ=UTC0 ps -o lstart= -p "$$" 2>/dev/null | tr -d \'[:space:]\')"'


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


def _doctor_ssh_failure_message(cfg: SSHRemoteConfig, detail: str) -> str:
    base = f"SSH connectivity check failed for {cfg.ssh_target}: {detail}"
    lowered = detail.lower()
    if "keyboard-interactive" not in lowered:
        return base
    auth_hint = (
        " Hint: BU SCC accepted publickey auth but still requires keyboard-interactive follow-up. "
        "Re-save the remote with `--no-batch-mode` or `batch_mode: false`, then establish "
        "`ssh scc1` or `ssh scc1.bu.edu` once in a terminal so the SSH ControlMaster socket is live "
        "before running `usr remotes doctor`, `usr diff`, `usr pull`, or `usr push`."
    )
    return base + auth_hint


def _status_recommendation(cfg: SSHRemoteConfig, status: SSHControlSessionStatus) -> str:
    if status.socket_live:
        return "ready for sync"
    if status.multiplex_enabled:
        return (
            f"run `usr remotes warm-auth --remote {cfg.name}` in a terminal, "
            f"or establish `ssh {cfg.host}` once before sync"
        )
    return "configure SSH ControlMaster plus ControlPath before relying on reusable auth sessions"


def _status_payload(cfg: SSHRemoteConfig, status: SSHControlSessionStatus) -> dict:
    return {
        "remote": cfg.name,
        "remotes_config": str(locate_config()),
        "ssh_target": cfg.ssh_target,
        "host": status.host,
        "user": status.user,
        "base_dir": cfg.base_dir,
        "batch_mode": bool(cfg.batch_mode),
        "control_master": status.control_master,
        "control_path": status.control_path,
        "control_persist": status.control_persist,
        "multiplex_enabled": bool(status.multiplex_enabled),
        "socket_exists": bool(status.socket_exists),
        "socket_live": bool(status.socket_live),
        "recommendation": _status_recommendation(cfg, status),
    }


def _print_status_payload(payload: dict) -> None:
    print(f"Remote       : {payload['remote']}")
    print(f"Config       : {payload['remotes_config']}")
    print(f"SSH          : {payload['ssh_target']}")
    print(f"base_dir     : {payload['base_dir']}")
    print(f"batch_mode   : {'yes' if payload['batch_mode'] else 'no'}")
    print(f"ControlMaster: {payload['control_master'] or '-'}")
    print(f"ControlPath  : {payload['control_path'] or '-'}")
    print(f"ControlPersist: {payload['control_persist'] or '-'}")
    print(f"Multiplex    : {'enabled' if payload['multiplex_enabled'] else 'disabled'}")
    if payload["multiplex_enabled"]:
        socket_state = (
            "live" if payload["socket_live"] else ("present-not-live" if payload["socket_exists"] else "absent")
        )
    else:
        socket_state = "n/a"
    print(f"Control socket: {socket_state}")
    print(f"Recommendation: {payload['recommendation']}")


def _emit_status(payload: dict, *, use_json: bool) -> None:
    if use_json:
        print(json.dumps(payload, separators=(",", ":")))
        return
    _print_status_payload(payload)


def cmd_remotes_list(args) -> None:
    remotes = load_all()
    if not remotes:
        print("(no remotes configured)")
        return
    for name, cfg in remotes.items():
        mode = "batch" if cfg.batch_mode else "interactive-auth"
        print(f"{name:20s} ssh {cfg.user}@{cfg.host}  base_dir={cfg.base_dir}  auth={mode}")


def cmd_remotes_show(args) -> None:
    cfg = get_remote(args.name)
    print(f"name     : {cfg.name}")
    print("type     : ssh")
    print(f"ssh      : {cfg.user}@{cfg.host}")
    print(f"base_dir : {cfg.base_dir}")
    print(f"batch    : {'yes' if cfg.batch_mode else 'no'}")
    print(f"ssh_key  : {cfg.ssh_key_env or '(ssh-agent or default key)'}")


def cmd_remotes_add(args) -> None:
    if args.type != "ssh":
        raise SystemExit("Only --type ssh is supported.")
    cfg = SSHRemoteConfig(
        name=args.name,
        host=args.host,
        user=args.user,
        base_dir=args.base_dir,
        batch_mode=bool(getattr(args, "batch_mode", True)),
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
        batch_mode=bool(getattr(args, "batch_mode", True)),
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
        raise SystemExit(_doctor_ssh_failure_message(cfg, detail))

    rc, _out, _err = remote._ssh_run("command -v rsync >/dev/null 2>&1", check=False)
    if rc != 0:
        raise SystemExit(f"Remote rsync is unavailable on {cfg.ssh_target}.")

    rc, _out, _err = remote._ssh_run("command -v flock >/dev/null 2>&1", check=False)
    if rc != 0:
        raise SystemExit(f"Remote flock is unavailable on {cfg.ssh_target}.")

    rc, _out, _err = remote._ssh_run(_REMOTE_PROCESS_START_PROBE, check=False)
    if rc != 0:
        raise SystemExit(
            f"Remote process-start identity is unavailable on {cfg.ssh_target}; "
            "USR requires `LC_ALL=C TZ=UTC0 ps -o lstart= -p $$` to return a value."
        )

    if bool(getattr(args, "check_base_dir", True)):
        base_dir = shlex.quote(cfg.base_dir)
        rc, _out, _err = remote._ssh_run(f"test -d {base_dir}", check=False)
        if rc != 0:
            raise SystemExit(f"Remote base_dir does not exist: {cfg.base_dir}")

    print(f"Remote: {cfg.name}")
    print(f"SSH: {cfg.ssh_target} (ok)")
    print("Remote rsync: ok")
    print("Remote flock: ok")
    print("Remote process identity: ok")
    if bool(getattr(args, "check_base_dir", True)):
        print(f"base_dir: {cfg.base_dir} (ok)")
    print("Doctor checks passed.")


def cmd_remotes_status(args) -> None:
    cfg = get_remote(args.remote)
    remote = SSHRemote(cfg)
    payload = _status_payload(cfg, remote.control_session_status())
    _emit_status(payload, use_json=bool(getattr(args, "json", False)))


def cmd_remotes_warm_auth(args) -> None:
    cfg = get_remote(args.remote)
    remote = SSHRemote(cfg)
    before = remote.control_session_status()
    if before.socket_live:
        payload = _status_payload(cfg, before)
        payload["bootstrap_state"] = "existing"
        _emit_status(payload, use_json=bool(getattr(args, "json", False)))
        return
    after = remote.warm_auth_session()
    payload = _status_payload(cfg, after)
    payload["bootstrap_state"] = "started"
    _emit_status(payload, use_json=bool(getattr(args, "json", False)))
