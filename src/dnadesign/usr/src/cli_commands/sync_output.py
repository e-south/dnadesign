"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/sync_output.py

Output rendering helpers for USR sync command summaries and audits.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..ui import render_diff_rich


def print_diff(summary, *, use_rich: bool | None = None) -> None:
    def fmt_sz(size):
        if size is None:
            return "?"
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.0f}{unit}"
            size /= 1024
        return f"{size:.0f}PB"

    if use_rich:
        render_diff_rich(summary)
        return
    pl, pr = summary.primary_local, summary.primary_remote
    changes = dict(getattr(summary, "changes", {}) or {})
    derived_local = len(getattr(summary, "derived_local_files", []) or [])
    derived_remote = len(getattr(summary, "derived_remote_files", []) or [])
    aux_local = len(getattr(summary, "aux_local_files", []) or [])
    aux_remote = len(getattr(summary, "aux_remote_files", []) or [])
    print(f"Dataset: {summary.dataset}")

    local_line = f"Local  : {pl.sha256 or '?'}  size={fmt_sz(pl.size)}  rows={pl.rows or '?'}  cols={pl.cols or '?'}"
    print(local_line)

    remote_line = f"Remote : {pr.sha256 or '?'}  size={fmt_sz(pr.size)}  rows={pr.rows or '?'}  cols={pr.cols or '?'}"
    print(remote_line)

    eq = "==" if (pl.sha256 and pr.sha256 and pl.sha256 == pr.sha256) else "≠"
    print(f"Primary sha: {pl.sha256 or '?'} {eq} {pr.sha256 or '?'}")
    print(f"meta.md     mtime: {summary.meta_local_mtime or '-'}  →  {summary.meta_remote_mtime or '-'}")
    delta_evt = max(0, summary.events_remote_lines - summary.events_local_lines)
    print(
        ".events.log lines: "
        f"local={summary.events_local_lines}  "
        f"remote={summary.events_remote_lines}  "
        f"(+{delta_evt} on remote)"
    )
    print(f"_snapshots : remote_count={summary.snapshots.count}  newer_than_local={summary.snapshots.newer_than_local}")
    print(
        "_derived   : "
        f"{'changed' if changes.get('derived_files_diff') else 'unchanged'}  "
        f"local_files={derived_local}  remote_files={derived_remote}"
    )
    print(
        "_auxiliary : "
        f"{'changed' if changes.get('aux_files_diff') else 'unchanged'}  "
        f"local_files={aux_local}  remote_files={aux_remote}"
    )
    print("Status     :", "CHANGES" if summary.has_change else "up-to-date")
    print("Verify     :", summary.verify_mode)


def print_verify_notes(summary) -> None:
    for note in summary.verify_notes:
        print(f"WARNING: {note}")


def build_sync_audit_payload(
    summary, *, action: str, dry_run: bool, verify_sidecars: bool, verify_derived_hashes: bool
) -> dict:
    has_change = bool(getattr(summary, "has_change", False))
    action_text = str(action)
    if action_text == "diff":
        transfer_state = "DIFF-ONLY"
    else:
        transfer_state = "DRY-RUN" if dry_run else ("TRANSFERRED" if has_change else "NO-OP")
    dataset_name = str(getattr(summary, "dataset", "<unknown>"))
    verify_mode = str(getattr(summary, "verify_mode", "auto"))
    changes = dict(getattr(summary, "changes", {}) or {})
    events_local = int(getattr(summary, "events_local_lines", 0))
    events_remote = int(getattr(summary, "events_remote_lines", 0))
    snapshots = getattr(summary, "snapshots", None)
    snapshot_count = int(getattr(snapshots, "count", 0)) if snapshots is not None else 0
    snapshots_newer = int(getattr(snapshots, "newer_than_local", 0)) if snapshots is not None else 0
    snapshots_changed = bool(changes.get("snapshots_name_diff")) or snapshots_newer > 0
    derived_local_files = [str(path) for path in (getattr(summary, "derived_local_files", []) or [])]
    derived_remote_files = [str(path) for path in (getattr(summary, "derived_remote_files", []) or [])]
    derived_local = len(derived_local_files)
    derived_remote = len(derived_remote_files)
    derived_local_only = sorted(set(derived_local_files) - set(derived_remote_files))
    derived_remote_only = sorted(set(derived_remote_files) - set(derived_local_files))
    derived_changed = bool(changes.get("derived_files_diff"))
    aux_local_files = [str(path) for path in (getattr(summary, "aux_local_files", []) or [])]
    aux_remote_files = [str(path) for path in (getattr(summary, "aux_remote_files", []) or [])]
    aux_local = len(aux_local_files)
    aux_remote = len(aux_remote_files)
    aux_local_only = sorted(set(aux_local_files) - set(aux_remote_files))
    aux_remote_only = sorted(set(aux_remote_files) - set(aux_local_files))
    aux_changed = bool(changes.get("aux_files_diff"))
    return {
        "action": action_text,
        "transfer_state": transfer_state,
        "dataset": dataset_name,
        "verify": {
            "primary": verify_mode,
            "sidecars": "strict" if verify_sidecars else "off",
            "content_hashes": "on" if verify_derived_hashes else "off",
        },
        "primary": {"changed": bool(changes.get("primary_sha_diff"))},
        "meta": {"changed": bool(changes.get("meta_mtime_diff"))},
        ".events.log": {"local": events_local, "remote": events_remote},
        "_snapshots": {
            "changed": snapshots_changed,
            "remote_count": snapshot_count,
            "newer_than_local": snapshots_newer,
        },
        "_derived": {
            "changed": derived_changed,
            "local_files": derived_local,
            "remote_files": derived_remote,
            "local_only": derived_local_only,
            "remote_only": derived_remote_only,
        },
        "_auxiliary": {
            "changed": aux_changed,
            "local_files": aux_local,
            "remote_files": aux_remote,
            "local_only": aux_local_only,
            "remote_only": aux_remote_only,
        },
    }


def print_sync_audit(
    summary, *, action: str, dry_run: bool, verify_sidecars: bool, verify_derived_hashes: bool
) -> None:
    payload = build_sync_audit_payload(
        summary,
        action=action,
        dry_run=dry_run,
        verify_sidecars=verify_sidecars,
        verify_derived_hashes=verify_derived_hashes,
    )
    print(f"{str(payload['action']).upper()} audit: {payload['transfer_state']}")
    print(f"Dataset    : {payload['dataset']}")
    print(
        "Verify     : "
        f"primary={payload['verify']['primary']} "
        f"sidecars={payload['verify']['sidecars']} "
        f"content_hashes={payload['verify']['content_hashes']}"
    )
    print(f"Primary    : {'changed' if payload['primary']['changed'] else 'unchanged'}")
    print(f"meta.md    : {'changed' if payload['meta']['changed'] else 'unchanged'}")
    print(f".events.log: local={payload['.events.log']['local']}  remote={payload['.events.log']['remote']}")
    print(
        "_snapshots : "
        f"{'changed' if payload['_snapshots']['changed'] else 'unchanged'}  "
        f"remote_count={payload['_snapshots']['remote_count']}  "
        f"newer_than_local={payload['_snapshots']['newer_than_local']}"
    )
    print(
        "_derived   : "
        f"{'changed' if payload['_derived']['changed'] else 'unchanged'}  "
        f"local_files={payload['_derived']['local_files']}  "
        f"remote_files={payload['_derived']['remote_files']}  "
        f"local_only={len(payload['_derived']['local_only'])}  "
        f"remote_only={len(payload['_derived']['remote_only'])}"
    )
    print(
        f"_auxiliary : {'changed' if payload['_auxiliary']['changed'] else 'unchanged'}  "
        f"local_files={payload['_auxiliary']['local_files']}  "
        f"remote_files={payload['_auxiliary']['remote_files']}  "
        f"local_only={len(payload['_auxiliary']['local_only'])}  "
        f"remote_only={len(payload['_auxiliary']['remote_only'])}"
    )
