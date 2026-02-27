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


def print_sync_audit(
    summary, *, action: str, dry_run: bool, verify_sidecars: bool, verify_derived_hashes: bool
) -> None:
    has_change = bool(getattr(summary, "has_change", False))
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
    derived_local = len(getattr(summary, "derived_local_files", []) or [])
    derived_remote = len(getattr(summary, "derived_remote_files", []) or [])
    derived_changed = bool(changes.get("derived_files_diff"))
    aux_local = len(getattr(summary, "aux_local_files", []) or [])
    aux_remote = len(getattr(summary, "aux_remote_files", []) or [])
    aux_changed = bool(changes.get("aux_files_diff"))
    print(f"{action.upper()} audit: {transfer_state}")
    print(f"Dataset    : {dataset_name}")
    print(
        "Verify     : "
        f"primary={verify_mode} sidecars={'strict' if verify_sidecars else 'off'} "
        f"derived_hashes={'on' if verify_derived_hashes else 'off'}"
    )
    print(f"Primary    : {'changed' if changes.get('primary_sha_diff') else 'unchanged'}")
    print(f"meta.md    : {'changed' if changes.get('meta_mtime_diff') else 'unchanged'}")
    print(f".events.log: local={events_local}  remote={events_remote}")
    print(
        "_snapshots : "
        f"{'changed' if snapshots_changed else 'unchanged'}  "
        f"remote_count={snapshot_count}  newer_than_local={snapshots_newer}"
    )
    print(
        "_derived   : "
        f"{'changed' if derived_changed else 'unchanged'}  "
        f"local_files={derived_local}  remote_files={derived_remote}"
    )
    print(
        f"_auxiliary : {'changed' if aux_changed else 'unchanged'}  local_files={aux_local}  remote_files={aux_remote}"
    )
