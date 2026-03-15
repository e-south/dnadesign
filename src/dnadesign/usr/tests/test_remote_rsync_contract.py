"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_remote_rsync_contract.py

Tests for rsync command construction on USR remotes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.usr.src.config import SSHRemoteConfig
from dnadesign.usr.src.remote import SSHRemote


def _remote(*, batch_mode: bool) -> SSHRemote:
    return SSHRemote(
        SSHRemoteConfig(
            name="bu-scc",
            host="scc1.bu.edu",
            user="alice",
            base_dir="/project/alice/dnadesign/src/dnadesign/usr/datasets",
            batch_mode=batch_mode,
        )
    )


def test_rsync_cmd_avoids_host_specific_permission_metadata() -> None:
    cmd = _remote(batch_mode=True)._rsync_cmd()

    assert "-rltz" in cmd
    assert "-az" not in cmd
    assert "--no-perms" in cmd
    assert "--no-owner" in cmd
    assert "--no-group" in cmd
    assert "--omit-dir-times" in cmd


def test_rsync_cmd_respects_batch_mode_toggle() -> None:
    strict_cmd = _remote(batch_mode=True)._rsync_cmd()
    interactive_cmd = _remote(batch_mode=False)._rsync_cmd()

    strict_ssh = strict_cmd[strict_cmd.index("-e") + 1]
    interactive_ssh = interactive_cmd[interactive_cmd.index("-e") + 1]

    assert "BatchMode=yes" in strict_ssh
    assert "BatchMode=yes" not in interactive_ssh
