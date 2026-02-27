"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/sync_policy.py

CLI policy helpers for sync verification defaults and sidecar strictness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

DEFAULT_SYNC_VERIFY = "hash"


def resolve_sync_verify(args) -> str:
    raw = getattr(args, "verify", None)
    if raw is None:
        return DEFAULT_SYNC_VERIFY
    text = str(raw).strip().lower()
    if not text:
        return DEFAULT_SYNC_VERIFY
    return text


def resolve_verify_sidecars(args, *, file_mode: bool) -> bool:
    explicit_on = bool(getattr(args, "verify_sidecars", False))
    explicit_off = bool(getattr(args, "no_verify_sidecars", False))
    if explicit_on and explicit_off:
        raise SystemExit("Cannot combine --verify-sidecars and --no-verify-sidecars.")
    if file_mode:
        if explicit_on or explicit_off:
            raise SystemExit("--verify-sidecars/--no-verify-sidecars are dataset-only flags (not valid in FILE mode).")
        return False
    if explicit_off:
        return False
    return True


def resolve_verify_derived_hashes(args, *, file_mode: bool, verify_sidecars: bool) -> bool:
    explicit_on = bool(getattr(args, "verify_derived_hashes", False))
    explicit_off = bool(getattr(args, "no_verify_derived_hashes", False))
    if explicit_on and explicit_off:
        raise SystemExit("Cannot combine --verify-derived-hashes and --no-verify-derived-hashes.")
    if file_mode:
        if explicit_on or explicit_off:
            raise SystemExit(
                "--verify-derived-hashes/--no-verify-derived-hashes are dataset-only flags (not valid in FILE mode)."
            )
        return False
    if not verify_sidecars:
        if explicit_on:
            raise SystemExit("--verify-derived-hashes requires sidecar verification.")
        return False
    if explicit_off:
        return False
    return True
