"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/ligandmpnn/preflight.py

Pinned LigandMPNN checkout and checkpoint preflight.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dnadesign.thread.adapters.ligandmpnn.models import LigandMpnnUpstreamPin
from dnadesign.thread.adapters.ligandmpnn.receipts import LigandMpnnProvenance


@dataclass(frozen=True)
class LigandMpnnPreflightIssue:
    """One actionable mismatch against the pinned upstream contract."""

    check_id: str
    message: str
    path: str


@dataclass(frozen=True)
class LigandMpnnPreflightReport:
    """Normalized, non-executing preflight result."""

    issues: tuple[LigandMpnnPreflightIssue, ...]
    provenance: LigandMpnnProvenance

    @property
    def ok(self) -> bool:
        return not self.issues


def preflight_ligandmpnn(
    checkout_root: Path,
    pin: LigandMpnnUpstreamPin,
    *,
    require_packing_checkpoint: bool = False,
) -> LigandMpnnPreflightReport:
    """Check entrypoint, exact Git commit, and exact checkpoint digests."""

    root = checkout_root.expanduser().resolve()
    issues: list[LigandMpnnPreflightIssue] = []
    entrypoint = root / "run.py"
    if not entrypoint.is_file():
        issues.append(_issue("missing_entrypoint", "checkout is missing official run.py", entrypoint))
    score_entrypoint = root / "score.py"
    if not score_entrypoint.is_file():
        issues.append(_issue("missing_score_entrypoint", "checkout is missing official score.py", score_entrypoint))
    parser_module = root / "data_utils.py"
    if not parser_module.is_file():
        issues.append(_issue("missing_parser_module", "checkout is missing official data_utils.py", parser_module))

    observed_commit = _git_commit(root)
    if observed_commit is None:
        issues.append(_issue("unreadable_upstream_commit", "checkout Git commit could not be read", root))
    elif observed_commit != pin.commit:
        issues.append(
            _issue("upstream_commit_mismatch", f"expected commit {pin.commit}, observed {observed_commit}", root)
        )
    if observed_commit == pin.commit:
        for entrypoint_path in (entrypoint, score_entrypoint):
            if not entrypoint_path.is_file():
                continue
            matches_pin = _git_path_matches_commit(root, pin.commit, entrypoint_path.name)
            if matches_pin is None:
                issues.append(
                    _issue(
                        "unreadable_entrypoint_blob",
                        "pinned checkout entrypoint bytes could not be read",
                        entrypoint_path,
                    )
                )
            elif not matches_pin:
                issues.append(
                    _issue(
                        "dirty_entrypoint",
                        "checkout entrypoint differs from the pinned commit",
                        entrypoint_path,
                    )
                )

    _check_digest(root / pin.checkpoint_path, pin.checkpoint_sha256, "checkpoint", issues)
    if require_packing_checkpoint:
        if pin.packing_checkpoint_sha256 is None:
            issues.append(
                _issue(
                    "packing_checkpoint_unpinned",
                    "side-chain packing requires an explicit packing checkpoint digest",
                    root / pin.packing_checkpoint_path,
                )
            )
        else:
            _check_digest(
                root / pin.packing_checkpoint_path,
                pin.packing_checkpoint_sha256,
                "packing_checkpoint",
                issues,
            )
    return LigandMpnnPreflightReport(issues=tuple(issues), provenance=LigandMpnnProvenance.from_pin(pin))


def _git_commit(root: Path) -> str | None:
    try:
        observed = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if isinstance(observed, bytes):
            observed = observed.decode("ascii")
        return observed.strip().lower()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_path_matches_commit(root: Path, commit: str, path: str) -> bool | None:
    try:
        pinned_bytes = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{commit}:{path}"],
            stderr=subprocess.DEVNULL,
        )
        return (root / path).read_bytes() == pinned_bytes
    except (OSError, subprocess.CalledProcessError, UnicodeError):
        return None


def _check_digest(
    path: Path,
    expected: str,
    label: str,
    issues: list[LigandMpnnPreflightIssue],
) -> None:
    if not path.is_file():
        issues.append(_issue(f"missing_{label}", f"checkout is missing pinned {label.replace('_', ' ')}", path))
        return
    observed = _sha256_file(path)
    if observed != expected:
        issues.append(_issue(f"{label}_hash_mismatch", f"expected sha256:{expected}, observed sha256:{observed}", path))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _issue(suffix: str, message: str, path: Path) -> LigandMpnnPreflightIssue:
    return LigandMpnnPreflightIssue(
        check_id=f"thread.ligandmpnn.{suffix}",
        message=message,
        path=str(path),
    )
