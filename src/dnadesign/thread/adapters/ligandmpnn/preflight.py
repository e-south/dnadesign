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
import stat
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from dnadesign.thread.adapters.ligandmpnn.models import LigandMpnnUpstreamPin
from dnadesign.thread.adapters.ligandmpnn.pinned_checkout import (
    index_path_matches_commit,
    working_tree_path_matches_commit,
)
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


class _RegularFileStatus(Enum):
    MISSING = "missing"
    REGULAR = "regular"
    NOT_REGULAR = "not_regular"
    UNREADABLE = "unreadable"


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
    parser_module_regular = _regular_file_status_no_follow(parser_module)
    if parser_module_regular is _RegularFileStatus.MISSING:
        issues.append(_issue("missing_parser_module", "checkout is missing official data_utils.py", parser_module))
    elif parser_module_regular is _RegularFileStatus.UNREADABLE:
        issues.append(
            _issue(
                "unreadable_parser_module",
                "pinned LigandMPNN data_utils.py status could not be read",
                parser_module,
            )
        )
    elif parser_module_regular is _RegularFileStatus.NOT_REGULAR:
        issues.append(
            _issue(
                "parser_module_not_regular",
                "pinned LigandMPNN data_utils.py must be a regular file",
                parser_module,
            )
        )

    observed_commit = _git_commit(root)
    if observed_commit is None:
        issues.append(_issue("unreadable_upstream_commit", "checkout Git commit could not be read", root))
    elif observed_commit != pin.commit:
        issues.append(
            _issue("upstream_commit_mismatch", f"expected commit {pin.commit}, observed {observed_commit}", root)
        )
    if observed_commit == pin.commit:
        if parser_module_regular is _RegularFileStatus.REGULAR:
            parser_tracked = _git_tracks_path(root, parser_module.name)
            if parser_tracked is None:
                issues.append(
                    _issue(
                        "unreadable_parser_index",
                        "pinned checkout data_utils.py index state could not be read",
                        parser_module,
                    )
                )
            elif parser_tracked is False:
                issues.append(
                    _issue(
                        "untracked_parser_module",
                        "pinned checkout must track data_utils.py in the Git index",
                        parser_module,
                    )
                )
            elif index_path_matches_commit(root, pin.commit, parser_module.name) is not True:
                issues.append(
                    _issue(
                        "dirty_parser_index",
                        "checkout parser module Git index differs from the pinned commit",
                        parser_module,
                    )
                )
        declared_sources = (
            (entrypoint, "entrypoint", "unreadable_entrypoint_blob", "dirty_entrypoint"),
            (score_entrypoint, "entrypoint", "unreadable_entrypoint_blob", "dirty_entrypoint"),
            (parser_module, "parser module", "unreadable_parser_blob", "dirty_parser_module"),
        )
        for source_path, label, unreadable_issue, dirty_issue in declared_sources:
            if source_path == parser_module and parser_module_regular is not _RegularFileStatus.REGULAR:
                continue
            if not source_path.is_file():
                continue
            matches_pin = working_tree_path_matches_commit(root, pin.commit, source_path.name)
            if matches_pin is None:
                issues.append(
                    _issue(
                        unreadable_issue,
                        f"pinned checkout {label} bytes could not be read",
                        source_path,
                    )
                )
            elif not matches_pin:
                issues.append(
                    _issue(
                        dirty_issue,
                        f"checkout {label} differs from the pinned commit",
                        source_path,
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


def _git_tracks_path(root: Path, relative_path: str) -> bool | None:
    try:
        observed = subprocess.check_output(
            [
                "git",
                "--no-replace-objects",
                "-C",
                str(root),
                "ls-files",
                "--error-unmatch",
                "--",
                relative_path,
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if isinstance(observed, bytes):
            observed = observed.decode("utf-8")
        return observed.strip() == relative_path
    except subprocess.CalledProcessError:
        return False
    except OSError:
        return None


def _regular_file_status_no_follow(path: Path) -> _RegularFileStatus:
    """Report a leaf's regular-file status without following a symlink."""

    try:
        is_regular = stat.S_ISREG(path.lstat().st_mode)
    except FileNotFoundError:
        return _RegularFileStatus.MISSING
    except OSError:
        return _RegularFileStatus.UNREADABLE
    return _RegularFileStatus.REGULAR if is_regular else _RegularFileStatus.NOT_REGULAR


def _check_digest(
    path: Path,
    expected: str,
    label: str,
    issues: list[LigandMpnnPreflightIssue],
) -> None:
    if path.is_symlink():
        issues.append(
            _issue(
                f"{label}_not_regular",
                f"pinned {label.replace('_', ' ')} must be a regular file, not a symlink",
                path,
            )
        )
        return
    if not path.is_file():
        issues.append(_issue(f"missing_{label}", f"checkout is missing pinned {label.replace('_', ' ')}", path))
        return
    try:
        observed = _sha256_file(path)
    except OSError:
        issues.append(_issue(f"unreadable_{label}", f"pinned {label.replace('_', ' ')} could not be read", path))
        return
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
