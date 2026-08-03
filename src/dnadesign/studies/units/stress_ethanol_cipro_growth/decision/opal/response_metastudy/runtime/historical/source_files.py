"""Resolve digest-bound files owned by the frozen response-metastudy replay."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ...evaluation.multistate_behavior_source_protocol import BehaviorSourceEquivalenceProtocol


@dataclass(frozen=True)
class HistoricalSourceFiles:
    reader_request: Path
    observation_policy: Path


def load_historical_source_files(
    repo_root: Path,
    *,
    protocol: BehaviorSourceEquivalenceProtocol,
) -> HistoricalSourceFiles:
    """Verify the immutable request and policy before replay reads either file."""

    return HistoricalSourceFiles(
        reader_request=_source_path(
            repo_root,
            relative_path=protocol.prior_observation_request_repo_path,
            expected_sha256=protocol.prior_observation_request_sha256,
            label="historical Reader request",
        ),
        observation_policy=_source_path(
            repo_root,
            relative_path=protocol.prior_observation_policy_repo_path,
            expected_sha256=protocol.prior_observation_policy_sha256,
            label="historical observation policy",
        ),
    )


def _source_path(
    repo_root: Path,
    *,
    relative_path: str,
    expected_sha256: str,
    label: str,
) -> Path:
    root = Path(repo_root).resolve()
    resolved = (root / relative_path).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"protocol-declared {label} is missing or escapes the repository.")
    if _sha256(resolved) != expected_sha256:
        raise ValueError(f"protocol-declared {label} digest mismatch.")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["HistoricalSourceFiles", "load_historical_source_files"]
