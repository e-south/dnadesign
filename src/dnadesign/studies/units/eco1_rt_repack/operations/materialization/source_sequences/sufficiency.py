"""Public sufficiency API and CLI for Eco1 conservation source FASTA bundles."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.checks import (
    SourceSequenceSufficiencyContext,
    collect_source_sequence_sufficiency_issues,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.paths import (
    CONSERVATION_SOURCES,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_BUNDLE_ROOT,
    DEFAULT_SOURCE_CACHE_ROOT,
)


@dataclass(frozen=True)
class SourceSequenceBundleSufficiencyReport:
    """Validation result for one source-sequence bundle sufficiency gate."""

    issues: tuple[ContractIssue, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "issue_count": len(self.issues),
            "issues": [issue.as_dict() for issue in self.issues],
        }


def validate_source_sequence_bundle_sufficiency(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    source_cache_root: Path | None = None,
    bundle_root: Path | None = None,
) -> SourceSequenceBundleSufficiencyReport:
    """Validate source FASTA bundles before any MAFFT alignment run."""

    root = (repo_root or _find_repo_root(Path.cwd())).expanduser().resolve()
    context = SourceSequenceSufficiencyContext(
        repo_root=root,
        output_root=_resolve_path(root, output_root or DEFAULT_OUTPUT_ROOT),
        source_cache_root=_resolve_path(root, source_cache_root or DEFAULT_SOURCE_CACHE_ROOT),
        bundle_root=_resolve_path(root, bundle_root or DEFAULT_SOURCE_BUNDLE_ROOT),
        conservation_sources_path=root / CONSERVATION_SOURCES,
        conservation_sources=_load_yaml(root / CONSERVATION_SOURCES),
    )
    return SourceSequenceBundleSufficiencyReport(
        issues=_dedupe_issues(collect_source_sequence_sufficiency_issues(context)),
    )


def _resolve_path(repo_root: Path, path: Path) -> Path:
    resolved = path.expanduser()
    return resolved if resolved.is_absolute() else (repo_root / resolved).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return loaded


def _find_repo_root(start: Path) -> Path:
    for parent in (start.resolve(), *start.resolve().parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("repo root with pyproject.toml not found")


def _dedupe_issues(issues: tuple[ContractIssue, ...]) -> tuple[ContractIssue, ...]:
    observed: set[tuple[str, str, str]] = set()
    deduped: list[ContractIssue] = []
    for issue in issues:
        key = (issue.check_id, issue.path, issue.message)
        if key not in observed:
            observed.add(key)
            deduped.append(issue)
    return tuple(deduped)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Eco1 RT conservation source-sequence bundle sufficiency.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-cache-root", type=Path, default=DEFAULT_SOURCE_CACHE_ROOT)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_SOURCE_BUNDLE_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = validate_source_sequence_bundle_sufficiency(
        repo_root=args.repo_root,
        output_root=args.output_root,
        source_cache_root=args.source_cache_root,
        bundle_root=args.bundle_root,
    )
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
