"""Public sufficiency API for Eco1 conservation source FASTA bundles."""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.io import (
    load_yaml_mapping,
    resolve_path,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.paths import (
    CONSERVATION_SOURCES,
    DEFAULT_OUTPUT_ROOT,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.sufficiency.context import (
    SourceSequenceSufficiencyContext,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.sufficiency.manifests import (
    collect_source_sequence_sufficiency_issues,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.sufficiency.report import (
    SourceSequenceBundleSufficiencyReport,
    dedupe_issues,
)


def validate_source_sequence_bundle_sufficiency(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    source_cache_root: Path | None = None,
    bundle_root: Path | None = None,
) -> SourceSequenceBundleSufficiencyReport:
    """Validate source FASTA bundles before any MAFFT alignment run."""

    root = (repo_root or _find_repo_root(Path.cwd())).expanduser().resolve()
    out_root = resolve_path(root, output_root or DEFAULT_OUTPUT_ROOT)
    context = SourceSequenceSufficiencyContext(
        repo_root=root,
        output_root=out_root,
        source_cache_root=resolve_path(root, source_cache_root)
        if source_cache_root is not None
        else out_root / "conservation_source_cache",
        bundle_root=resolve_path(root, bundle_root) if bundle_root is not None else out_root / "conservation_sources",
        conservation_sources_path=root / CONSERVATION_SOURCES,
        conservation_sources=load_yaml_mapping(root / CONSERVATION_SOURCES),
    )
    return SourceSequenceBundleSufficiencyReport(
        issues=dedupe_issues(collect_source_sequence_sufficiency_issues(context)),
    )


def _find_repo_root(start: Path) -> Path:
    for parent in (start.resolve(), *start.resolve().parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("repo root with pyproject.toml not found")
