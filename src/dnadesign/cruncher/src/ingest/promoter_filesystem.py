"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/ingest/promoter_filesystem.py

Discover parseable promoter sources from an explicit data repository root.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path

from .promoter_contracts import PromoterAssociationSourceFile, PromoterSourceFile


def iter_promoter_source_files(root: Path | None) -> tuple[PromoterSourceFile, ...]:
    """Return one parseable promoter table per declared RegulonDB release."""

    data_root = _require_root(root)
    by_release: dict[str, list[Path]] = {}
    pattern = "sources/databases/regulondb/*/promoters/PromoterSet.*"
    for candidate in sorted(data_root.glob(pattern)):
        if candidate.is_file() and candidate.suffix.lower() in {".csv", ".tsv"}:
            by_release.setdefault(candidate.parent.parent.name, []).append(candidate)

    sources = []
    for release in sorted(by_release, key=_release_key, reverse=True):
        candidates = sorted(by_release[release], key=lambda path: (path.suffix.lower() != ".tsv", path.name))
        selected = candidates[0]
        file_format = selected.suffix.lower().removeprefix(".")
        sources.append(
            PromoterSourceFile(
                source_id=f"regulondb_{_release_id(release)}_promoter_set",
                source="regulondb",
                release=release,
                path=selected.relative_to(data_root).as_posix(),
                table=selected.name,
                stratum="local_release_pinned_curated",
                role="curated_base",
                file_format=file_format,
                parser_hint="regulondb_promoter_set",
                creates_base_rows=True,
            )
        )
    return tuple(sources)


def iter_promoter_association_source_files(root: Path | None) -> tuple[PromoterAssociationSourceFile, ...]:
    """Prefer the newest direct TF-promoter table, then historical network tables."""

    data_root = _require_root(root)
    direct = sorted(
        (path for path in data_root.glob("sources/databases/regulondb/*/binding_sites/TF-RISet.tsv") if path.is_file()),
        key=lambda path: _release_key(path.parent.parent.name),
        reverse=True,
    )
    if direct:
        selected = direct[0]
        release = selected.parent.parent.name
        return (
            PromoterAssociationSourceFile(
                source_id=f"regulondb_{_release_id(release)}_tf_riset",
                source="regulondb",
                release=release,
                path=selected.relative_to(data_root).as_posix(),
                table=selected.name,
                stratum="current_curated_regulatory_interaction",
                role="tf_promoter_association_overlay",
                file_format="tsv",
                parser_hint="regulondb_tf_riset",
            ),
        )

    historical = []
    for path in sorted(data_root.glob("sources/databases/regulondb/*/network_associations/network_tf_tu.txt")):
        if not path.is_file():
            continue
        release = path.parent.parent.name
        historical.append(
            PromoterAssociationSourceFile(
                source_id=f"regulondb_{_release_id(release)}_network_tf_tu",
                source="regulondb",
                release=release,
                path=path.relative_to(data_root).as_posix(),
                table=path.name,
                stratum="historical_curated_network_association",
                role="tf_promoter_association_overlay",
                file_format="tsv",
                parser_hint="regulondb_network_tf_tu",
            )
        )
    return tuple(historical)


def _require_root(root: Path | None) -> Path:
    if root is None:
        raise ValueError("filesystem promoter discovery requires an explicit data root")
    data_root = Path(root).expanduser().resolve()
    if not data_root.is_dir():
        raise ValueError(f"promoter data root does not exist: {data_root}")
    return data_root


def _release_id(release: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", release.lower()).strip("_")
    return normalized.removesuffix("_0")


def _release_key(release: str) -> tuple[tuple[int, int | str], ...]:
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower()) for part in re.split(r"[._-]", release))


__all__ = ["iter_promoter_association_source_files", "iter_promoter_source_files"]
