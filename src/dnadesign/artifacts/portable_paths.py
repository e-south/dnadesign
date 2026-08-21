"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/artifacts/portable_paths.py

Conservative cross-filesystem identity for artifact bundle paths.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

from .errors import PublicationError


def portable_path_identity(path: str | Path) -> tuple[str, ...]:
    """Return a conservative portable identity for a relative bundle path.

    This is a portability contract, not a probe of the host filesystem. NFKC
    normalization and case folding deliberately reject names that may alias on
    common case-insensitive or Unicode-normalizing filesystems.
    """

    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise PublicationError(f"Portable bundle path must be relative and contained: {path}")
    return tuple(
        unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", part).casefold()) for part in relative.parts
    )


def _entries_by_identity(stage: Path) -> dict[tuple[str, ...], Path]:
    entries: dict[tuple[str, ...], Path] = {}
    for entry in sorted(stage.rglob("*")):
        relative = entry.relative_to(stage)
        identity = portable_path_identity(relative)
        previous = entries.get(identity)
        if previous is not None:
            raise PublicationError(
                "Artifact bundle staging contains paths with the same portable filesystem identity: "
                f"{previous.relative_to(stage)} and {relative}"
            )
        entries[identity] = entry
    return entries


def validate_publication_metadata_paths(
    stage: Path,
    *,
    required_manifest: Path,
    owner_file: str,
    require_owner: bool = True,
) -> None:
    """Validate required and reserved metadata under portable path identity.

    Staging publications require their transaction owner. Published bundles
    require that owner to be absent. Both states reject portable aliases of the
    reserved manifest and owner names.
    """

    canonical_manifest = Path("manifest.json")
    if (
        portable_path_identity(required_manifest) == portable_path_identity(canonical_manifest)
        and required_manifest != canonical_manifest
    ):
        raise PublicationError(
            "Artifact bundle required manifest uses a reserved metadata alias under the portable filesystem identity: "
            f"{required_manifest}"
        )
    entries = _entries_by_identity(stage)
    owner_identity = portable_path_identity(Path(owner_file))[0]
    for identity, entry in entries.items():
        relative = entry.relative_to(stage)
        if owner_identity in identity and relative != Path(owner_file):
            raise PublicationError(
                f"Artifact bundle staging contains a reserved publication owner metadata name: {relative}"
            )
    reserved = (canonical_manifest, Path(owner_file))
    for canonical in reserved:
        match = entries.get(portable_path_identity(canonical))
        if match is not None and match.relative_to(stage) != canonical:
            raise PublicationError(
                "Artifact bundle staging contains a reserved metadata alias under the portable filesystem identity: "
                f"{match.relative_to(stage)}"
            )
    manifest = entries.get(portable_path_identity(required_manifest))
    if manifest is None or manifest.relative_to(stage) != required_manifest or not manifest.is_file():
        raise PublicationError(f"Artifact bundle staging is incomplete: {stage / required_manifest}")
    owner = entries.get(portable_path_identity(Path(owner_file)))
    if require_owner:
        if owner is None or owner.relative_to(stage) != Path(owner_file) or not owner.is_file():
            raise PublicationError("Artifact bundle publication owner sentinel is unavailable or unsafe")
    elif owner is not None:
        raise PublicationError("Published artifact bundle cannot contain publication transaction metadata")


__all__ = ["portable_path_identity", "validate_publication_metadata_paths"]
