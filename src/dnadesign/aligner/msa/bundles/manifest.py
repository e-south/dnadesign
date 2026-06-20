"""Manifest records for aligned FASTA bundles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AlignedFastaBundleManifest:
    """Provenance for one aligned FASTA emitted by an MSA backend."""

    backend_id: str
    backend_version: str
    executable_path: str
    command: list[str]
    input_fasta: str
    output_fasta: str
    input_fasta_sha256: str
    output_fasta_sha256: str
    target_row_id: str | None
    environment: str
    pixi_lock_sha256: str | None
    failure_policy: str

    def to_mapping(self) -> dict[str, object]:
        """Return a YAML-serializable manifest mapping."""

        return {
            "schema_id": "dnadesign.aligner.msa.aligned_fasta_bundle",
            "schema_version": 1,
            **asdict(self),
        }


def write_bundle_manifest(path: Path, manifest: AlignedFastaBundleManifest) -> None:
    """Write a deterministic YAML manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest.to_mapping(), sort_keys=True), encoding="utf-8")
