"""Contracts for MSA backend execution and aligned FASTA bundles."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MsaBackendSpec:
    """Declared MSA backend and execution policy."""

    backend_id: str = "mafft"
    executable: str = "mafft"
    environment: str = "pixi"
    failure_policy: str = "fail_if_missing"

    def __post_init__(self) -> None:
        if self.backend_id != "mafft":
            raise ValueError(f"unsupported MSA backend_id: {self.backend_id!r}")
        if self.failure_policy != "fail_if_missing":
            raise ValueError(f"unsupported MSA failure_policy: {self.failure_policy!r}")


@dataclass(frozen=True)
class MsaRequest:
    """Inputs required to run one explicit MSA backend pass."""

    input_fasta: Path
    output_fasta: Path
    manifest_path: Path
    target_row_id: str | None = None
    backend: MsaBackendSpec = field(default_factory=MsaBackendSpec)
    command_args: tuple[str, ...] = ("--globalpair", "--maxiterate", "1000", "--reorder")

    def __post_init__(self) -> None:
        if not self.command_args:
            raise ValueError("MSA command_args must be explicit and non-empty")
        if any(not arg or not isinstance(arg, str) for arg in self.command_args):
            raise ValueError("MSA command_args must contain only non-empty strings")


@dataclass(frozen=True)
class MsaRunResult:
    """Result and provenance emitted by one MSA backend pass."""

    aligned_fasta: Path
    manifest_path: Path
    backend_id: str
    backend_version: str
    command: tuple[str, ...]
    input_fasta_sha256: str
    output_fasta_sha256: str
    pixi_lock_sha256: str | None
