"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/msa/contracts.py

Contracts for MSA backend execution and aligned FASTA bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MsaBackendSpec:
    """Declared MSA backend and execution policy."""

    backend_id: str = "mafft"
    executable: str | None = None
    environment: str = "pixi"
    failure_policy: str = "fail_if_missing"

    def __post_init__(self) -> None:
        if self.backend_id not in {"clustalo", "mafft"}:
            raise ValueError(f"unsupported MSA backend_id: {self.backend_id!r}")
        if self.failure_policy != "fail_if_missing":
            raise ValueError(f"unsupported MSA failure_policy: {self.failure_policy!r}")

    @property
    def executable_name(self) -> str:
        """Return the declared executable or the backend default."""

        return self.executable or self.backend_id


@dataclass(frozen=True)
class MsaRequest:
    """Inputs required to run one explicit MSA backend pass."""

    input_fasta: Path
    output_fasta: Path
    manifest_path: Path
    target_row_id: str | None = None
    backend: MsaBackendSpec = field(default_factory=MsaBackendSpec)
    command_args: tuple[str, ...] = ("--globalpair", "--maxiterate", "1000", "--reorder")
    timeout_seconds: float | None = None
    stderr_path: Path | None = None
    run_label: str | None = None

    def __post_init__(self) -> None:
        if not self.command_args:
            raise ValueError("MSA command_args must be explicit and non-empty")
        if any(not arg or not isinstance(arg, str) for arg in self.command_args):
            raise ValueError("MSA command_args must contain only non-empty strings")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("MSA timeout_seconds must be positive when provided")
        if self.run_label is not None and not self.run_label.strip():
            raise ValueError("MSA run_label must be non-empty when provided")


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
    elapsed_seconds: float
    return_code: int
    stderr_path: Path | None
    run_label: str | None
