"""Normalized LigandMPNN provenance and non-executing request receipts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from dnadesign.thread.adapters.ligandmpnn.models import (
    UPSTREAM_REPOSITORY,
    LigandMpnnCommand,
    LigandMpnnRequest,
    LigandMpnnUpstreamPin,
)


@dataclass(frozen=True)
class LigandMpnnProvenance:
    """Normalized source and weight identities for an adapted request."""

    upstream_repository: str
    upstream_commit: str
    checkpoint_sha256: str
    packing_checkpoint_sha256: str | None

    @classmethod
    def from_pin(cls, pin: LigandMpnnUpstreamPin) -> LigandMpnnProvenance:
        return cls(
            upstream_repository=UPSTREAM_REPOSITORY,
            upstream_commit=pin.commit,
            checkpoint_sha256=f"sha256:{pin.checkpoint_sha256}",
            packing_checkpoint_sha256=(
                f"sha256:{pin.packing_checkpoint_sha256}" if pin.packing_checkpoint_sha256 is not None else None
            ),
        )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "upstream_repository": self.upstream_repository,
            "upstream_commit": self.upstream_commit,
            "checkpoint_sha256": self.checkpoint_sha256,
            "packing_checkpoint_sha256": self.packing_checkpoint_sha256,
        }


@dataclass(frozen=True)
class LigandMpnnRunReceipt:
    """Portable receipt for a planned command set; it makes no run claim."""

    request_id: str
    request_hash: str
    commands: tuple[LigandMpnnCommand, ...]
    expected_sequence_count: int
    provenance: LigandMpnnProvenance

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": "thread.ligandmpnn.run_receipt",
            "schema_version": 1,
            "status": "planned_not_run",
            "model_type": "ligand_mpnn",
            "request_id": self.request_id,
            "request_hash": self.request_hash,
            "expected_sequence_count": self.expected_sequence_count,
            "provenance": self.provenance.to_dict(),
            "commands": [command.to_dict() for command in self.commands],
        }


def build_planned_receipt(
    request: LigandMpnnRequest,
    commands: tuple[LigandMpnnCommand, ...],
) -> LigandMpnnRunReceipt:
    """Normalize a validated request and its deterministic commands."""

    command_payload = [command.to_dict() for command in commands]
    canonical = json.dumps(
        {"request_id": request.request_id, "commands": command_payload},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return LigandMpnnRunReceipt(
        request_id=request.request_id,
        request_hash="sha256:" + hashlib.sha256(canonical).hexdigest(),
        commands=commands,
        expected_sequence_count=request.expected_sequence_count,
        provenance=LigandMpnnProvenance.from_pin(request.upstream),
    )
