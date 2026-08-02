"""Tool-owned TriJunction bundle manifest contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dnadesign.trijunction.contracts.identity import sha256_bytes

BUNDLE_SCHEMA = "dnadesign.trijunction.bundle.v1"


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    path: str
    sha256: str
    bytes: int

    @classmethod
    def from_content(cls, path: str, content: bytes) -> ArtifactIdentity:
        return cls(path=path, sha256=sha256_bytes(content), bytes=len(content))

    def to_mapping(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


def manifest_mapping(
    *,
    plan_id: str,
    request_sha256: str,
    artifacts: dict[str, ArtifactIdentity],
) -> dict[str, Any]:
    return {
        "schema": BUNDLE_SCHEMA,
        "plan_id": plan_id,
        "request_sha256": request_sha256,
        "artifacts": {key: artifacts[key].to_mapping() for key in sorted(artifacts)},
    }
