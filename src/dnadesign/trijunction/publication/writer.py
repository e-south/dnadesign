"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/publication/writer.py

Writes verified TriJunction bundles without overwriting existing directories.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dnadesign.artifacts import (
    CreateOnlyDirectoryPublication,
    PublicationError,
    preflight_create_only_directory_publication,
)
from dnadesign.trijunction.contracts.identity import canonical_json_bytes, mapping_sha256
from dnadesign.trijunction.contracts.plan import TriJunctionPlan
from dnadesign.trijunction.contracts.request import TriJunctionRequest
from dnadesign.trijunction.errors import TriJunctionBundleError
from dnadesign.trijunction.publication.manifest import ArtifactIdentity, manifest_mapping
from dnadesign.trijunction.publication.payloads import ARTIFACT_PATHS, bundle_payloads
from dnadesign.trijunction.publication.verify import _verify_published_bundle, _verify_staged_bundle


@dataclass(frozen=True, slots=True)
class PublishedTriJunctionBundle:
    path: Path
    plan_id: str
    request_sha256: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "status": "published",
            "path": str(self.path),
            "plan_id": self.plan_id,
            "request_sha256": self.request_sha256,
        }


def _preflight_bundle_destination(destination: str | Path) -> Path:
    """Reject a known-invalid publication destination without mutation."""

    try:
        return preflight_create_only_directory_publication(destination)
    except PublicationError as exc:
        raise TriJunctionBundleError(str(exc).replace("Artifact bundle", "TriJunction bundle")) from exc
    except OSError as exc:
        raise TriJunctionBundleError(f"TriJunction bundle publication failed: {exc}") from exc


def _publish_bundle(
    request: TriJunctionRequest,
    plan: TriJunctionPlan,
    destination: str | Path,
) -> PublishedTriJunctionBundle:
    """Write, verify, and atomically install a bundle in a new directory."""

    if not plan.request_sha256:
        raise TriJunctionBundleError("TriJunction plan has no request identity.")
    if plan.request_sha256 != mapping_sha256(request.to_mapping()):
        raise TriJunctionBundleError("TriJunction request does not match the supplied plan.")
    payloads = bundle_payloads(request, plan)
    try:
        publication = CreateOnlyDirectoryPublication.prepare(destination, published_root_mode=0o700)
    except PublicationError as exc:
        raise TriJunctionBundleError(str(exc).replace("Artifact bundle", "TriJunction bundle")) from exc
    except OSError as exc:
        raise TriJunctionBundleError(f"TriJunction bundle publication failed: {exc}") from exc

    try:
        with publication:
            artifacts: dict[str, ArtifactIdentity] = {}
            for key, content in payloads.items():
                relative = ARTIFACT_PATHS[key]
                path = publication.stage / relative
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                path.write_bytes(content)
                path.chmod(0o600)
                artifacts[key] = ArtifactIdentity.from_content(relative, content)
            manifest = manifest_mapping(
                plan_id=plan.plan_id,
                request_sha256=plan.request_sha256,
                artifacts=artifacts,
            )
            manifest_path = publication.stage / "manifest.json"
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            manifest_path.chmod(0o600)
            _verify_staged_bundle(
                publication.stage,
                expected_payloads=payloads,
                plan_id=plan.plan_id,
                request_sha256=plan.request_sha256,
            )
            publication.publish(required_manifest="manifest.json")
            try:
                _verify_published_bundle(publication.final)
            except BaseException:
                publication.rollback()
                raise
    except PublicationError as exc:
        raise TriJunctionBundleError(str(exc).replace("Artifact bundle", "TriJunction bundle")) from exc
    except OSError as exc:
        raise TriJunctionBundleError(f"TriJunction bundle publication failed: {exc}") from exc
    return PublishedTriJunctionBundle(
        path=publication.final,
        plan_id=plan.plan_id,
        request_sha256=plan.request_sha256,
    )
