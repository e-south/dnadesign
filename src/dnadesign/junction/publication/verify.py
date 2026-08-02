"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/publication/verify.py

Offline semantic verification of one published junction bundle.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dnadesign.junction.contracts.identity import canonical_json_bytes, sha256_bytes
from dnadesign.junction.contracts.publication.limits import ARTIFACT_BYTE_LIMITS, MANIFEST_BYTE_LIMIT
from dnadesign.junction.contracts.request import parse_request
from dnadesign.junction.design.planner import design_junction
from dnadesign.junction.errors import JunctionBundleError, JunctionError
from dnadesign.junction.publication.manifest import BUNDLE_SCHEMA, ArtifactIdentity
from dnadesign.junction.publication.payloads import ARTIFACT_PATHS, render_artifact_bytes
from dnadesign.junction.publication.snapshot import open_bundle_snapshot

_EXPECTED_FILES = frozenset({"manifest.json", *ARTIFACT_PATHS.values()})


@dataclass(frozen=True, slots=True)
class BundleVerification:
    status: str
    bundle: Path
    plan_id: str
    request_sha256: str
    artifact_count: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "status": self.status,
            "bundle": str(self.bundle),
            "plan_id": self.plan_id,
            "request_sha256": self.request_sha256,
            "artifact_count": self.artifact_count,
        }


@dataclass(frozen=True, slots=True)
class _StagedBundleExpectation:
    artifacts: dict[str, ArtifactIdentity]
    plan_id: str
    request_sha256: str


def _load_json(content: bytes, *, path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise JunctionBundleError(f"{context} is not valid UTF-8 JSON: {path}") from None
    if not isinstance(value, dict):
        raise JunctionBundleError(f"{context} must contain one JSON object: {path}")
    return value


def _artifact_relative_path(relative: object, *, key: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise JunctionBundleError(f"Bundle artifact '{key}' path must be a non-empty relative string.")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != relative:
        raise JunctionBundleError(f"Bundle artifact '{key}' path is not portable: {relative!r}")
    return candidate


def _verify_bundle(
    bundle: str | Path,
    *,
    reject_undeclared_entries: bool,
    staged_expectation: _StagedBundleExpectation | None = None,
) -> BundleVerification:
    """Verify one stable bundle snapshot against replayed or staged evidence."""

    root = Path(bundle).expanduser().absolute()
    manifest_path = root / "manifest.json"
    with open_bundle_snapshot(
        root,
        expected_files=_EXPECTED_FILES,
        reject_undeclared_entries=reject_undeclared_entries,
    ) as snapshot:
        manifest_read = snapshot.read_file(
            Path("manifest.json"),
            limit=MANIFEST_BYTE_LIMIT,
            context="junction manifest",
            retain_content=True,
        )
        assert manifest_read.content is not None
        manifest = _load_json(
            manifest_read.content,
            path=manifest_path,
            context="junction manifest",
        )
        required_fields = {"schema", "plan_id", "request_sha256", "artifacts"}
        if set(manifest) != required_fields:
            raise JunctionBundleError("junction manifest fields do not match the v1 contract.")
        if manifest["schema"] != BUNDLE_SCHEMA:
            raise JunctionBundleError(f"junction manifest schema must be {BUNDLE_SCHEMA!r}.")
        artifacts = manifest["artifacts"]
        if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_PATHS):
            raise JunctionBundleError("junction manifest must declare the complete v1 artifact set.")

        identities: dict[str, dict[str, object]] = {}
        request_content: bytes | None = None
        for key, expected_relative in ARTIFACT_PATHS.items():
            identity = artifacts[key]
            if not isinstance(identity, dict) or set(identity) != {"path", "sha256", "bytes"}:
                raise JunctionBundleError(f"Bundle artifact '{key}' identity is malformed.")
            declared_bytes = identity["bytes"]
            if isinstance(declared_bytes, bool) or not isinstance(declared_bytes, int) or declared_bytes < 0:
                raise JunctionBundleError(f"Bundle artifact '{key}' byte length must be a nonnegative integer.")
            if declared_bytes > ARTIFACT_BYTE_LIMITS[key]:
                raise JunctionBundleError(
                    f"Bundle artifact '{key}' exceeds the {ARTIFACT_BYTE_LIMITS[key]}-byte verification limit."
                )
            if identity["path"] != expected_relative:
                raise JunctionBundleError(f"Bundle artifact '{key}' must use path {expected_relative!r}.")
            artifact_read = snapshot.read_file(
                _artifact_relative_path(identity["path"], key=key),
                limit=ARTIFACT_BYTE_LIMITS[key],
                context=f"Bundle artifact '{key}'",
                retain_content=key == "request" and staged_expectation is None,
            )
            if declared_bytes != artifact_read.observed_bytes or identity["sha256"] != artifact_read.sha256:
                raise JunctionBundleError(f"Bundle artifact '{key}' content identity does not match manifest.")
            identities[key] = identity
            if key == "request":
                request_content = artifact_read.content

        if staged_expectation is None:
            assert request_content is not None
            try:
                request_payload = json.loads(request_content.decode("utf-8"))
            except (UnicodeDecodeError, ValueError, RecursionError):
                raise JunctionBundleError("Bundle request cannot reproduce a valid junction plan.") from None
            finally:
                del request_content
            try:
                request = parse_request(request_payload)
                del request_payload
                recomputed_plan = design_junction(request)
            except JunctionError as exc:
                raise JunctionBundleError("Bundle request cannot reproduce a valid junction plan.") from exc
            plan_id = recomputed_plan.plan_id
            request_sha256 = recomputed_plan.request_sha256
        else:
            plan_id = staged_expectation.plan_id
            request_sha256 = staged_expectation.request_sha256
        expectation_label = "reproduced plan" if staged_expectation is None else "staged canonical payload"
        if staged_expectation is None:
            for key in ARTIFACT_PATHS:
                expected = render_artifact_bytes(key, request, recomputed_plan)
                try:
                    identity = identities[key]
                    if identity["bytes"] != len(expected) or identity["sha256"] != sha256_bytes(expected):
                        raise JunctionBundleError(f"Bundle artifact '{key}' does not match the {expectation_label}.")
                finally:
                    del expected
        else:
            if set(staged_expectation.artifacts) != set(ARTIFACT_PATHS):
                raise JunctionBundleError("Expected junction artifacts do not match the v1 artifact set.")
            for key, expected in staged_expectation.artifacts.items():
                if identities[key] != expected.to_mapping():
                    raise JunctionBundleError(f"Bundle artifact '{key}' does not match the {expectation_label}.")
        if manifest["plan_id"] != plan_id:
            raise JunctionBundleError("junction manifest plan_id does not match the verified plan identity.")
        if manifest["request_sha256"] != request_sha256:
            raise JunctionBundleError("junction manifest request_sha256 does not match the verified request identity.")
        if canonical_json_bytes(manifest) != manifest_read.content:
            raise JunctionBundleError("junction manifest is not canonically serialized.")
        snapshot.assert_stable()
        return BundleVerification(
            status="verified",
            bundle=root,
            plan_id=plan_id,
            request_sha256=request_sha256,
            artifact_count=len(identities),
        )


def _verify_published_bundle(bundle: str | Path) -> BundleVerification:
    """Strictly verify one published bundle, including its complete inventory."""

    try:
        return _verify_bundle(bundle, reject_undeclared_entries=True)
    except JunctionBundleError:
        raise
    except OSError as exc:
        raise JunctionBundleError(f"Unable to read junction bundle: {exc}") from exc


def _verify_staged_bundle(
    bundle: str | Path,
    *,
    expected_artifacts: dict[str, ArtifactIdentity],
    plan_id: str,
    request_sha256: str,
) -> BundleVerification:
    """Verify staged bytes against the already-derived canonical payloads."""

    try:
        return _verify_bundle(
            bundle,
            reject_undeclared_entries=False,
            staged_expectation=_StagedBundleExpectation(
                artifacts=expected_artifacts,
                plan_id=plan_id,
                request_sha256=request_sha256,
            ),
        )
    except JunctionBundleError:
        raise
    except OSError as exc:
        raise JunctionBundleError(f"Unable to read staged junction bundle: {exc}") from exc
