"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/publication/verify.py

Offline semantic verification of one published TriJunction bundle.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dnadesign.trijunction.contracts.identity import canonical_json_bytes, sha256_bytes
from dnadesign.trijunction.contracts.publication.limits import ARTIFACT_BYTE_LIMITS, MANIFEST_BYTE_LIMIT
from dnadesign.trijunction.contracts.request import parse_request
from dnadesign.trijunction.design.planner import design_trijunction
from dnadesign.trijunction.errors import TriJunctionBundleError, TriJunctionError
from dnadesign.trijunction.publication.manifest import BUNDLE_SCHEMA
from dnadesign.trijunction.publication.payloads import ARTIFACT_PATHS, bundle_payloads
from dnadesign.trijunction.publication.snapshot import open_bundle_snapshot

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
    payloads: Mapping[str, bytes]
    plan_id: str
    request_sha256: str


def _load_json(content: bytes, *, path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise TriJunctionBundleError(f"{context} is not valid UTF-8 JSON: {path}") from None
    if not isinstance(value, dict):
        raise TriJunctionBundleError(f"{context} must contain one JSON object: {path}")
    return value


def _artifact_relative_path(relative: object, *, key: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise TriJunctionBundleError(f"Bundle artifact '{key}' path must be a non-empty relative string.")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != relative:
        raise TriJunctionBundleError(f"Bundle artifact '{key}' path is not portable: {relative!r}")
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
            context="TriJunction manifest",
            retain_content=True,
        )
        assert manifest_read.content is not None
        manifest = _load_json(
            manifest_read.content,
            path=manifest_path,
            context="TriJunction manifest",
        )
        required_fields = {"schema", "plan_id", "request_sha256", "artifacts"}
        if set(manifest) != required_fields:
            raise TriJunctionBundleError("TriJunction manifest fields do not match the v1 contract.")
        if manifest["schema"] != BUNDLE_SCHEMA:
            raise TriJunctionBundleError(f"TriJunction manifest schema must be {BUNDLE_SCHEMA!r}.")
        artifacts = manifest["artifacts"]
        if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_PATHS):
            raise TriJunctionBundleError("TriJunction manifest must declare the complete v1 artifact set.")

        identities: dict[str, dict[str, object]] = {}
        request_content: bytes | None = None
        for key, expected_relative in ARTIFACT_PATHS.items():
            identity = artifacts[key]
            if not isinstance(identity, dict) or set(identity) != {"path", "sha256", "bytes"}:
                raise TriJunctionBundleError(f"Bundle artifact '{key}' identity is malformed.")
            declared_bytes = identity["bytes"]
            if isinstance(declared_bytes, bool) or not isinstance(declared_bytes, int) or declared_bytes < 0:
                raise TriJunctionBundleError(f"Bundle artifact '{key}' byte length must be a nonnegative integer.")
            if declared_bytes > ARTIFACT_BYTE_LIMITS[key]:
                raise TriJunctionBundleError(
                    f"Bundle artifact '{key}' exceeds the {ARTIFACT_BYTE_LIMITS[key]}-byte verification limit."
                )
            if identity["path"] != expected_relative:
                raise TriJunctionBundleError(f"Bundle artifact '{key}' must use path {expected_relative!r}.")
            artifact_read = snapshot.read_file(
                _artifact_relative_path(identity["path"], key=key),
                limit=ARTIFACT_BYTE_LIMITS[key],
                context=f"Bundle artifact '{key}'",
                retain_content=key == "request" and staged_expectation is None,
            )
            if declared_bytes != artifact_read.observed_bytes or identity["sha256"] != artifact_read.sha256:
                raise TriJunctionBundleError(f"Bundle artifact '{key}' content identity does not match manifest.")
            identities[key] = identity
            if key == "request":
                request_content = artifact_read.content

        if staged_expectation is None:
            assert request_content is not None
            try:
                request_payload = json.loads(request_content.decode("utf-8"))
            except (UnicodeDecodeError, ValueError, RecursionError):
                raise TriJunctionBundleError("Bundle request cannot reproduce a valid TriJunction plan.") from None
            try:
                request = parse_request(request_payload)
                recomputed_plan = design_trijunction(request)
            except TriJunctionError as exc:
                raise TriJunctionBundleError("Bundle request cannot reproduce a valid TriJunction plan.") from exc
            expected_payloads = bundle_payloads(request, recomputed_plan)
            plan_id = recomputed_plan.plan_id
            request_sha256 = recomputed_plan.request_sha256
        else:
            expected_payloads = staged_expectation.payloads
            plan_id = staged_expectation.plan_id
            request_sha256 = staged_expectation.request_sha256
        expectation_label = "reproduced plan" if staged_expectation is None else "staged canonical payload"
        if set(expected_payloads) != set(ARTIFACT_PATHS):
            raise TriJunctionBundleError("Expected TriJunction payloads do not match the v1 artifact set.")
        for key, expected in expected_payloads.items():
            identity = identities[key]
            if identity["bytes"] != len(expected) or identity["sha256"] != sha256_bytes(expected):
                raise TriJunctionBundleError(f"Bundle artifact '{key}' does not match the {expectation_label}.")
        if manifest["plan_id"] != plan_id:
            raise TriJunctionBundleError("TriJunction manifest plan_id does not match the verified plan identity.")
        if manifest["request_sha256"] != request_sha256:
            raise TriJunctionBundleError(
                "TriJunction manifest request_sha256 does not match the verified request identity."
            )
        if canonical_json_bytes(manifest) != manifest_read.content:
            raise TriJunctionBundleError("TriJunction manifest is not canonically serialized.")
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
    except TriJunctionBundleError:
        raise
    except OSError as exc:
        raise TriJunctionBundleError(f"Unable to read TriJunction bundle: {exc}") from exc


def _verify_staged_bundle(
    bundle: str | Path,
    *,
    expected_payloads: Mapping[str, bytes],
    plan_id: str,
    request_sha256: str,
) -> BundleVerification:
    """Verify staged bytes against the already-derived canonical payloads."""

    try:
        return _verify_bundle(
            bundle,
            reject_undeclared_entries=False,
            staged_expectation=_StagedBundleExpectation(
                payloads=expected_payloads,
                plan_id=plan_id,
                request_sha256=request_sha256,
            ),
        )
    except TriJunctionBundleError:
        raise
    except OSError as exc:
        raise TriJunctionBundleError(f"Unable to read staged TriJunction bundle: {exc}") from exc
