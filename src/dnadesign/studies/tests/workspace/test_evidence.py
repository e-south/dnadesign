"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/workspace/test_evidence.py

Contract tests for study evidence indexes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from dnadesign.studies.core.workspace import load_study_evidence_index


def _digest(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _evidence_index(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    study_root = tmp_path / "study"
    artifact_path = study_root / "evidence" / "review" / "summary.svg"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    content = b"<svg xmlns='http://www.w3.org/2000/svg'/>\n"
    artifact_path.write_bytes(content)
    index_path = study_root / "evidence" / "index.yaml"
    payload: dict[str, object] = {
        "schema": "study-evidence-index/v1",
        "study_id": "demo_study",
        "artifacts": [
            {
                "artifact_id": "review_summary",
                "artifact_type": "review-figure",
                "status": "available",
                "path": "review/summary.svg",
                "media_type": "image/svg+xml",
                "content_digest": _digest(content),
                "source_revisions": {
                    "dnadesign": "git:0123456789abcdef0123456789abcdef01234567",
                    "reader": "record:response_window@sha256:abc",
                },
                "generated_by": ["uv", "run", "study", "render", "demo_study"],
            },
            {
                "artifact_id": "external_bundle",
                "artifact_type": "data-bundle",
                "status": "available",
                "uri": "s3://private-study-artifacts/demo/bundle.parquet",
                "media_type": "application/vnd.apache.parquet",
                "content_digest": "sha256:" + "a" * 64,
                "source_revisions": {"reader": "record:aggregate@sha256:def"},
                "generated_by": ["uv", "run", "study", "build", "demo_study"],
            },
            {
                "artifact_id": "future_model",
                "artifact_type": "model",
                "status": "blocked",
                "blocker": "Required measurements are not available.",
                "source_revisions": {"reader": "missing"},
            },
        ],
    }
    _write_yaml(index_path, payload)
    return study_root, index_path, payload


def test_load_evidence_index_verifies_tracked_content(tmp_path: Path) -> None:
    study_root, index_path, _ = _evidence_index(tmp_path)

    index = load_study_evidence_index(index_path, study_root=study_root, expected_study_id="demo_study")

    assert index.artifacts[0].path == (study_root / "evidence/review/summary.svg").resolve()
    assert index.artifacts[1].uri == "s3://private-study-artifacts/demo/bundle.parquet"
    assert index.artifacts[2].blocker == "Required measurements are not available."


def test_evidence_index_rejects_digest_mismatch(tmp_path: Path) -> None:
    study_root, index_path, payload = _evidence_index(tmp_path)
    payload["artifacts"][0]["content_digest"] = "sha256:" + "0" * 64
    _write_yaml(index_path, payload)

    with pytest.raises(ValueError, match="content digest mismatch"):
        load_study_evidence_index(index_path, study_root=study_root, expected_study_id="demo_study")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda item: item.update({"uri": "https://example.invalid/also.svg"}), "exactly one of path or uri"),
        (lambda item: item.pop("source_revisions"), "missing required key.*source_revisions"),
        (lambda item: item.update({"generated_by": "uv run study"}), "generated_by must be a non-empty list"),
        (lambda item: item.update({"status": "ready"}), "unsupported status"),
    ],
)
def test_available_evidence_rejects_ambiguous_or_incomplete_records(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    study_root, index_path, payload = _evidence_index(tmp_path)
    mutation(payload["artifacts"][0])
    _write_yaml(index_path, payload)

    with pytest.raises(ValueError, match=message):
        load_study_evidence_index(index_path, study_root=study_root, expected_study_id="demo_study")


def test_blocked_evidence_requires_blocker_and_forbids_location(tmp_path: Path) -> None:
    study_root, index_path, payload = _evidence_index(tmp_path)
    blocked = payload["artifacts"][2]
    blocked.pop("blocker")
    blocked["uri"] = "https://example.invalid/not-produced"
    _write_yaml(index_path, payload)

    with pytest.raises(ValueError, match="blocked artifact requires blocker and must not define a location"):
        load_study_evidence_index(index_path, study_root=study_root, expected_study_id="demo_study")


def test_evidence_index_rejects_path_escape(tmp_path: Path) -> None:
    study_root, index_path, payload = _evidence_index(tmp_path)
    payload["artifacts"][0]["path"] = "../../outside.svg"
    _write_yaml(index_path, payload)

    with pytest.raises(ValueError, match="repository-relative path"):
        load_study_evidence_index(index_path, study_root=study_root, expected_study_id="demo_study")


def test_evidence_index_rejects_identity_drift(tmp_path: Path) -> None:
    study_root, index_path, _ = _evidence_index(tmp_path)

    with pytest.raises(ValueError, match="does not match expected study_id"):
        load_study_evidence_index(index_path, study_root=study_root, expected_study_id="another_study")
