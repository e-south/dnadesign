"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_publication.py

Publication-contract tests for the response metric metastudy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime import audit
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime.publication import (
    METASTUDY_SCHEMA_VERSION,
    artifact_inventory,
    create_staging_dir,
    publish_staging_dir,
    verify_bundle_artifacts,
)


def test_publish_replaces_bundle_only_after_staging_is_complete(tmp_path: Path) -> None:
    final = tmp_path / "latest"
    final.mkdir()
    (final / "old.txt").write_text("old", encoding="utf-8")
    stage = create_staging_dir(final, overwrite=True)
    (stage / "new.txt").write_text("new", encoding="utf-8")

    publish_staging_dir(stage, final, overwrite=True)

    assert not (final / "old.txt").exists()
    assert (final / "new.txt").read_text(encoding="utf-8") == "new"


def test_artifact_inventory_rejects_unregistered_sprawl(tmp_path: Path) -> None:
    expected = tmp_path / "expected.txt"
    expected.write_text("expected", encoding="utf-8")
    (tmp_path / "stale.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unregistered artifacts"):
        artifact_inventory(tmp_path, {"expected": expected})


def test_review_bundle_verifier_rejects_post_publication_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "report.md"
    artifact.write_text("artifact", encoding="utf-8")
    manifest = {
        "schema_version": METASTUDY_SCHEMA_VERSION,
        "artifacts": artifact_inventory(tmp_path, {"report": artifact}),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert verify_bundle_artifacts(tmp_path)["schema_version"] == manifest["schema_version"]
    artifact.write_text("drifted", encoding="utf-8")

    with pytest.raises(RuntimeError, match="wrong size|digest mismatch"):
        verify_bundle_artifacts(tmp_path)


def test_review_bundle_verifier_ignores_only_marimo_runtime_state(tmp_path: Path) -> None:
    artifact = tmp_path / "report.md"
    artifact.write_text("artifact", encoding="utf-8")
    manifest = {
        "schema_version": METASTUDY_SCHEMA_VERSION,
        "artifacts": artifact_inventory(tmp_path, {"report": artifact}),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    session = tmp_path / "__marimo__" / "session" / "review.py.json"
    session.parent.mkdir(parents=True)
    session.write_text("{}", encoding="utf-8")

    assert verify_bundle_artifacts(tmp_path)["schema_version"] == manifest["schema_version"]

    (tmp_path / "unexpected.txt").write_text("drift", encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"unexpected=\['unexpected.txt'\]"):
        verify_bundle_artifacts(tmp_path)


def test_review_bundle_verifier_rejects_duplicate_manifest_keys(tmp_path: Path) -> None:
    artifact = tmp_path / "report.md"
    artifact.write_text("artifact", encoding="utf-8")
    artifacts = json.dumps(artifact_inventory(tmp_path, {"report": artifact}))
    (tmp_path / "manifest.json").write_text(
        "{" + f'"schema_version":"{METASTUDY_SCHEMA_VERSION}",' * 2 + f'"artifacts":{artifacts}' + "}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key 'schema_version'"):
        verify_bundle_artifacts(tmp_path)


def test_review_bundle_verifier_rejects_non_finite_manifest_numbers(tmp_path: Path) -> None:
    artifact = tmp_path / "report.md"
    artifact.write_text("artifact", encoding="utf-8")
    manifest = {
        "schema_version": METASTUDY_SCHEMA_VERSION,
        "artifacts": artifact_inventory(tmp_path, {"report": artifact}),
    }
    rendered = json.dumps(manifest)[:-1] + ', "invalid_metric": NaN}'
    (tmp_path / "manifest.json").write_text(rendered, encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite JSON value 'NaN'"):
        verify_bundle_artifacts(tmp_path)


def test_review_bundle_verifier_rejects_pre_v12_schema(tmp_path: Path) -> None:
    artifact = tmp_path / "report.md"
    artifact.write_text("artifact", encoding="utf-8")
    manifest = {
        "schema_version": "stress_ethanol_cipro_growth.response_metastudy.v11",
        "artifacts": artifact_inventory(tmp_path, {"report": artifact}),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="schema is missing or unsupported"):
        verify_bundle_artifacts(tmp_path)


def test_metastudy_run_removes_staging_directory_on_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt(**_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(audit, "_materialize_metastudy", interrupt)

    with pytest.raises(KeyboardInterrupt):
        audit.run_metastudy(
            repo_root=tmp_path,
            reader_bundle_root=tmp_path,
            candidate_binding_bundle_root=tmp_path,
            out_dir=tmp_path / "latest",
            overwrite=True,
        )

    assert list(tmp_path.glob(".latest.staging-*")) == []
