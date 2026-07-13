"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/promoter_candidate_bindings/test_artifact.py

Bundle and failure-recovery tests for promoter candidate bindings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings import (
    BINDINGS_RECORD_ID,
    SCHEMA_ID,
    SCHEMA_VERSION,
    BindingSourceArtifact,
    PromoterCandidateBindingsError,
    bundle_io,
    materialize_promoter_candidate_bindings,
    preview_promoter_candidate_bindings,
    verify_promoter_candidate_bindings,
)

from .test_resolution import aliases, densegen_candidate


def preview(*, one_row: bool = False):
    alias_rows = aliases().iloc[:1].copy() if one_row else aliases()
    return preview_promoter_candidate_bindings(
        alias_rows=alias_rows,
        candidate_records=pd.DataFrame([densegen_candidate()]),
        genbank_annotations=pd.DataFrame(),
        candidate_table_id="candidate-table",
        candidate_selection_sha256="b" * 64,
        source_artifacts=(BindingSourceArtifact("alias-authority", "inputs/aliases.parquet", "a" * 64),),
    )


def test_materialize_and_verify_round_trip(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    bundle = allowed_root / "bundle"

    result = materialize_promoter_candidate_bindings(
        preview(),
        out_dir=bundle,
        allowed_output_root=allowed_root,
    )
    verified = verify_promoter_candidate_bindings(bundle, allowed_root=allowed_root)

    assert result.bindings_parquet == bundle / "bindings.parquet"
    assert result.binding_count == 2
    assert result.candidate_count == 1
    assert verified.schema_id == SCHEMA_ID
    assert verified.schema_version == SCHEMA_VERSION
    assert verified.binding_count == 2
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    assert manifest["record"]["record_id"] == BINDINGS_RECORD_ID
    assert manifest["record"]["path"] == "bindings.parquet"
    assert "x_projection" not in manifest


def test_overwrite_restores_prior_bundle_when_publication_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed_root = tmp_path / "allowed"
    bundle = allowed_root / "bundle"
    first = materialize_promoter_candidate_bindings(preview(), out_dir=bundle, allowed_output_root=allowed_root)
    original = (first.manifest_json.read_bytes(), first.bindings_parquet.read_bytes())
    real_replace = bundle_io.os.replace

    def fail_publish(source: str | Path, target: str | Path) -> None:
        if Path(target) == bundle and Path(source).name.startswith(".bundle.staging-"):
            raise OSError("injected publication failure")
        real_replace(source, target)

    monkeypatch.setattr(bundle_io.os, "replace", fail_publish)
    with pytest.raises(PromoterCandidateBindingsError, match="restored prior bundle"):
        materialize_promoter_candidate_bindings(
            preview(),
            out_dir=bundle,
            allowed_output_root=allowed_root,
            overwrite=True,
        )

    assert ((bundle / "manifest.json").read_bytes(), (bundle / "bindings.parquet").read_bytes()) == original


def test_double_publication_failure_keeps_durable_recoverable_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed_root = tmp_path / "allowed"
    bundle = allowed_root / "bundle"
    first = materialize_promoter_candidate_bindings(preview(), out_dir=bundle, allowed_output_root=allowed_root)
    original = (first.manifest_json.read_bytes(), first.bindings_parquet.read_bytes())
    real_replace = bundle_io.os.replace

    def fail_publish_and_rollback(source: str | Path, target: str | Path) -> None:
        source_path = Path(source)
        if Path(target) == bundle and (
            source_path.name.startswith(".bundle.staging-") or source_path.name.startswith(".bundle.backup-")
        ):
            raise OSError("injected publish or rollback failure")
        real_replace(source, target)

    monkeypatch.setattr(bundle_io.os, "replace", fail_publish_and_rollback)
    with pytest.raises(PromoterCandidateBindingsError, match="remains recoverable"):
        materialize_promoter_candidate_bindings(
            preview(),
            out_dir=bundle,
            allowed_output_root=allowed_root,
            overwrite=True,
        )

    backups = list(allowed_root.glob(".bundle.backup-*"))
    assert len(backups) == 1
    assert ((backups[0] / "manifest.json").read_bytes(), (backups[0] / "bindings.parquet").read_bytes()) == original


def test_materialization_rejects_output_escape(tmp_path: Path) -> None:
    with pytest.raises(PromoterCandidateBindingsError, match="outside allowed output root"):
        materialize_promoter_candidate_bindings(
            preview(),
            out_dir=tmp_path / "escape",
            allowed_output_root=tmp_path / "allowed",
        )


@pytest.mark.parametrize(
    "source_path",
    ["../aliases.parquet", "/private/aliases.parquet", r"C:\\aliases.parquet", "~/aliases.parquet"],
)
def test_preview_rejects_nonportable_source_paths(source_path: str) -> None:
    with pytest.raises(PromoterCandidateBindingsError, match="relative POSIX path"):
        preview_promoter_candidate_bindings(
            alias_rows=aliases(),
            candidate_records=pd.DataFrame([densegen_candidate()]),
            genbank_annotations=pd.DataFrame(),
            candidate_table_id="candidate-table",
            candidate_selection_sha256="b" * 64,
            source_artifacts=(BindingSourceArtifact("source", source_path, "a" * 64),),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("created_at", "2026-07-13T12:00:00", "UTC offset"),
        ("row_count", True, "positive integer"),
    ],
)
def test_verify_rejects_ambiguous_manifest_scalars(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    allowed_root = tmp_path / "allowed"
    bundle = allowed_root / "bundle"
    result = materialize_promoter_candidate_bindings(
        preview(one_row=True),
        out_dir=bundle,
        allowed_output_root=allowed_root,
    )
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    if field == "created_at":
        manifest[field] = value
    else:
        manifest["record"][field] = value
    result.manifest_json.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PromoterCandidateBindingsError, match=message):
        verify_promoter_candidate_bindings(bundle, allowed_root=allowed_root)


@pytest.mark.parametrize("tamper", ["schema", "digest", "path"])
def test_verify_rejects_contract_drift(tmp_path: Path, tamper: str) -> None:
    allowed_root = tmp_path / "allowed"
    bundle = allowed_root / "bundle"
    result = materialize_promoter_candidate_bindings(preview(), out_dir=bundle, allowed_output_root=allowed_root)
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    if tamper == "schema":
        manifest["schema_version"] = "999"
        message = "identity"
    elif tamper == "path":
        manifest["record"]["path"] = "../escape.parquet"
        message = "record path"
    else:
        result.bindings_parquet.write_bytes(result.bindings_parquet.read_bytes() + b"tamper")
        message = "digest mismatch"
    result.manifest_json.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PromoterCandidateBindingsError, match=message):
        verify_promoter_candidate_bindings(bundle, allowed_root=allowed_root)
