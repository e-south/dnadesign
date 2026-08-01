"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/source_ingest/test_msd_selected_variant_lineage.py

Tests for the selected materialized-variant lineage projection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.retron_hairpin_design.source_ingest.selected_lineage import (
    MaterializedVariantLineageError,
    load_materialized_variant_lineage,
)

from ..support.paths import repo_root_from


def test_checked_in_lineage_exactly_links_selected_variants_195_through_204() -> None:
    repo_root = repo_root_from(__file__)
    index_path = (
        repo_root
        / "docs/studies/retron_hairpin_design/workbench/provenance/materialized_variant_lineage"
        / "pes_retron_195_204.yaml"
    )

    lineage = load_materialized_variant_lineage(index_path, repo_root=repo_root)

    expected = {f"retron{number}" for number in range(195, 205)}
    assert lineage.owner_study_id == "retron_hairpin_design"
    assert lineage.expected_selected_variant_count == 10
    assert set(lineage.selected_variant_ids) == expected
    assert {entry.variant_id for entry in lineage.entries} == expected


def test_lineage_rejects_missing_selected_variant_ids(tmp_path: Path) -> None:
    repo_root, lineage_path = _single_entry_fixture(tmp_path, source_root=repo_root_from(__file__))
    payload = yaml.safe_load(lineage_path.read_text(encoding="utf-8"))
    payload.pop("selected_variant_ids")
    lineage_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(MaterializedVariantLineageError, match="selected_variant_ids"):
        load_materialized_variant_lineage(lineage_path, repo_root=repo_root)


def test_lineage_rejects_selected_ids_that_do_not_match_entry_ids(tmp_path: Path) -> None:
    repo_root, lineage_path = _single_entry_fixture(tmp_path, source_root=repo_root_from(__file__))
    payload = yaml.safe_load(lineage_path.read_text(encoding="utf-8"))
    payload["selected_variant_ids"] = ["retron999"]
    lineage_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(MaterializedVariantLineageError, match="selected_variant_ids must exactly match"):
        load_materialized_variant_lineage(lineage_path, repo_root=repo_root)


def test_lineage_rejects_selected_variant_absent_from_source_manifest(tmp_path: Path) -> None:
    repo_root, lineage_path = _single_entry_fixture(tmp_path, source_root=repo_root_from(__file__))
    manifest_path = _source_manifest_path(repo_root, lineage_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["records"] = [row for row in manifest["records"] if row["variant_id"] != "retron195"]
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    with pytest.raises(MaterializedVariantLineageError, match="missing selected variant IDs"):
        load_materialized_variant_lineage(lineage_path, repo_root=repo_root)


def test_lineage_rejects_duplicate_source_manifest_record_ids(tmp_path: Path) -> None:
    repo_root, lineage_path = _single_entry_fixture(tmp_path, source_root=repo_root_from(__file__))
    manifest_path = _source_manifest_path(repo_root, lineage_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["records"].append(dict(manifest["records"][0]))
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    with pytest.raises(MaterializedVariantLineageError, match="record variant IDs must be unique"):
        load_materialized_variant_lineage(lineage_path, repo_root=repo_root)


def test_lineage_accepts_unrelated_extra_source_manifest_rows(tmp_path: Path) -> None:
    repo_root, lineage_path = _single_entry_fixture(tmp_path, source_root=repo_root_from(__file__))

    lineage = load_materialized_variant_lineage(lineage_path, repo_root=repo_root)

    assert lineage.selected_variant_ids == ("retron195",)


def test_lineage_rejects_repo_escape_before_reading_source_genbank(tmp_path: Path) -> None:
    source_root = repo_root_from(__file__)
    repo_root, index_path = _single_entry_fixture(tmp_path, source_root=source_root)
    outside = tmp_path / "outside.gb"
    outside.write_bytes(b"not a valid GenBank")
    payload = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    payload["entries"][0]["source_genbank_ref"] = "../outside.gb"
    index_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(MaterializedVariantLineageError, match="path escapes the repository"):
        load_materialized_variant_lineage(index_path, repo_root=repo_root)


def test_lineage_rejects_missing_selected_source_file(tmp_path: Path) -> None:
    repo_root, lineage_path = _single_entry_fixture(tmp_path, source_root=repo_root_from(__file__))
    payload = yaml.safe_load(lineage_path.read_text(encoding="utf-8"))
    (repo_root / payload["entries"][0]["source_genbank_ref"]).unlink()

    with pytest.raises(MaterializedVariantLineageError, match="source_genbank_ref path does not exist"):
        load_materialized_variant_lineage(lineage_path, repo_root=repo_root)


def test_lineage_rejects_source_genbank_digest_drift(tmp_path: Path) -> None:
    source_root = repo_root_from(__file__)
    repo_root, index_path = _single_entry_fixture(tmp_path, source_root=source_root)
    payload = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    payload["entries"][0]["source_genbank_sha256"] = "0" * 64
    index_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(MaterializedVariantLineageError, match="source GenBank digest drift"):
        load_materialized_variant_lineage(index_path, repo_root=repo_root)


def test_lineage_rejects_primitive_drift_from_design_set(tmp_path: Path) -> None:
    source_root = repo_root_from(__file__)
    repo_root, index_path = _single_entry_fixture(tmp_path, source_root=source_root)
    payload = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    payload["entries"][0]["primitives"]["cap_id"] = "C999"
    index_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(MaterializedVariantLineageError, match="design set cap_id drift"):
        load_materialized_variant_lineage(index_path, repo_root=repo_root)


def _single_entry_fixture(tmp_path: Path, *, source_root: Path) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    source_index = (
        source_root
        / "docs/studies/retron_hairpin_design/workbench/provenance/materialized_variant_lineage"
        / "pes_retron_195_204.yaml"
    )
    payload = yaml.safe_load(source_index.read_text(encoding="utf-8"))
    payload["entries"] = [payload["entries"][0]]
    payload["selected_variant_ids"] = [payload["entries"][0]["variant_id"]]
    payload["expected_selected_variant_count"] = 1
    entry = payload["entries"][0]
    refs = {
        payload["source_bundle_manifest_ref"],
        entry["design_set_ref"],
        entry["compiler_spec_ref"],
        entry["deliverable_plan_ref"],
        entry["source_genbank_ref"],
        entry["msd_region_record_ref"],
    }
    for ref in refs:
        destination = repo_root / ref
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / ref, destination)
    index_path = repo_root / source_index.relative_to(source_root)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return repo_root, index_path


def _source_manifest_path(repo_root: Path, lineage_path: Path) -> Path:
    payload = yaml.safe_load(lineage_path.read_text(encoding="utf-8"))
    return repo_root / payload["source_bundle_manifest_ref"]
