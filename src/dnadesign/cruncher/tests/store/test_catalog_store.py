"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/store/test_catalog_store.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging

from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex
from dnadesign.cruncher.store.catalog_store import CatalogMotifStore
from dnadesign.cruncher.store.motif_store import MotifRef


def test_pwm_from_sites_warns_with_override(tmp_path, caplog):
    sites_dir = tmp_path / "normalized" / "sites" / "regulondb"
    sites_dir.mkdir(parents=True)
    site_path = sites_dir / "RDB0002.jsonl"
    site_path.write_text(json.dumps({"sequence": "ACGT"}) + "\n")

    catalog = CatalogIndex(
        entries={
            "regulondb:RDB0002": CatalogEntry(
                source="regulondb",
                motif_id="RDB0002",
                tf_name="lexA",
                kind="PFM",
                has_sites=True,
                site_count=1,
                site_total=1,
            )
        }
    )
    catalog.save(tmp_path)

    caplog.set_level(logging.WARNING)
    store = CatalogMotifStore(tmp_path, pwm_source="sites", min_sites_for_pwm=2, allow_low_sites=True)
    pwm = store.get_pwm(MotifRef(source="regulondb", motif_id="RDB0002"))
    assert pwm.length == 4
    assert any("Only 1 binding-site sequences available" in rec.message for rec in caplog.records)


def test_pwm_from_sites_strict_min_sites(tmp_path):
    sites_dir = tmp_path / "normalized" / "sites" / "regulondb"
    sites_dir.mkdir(parents=True)
    site_path = sites_dir / "RDB0003.jsonl"
    site_path.write_text(json.dumps({"sequence": "ACGT"}) + "\n")

    catalog = CatalogIndex(
        entries={
            "regulondb:RDB0003": CatalogEntry(
                source="regulondb",
                motif_id="RDB0003",
                tf_name="lexA",
                kind="PFM",
                has_sites=True,
                site_count=1,
                site_total=1,
            )
        }
    )
    catalog.save(tmp_path)

    store = CatalogMotifStore(tmp_path, pwm_source="sites", min_sites_for_pwm=2, allow_low_sites=False)
    try:
        store.get_pwm(MotifRef(source="regulondb", motif_id="RDB0003"))
    except ValueError as exc:
        assert "Only 1 binding-site sequences available" in str(exc)
    else:
        raise AssertionError("Expected ValueError for insufficient binding sites")


def test_pwm_from_ht_sites_requires_window_length_for_variable_lengths(tmp_path):
    sites_dir = tmp_path / "normalized" / "sites" / "regulondb"
    sites_dir.mkdir(parents=True)
    site_path = sites_dir / "DATASET1.jsonl"
    site_path.write_text(
        json.dumps({"sequence": "ACGTACGTAA"}) + "\n" + json.dumps({"sequence": "ACGTACGTACGT"}) + "\n"
    )
    catalog = CatalogIndex(
        entries={
            "regulondb:DATASET1": CatalogEntry(
                source="regulondb",
                motif_id="DATASET1",
                tf_name="lexA",
                kind="PFM",
                has_sites=True,
                site_count=2,
                site_total=2,
                site_kind="ht_peak",
            )
        }
    )
    catalog.save(tmp_path)
    store = CatalogMotifStore(tmp_path, pwm_source="sites", min_sites_for_pwm=2, allow_low_sites=False)
    try:
        store.get_pwm(MotifRef(source="regulondb", motif_id="DATASET1"))
    except ValueError as exc:
        assert "site_window_lengths" in str(exc)
    else:
        raise AssertionError("Expected ValueError for variable-length HT sites without window length")


def test_pwm_from_ht_sites_applies_window_length(tmp_path):
    sites_dir = tmp_path / "normalized" / "sites" / "regulondb"
    sites_dir.mkdir(parents=True)
    site_path = sites_dir / "DATASET2.jsonl"
    site_path.write_text(
        json.dumps({"sequence": "AACCGGTTAACC"}) + "\n" + json.dumps({"sequence": "TTGGCCAATTGG"}) + "\n"
    )
    catalog = CatalogIndex(
        entries={
            "regulondb:DATASET2": CatalogEntry(
                source="regulondb",
                motif_id="DATASET2",
                tf_name="lexA",
                kind="PFM",
                has_sites=True,
                site_count=2,
                site_total=2,
                site_kind="ht_peak",
                dataset_id="DATASET2",
            )
        }
    )
    catalog.save(tmp_path)
    store = CatalogMotifStore(
        tmp_path,
        pwm_source="sites",
        min_sites_for_pwm=2,
        allow_low_sites=False,
        site_window_lengths={"lexA": 8},
        site_window_center="midpoint",
    )
    pwm = store.get_pwm(MotifRef(source="regulondb", motif_id="DATASET2"))
    assert pwm.length == 8


def test_pwm_from_sites_requires_sequences(tmp_path):
    sites_dir = tmp_path / "normalized" / "sites" / "regulondb"
    sites_dir.mkdir(parents=True)
    site_path = sites_dir / "RDB0004.jsonl"
    site_path.write_text(json.dumps({"site_id": "s1"}) + "\n")

    catalog = CatalogIndex(
        entries={
            "regulondb:RDB0004": CatalogEntry(
                source="regulondb",
                motif_id="RDB0004",
                tf_name="lexA",
                kind="PFM",
                has_sites=True,
                site_count=0,
                site_total=1,
            )
        }
    )
    catalog.save(tmp_path)

    store = CatalogMotifStore(tmp_path, pwm_source="sites", min_sites_for_pwm=1, allow_low_sites=False)
    try:
        store.get_pwm(MotifRef(source="regulondb", motif_id="RDB0004"))
    except ValueError as exc:
        assert "missing a sequence" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing site sequences")


def test_matrix_pwm_window_can_be_disabled_for_logo_views(tmp_path):
    motifs_dir = tmp_path / "normalized" / "motifs" / "demo_source"
    motifs_dir.mkdir(parents=True)
    motif_path = motifs_dir / "lexA_demo.json"
    motif_path.write_text(
        json.dumps(
            {
                "descriptor": {"tf_name": "lexA"},
                "matrix": [
                    [0.97, 0.01, 0.01, 0.01],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.97, 0.01, 0.01, 0.01],
                ],
            }
        )
    )
    catalog = CatalogIndex(
        entries={
            "demo_source:lexA_demo": CatalogEntry(
                source="demo_source",
                motif_id="lexA_demo",
                tf_name="lexA",
                kind="PFM",
                has_matrix=True,
                matrix_length=4,
            )
        }
    )
    catalog.save(tmp_path)

    windowed_store = CatalogMotifStore(
        tmp_path,
        pwm_source="matrix",
        pwm_window_lengths={"lexA": 2},
        pwm_window_strategy="max_info",
    )
    windowed = windowed_store.get_pwm(MotifRef(source="demo_source", motif_id="lexA_demo"))
    assert windowed.length == 2

    full_store = CatalogMotifStore(
        tmp_path,
        pwm_source="matrix",
        pwm_window_lengths={"lexA": 2},
        pwm_window_strategy="max_info",
        apply_pwm_window=False,
    )
    full = full_store.get_pwm(MotifRef(source="demo_source", motif_id="lexA_demo"))
    assert full.length == 4
