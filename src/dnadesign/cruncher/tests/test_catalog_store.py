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
