"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_target_service.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from dnadesign.cruncher.app.target_service import (
    has_blocking_target_errors,
    target_candidates,
    target_candidates_fuzzy,
    target_statuses,
)
from dnadesign.cruncher.config.schema_v2 import CruncherConfig, IngestConfig, MotifStoreConfig, ParseConfig, PlotConfig
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex


def _write_lock(tmp_path: Path, payload: dict) -> None:
    lock_path = tmp_path / ".cruncher" / "locks" / "config.lock.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(payload)


def _config(
    tmp_path: Path,
    *,
    pwm_source: str = "sites",
    min_sites: int = 2,
    allow_low_sites: bool = False,
    combine_sites: bool = False,
    regulator_sets: list[list[str]] | None = None,
) -> CruncherConfig:
    regulator_sets = regulator_sets or [["lexA", "oxyR"]]
    return CruncherConfig(
        out_dir=Path("out"),
        regulator_sets=regulator_sets,
        motif_store=MotifStoreConfig(
            catalog_root=(tmp_path / ".cruncher"),
            pwm_source=pwm_source,
            min_sites_for_pwm=min_sites,
            allow_low_sites=allow_low_sites,
            combine_sites=combine_sites,
        ),
        ingest=IngestConfig(),
        parse=ParseConfig(plot=PlotConfig(logo=False, bits_mode="information", dpi=100)),
        sample=None,
        analysis=None,
    )


def test_target_status_warning_and_ready(tmp_path: Path) -> None:
    cfg = _config(tmp_path, pwm_source="sites", min_sites=2, allow_low_sites=True)
    lock_payload = {
        "pwm_source": "sites",
        "resolved": {
            "lexA": {"source": "regulondb", "motif_id": "RBM1", "sha256": "aaa"},
            "oxyR": {"source": "regulondb", "motif_id": "RBM2", "sha256": "bbb"},
        },
    }
    _write_lock(tmp_path, json.dumps(lock_payload))
    catalog = CatalogIndex(
        entries={
            "regulondb:RBM1": CatalogEntry(
                source="regulondb",
                motif_id="RBM1",
                tf_name="lexA",
                kind="PFM",
                has_sites=True,
                site_count=1,
                site_total=1,
            ),
            "regulondb:RBM2": CatalogEntry(
                source="regulondb",
                motif_id="RBM2",
                tf_name="oxyR",
                kind="PFM",
                has_sites=True,
                site_count=3,
                site_total=3,
            ),
        }
    )
    catalog.save(tmp_path / ".cruncher")
    sites_dir = tmp_path / ".cruncher" / "normalized" / "sites" / "regulondb"
    sites_dir.mkdir(parents=True, exist_ok=True)
    (sites_dir / "RBM1.jsonl").write_text(json.dumps({"sequence": "ACGT"}) + "\n")
    (sites_dir / "RBM2.jsonl").write_text(json.dumps({"sequence": "ACGT"}) + "\n")
    statuses = target_statuses(cfg=cfg, config_path=tmp_path / "config.yaml")
    status_map = {s.tf_name: s.status for s in statuses}
    assert status_map["lexA"] == "warning"
    assert status_map["oxyR"] == "ready"
    assert has_blocking_target_errors(statuses) is False


def test_target_status_missing_sequences_blocks(tmp_path: Path) -> None:
    cfg = _config(tmp_path, pwm_source="sites", min_sites=1, allow_low_sites=True, regulator_sets=[["lexA"]])
    lock_payload = {
        "pwm_source": "sites",
        "resolved": {"lexA": {"source": "regulondb", "motif_id": "RBM1", "sha256": "aaa"}},
    }
    _write_lock(tmp_path, json.dumps(lock_payload))
    catalog = CatalogIndex(
        entries={
            "regulondb:RBM1": CatalogEntry(
                source="regulondb",
                motif_id="RBM1",
                tf_name="lexA",
                kind="PFM",
                has_sites=True,
                site_count=1,
                site_total=2,
            )
        }
    )
    catalog.save(tmp_path / ".cruncher")
    sites_dir = tmp_path / ".cruncher" / "normalized" / "sites" / "regulondb"
    sites_dir.mkdir(parents=True, exist_ok=True)
    (sites_dir / "RBM1.jsonl").write_text(json.dumps({"sequence": "ACGT"}) + "\n")
    statuses = target_statuses(cfg=cfg, config_path=tmp_path / "config.yaml")
    assert statuses[0].status == "missing-sequences"
    assert has_blocking_target_errors(statuses) is True


def test_target_status_missing_catalog(tmp_path: Path) -> None:
    cfg = _config(tmp_path, pwm_source="matrix", min_sites=2, regulator_sets=[["lexA"]])
    lock_payload = {
        "pwm_source": "matrix",
        "resolved": {"lexA": {"source": "regulondb", "motif_id": "RBM1", "sha256": "aaa"}},
    }
    _write_lock(tmp_path, json.dumps(lock_payload))
    statuses = target_statuses(cfg=cfg, config_path=tmp_path / "config.yaml")
    assert statuses[0].status == "missing-catalog"
    assert has_blocking_target_errors(statuses) is True


def test_target_status_insufficient_sites_blocks(tmp_path: Path) -> None:
    cfg = _config(
        tmp_path,
        pwm_source="sites",
        min_sites=2,
        allow_low_sites=False,
        regulator_sets=[["lexA"]],
    )
    lock_payload = {
        "pwm_source": "sites",
        "resolved": {"lexA": {"source": "regulondb", "motif_id": "RBM1", "sha256": "aaa"}},
    }
    _write_lock(tmp_path, json.dumps(lock_payload))
    catalog = CatalogIndex(
        entries={
            "regulondb:RBM1": CatalogEntry(
                source="regulondb",
                motif_id="RBM1",
                tf_name="lexA",
                kind="PFM",
                has_sites=True,
                site_count=1,
                site_total=1,
            )
        }
    )
    catalog.save(tmp_path / ".cruncher")
    sites_dir = tmp_path / ".cruncher" / "normalized" / "sites" / "regulondb"
    sites_dir.mkdir(parents=True, exist_ok=True)
    (sites_dir / "RBM1.jsonl").write_text(json.dumps({"sequence": "ACGT"}) + "\n")
    statuses = target_statuses(cfg=cfg, config_path=tmp_path / "config.yaml")
    assert statuses[0].status == "insufficient-sites"
    assert has_blocking_target_errors(statuses) is True


def test_target_status_missing_matrix_file(tmp_path: Path) -> None:
    cfg = _config(tmp_path, pwm_source="matrix", min_sites=2, regulator_sets=[["lexA"]])
    lock_payload = {
        "pwm_source": "matrix",
        "resolved": {"lexA": {"source": "regulondb", "motif_id": "RBM1", "sha256": "aaa"}},
    }
    _write_lock(tmp_path, json.dumps(lock_payload))
    catalog = CatalogIndex(
        entries={
            "regulondb:RBM1": CatalogEntry(
                source="regulondb",
                motif_id="RBM1",
                tf_name="lexA",
                kind="PFM",
                has_matrix=True,
            )
        }
    )
    catalog.save(tmp_path / ".cruncher")
    statuses = target_statuses(cfg=cfg, config_path=tmp_path / "config.yaml")
    assert statuses[0].status == "missing-matrix-file"
    assert has_blocking_target_errors(statuses) is True


def test_target_status_merges_sites_for_matrix_targets(tmp_path: Path) -> None:
    cfg = _config(tmp_path, pwm_source="matrix", min_sites=2, combine_sites=True, regulator_sets=[["lexA"]])
    lock_payload = {
        "pwm_source": "matrix",
        "resolved": {"lexA": {"source": "regulondb", "motif_id": "RBM1", "sha256": "aaa"}},
    }
    _write_lock(tmp_path, json.dumps(lock_payload))
    catalog = CatalogIndex(
        entries={
            "regulondb:RBM1": CatalogEntry(
                source="regulondb",
                motif_id="RBM1",
                tf_name="lexA",
                kind="PFM",
                has_matrix=True,
                has_sites=True,
                site_count=5,
                site_total=5,
                site_kind="curated",
                dataset_id="regulondb_curated",
            ),
            "demo_local_meme:lexA_demo": CatalogEntry(
                source="demo_local_meme",
                motif_id="lexA_demo",
                tf_name="lexA",
                kind="PFM",
                has_sites=True,
                site_count=7,
                site_total=8,
                site_kind="meme_blocks",
                dataset_id="dapseq_demo",
            ),
        }
    )
    catalog.save(tmp_path / ".cruncher")
    motif_path = tmp_path / ".cruncher" / "normalized" / "motifs" / "regulondb" / "RBM1.json"
    motif_path.parent.mkdir(parents=True, exist_ok=True)
    motif_path.write_text("{}")
    statuses = target_statuses(cfg=cfg, config_path=tmp_path / "config.yaml")
    assert statuses[0].site_count == 12
    assert statuses[0].site_total == 13
    assert statuses[0].site_kind == "mixed"
    assert statuses[0].dataset_id == "mixed"


def test_target_status_needs_window_for_ht_sites(tmp_path: Path) -> None:
    cfg = _config(tmp_path, pwm_source="sites", min_sites=2, regulator_sets=[["lexA"]])
    lock_payload = {
        "pwm_source": "sites",
        "resolved": {"lexA": {"source": "regulondb", "motif_id": "DATASET1", "sha256": "aaa"}},
    }
    _write_lock(tmp_path, json.dumps(lock_payload))
    catalog = CatalogIndex(
        entries={
            "regulondb:DATASET1": CatalogEntry(
                source="regulondb",
                motif_id="DATASET1",
                tf_name="lexA",
                kind="PFM",
                has_sites=True,
                site_count=3,
                site_total=3,
                site_kind="ht_peak",
                site_length_min=10,
                site_length_max=12,
                site_length_count=3,
            )
        }
    )
    catalog.save(tmp_path / ".cruncher")
    sites_dir = tmp_path / ".cruncher" / "normalized" / "sites" / "regulondb"
    sites_dir.mkdir(parents=True, exist_ok=True)
    (sites_dir / "DATASET1.jsonl").write_text(json.dumps({"sequence": "ACGTACGTAA"}) + "\n")
    statuses = target_statuses(cfg=cfg, config_path=tmp_path / "config.yaml")
    assert statuses[0].status == "needs-window"
    assert has_blocking_target_errors(statuses) is True


def test_target_candidates_lists_catalog(tmp_path: Path) -> None:
    cfg = _config(tmp_path, pwm_source="matrix", min_sites=2)
    catalog = CatalogIndex(
        entries={
            "regulondb:RBM1": CatalogEntry(
                source="regulondb",
                motif_id="RBM1",
                tf_name="lexA",
                kind="PFM",
                has_matrix=True,
            )
        }
    )
    catalog.save(tmp_path / ".cruncher")
    candidates = target_candidates(cfg=cfg, config_path=tmp_path / "config.yaml")
    assert candidates[0].tf_name == "lexA"
    assert len(candidates[0].candidates) == 1


def test_target_candidates_fuzzy(tmp_path: Path) -> None:
    cfg = _config(tmp_path, pwm_source="matrix", min_sites=2, regulator_sets=[["lex"]])
    catalog = CatalogIndex(
        entries={
            "regulondb:RBM1": CatalogEntry(
                source="regulondb",
                motif_id="RBM1",
                tf_name="lexA",
                kind="PFM",
                has_matrix=True,
            )
        }
    )
    catalog.save(tmp_path / ".cruncher")
    candidates = target_candidates_fuzzy(cfg=cfg, config_path=tmp_path / "config.yaml", min_score=0.5, limit=5)
    assert candidates[0].tf_name == "lex"
    assert len(candidates[0].candidates) == 1


def test_target_status_combine_sites_needs_window(tmp_path: Path) -> None:
    cfg = _config(
        tmp_path,
        pwm_source="sites",
        min_sites=1,
        allow_low_sites=True,
        combine_sites=True,
        regulator_sets=[["lexA"]],
    )
    lock_payload = {
        "pwm_source": "sites",
        "combine_sites": True,
        "resolved": {"lexA": {"source": "regulondb", "motif_id": "DATASET1", "sha256": "aaa"}},
    }
    _write_lock(tmp_path, json.dumps(lock_payload))
    catalog = CatalogIndex(
        entries={
            "regulondb:DATASET1": CatalogEntry(
                source="regulondb",
                motif_id="DATASET1",
                tf_name="lexA",
                kind="PFM",
                has_sites=True,
                site_count=3,
                site_total=3,
                site_length_min=10,
                site_length_max=10,
            ),
            "regulondb:DATASET2": CatalogEntry(
                source="regulondb",
                motif_id="DATASET2",
                tf_name="lexA",
                kind="PFM",
                has_sites=True,
                site_count=2,
                site_total=2,
                site_length_min=12,
                site_length_max=12,
            ),
        }
    )
    catalog.save(tmp_path / ".cruncher")
    sites_dir = tmp_path / ".cruncher" / "normalized" / "sites" / "regulondb"
    sites_dir.mkdir(parents=True, exist_ok=True)
    (sites_dir / "DATASET1.jsonl").write_text(json.dumps({"sequence": "ACGTACGTAA"}) + "\n")
    (sites_dir / "DATASET2.jsonl").write_text(json.dumps({"sequence": "ACGTACGTACGT"}) + "\n")

    statuses = target_statuses(cfg=cfg, config_path=tmp_path / "config.yaml")
    assert statuses[0].status == "needs-window"


def test_target_status_combine_sites_aggregates_counts(tmp_path: Path) -> None:
    cfg = _config(
        tmp_path,
        pwm_source="sites",
        min_sites=5,
        allow_low_sites=False,
        combine_sites=True,
        regulator_sets=[["lexA"]],
    )
    lock_payload = {
        "pwm_source": "sites",
        "combine_sites": True,
        "resolved": {"lexA": {"source": "regulondb", "motif_id": "RDB1", "sha256": "aaa"}},
    }
    _write_lock(tmp_path, json.dumps(lock_payload))
    catalog = CatalogIndex(
        entries={
            "regulondb:RDB1": CatalogEntry(
                source="regulondb",
                motif_id="RDB1",
                tf_name="lexA",
                kind="PFM",
                has_sites=True,
                site_count=3,
                site_total=3,
                site_length_min=10,
                site_length_max=10,
            ),
            "regulondb:RDB2": CatalogEntry(
                source="regulondb",
                motif_id="RDB2",
                tf_name="lexA",
                kind="PFM",
                has_sites=True,
                site_count=4,
                site_total=4,
                site_length_min=10,
                site_length_max=10,
            ),
        }
    )
    catalog.save(tmp_path / ".cruncher")
    sites_dir = tmp_path / ".cruncher" / "normalized" / "sites" / "regulondb"
    sites_dir.mkdir(parents=True, exist_ok=True)
    (sites_dir / "RDB1.jsonl").write_text(json.dumps({"sequence": "ACGTACGTAA"}) + "\n")
    (sites_dir / "RDB2.jsonl").write_text(json.dumps({"sequence": "ACGTACGTAA"}) + "\n")

    statuses = target_statuses(cfg=cfg, config_path=tmp_path / "config.yaml")
    assert statuses[0].status == "ready"
    assert statuses[0].site_count == 7
