from __future__ import annotations

import json
from pathlib import Path

import pytest

from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex
from dnadesign.cruncher.store.lockfile import LockedMotif, Lockfile, verify_lockfile_hashes
from dnadesign.cruncher.utils.hashing import sha256_path


def _write_motif(path: Path, *, source: str, motif_id: str, tf_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "descriptor": {"source": source, "motif_id": motif_id, "tf_name": tf_name},
        "matrix": [[0.25, 0.25, 0.25, 0.25]],
        "checksums": {"sha256_norm": "good"},
    }
    path.write_text(json.dumps(payload))


def _write_sites(path: Path, seq: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sequence": seq}) + "\n")


def test_verify_lockfile_hashes_matrix_mismatch(tmp_path: Path) -> None:
    catalog_root = tmp_path
    entry = CatalogEntry(
        source="regulondb",
        motif_id="RBM1",
        tf_name="lexA",
        kind="PFM",
        has_matrix=True,
    )
    CatalogIndex(entries={entry.key: entry}).save(catalog_root)

    motif_path = catalog_root / "normalized" / "motifs" / "regulondb" / "RBM1.json"
    _write_motif(motif_path, source="regulondb", motif_id="RBM1", tf_name="lexA")

    lockfile = Lockfile(
        resolved={"lexA": LockedMotif(source="regulondb", motif_id="RBM1", sha256="bad")},
        pwm_source="matrix",
        site_kinds=None,
        combine_sites=None,
        generated_at=None,
    )
    with pytest.raises(ValueError):
        verify_lockfile_hashes(lockfile=lockfile, catalog_root=catalog_root, expected_pwm_source="matrix")


def test_verify_lockfile_hashes_sites_dataset_mismatch(tmp_path: Path) -> None:
    catalog_root = tmp_path
    entry = CatalogEntry(
        source="regulondb",
        motif_id="DATASET1",
        tf_name="lexA",
        kind="PFM",
        has_sites=True,
        dataset_id="DATASET1",
    )
    CatalogIndex(entries={entry.key: entry}).save(catalog_root)

    sites_path = catalog_root / "normalized" / "sites" / "regulondb" / "DATASET1.jsonl"
    _write_sites(sites_path, "ACGT")
    checksum = sha256_path(sites_path)

    lockfile = Lockfile(
        resolved={
            "lexA": LockedMotif(
                source="regulondb",
                motif_id="DATASET1",
                sha256=checksum,
                dataset_id="DATASET2",
            )
        },
        pwm_source="sites",
        site_kinds=None,
        combine_sites=None,
        generated_at=None,
    )
    with pytest.raises(ValueError):
        verify_lockfile_hashes(lockfile=lockfile, catalog_root=catalog_root, expected_pwm_source="sites")
