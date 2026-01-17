"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_lockfile_hashes.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dnadesign.cruncher.app.lock_service import resolve_lock
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex
from dnadesign.cruncher.store.lockfile import LockedMotif, Lockfile, read_lockfile, verify_lockfile_hashes
from dnadesign.cruncher.utils.hashing import sha256_lines, sha256_path


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


def test_resolve_lock_combines_site_hashes(tmp_path: Path) -> None:
    catalog_root = tmp_path
    entries = {
        "regulondb:RBM1": CatalogEntry(
            source="regulondb",
            motif_id="RBM1",
            tf_name="lexA",
            kind="PFM",
            has_sites=True,
        ),
        "regulondb:RBM2": CatalogEntry(
            source="regulondb",
            motif_id="RBM2",
            tf_name="lexA",
            kind="PFM",
            has_sites=True,
        ),
    }
    CatalogIndex(entries=entries).save(catalog_root)

    sites_dir = catalog_root / "normalized" / "sites" / "regulondb"
    sites_dir.mkdir(parents=True, exist_ok=True)
    (sites_dir / "RBM1.jsonl").write_text(json.dumps({"sequence": "ACGT"}) + "\n")
    (sites_dir / "RBM2.jsonl").write_text(json.dumps({"sequence": "TGCA"}) + "\n")

    lock_path = catalog_root / "locks" / "config.lock.json"
    resolve_lock(
        names=["lexA"],
        catalog_root=catalog_root,
        pwm_source="sites",
        combine_sites=True,
        lock_path=lock_path,
    )

    payload = json.loads(lock_path.read_text())
    resolved = payload["resolved"]["lexA"]
    lines = []
    for motif_id in ("RBM1", "RBM2"):
        path = sites_dir / f"{motif_id}.jsonl"
        lines.append(f"regulondb:{motif_id}:{sha256_path(path)}")
    expected = sha256_lines(lines)
    assert resolved["sha256"] == expected

    lockfile = read_lockfile(lock_path)
    verify_lockfile_hashes(lockfile=lockfile, catalog_root=catalog_root, expected_pwm_source="sites")
