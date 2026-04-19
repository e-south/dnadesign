from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa

from dnadesign.usr.dataset import Dataset
from dnadesign.usr.overlay_digest_ledger import overlay_digest_ledger_path
from dnadesign.usr.tests.registry_helpers import register_test_namespace


def _make_dataset(root: Path) -> Dataset:
    register_test_namespace(root, namespace="audit", columns_spec="audit__score:float64")
    dataset = Dataset(root, "demo")
    dataset.init(source="test")
    dataset.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
            {"sequence": "TGCA", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
        ],
        source="unit",
    )
    return dataset


def test_write_overlay_digest_ledger_is_explicit_and_write_overlay_part_maintains_it(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    dataset = _make_dataset(root)
    ids = dataset.head(2)["id"].tolist()

    dataset.write_overlay_part(
        "audit",
        pa.table({"id": ids, "audit__score": [0.1, 0.2]}),
        key="id",
        allow_missing=False,
    )

    overlay_dir = dataset.dir / "_derived" / "audit"
    ledger_path = overlay_digest_ledger_path(overlay_dir)
    assert ledger_path is not None
    assert not ledger_path.exists()

    written_ledger = dataset.write_overlay_digest_ledger("audit")
    assert written_ledger == ledger_path

    first_payload = json.loads(written_ledger.read_text(encoding="utf-8"))
    assert first_payload["schema_version"] == "usr.overlay_digest_ledger.v1"
    assert len(first_payload["parts"]) == 1

    dataset.write_overlay_part(
        "audit",
        pa.table({"id": ids, "audit__score": [0.3, 0.4]}),
        key="id",
        allow_missing=False,
    )

    second_payload = json.loads(written_ledger.read_text(encoding="utf-8"))
    assert len(second_payload["parts"]) == 2
    assert all(str(entry["digest"]).startswith("sha256:") for entry in second_payload["parts"])
