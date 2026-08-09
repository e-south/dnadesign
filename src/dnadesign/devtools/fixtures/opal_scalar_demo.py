"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/fixtures/opal_scalar_demo.py

Build the small synthetic dataset shared by OPAL's portable demos.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

RECORD_COUNT = 96
LABEL_COUNT = 32
FEATURE_COUNT = 12


def _sequence(index: int, *, length: int = 32) -> str:
    digest = hashlib.sha256(f"opal-synthetic-{index}".encode()).digest()
    alphabet = "ACGT"
    sequence = "".join(alphabet[(digest[position // 4] >> (2 * (position % 4))) & 0b11] for position in range(length))
    # ACCA followed by sixteen DNA bases matches a real AWS access-key prefix.
    # Keep the synthetic fixture visibly DNA-like without emitting that token.
    return sequence.replace("ACCA", "ACCG")


def _features(index: int) -> list[float]:
    digest = hashlib.sha256(f"opal-features-{index}".encode()).digest()
    return [2.0 * (digest[position] / 255.0) - 1.0 for position in range(FEATURE_COUNT)]


def _response(features: list[float]) -> float:
    return 1.3 * features[0] - 0.7 * features[1] + 0.4 * features[2] * features[3] + 0.1 * math.sin(features[4])


def build_fixture(output_dir: Path) -> tuple[Path, Path]:
    """Write deterministic candidate records and a labeled subset."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ids = [f"synthetic-{index:03d}" for index in range(RECORD_COUNT)]
    sequences = [_sequence(index) for index in range(RECORD_COUNT)]
    features = [_features(index) for index in range(RECORD_COUNT)]

    records_path = output_dir / "records.parquet"
    table = pa.table(
        {
            "id": pa.array(ids, type=pa.string()),
            "sequence": pa.array(sequences, type=pa.string()),
            "bio_type": pa.array(["dna"] * RECORD_COUNT, type=pa.string()),
            "alphabet": pa.array(["dna_4"] * RECORD_COUNT, type=pa.string()),
            "fixture_kind": pa.array(["synthetic"] * RECORD_COUNT, type=pa.string()),
            "X": pa.array(features, type=pa.list_(pa.float32(), list_size=FEATURE_COUNT)),
        }
    )
    pq.write_table(table, records_path, compression="zstd")

    labels_path = output_dir / "labels.csv"
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "y"], lineterminator="\n")
        writer.writeheader()
        for sequence, row_features in zip(sequences[:LABEL_COUNT], features[:LABEL_COUNT], strict=True):
            writer.writerow({"sequence": sequence, "y": f"{_response(row_features):.12g}"})

    return records_path, labels_path


def main() -> None:
    output_dir = Path(__file__).resolve().parents[2] / "opal" / "campaigns" / "_fixtures" / "scalar-regression"
    records_path, labels_path = build_fixture(output_dir)
    print(records_path)
    print(labels_path)


if __name__ == "__main__":
    main()
