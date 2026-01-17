"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/adapters/outputs/usr_writer.py

USR adapter: buffered import + attach for DenseGen.

- Ensures a USR dataset exists (creates if missing).
    - Buffers sequences (essential columns) and derived metadata (namespaced).
    - De-duplicates against existing ids in records.parquet before import.
    - Imports via Dataset.import_rows (no JSONL intermediates).
    - Attaches namespaced columns keyed by 'id' (optionally overwriting).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pyarrow.parquet as pq

from dnadesign.usr.src.dataset import Dataset

from ...core.metadata_schema import validate_metadata
from .base import AlignmentDigest
from .id_index import compute_alignment_digest_from_ids
from .record import OutputRecord


@dataclass
class USRWriter:
    dataset: str
    root: Optional[Path] = None
    namespace: str = "densegen"
    chunk_size: int = 128
    allow_overwrite: bool = True
    default_bio_type: str = "dna"
    default_alphabet: str = "dna_4"
    deduplicate: bool = True

    # internal buffers
    _records: List[OutputRecord] = field(default_factory=list)
    _seen_ids: set = field(default_factory=set)
    _existing_ids: Optional[set] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.root is None:
            raise ValueError("USRWriter requires an explicit root path.")
        self.root = Path(self.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ds = Dataset(self.root, self.dataset)
        if not self.ds.records_path.exists():
            self.ds.init(source="densegen init")

    def add(self, record: OutputRecord) -> bool:
        if record.bio_type != self.default_bio_type or record.alphabet != self.default_alphabet:
            raise ValueError(
                "OutputRecord bio_type/alphabet mismatch for USR sink. "
                f"record=({record.bio_type}, {record.alphabet}) "
                f"sink=({self.default_bio_type}, {self.default_alphabet})"
            )
        validate_metadata(record.meta)
        seq_id = record.id
        if self.deduplicate:
            existing = self._load_existing_ids()
            if seq_id in existing or seq_id in self._seen_ids:
                return False  # skip existing ids or local duplicates
        elif seq_id in self._seen_ids:
            return False

        self._seen_ids.add(seq_id)
        self._records.append(record)

        if len(self._records) >= self.chunk_size:
            self.flush()
        return True

    def flush(self) -> None:
        if not self._records:
            return

        rows = [
            {
                "id": r.id,
                "sequence": r.sequence,
                "bio_type": r.bio_type,
                "alphabet": r.alphabet,
                "source": r.source,
            }
            for r in self._records
        ]

        incoming_ids = [r.id for r in self._records]
        if self.deduplicate:
            existing_ids = self._load_existing_ids()
            mask = [rid not in existing_ids for rid in incoming_ids]
            rows_new = [rows[i] for i, keep in enumerate(mask) if keep]
            records_new = [self._records[i] for i, keep in enumerate(mask) if keep]
        else:
            rows_new = rows
            records_new = self._records

        meta_new = []
        for rec in records_new:
            meta_row = {"id": rec.id}
            for k, v in rec.meta.items():
                if isinstance(v, (list, dict)):
                    meta_row[k] = json.dumps(v)
                else:
                    meta_row[k] = v
            meta_new.append(meta_row)

        if rows_new:
            self.ds.import_rows(
                rows_new,
                default_bio_type=self.default_bio_type,
                default_alphabet=self.default_alphabet,
                source=None,
                strict_id_check=True,
            )
            if self.deduplicate:
                existing_ids.update({rec.id for rec in records_new})

        if meta_new:
            all_keys = set().union(*(m.keys() for m in meta_new))
            columns = ["id"] + [k for k in sorted(all_keys) if k != "id"]
            with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmpc:
                tmpc.write(",".join(columns) + "\n")
                for row in meta_new:
                    vals = []
                    for col in columns:
                        v = row.get(col, "")
                        if isinstance(v, str):
                            v = v.replace('"', '""')
                            vals.append(f'"{v}"')
                        else:
                            vals.append("" if v is None else str(v))
                    tmpc.write(",".join(vals) + "\n")
                csv_path = Path(tmpc.name)
            self.ds.attach(
                csv_path,
                namespace=self.namespace,
                id_col="id",
                allow_overwrite=self.allow_overwrite,
            )
            csv_path.unlink(missing_ok=True)

        self._records.clear()
        self._seen_ids.clear()

    def existing_ids(self) -> set:
        return set(self._load_existing_ids())

    def alignment_digest(self):
        rp = self.ds.records_path
        if not rp.exists():
            return compute_alignment_digest_from_ids([])
        try:
            import pyarrow.dataset as ds
        except Exception as e:
            raise RuntimeError(f"Parquet support is not available: {e}") from e
        dataset = ds.dataset(rp, format="parquet")
        scanner = ds.Scanner.from_dataset(dataset, columns=["id"], batch_size=4096)
        xor_val = 0
        count = 0
        for batch in scanner.to_batches():
            ids = [str(x) for x in batch.column(0).to_pylist() if x is not None]
            if not ids:
                continue
            digest = compute_alignment_digest_from_ids(ids)
            xor_val ^= int(digest.xor_hash, 16)
            count += digest.id_count
        return AlignmentDigest(count, f"{xor_val:032x}")

    def _load_existing_ids(self) -> set:
        if self._existing_ids is not None:
            return self._existing_ids
        rp = self.ds.records_path
        if not rp.exists():
            self._existing_ids = set()
            return self._existing_ids
        try:
            tbl = pq.read_table(rp, columns=["id"])
        except Exception as e:
            raise RuntimeError(f"Failed to read existing USR records at {rp}: {e}") from e
        self._existing_ids = set(tbl.column("id").to_pylist())
        return self._existing_ids
