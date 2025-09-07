"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/src/usr_adapter.py

USR adapter: buffered import + attach for DenseGen.

- Ensures a USR dataset exists (creates if missing).
- Buffers sequences (essential columns) and derived metadata (namespaced).
- De-duplicates against existing ids in records.parquet before import.
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
from typing import Dict, List, Optional, Tuple

import pyarrow.parquet as pq

from dnadesign.usr.src.dataset import Dataset
from dnadesign.usr.src.normalize import compute_id, normalize_sequence


def _repo_root_from(file_path: Path) -> Path:
    return file_path.resolve().parents[2]


def _default_usr_root() -> Path:
    return _repo_root_from(Path(__file__)) / "usr" / "datasets"


@dataclass
class USRWriter:
    dataset: str
    root: Optional[Path] = None
    namespace: str = "densegen"
    chunk_size: int = 128
    allow_overwrite: bool = True
    default_bio_type: str = "dna"
    default_alphabet: str = "dna_4"

    # internal buffers
    _seq_buf: List[Tuple[str, str]] = field(
        default_factory=list
    )  # (sequence, source_label)
    _meta_buf: List[Dict] = field(
        default_factory=list
    )  # each has 'id' + free-form keys
    _seen_ids: set = field(default_factory=set)

    def __post_init__(self):
        self.root = (
            Path(self.root).resolve() if self.root else _default_usr_root().resolve()
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.ds = Dataset(self.root, self.dataset)
        if not self.ds.records_path.exists():
            self.ds.init(source="densegen init")

    def add(self, sequence: str, meta: Dict, source_label: str) -> None:
        seq_norm = normalize_sequence(
            sequence, self.default_bio_type, self.default_alphabet
        )
        seq_id = compute_id(self.default_bio_type, seq_norm)
        if seq_id in self._seen_ids:
            return  # de-dup within this buffer batch
        self._seen_ids.add(seq_id)

        self._seq_buf.append((seq_norm, source_label))

        meta_row = {"id": seq_id}
        for k, v in meta.items():
            if isinstance(v, (list, dict)):
                meta_row[k] = json.dumps(v)
            else:
                meta_row[k] = v
        self._meta_buf.append(meta_row)

        if len(self._seq_buf) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self._seq_buf:
            return

        rows = [
            {
                "sequence": s,
                "bio_type": self.default_bio_type,
                "alphabet": self.default_alphabet,
                "source": src,
            }
            for s, src in self._seq_buf
        ]

        # de-dup vs existing
        existing_ids = self._load_existing_ids()
        incoming_ids = [compute_id(self.default_bio_type, r["sequence"]) for r in rows]
        rows_new = [
            rows[i] for i, rid in enumerate(incoming_ids) if rid not in existing_ids
        ]

        if rows_new:
            with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp:
                for r in rows_new:
                    tmp.write(json.dumps(r) + "\n")
                tmp_path = Path(tmp.name)
            self.ds.import_jsonl(
                tmp_path,
                default_bio_type=self.default_bio_type,
                default_alphabet=self.default_alphabet,
            )
            tmp_path.unlink(missing_ok=True)

        if self._meta_buf:
            all_keys = set().union(*(m.keys() for m in self._meta_buf))
            columns = ["id"] + [k for k in sorted(all_keys) if k != "id"]
            with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmpc:
                tmpc.write(",".join(columns) + "\n")
                for row in self._meta_buf:
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

        self._seq_buf.clear()
        self._meta_buf.clear()
        self._seen_ids.clear()

    def _load_existing_ids(self) -> set:
        rp = self.ds.records_path
        if not rp.exists():
            return set()
        try:
            tbl = pq.read_table(rp, columns=["id"])
            return set(tbl.column("id").to_pylist())
        except Exception:
            return set()
