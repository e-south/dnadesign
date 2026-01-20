"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/adapters/sources/usr_sequences.py

USR sequences input source.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .base import BaseDataSource, resolve_path


@dataclass
class USRSequencesDataSource(BaseDataSource):
    dataset: str
    cfg_path: Path
    root: str
    limit: Optional[int] = None

    def load_data(self, *, rng=None, outputs_root: Path | None = None):
        try:
            from dnadesign.usr.src.dataset import Dataset as USRDataset  # type: ignore
        except Exception as e:  # pragma: no cover - depends on optional USR install
            raise RuntimeError(f"USR dataset reader not available: {e}") from e

        if not self.root:
            raise ValueError("usr_sequences requires an explicit root path.")
        root_path = resolve_path(self.cfg_path, self.root)

        ds = USRDataset(root_path, self.dataset)
        if not ds.records_path.exists():
            raise FileNotFoundError(f"USR records not found at {ds.records_path}")

        import pyarrow.parquet as pq

        tbl = pq.read_table(ds.records_path, columns=["sequence"])
        seqs_raw = tbl.column("sequence").to_pylist()
        bad_rows = [i for i, s in enumerate(seqs_raw) if not isinstance(s, str) or not s.strip()]
        if bad_rows:
            preview = ", ".join(str(i) for i in bad_rows[:10])
            raise ValueError(
                f"USR dataset contains null/empty sequences (rows: {preview}). "
                "DenseGen requires non-empty sequence strings."
            )
        seqs = [s.strip().upper() for s in seqs_raw]
        invalid_rows = [i for i, s in enumerate(seqs) if any(ch not in {"A", "C", "G", "T"} for ch in s)]
        if invalid_rows:
            preview = ", ".join(str(i) for i in invalid_rows[:10])
            raise ValueError(
                f"USR dataset contains invalid sequences (rows: {preview}). DenseGen requires A/C/G/T only."
            )
        if self.limit:
            seqs = seqs[: max(0, int(self.limit))]
        return seqs, None
