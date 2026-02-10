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

_VALID_DNA_BASES = {"A", "C", "G", "T"}


@dataclass
class USRSequencesDataSource(BaseDataSource):
    dataset: str
    cfg_path: Path
    root: str
    limit: Optional[int] = None

    def load_data(self, *, rng=None, outputs_root: Path | None = None, run_id: str | None = None):
        try:
            from dnadesign.usr import Dataset as USRDataset  # type: ignore
        except Exception as e:  # pragma: no cover - depends on optional USR install
            raise RuntimeError(f"USR dataset reader not available: {e}") from e

        if not self.root:
            raise ValueError("usr_sequences requires an explicit root path.")
        root_path = resolve_path(self.cfg_path, self.root)

        ds = USRDataset(root_path, self.dataset)
        if not ds.records_path.exists():
            raise FileNotFoundError(f"USR records not found at {ds.records_path}")

        import pyarrow.dataset as ds_scan

        scanner = ds_scan.Scanner.from_dataset(
            ds_scan.dataset(ds.records_path, format="parquet"),
            columns=["sequence"],
            batch_size=4096,
        )
        seqs: list[str] = []
        limit = max(0, int(self.limit)) if self.limit is not None else None
        row_idx = 0
        for batch in scanner.to_batches():
            for raw in batch.column(0).to_pylist():
                if not isinstance(raw, str) or not raw.strip():
                    raise ValueError(
                        f"USR dataset contains null/empty sequences (rows: {row_idx}). "
                        "DenseGen requires non-empty sequence strings."
                    )
                seq = raw.strip().upper()
                if any(ch not in _VALID_DNA_BASES for ch in seq):
                    raise ValueError(
                        f"USR dataset contains invalid sequences (rows: {row_idx}). DenseGen requires A/C/G/T only."
                    )
                seqs.append(seq)
                row_idx += 1
                if limit is not None and len(seqs) >= limit:
                    return seqs, None, None
        return seqs, None, None
