"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/src/data_ingestor.py

Data ingestion utilities for DenseGen.

Supported source types:
- "csv_tfbs": local CSV with columns for TF and TFBS (required).
- "csv_sequences": local CSV containing raw sequences (column: 'sequence').
- "usr_sequences": read sequences from a USR dataset (records.parquet).

Path resolution
---------------
Relative paths are resolved in this order:

  1) DENSEGEN_ROOT / <relative>      # e.g., .../dnadesign/src/dnadesign/densegen/<relative>
  2) PROJECT_ROOT / <relative>       # repo root (has pyproject.toml)
  3) SRC_ROOT / <relative>           # .../dnadesign/src

This matches how configs and inputs are laid out in the repo, so
`path: inputs/xxx.csv` will correctly find `densegen/inputs/xxx.csv`.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    # optional torch import for legacy use; not required otherwise
    import torch  # noqa: F401
except Exception:
    torch = None  # type: ignore

# ---- Roots (robust, no hard-coded depths) ------------------------------------

_THIS = Path(__file__).resolve()
# .../src/dnadesign/densegen/src/data_ingestor.py
SRC_ROOT = _THIS.parents[3]  # .../dnadesign/src
PROJECT_ROOT = SRC_ROOT.parent  # .../dnadesign
DENSEGEN_ROOT = _THIS.parents[1]  # .../dnadesign/src/dnadesign/densegen

# USR reading for inputs (optional)
try:
    from dnadesign.usr.src.dataset import Dataset as USRDataset  # type: ignore
except Exception:
    USRDataset = None  # type: ignore


def resolve_path(given: str) -> Path:
    """
    Resolve a path string. If absolute, return as-is.
    If relative, try DenseGen root, then project root, then src root.
    Return the first existing candidate; otherwise return the DenseGen-root candidate.
    """
    p = Path(given)
    if p.is_absolute():
        return p

    candidates = [
        (DENSEGEN_ROOT / p).resolve(),
        (PROJECT_ROOT / p).resolve(),
        (SRC_ROOT / p).resolve(),
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fall back to DenseGen-root candidate so error messages are informative
    return candidates[0]


# ---------- Base interface ----------


class BaseDataSource(abc.ABC):
    @abc.abstractmethod
    def load_data(self) -> Tuple[List, Optional[pd.DataFrame]]:
        """
        Returns:
            (data_entries, meta_df)
            - For TF/TFBS CSV: meta_df is a DataFrame with 'tf' and 'tfbs' columns.
            - For sequences inputs: data_entries is a list of sequences; meta_df None.
        """
        raise NotImplementedError


# ---------- CSV: TF/TFBS ----------


@dataclass
class CSVTFBSDataSource(BaseDataSource):
    path: str
    columns: Optional[Dict[str, str]] = None  # {'tf': 'TF', 'tfbs': 'binding_site'}

    def load_data(self):
        csv_path = resolve_path(self.path)
        if not (csv_path.exists() and csv_path.is_file()):
            raise FileNotFoundError(
                "CSV not found for 'csv_tfbs' source. Looked here:\n"
                f"  - {(DENSEGEN_ROOT / self.path).resolve()}\n"
                f"  - {(PROJECT_ROOT / self.path).resolve()}\n"
                f"  - {(SRC_ROOT / self.path).resolve()}"
            )

        df = pd.read_csv(csv_path)

        tf_col = (self.columns or {}).get("tf", "tf")
        tfbs_col = (self.columns or {}).get("tfbs", "tfbs")

        for c in [tf_col, tfbs_col]:
            if c not in df.columns:
                raise ValueError(f"Required column '{c}' missing in {csv_path}")

        df = df.rename(columns={tf_col: "tf", tfbs_col: "tfbs"})
        df = df.dropna(subset=["tf", "tfbs"]).copy()
        df["tf"] = df["tf"].astype(str).str.strip()
        df["tfbs"] = df["tfbs"].astype(str).str.strip()
        df = df[(df["tf"] != "") & (df["tfbs"] != "")]
        df = df.drop_duplicates(subset=["tf", "tfbs"]).reset_index(drop=True)
        entries = list(zip(df["tf"].tolist(), df["tfbs"].tolist(), [str(csv_path)] * len(df)))
        return entries, df


# ---------- CSV: raw sequences ----------


@dataclass
class CSVSequencesDataSource(BaseDataSource):
    path: str
    sequence_column: str = "sequence"

    def load_data(self):
        csv_path = resolve_path(self.path)
        if not (csv_path.exists() and csv_path.is_file()):
            raise FileNotFoundError(
                "CSV not found for 'csv_sequences' source. Looked here:\n"
                f"  - {(DENSEGEN_ROOT / self.path).resolve()}\n"
                f"  - {(PROJECT_ROOT / self.path).resolve()}\n"
                f"  - {(SRC_ROOT / self.path).resolve()}"
            )

        df = pd.read_csv(csv_path)
        if self.sequence_column not in df.columns:
            raise ValueError(f"Column '{self.sequence_column}' missing in {csv_path}")
        seqs = [str(s).strip().upper() for s in df[self.sequence_column].dropna().tolist() if str(s).strip()]
        return seqs, None


# ---------- USR: sequences ----------


@dataclass
class USRSequencesDataSource(BaseDataSource):
    dataset: str
    root: Optional[str] = None
    limit: Optional[int] = None

    def load_data(self):
        if USRDataset is None:
            raise RuntimeError("USR dataset reader not available.")
        # Default USR datasets root: <project>/src/dnadesign/usr/datasets
        default_usr_root = PROJECT_ROOT / "src" / "dnadesign" / "usr" / "datasets"
        root_path = Path(self.root).resolve() if self.root else default_usr_root

        ds = USRDataset(root_path, self.dataset)
        if not ds.records_path.exists():
            raise FileNotFoundError(f"USR records not found at {ds.records_path}")

        import pyarrow.parquet as pq  # local import to avoid hard dep at import time

        tbl = pq.read_table(ds.records_path, columns=["sequence"])
        seqs = [s.upper() for s in tbl.column("sequence").to_pylist() if isinstance(s, str) and s.strip()]
        if self.limit:
            seqs = seqs[: max(0, int(self.limit))]
        return seqs, None


# ---------- factory ----------


def data_source_factory(cfg: dict) -> BaseDataSource:
    t = (cfg.get("type") or "").lower()
    if t == "csv_tfbs":
        if "path" not in cfg:
            raise ValueError("csv_tfbs requires 'path'")
        return CSVTFBSDataSource(path=cfg["path"], columns=cfg.get("columns"))
    if t == "csv_sequences":
        if "path" not in cfg:
            raise ValueError("csv_sequences requires 'path'")
        return CSVSequencesDataSource(path=cfg["path"], sequence_column=cfg.get("sequence_column", "sequence"))
    if t == "usr_sequences":
        if "dataset" not in cfg:
            raise ValueError("usr_sequences requires 'dataset'")
        return USRSequencesDataSource(dataset=cfg["dataset"], root=cfg.get("root"), limit=cfg.get("limit"))
    raise ValueError(f"Unsupported source type: {t}")
