"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/adapters/sources/binding_sites.py

Binding-site table input source (CSV/Parquet).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .base import BaseDataSource, infer_format, resolve_path


@dataclass
class BindingSitesDataSource(BaseDataSource):
    path: str
    cfg_path: Path
    fmt: Optional[str] = None
    columns: Optional[dict[str, Optional[str]]] = None  # regulator, sequence, site_id, source

    def _resolve_format(self, path: Path) -> str:
        fmt = (self.fmt or "").strip().lower() if self.fmt is not None else None
        if fmt:
            if fmt not in {"csv", "parquet"}:
                raise ValueError(f"binding_sites.format must be 'csv' or 'parquet', got: {fmt!r}")
            return fmt
        inferred = infer_format(path)
        if inferred is None:
            raise ValueError(
                f"binding_sites.format is required when file extension is not .csv/.parquet. Got path: {path}"
            )
        return inferred

    def _load_table(self, path: Path, fmt: str) -> pd.DataFrame:
        if fmt == "csv":
            return pd.read_csv(path)
        if fmt == "parquet":
            import pyarrow.parquet as pq

            return pq.read_table(path).to_pandas()
        raise ValueError(f"Unsupported binding_sites.format: {fmt}")

    def load_data(self, *, rng=None):
        data_path = resolve_path(self.cfg_path, self.path)
        if not (data_path.exists() and data_path.is_file()):
            raise FileNotFoundError(f"Binding sites file not found. Looked here:\n  - {data_path}")

        fmt = self._resolve_format(data_path)
        df = self._load_table(data_path, fmt)

        cols = self.columns or {}
        tf_col = cols.get("regulator") or "tf"
        seq_col = cols.get("sequence") or "tfbs"
        site_id_col = cols.get("site_id")
        source_col = cols.get("source")

        for c in [tf_col, seq_col]:
            if c not in df.columns:
                raise ValueError(f"Required column '{c}' missing in {data_path}")

        if site_id_col and site_id_col not in df.columns:
            raise ValueError(f"site_id column '{site_id_col}' missing in {data_path}")
        if source_col and source_col not in df.columns:
            raise ValueError(f"source column '{source_col}' missing in {data_path}")

        tf_raw = df[tf_col]
        seq_raw = df[seq_col]
        if tf_raw.isna().any() or seq_raw.isna().any():
            bad_rows = df[tf_raw.isna() | seq_raw.isna()].index.tolist()
            preview = ", ".join(str(i) for i in bad_rows[:10])
            raise ValueError(
                f"Null regulator/sequence values in {data_path} (rows: {preview}). "
                "DenseGen requires non-empty regulator and binding-site strings."
            )

        tf_clean = tf_raw.astype(str).str.strip()
        seq_clean = seq_raw.astype(str).str.strip().str.upper()
        empty_mask = (tf_clean == "") | (seq_clean == "")
        if empty_mask.any():
            bad_rows = df[empty_mask].index.tolist()
            preview = ", ".join(str(i) for i in bad_rows[:10])
            raise ValueError(
                f"Empty regulator/sequence values in {data_path} (rows: {preview}). "
                "DenseGen requires non-empty regulator and binding-site strings."
            )

        dup_mask = pd.DataFrame({"tf": tf_clean, "tfbs": seq_clean}).duplicated()
        if dup_mask.any():
            bad_rows = df[dup_mask].index.tolist()
            preview = ", ".join(str(i) for i in bad_rows[:10])
            raise ValueError(
                f"Duplicate regulator/binding-site pairs found in {data_path} (rows: {preview}). "
                "Remove duplicates or pre-aggregate before running DenseGen."
            )

        invalid_mask = ~seq_clean.str.fullmatch(r"[ACGT]+")
        if invalid_mask.any():
            bad_rows = df[invalid_mask].index.tolist()
            preview = ", ".join(str(i) for i in bad_rows[:10])
            raise ValueError(
                f"Invalid binding-site motifs in {data_path} (rows: {preview}). DenseGen requires A/C/G/T."
            )

        out = pd.DataFrame({"tf": tf_clean, "tfbs": seq_clean})
        if site_id_col:
            out["site_id"] = df[site_id_col].astype(str).str.strip()
        if source_col:
            out["source"] = df[source_col].astype(str).str.strip()

        out = out.reset_index(drop=True)
        source_default = str(data_path)
        src_vals = out.get("source")
        if src_vals is not None:
            src_list = [s if s else source_default for s in src_vals.tolist()]
        else:
            src_list = [source_default] * len(out)
        entries = list(zip(out["tf"].tolist(), out["tfbs"].tolist(), src_list))
        return entries, out
