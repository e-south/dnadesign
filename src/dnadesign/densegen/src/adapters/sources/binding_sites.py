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

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from ...core.artifacts.ids import hash_label_motif, hash_tfbs_id
from .base import BaseDataSource, infer_format, resolve_path

log = logging.getLogger(__name__)


@dataclass
class BindingSitesDataSource(BaseDataSource):
    path: str
    cfg_path: Path
    fmt: Optional[str] = None
    columns: Optional[dict[str, Optional[str]]] = None  # regulator, sequence, site_id, source

    def _resolve_format(self, path: Path) -> str:
        fmt = (self.fmt or "").strip().lower() if self.fmt is not None else None
        if fmt:
            if fmt not in {"csv", "parquet", "xlsx"}:
                raise ValueError(f"binding_sites.format must be 'csv', 'parquet', or 'xlsx', got: {fmt!r}")
            return fmt
        inferred = infer_format(path)
        if inferred is None and path.suffix.lower() in {".xlsx", ".xls"}:
            inferred = "xlsx"
        if inferred is None:
            raise ValueError(
                f"binding_sites.format is required when file extension is not .csv/.parquet/.xlsx. Got path: {path}"
            )
        return inferred

    def _load_table(self, path: Path, fmt: str) -> pd.DataFrame:
        if fmt == "csv":
            return pd.read_csv(path)
        if fmt == "parquet":
            import pyarrow.parquet as pq

            return pq.read_table(path).to_pandas()
        if fmt == "xlsx":
            return pd.read_excel(path)
        raise ValueError(f"Unsupported binding_sites.format: {fmt}")

    def load_data(self, *, rng=None, outputs_root: Path | None = None, run_id: str | None = None):
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
            dup_count = int(dup_mask.sum())
            log.warning(
                "Binding sites input contains %d duplicate regulator/binding-site pairs in %s. "
                "Duplicates are retained; set generation.sampling.unique_binding_sites=true or "
                "generation.sampling.unique_binding_cores=true to dedupe at Stage-B sampling.",
                dup_count,
                data_path,
            )

        invalid_mask = ~seq_clean.str.fullmatch(r"[ACGT]+")
        if invalid_mask.any():
            bad_rows = df[invalid_mask].index.tolist()
            preview = ", ".join(str(i) for i in bad_rows[:10])
            raise ValueError(
                f"Invalid binding-site motifs in {data_path} (rows: {preview}). DenseGen requires A/C/G/T."
            )

        out = pd.DataFrame({"tf": tf_clean, "tfbs": seq_clean})
        out["tfbs_core"] = seq_clean
        source_default = str(data_path)
        if site_id_col:
            out["site_id"] = df[site_id_col].astype(str).str.strip()
        if source_col:
            source_values = df[source_col].fillna("").astype(str).str.strip()
            out["source"] = source_values.where(source_values != "", other=source_default)
        else:
            out["source"] = source_default

        motif_id_map = {tf: hash_label_motif(label=tf, source_kind="binding_sites") for tf in tf_clean.unique()}
        out["motif_id"] = tf_clean.map(motif_id_map)
        out["tfbs_id"] = [
            hash_tfbs_id(
                motif_id=motif_id_map[tf],
                sequence=seq,
                scoring_backend="binding_sites",
            )
            for tf, seq in zip(tf_clean.tolist(), seq_clean.tolist())
        ]

        out = out.reset_index(drop=True)
        entries = list(zip(out["tf"].tolist(), out["tfbs"].tolist(), out["source"].tolist()))
        return entries, out, None
