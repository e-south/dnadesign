"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/storage/ledger.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence
from uuid import uuid4

import pandas as pd

from .. import LEDGER_SCHEMA_VERSION
from ..core.utils import LedgerError, ensure_dir
from .parquet_io import (
    dataset_from_dir,
    pyarrow_compute,
    read_parquet_df,
    schema_signature,
    table_from_pandas,
    write_parquet_df,
    write_parquet_table,
)
from .workspace import CampaignWorkspace

# ---- schema allow-lists (Ledger v1.1) ----
ALLOW: dict[str, set[str]] = {
    "run_pred": {
        "event",
        "run_id",
        "as_of_round",
        "id",
        "sequence",
        "pred__y_dim",
        "pred__y_hat_model",
        "pred__y_obj_scalar",
        "sel__rank_competition",
        "sel__is_selected",
        # row-level objective diagnostics only
        "obj__logic_fidelity",
        "obj__effect_raw",
        "obj__effect_scaled",
        "obj__clip_lo_mask",
        "obj__clip_hi_mask",
    },
    "run_meta": {
        "event",
        "run_id",
        "as_of_round",
        "model__name",
        "model__params",
        "training__y_ops",
        "x_transform__name",
        "x_transform__params",
        "y_ingest__name",
        "y_ingest__params",
        "objective__name",
        "objective__params",
        "objective__summary_stats",
        "objective__denom_value",
        "objective__denom_percentile",
        "selection__name",
        "selection__params",
        "selection__score_field",
        "selection__objective",
        "selection__tie_handling",
        "stats__n_train",
        "stats__n_scored",
        "stats__unc_mean_sd_targets",
        "artifacts",
        "pred__preview",
        "schema__version",
        "opal__version",
    },
    "label": {
        "event",
        "observed_round",
        "id",
        "sequence",
        "y_obs",
        "src",
        "note",
    },
}

REQUIRED: dict[str, set[str]] = {
    "run_pred": {
        "event",
        "run_id",
        "as_of_round",
        "id",
        "pred__y_dim",
        "pred__y_hat_model",
        "pred__y_obj_scalar",
        "sel__rank_competition",
        "sel__is_selected",
    },
    "run_meta": {
        "event",
        "run_id",
        "as_of_round",
        "model__name",
        "model__params",
        "objective__name",
        "selection__name",
        "selection__params",
        "schema__version",
        "opal__version",
    },
    "label": {
        "event",
        "observed_round",
        "id",
        "y_obs",
        "src",
    },
}


def _allow_extra_columns() -> bool:
    return str(os.getenv("OPAL_LEDGER_ALLOW_EXTRA", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _assert_schema_match(existing_schema, new_schema, *, ctx: str) -> None:
    if schema_signature(existing_schema) != schema_signature(new_schema):
        existing_sig = schema_signature(existing_schema)
        new_sig = schema_signature(new_schema)
        raise LedgerError(f"{ctx} schema mismatch.\n  existing: {existing_sig}\n  new     : {new_sig}")


def _validate_columns(df: pd.DataFrame, kind: str) -> None:
    allow = ALLOW.get(kind)
    required = REQUIRED.get(kind, set())
    if allow is None:
        raise LedgerError(f"Unknown ledger event kind: {kind!r}")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise LedgerError(f"[ledger:{kind}] missing required columns: {sorted(missing)}")
    unknown = [c for c in df.columns if c not in allow]
    if unknown and not _allow_extra_columns():
        raise LedgerError(
            f"[ledger:{kind}] unknown columns {sorted(unknown)} (schema v{LEDGER_SCHEMA_VERSION}). "
            "Set OPAL_LEDGER_ALLOW_EXTRA=1 to override."
        )


def _append_parquet_dedupe(path: Path, df: pd.DataFrame, *, key: Sequence[str]) -> None:
    ensure_dir(path.parent)
    if not path.exists():
        write_parquet_df(path, df, index=False)
        return
    existing = read_parquet_df(path)
    if list(existing.columns) != list(df.columns):
        raise LedgerError(
            f"Ledger sink schema mismatch.\n  existing: {list(existing.columns)}\n  new     : {list(df.columns)}"
        )
    out = pd.concat([existing, df], ignore_index=True)
    out = out.drop_duplicates(subset=list(key), keep="last")
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_parquet_df(tmp, out, index=False)
    tmp.replace(path)


def _append_parquet_dedupe_labels(path: Path, df: pd.DataFrame) -> None:
    """
    Append label events while preserving distinct provenance.
    Dedupes only exact duplicates (including y_obs content), not just (round, id).
    """
    ensure_dir(path.parent)
    if not path.exists():
        write_parquet_df(path, df, index=False)
        return
    existing = read_parquet_df(path)
    if list(existing.columns) != list(df.columns):
        raise LedgerError(
            f"Ledger sink schema mismatch.\n  existing: {list(existing.columns)}\n  new     : {list(df.columns)}"
        )

    def _sig(val) -> str | None:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        try:
            if hasattr(val, "tolist"):
                val = val.tolist()
            return json.dumps(val, separators=(",", ":"), ensure_ascii=True)
        except Exception:
            return str(val)

    def _with_sig(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["__y_obs_sig"] = out["y_obs"].apply(_sig) if "y_obs" in out.columns else None
        return out

    def _dedupe(frame: pd.DataFrame) -> pd.DataFrame:
        key_cols = [c for c in frame.columns if c not in {"event", "y_obs"}]
        if "__y_obs_sig" not in key_cols:
            key_cols.append("__y_obs_sig")
        out = frame.drop_duplicates(subset=key_cols, keep="last")
        return out.drop(columns=["__y_obs_sig"])

    combined = pd.concat([_with_sig(existing), _with_sig(df)], ignore_index=True)
    out = _dedupe(combined)
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_parquet_df(tmp, out, index=False)
    tmp.replace(path)


def _ensure_event_value(df: pd.DataFrame, kind: str) -> None:
    if "event" not in df.columns:
        raise LedgerError(f"[ledger:{kind}] missing 'event' column.")
    vals = set(df["event"].dropna().astype(str).unique().tolist())
    if vals != {kind}:
        raise LedgerError(f"[ledger:{kind}] expected event='{kind}', found {sorted(vals)}")


@dataclass(frozen=True)
class LedgerPaths:
    predictions_dir: Path
    runs_path: Path
    labels_path: Path


class LedgerWriter:
    def __init__(self, workspace: CampaignWorkspace):
        self._ws = workspace
        self._paths = LedgerPaths(
            predictions_dir=workspace.ledger_predictions_dir,
            runs_path=workspace.ledger_runs_path,
            labels_path=workspace.ledger_labels_path,
        )

    @property
    def paths(self) -> LedgerPaths:
        return self._paths

    def append_run_pred(self, df: pd.DataFrame) -> None:
        _ensure_event_value(df, "run_pred")
        _validate_columns(df, "run_pred")
        if "run_id" in df.columns:
            run_ids = df["run_id"].astype(str).unique().tolist()
            if len(run_ids) != 1:
                raise LedgerError(f"[ledger:run_pred] expected single run_id, found {run_ids}")
        if {"run_id", "id"}.issubset(set(df.columns)):
            dup = df.duplicated(subset=["run_id", "id"]).any()
            if dup:
                raise LedgerError("[ledger:run_pred] duplicate (run_id, id) rows are not allowed.")
        ensure_dir(self._paths.predictions_dir)
        tbl = table_from_pandas(df)
        if any(self._paths.predictions_dir.rglob("*.parquet")):
            dset = dataset_from_dir(self._paths.predictions_dir)
            _assert_schema_match(dset.schema, tbl.schema, ctx="[ledger:run_pred]")
        out = self._paths.predictions_dir / f"part-{uuid4().hex}.parquet"
        write_parquet_table(out, tbl)

    def append_run_meta(self, df: pd.DataFrame) -> None:
        _ensure_event_value(df, "run_meta")
        _validate_columns(df, "run_meta")
        _append_parquet_dedupe(self._paths.runs_path, df, key=("run_id",))

    def append_label(self, df: pd.DataFrame) -> None:
        _ensure_event_value(df, "label")
        _validate_columns(df, "label")
        _append_parquet_dedupe_labels(self._paths.labels_path, df)


class LedgerReader:
    def __init__(self, workspace: CampaignWorkspace):
        self._ws = workspace
        self._paths = LedgerPaths(
            predictions_dir=workspace.ledger_predictions_dir,
            runs_path=workspace.ledger_runs_path,
            labels_path=workspace.ledger_labels_path,
        )

    @property
    def paths(self) -> LedgerPaths:
        return self._paths

    def predictions_schema_columns(self) -> list[str]:
        if not self._paths.predictions_dir.exists():
            raise LedgerError(f"Missing predictions sink: {self._paths.predictions_dir}")
        dset = dataset_from_dir(self._paths.predictions_dir)
        return [f.name for f in dset.schema]

    def read_predictions(
        self,
        *,
        columns: Optional[Iterable[str]] = None,
        round_selector: Optional[object] = None,
        run_id: Optional[str] = None,
        id_value: Optional[str] = None,
        require_run_id: bool = True,
    ) -> pd.DataFrame:
        if not self._paths.predictions_dir.exists():
            raise LedgerError(f"Missing predictions sink: {self._paths.predictions_dir}")
        round_sel = "latest" if round_selector is None else round_selector
        if require_run_id and run_id is None:
            runs_df = self.read_runs(columns=["run_id", "as_of_round"])
            if runs_df.empty:
                raise LedgerError("Run ID is required to disambiguate ledger predictions, but ledger.runs is empty.")
            if round_sel in ("unspecified", "latest"):
                round_vals = [int(runs_df["as_of_round"].max())]
            elif round_sel == "all":
                round_vals = sorted({int(x) for x in runs_df["as_of_round"].to_list()})
            elif isinstance(round_sel, list):
                round_vals = [int(x) for x in round_sel]
            else:
                round_vals = [int(round_sel)]
            df_sel = runs_df[runs_df["as_of_round"].isin(round_vals)]
            if df_sel.empty:
                raise LedgerError(f"No runs found for selected rounds {round_vals}.")
            counts = df_sel.groupby("as_of_round")["run_id"].nunique()
            multi = counts[counts > 1]
            if not multi.empty:
                raise LedgerError(
                    "Multiple run_id values found for round(s) "
                    f"{sorted(int(x) for x in multi.index.tolist())}. "
                    "Specify run_id to avoid mixing reruns."
                )
        dset = dataset_from_dir(self._paths.predictions_dir)
        cols = list(columns) if columns is not None else None

        filt = None
        pc = pyarrow_compute()
        if round_sel == "all":
            filt = None
        elif round_sel in ("latest", "unspecified"):
            t = dset.to_table(columns=["as_of_round"])
            if t.num_rows > 0:
                latest = int(pd.Series(t.column("as_of_round").to_pylist()).max())
                filt = pc.field("as_of_round") == latest
        elif isinstance(round_sel, list):
            vals = [int(x) for x in round_sel]
            filt = pc.field("as_of_round").isin(vals)
        else:
            filt = pc.field("as_of_round") == int(round_sel)
        if run_id is not None:
            pc = pyarrow_compute()
            cond = pc.field("run_id") == str(run_id)
            filt = cond if filt is None else (filt & cond)
        if id_value is not None:
            pc = pyarrow_compute()
            cond = pc.field("id") == str(id_value)
            filt = cond if filt is None else (filt & cond)

        tbl = dset.to_table(columns=cols, filter=filt)
        return tbl.to_pandas()

    def read_runs(self, *, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
        if not self._paths.runs_path.exists():
            raise LedgerError(f"Missing runs sink: {self._paths.runs_path}")
        return read_parquet_df(self._paths.runs_path, columns=list(columns) if columns else None)

    def read_labels(self, *, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
        if not self._paths.labels_path.exists():
            raise LedgerError(f"Missing labels sink: {self._paths.labels_path}")
        return read_parquet_df(self._paths.labels_path, columns=list(columns) if columns else None)
