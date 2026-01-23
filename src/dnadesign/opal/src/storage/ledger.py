# ABOUTME: Handles append-only ledger sinks and schema validation for OPAL.
# ABOUTME: Reads/writes run metadata, predictions, and label events.
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
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
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


def _ensure_dataset_dir(path: Path, *, ctx: str) -> None:
    if path.exists() and not path.is_dir():
        raise LedgerError(f"{ctx} expects a directory sink at {path}; remove the file to proceed.")
    ensure_dir(path)


def _dataset_schema(path: Path):
    if any(path.rglob("*.parquet")):
        return dataset_from_dir(path).schema
    return None


def _append_dataset_table(path: Path, table, *, ctx: str) -> None:
    _ensure_dataset_dir(path, ctx=ctx)
    schema = _dataset_schema(path)
    if schema is not None:
        _assert_schema_match(schema, table.schema, ctx=ctx)
    out = path / f"part-{uuid4().hex}.parquet"
    write_parquet_table(out, table)


def _rewrite_dataset(path: Path, df: pd.DataFrame) -> None:
    tmp_dir = path.with_name(f"{path.name}.tmp-{uuid4().hex}")
    ensure_dir(tmp_dir)
    table = table_from_pandas(df)
    out = tmp_dir / f"part-{uuid4().hex}.parquet"
    write_parquet_table(out, table)
    if path.exists():
        shutil.rmtree(path)
    tmp_dir.replace(path)


def _dedupe_labels_frame(df: pd.DataFrame) -> pd.DataFrame:
    def _sig(val) -> str | None:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        try:
            if hasattr(val, "tolist"):
                val = val.tolist()
            return json.dumps(val, separators=(",", ":"), ensure_ascii=True)
        except Exception:
            return str(val)

    out = df.copy()
    out["__y_obs_sig"] = out["y_obs"].apply(_sig) if "y_obs" in out.columns else None
    key_cols = [c for c in out.columns if c not in {"event", "y_obs"}]
    if "__y_obs_sig" not in key_cols:
        key_cols.append("__y_obs_sig")
    out = out.drop_duplicates(subset=key_cols, keep="last")
    return out.drop(columns=["__y_obs_sig"])


def _append_run_meta_dataset(path: Path, df: pd.DataFrame) -> None:
    ctx = "[ledger:run_meta]"
    _ensure_dataset_dir(path, ctx=ctx)
    run_ids = set(df["run_id"].astype(str).tolist())
    schema = _dataset_schema(path)
    table = table_from_pandas(df, schema=schema) if schema is not None else table_from_pandas(df)
    if schema is not None:
        _assert_schema_match(schema, table.schema, ctx=ctx)
        existing_ids = set(read_parquet_df(path, columns=["run_id"])["run_id"].astype(str).tolist())
        if run_ids & existing_ids:
            existing = read_parquet_df(path)
            out = pd.concat([existing, df], ignore_index=True)
            out = out.drop_duplicates(subset=["run_id"], keep="last")
            _rewrite_dataset(path, out)
            return
    _append_dataset_table(path, table, ctx=ctx)


def compact_runs_ledger(path: Path) -> dict[str, int]:
    ctx = "[ledger:run_meta]"
    if not path.exists():
        raise LedgerError(f"{ctx} missing dataset: {path}")
    df = read_parquet_df(path)
    if df.empty:
        return {"rows_before": 0, "rows_after": 0, "duplicates_removed": 0}
    if "run_id" not in df.columns or "as_of_round" not in df.columns:
        raise LedgerError(f"{ctx} missing required columns run_id/as_of_round.")
    multi_round = df.groupby("run_id")["as_of_round"].nunique()
    if (multi_round > 1).any():
        bad_ids = multi_round[multi_round > 1].index.tolist()[:10]
        raise LedgerError(f"{ctx} run_id appears in multiple rounds (sample={bad_ids}).")
    before = int(len(df))
    df2 = df.drop_duplicates(subset=["run_id"], keep="last")
    after = int(len(df2))
    if before != after:
        _rewrite_dataset(path, df2)
    return {"rows_before": before, "rows_after": after, "duplicates_removed": before - after}


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
        tbl = table_from_pandas(df)
        _append_dataset_table(self._paths.predictions_dir, tbl, ctx="[ledger:run_pred]")

    def append_run_meta(self, df: pd.DataFrame) -> None:
        _ensure_event_value(df, "run_meta")
        _validate_columns(df, "run_meta")
        _append_run_meta_dataset(self._paths.runs_path, df)

    def append_label(self, df: pd.DataFrame) -> None:
        _ensure_event_value(df, "label")
        _validate_columns(df, "label")
        deduped = _dedupe_labels_frame(df)
        tbl = table_from_pandas(deduped)
        _append_dataset_table(self._paths.labels_path, tbl, ctx="[ledger:label]")


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
        runs_df = None
        if run_id is not None or (require_run_id and run_id is None):
            runs_df = self.read_runs(columns=["run_id", "as_of_round"])
            if runs_df.empty:
                raise LedgerError("outputs/ledger/runs.parquet is empty; cannot resolve run_id or rounds.")

        if run_id is not None:
            df_run = runs_df[runs_df["run_id"].astype(str) == str(run_id)] if runs_df is not None else pd.DataFrame()
            if df_run.empty:
                raise LedgerError(f"run_id {run_id!r} not found in outputs/ledger/runs.parquet.")
            run_rounds = sorted({int(x) for x in df_run["as_of_round"].to_list()})
            if len(run_rounds) > 1:
                raise LedgerError(
                    f"run_id {run_id!r} appears in multiple rounds {run_rounds}; "
                    "outputs/ledger/runs.parquet is inconsistent."
                )
            run_round = run_rounds[0]
            if round_sel in ("unspecified", "latest"):
                round_sel = run_round
            elif round_sel != "all":
                if isinstance(round_sel, list):
                    sel_rounds = [int(x) for x in round_sel]
                else:
                    sel_rounds = [int(round_sel)]
                if run_round not in sel_rounds:
                    raise LedgerError(
                        f"run_id {run_id!r} belongs to as_of_round={run_round}, "
                        f"but round_selector={round_sel!r} excludes it."
                    )

        if require_run_id and run_id is None:
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
