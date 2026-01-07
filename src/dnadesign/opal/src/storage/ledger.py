"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/storage/ledger.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence
from uuid import uuid4

import pandas as pd

from .. import LEDGER_SCHEMA_VERSION
from ..core.stderr_filter import maybe_install_pyarrow_sysctl_filter
from ..core.utils import LedgerError, ensure_dir
from .workspace import CampaignWorkspace


def _import_pyarrow():
    maybe_install_pyarrow_sysctl_filter()
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from pyarrow import dataset as ds

    return pa, pc, pq, ds


pa, pc, pq, ds = _import_pyarrow()

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
    return str(os.getenv("OPAL_LEDGER_ALLOW_EXTRA", "")).strip().lower() in ("1", "true", "yes", "on")


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
        df.to_parquet(path, index=False)
        return
    existing = pd.read_parquet(path)
    if list(existing.columns) != list(df.columns):
        raise LedgerError(
            f"Ledger sink schema mismatch.\n  existing: {list(existing.columns)}\n  new     : {list(df.columns)}"
        )
    out = pd.concat([existing, df], ignore_index=True)
    out = out.drop_duplicates(subset=list(key), keep="last")
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.to_parquet(tmp, index=False)
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
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        out = self._paths.predictions_dir / f"part-{uuid4().hex}.parquet"
        pq.write_table(tbl, out)

    def append_run_meta(self, df: pd.DataFrame) -> None:
        _ensure_event_value(df, "run_meta")
        _validate_columns(df, "run_meta")
        _append_parquet_dedupe(self._paths.runs_path, df, key=("run_id",))

    def append_label(self, df: pd.DataFrame) -> None:
        _ensure_event_value(df, "label")
        _validate_columns(df, "label")
        _append_parquet_dedupe(self._paths.labels_path, df, key=("observed_round", "id"))


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
        dset = ds.dataset(str(self._paths.predictions_dir))
        return [f.name for f in dset.schema]

    def read_predictions(
        self,
        *,
        columns: Optional[Iterable[str]] = None,
        round_selector: Optional[object] = None,
        run_id: Optional[str] = None,
        id_value: Optional[str] = None,
    ) -> pd.DataFrame:
        if not self._paths.predictions_dir.exists():
            raise LedgerError(f"Missing predictions sink: {self._paths.predictions_dir}")
        dset = ds.dataset(str(self._paths.predictions_dir))
        cols = list(columns) if columns is not None else None

        filt = None
        if round_selector is not None:
            if round_selector == "all":
                filt = None
            elif round_selector in ("latest", "unspecified"):
                t = dset.to_table(columns=["as_of_round"])
                if t.num_rows > 0:
                    latest = int(pd.Series(t.column("as_of_round").to_pylist()).max())
                    filt = pc.field("as_of_round") == latest
            elif isinstance(round_selector, list):
                vals = [int(x) for x in round_selector]
                filt = pc.field("as_of_round").isin(vals)
            else:
                filt = pc.field("as_of_round") == int(round_selector)
        if run_id is not None:
            cond = pc.field("run_id") == str(run_id)
            filt = cond if filt is None else (filt & cond)
        if id_value is not None:
            cond = pc.field("id") == str(id_value)
            filt = cond if filt is None else (filt & cond)

        tbl = dset.to_table(columns=cols, filter=filt)
        return tbl.to_pandas()

    def read_runs(self, *, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
        if not self._paths.runs_path.exists():
            raise LedgerError(f"Missing runs sink: {self._paths.runs_path}")
        return pd.read_parquet(self._paths.runs_path, columns=list(columns) if columns else None)

    def read_labels(self, *, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
        if not self._paths.labels_path.exists():
            raise LedgerError(f"Missing labels sink: {self._paths.labels_path}")
        return pd.read_parquet(self._paths.labels_path, columns=list(columns) if columns else None)
