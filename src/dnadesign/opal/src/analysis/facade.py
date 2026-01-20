"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/facade.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import polars as pl

from ..config import load_config
from ..config.types import RootConfig
from ..core.config_resolve import resolve_campaign_config_path
from ..core.utils import ExitCodes, OpalError
from ..storage.store_factory import records_store_from_config
from ..storage.workspace import CampaignWorkspace

RoundSelector = Union[str, List[int]]


def parse_round_selector(sel: Optional[str]) -> RoundSelector:
    if not sel:
        return "unspecified"
    s = str(sel).strip().lower()
    if s in {"latest", "all"}:
        return s
    if "-" in s:
        a, b = s.split("-", 1)
        if not a or not b:
            raise OpalError(
                f"Invalid round selector '{sel}'. Use 'latest', 'all', '3', '1,3', or '2-5'.",
                ExitCodes.BAD_ARGS,
            )
        try:
            return list(range(int(a), int(b) + 1))
        except Exception as exc:
            raise OpalError(
                f"Invalid round selector '{sel}'. Use 'latest', 'all', '3', '1,3', or '2-5'.",
                ExitCodes.BAD_ARGS,
            ) from exc
    if "," in s:
        try:
            return [int(x) for x in s.split(",") if x]
        except Exception as exc:
            raise OpalError(
                f"Invalid round selector '{sel}'. Use 'latest', 'all', '3', '1,3', or '2-5'.",
                ExitCodes.BAD_ARGS,
            ) from exc
    try:
        return [int(s)]
    except Exception as exc:
        raise OpalError(
            f"Invalid round selector '{sel}'. Use 'latest', 'all', '3', '1,3', or '2-5'.",
            ExitCodes.BAD_ARGS,
        ) from exc


def round_suffix(rounds: RoundSelector) -> str:
    if rounds == "unspecified":
        return ""
    if rounds == "latest":
        return "_rlatest"
    if rounds == "all":
        return "_rall"
    if isinstance(rounds, list) and len(rounds) == 1:
        return f"_r{rounds[0]}"
    if isinstance(rounds, list):
        return f"_r{','.join(map(str, rounds))}"
    return ""


def require_columns(df: pl.DataFrame, columns: Iterable[str], *, ctx: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise OpalError(
            f"{ctx}: missing required columns {sorted(missing)}",
            ExitCodes.CONTRACT_VIOLATION,
        )


def available_rounds(runs_df: pl.DataFrame) -> List[int]:
    if runs_df.is_empty():
        return []
    return sorted({int(x) for x in runs_df["as_of_round"].to_list()})


def latest_round(runs_df: pl.DataFrame) -> int:
    if runs_df.is_empty():
        raise OpalError("No runs available. Run `opal run ...` first.", ExitCodes.BAD_ARGS)
    return int(max(runs_df["as_of_round"].to_list()))


def latest_run_id(runs_df: pl.DataFrame, *, round_k: Optional[int] = None) -> str:
    if runs_df.is_empty():
        raise OpalError("No runs available. Run `opal run ...` first.", ExitCodes.BAD_ARGS)
    df = runs_df
    if round_k is not None:
        df = df.filter(pl.col("as_of_round") == int(round_k))
        if df.is_empty():
            raise OpalError(f"No runs found for round {round_k}.", ExitCodes.BAD_ARGS)
    # Use last run_id (lexicographic) as a stable "latest" within round
    return str(df.sort("run_id").tail(1)["run_id"][0])


def _pred_parquet_paths(pred_dir: Path) -> list[Path]:
    return sorted({p for p in pred_dir.rglob("*.parquet") if p.is_file()})


def _ensure_predictions_dir(pred_dir: Path) -> None:
    if not pred_dir.exists():
        raise OpalError(
            f"Missing predictions sink: {pred_dir}. Run `opal run -c <campaign.yaml> --round <k>` first.",
            ExitCodes.BAD_ARGS,
        )
    if not _pred_parquet_paths(pred_dir):
        raise OpalError(
            f"Predictions sink is empty: {pred_dir}. Run `opal run -c <campaign.yaml> --round <k>` first.",
            ExitCodes.BAD_ARGS,
        )


def _ensure_runs_path(runs_path: Path) -> None:
    if not runs_path.exists():
        raise OpalError(
            f"Missing runs sink: {runs_path}. Run `opal run -c <campaign.yaml> --round <k>` first.",
            ExitCodes.BAD_ARGS,
        )


def _ensure_labels_path(labels_path: Path) -> None:
    if not labels_path.exists():
        raise OpalError(
            f"Missing labels sink: {labels_path}. Run `opal ingest-y -c <campaign.yaml> --round <k>` first.",
            ExitCodes.BAD_ARGS,
        )


def scan_predictions(pred_dir: Path) -> pl.LazyFrame:
    _ensure_predictions_dir(pred_dir)
    files = _pred_parquet_paths(pred_dir)
    return pl.scan_parquet([str(path) for path in files])


def read_runs(runs_path: Path) -> pl.DataFrame:
    _ensure_runs_path(runs_path)
    return pl.read_parquet(runs_path)


def read_labels(labels_path: Path) -> pl.DataFrame:
    _ensure_labels_path(labels_path)
    return pl.read_parquet(labels_path)


def ensure_predictions_dir(pred_dir: Path) -> None:
    _ensure_predictions_dir(pred_dir)


def ensure_runs_path(runs_path: Path) -> None:
    _ensure_runs_path(runs_path)


def ensure_labels_path(labels_path: Path) -> None:
    _ensure_labels_path(labels_path)


def _apply_round_filter(
    lf: pl.LazyFrame,
    *,
    round_selector: Optional[RoundSelector],
    runs_df: Optional[pl.DataFrame] = None,
) -> pl.LazyFrame:
    if round_selector in (None, "unspecified", "latest"):
        if runs_df is not None and not runs_df.is_empty():
            latest = latest_round(runs_df)
            return lf.filter(pl.col("as_of_round") == latest)
        latest = lf.select(pl.col("as_of_round").max()).collect()
        if latest.is_empty():
            return lf
        latest_val = int(latest["as_of_round"][0])
        return lf.filter(pl.col("as_of_round") == latest_val)
    if round_selector == "all":
        return lf
    if isinstance(round_selector, list):
        return lf.filter(pl.col("as_of_round").is_in([int(x) for x in round_selector]))
    return lf.filter(pl.col("as_of_round") == int(round_selector))


def _selected_rounds(
    round_selector: Optional[RoundSelector],
    runs_df: Optional[pl.DataFrame],
) -> list[int]:
    if runs_df is None or runs_df.is_empty():
        return []
    if round_selector in (None, "unspecified", "latest"):
        return [latest_round(runs_df)]
    if round_selector == "all":
        return sorted({int(x) for x in runs_df["as_of_round"].to_list()})
    if isinstance(round_selector, list):
        return [int(x) for x in round_selector]
    return [int(round_selector)]


def _resolve_round_for_run_id(run_id: str, runs_df: pl.DataFrame) -> int:
    if runs_df.is_empty():
        raise OpalError("ledger.runs is empty; cannot resolve run_id.", ExitCodes.BAD_ARGS)
    if "run_id" not in runs_df.columns or "as_of_round" not in runs_df.columns:
        raise OpalError("ledger.runs missing required columns (run_id, as_of_round).", ExitCodes.BAD_ARGS)
    df = runs_df.filter(pl.col("run_id") == str(run_id)).select(pl.col("as_of_round").drop_nulls().unique())
    if df.is_empty():
        raise OpalError(f"run_id {run_id!r} not found in ledger.runs.", ExitCodes.BAD_ARGS)
    rounds = sorted({int(x) for x in df.to_series().to_list()})
    if len(rounds) > 1:
        raise OpalError(
            f"run_id {run_id!r} appears in multiple rounds {rounds}; ledger.runs is inconsistent.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return rounds[0]


def _require_run_id_if_ambiguous(
    *,
    runs_df: Optional[pl.DataFrame],
    round_selector: Optional[RoundSelector],
    run_id: Optional[str],
    require_run_id: bool,
) -> None:
    if not require_run_id or run_id is not None:
        return
    if runs_df is None or runs_df.is_empty():
        raise OpalError(
            "Run ID is required to disambiguate ledger predictions, but ledger.runs is missing or empty. "
            "Provide run_id explicitly (e.g., --run-id) or generate ledger runs first.",
            ExitCodes.BAD_ARGS,
        )
    rounds = _selected_rounds(round_selector, runs_df)
    if not rounds:
        raise OpalError(
            "Run ID is required to disambiguate ledger predictions, but no runs were found for selection.",
            ExitCodes.BAD_ARGS,
        )
    df = runs_df.filter(pl.col("as_of_round").is_in(rounds))
    if df.is_empty():
        raise OpalError(
            f"No runs found in ledger.runs for selected rounds {rounds}.",
            ExitCodes.BAD_ARGS,
        )
    counts = df.group_by("as_of_round").agg(pl.col("run_id").n_unique().alias("n_runs"))
    multi = counts.filter(pl.col("n_runs") > 1).select(pl.col("as_of_round")).to_series().to_list()
    if multi:
        raise OpalError(
            "Multiple run_id values found for round(s) "
            f"{sorted(int(x) for x in multi)}. Specify run_id to avoid mixing reruns "
            "(ledger is append-only; rerunning a round creates a new run_id). "
            "Use `opal runs list` or `opal status --with-ledger` to find valid run_id values.",
            ExitCodes.BAD_ARGS,
        )


def read_predictions(
    pred_dir: Path,
    *,
    columns: Optional[Sequence[str]] = None,
    round_selector: Optional[RoundSelector] = None,
    run_id: Optional[str] = None,
    runs_df: Optional[pl.DataFrame] = None,
    allow_missing: bool = False,
    require_run_id: bool = True,
) -> pl.DataFrame:
    lf = scan_predictions(pred_dir)
    if run_id is not None:
        if runs_df is None or runs_df.is_empty():
            raise OpalError(
                "run_id was provided but ledger.runs context is missing or empty. "
                "Pass runs_df (ledger.runs) or call CampaignAnalysis.read_predictions "
                "so OPAL can resolve run_id → as_of_round. "
                "Use `opal runs list` or `opal status --with-ledger` to find valid run_id values.",
                ExitCodes.BAD_ARGS,
            )
        run_round = _resolve_round_for_run_id(str(run_id), runs_df)
        if round_selector in (None, "unspecified", "latest"):
            round_selector = [run_round]
        elif round_selector != "all":
            selected = _selected_rounds(round_selector, runs_df)
            if run_round not in selected:
                raise OpalError(
                    f"run_id {run_id!r} belongs to as_of_round={run_round}, "
                    f"but round_selector={round_selector!r} excludes it.",
                    ExitCodes.BAD_ARGS,
                )
    _require_run_id_if_ambiguous(
        runs_df=runs_df,
        round_selector=round_selector,
        run_id=run_id,
        require_run_id=require_run_id,
    )
    if columns:
        want = [c for c in columns if c]
        try:
            schema_cols = set(lf.collect_schema().keys())
        except Exception:
            schema_cols = set(lf.schema.keys())
        missing = [c for c in want if c not in schema_cols]
        if missing and not allow_missing:
            raise OpalError(
                f"ledger.predictions is missing columns: {sorted(missing)}",
                ExitCodes.CONTRACT_VIOLATION,
            )
        if allow_missing:
            want = [c for c in want if c in schema_cols]
        lf = lf.select(want)
    lf = _apply_round_filter(lf, round_selector=round_selector, runs_df=runs_df)
    if run_id is not None:
        lf = lf.filter(pl.col("run_id") == str(run_id))
    return lf.collect()


def load_predictions_with_setpoint(
    outputs_dir: Path,
    base_columns: Iterable[str],
    *,
    round_selector: Optional[RoundSelector] = None,
    run_id: Optional[str] = None,
    require_run_id: bool = True,
) -> pl.DataFrame:
    """
    Read predictions (ledger.predictions) and join setpoint from ledger.runs.
    Returns a polars DataFrame with an added obj__diag__setpoint column.
    """
    pred_dir = outputs_dir / "ledger.predictions"
    runs_path = outputs_dir / "ledger.runs.parquet"
    runs_df = read_runs(runs_path)

    want = set(map(str, base_columns)) | {"run_id"}
    df = read_predictions(
        pred_dir,
        columns=sorted(want),
        round_selector=round_selector,
        run_id=run_id,
        runs_df=runs_df,
        require_run_id=require_run_id,
    )
    if df.is_empty():
        raise OpalError("ledger.predictions had zero rows after projection.", ExitCodes.BAD_ARGS)

    def _extract_setpoint(obj) -> Optional[List[float]]:
        vec = (obj or {}).get("setpoint_vector")
        if vec is None:
            return None
        try:
            vals = [float(x) for x in vec]
        except Exception:
            return None
        if not vals:
            return None
        if not all(math.isfinite(v) for v in vals):
            return None
        return vals

    if "objective__params" not in runs_df.columns:
        raise OpalError(
            "ledger.runs is missing objective__params (cannot resolve setpoints).",
            ExitCodes.BAD_ARGS,
        )

    meta = runs_df.select(["run_id", "objective__params"]).with_columns(
        pl.col("objective__params")
        .map_elements(_extract_setpoint, return_dtype=pl.List(pl.Float64))
        .alias("obj__diag__setpoint")
    )
    out = df.join(meta.select(["run_id", "obj__diag__setpoint"]), on="run_id", how="left")
    if out["obj__diag__setpoint"].drop_nulls().is_empty():
        raise OpalError(
            "Could not resolve setpoint for any rows: run_meta lacks objective__params.setpoint_vector.",
            ExitCodes.BAD_ARGS,
        )
    missing = (
        out.filter(pl.col("obj__diag__setpoint").is_null()).select(pl.col("run_id").unique()).to_series().to_list()
    )
    if missing:
        raise OpalError(
            f"Missing objective__params.setpoint_vector for run_id(s): {sorted(missing)}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    setpoints = [tuple(sp) for sp in out["obj__diag__setpoint"].drop_nulls().to_list()]
    unique = {sp for sp in setpoints}
    if len(unique) > 1:
        raise OpalError(
            f"Multiple setpoint vectors found for selected rows: {sorted(unique)}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return out


@dataclass(frozen=True)
class CampaignAnalysis:
    config_path: Path
    config: RootConfig
    workspace: CampaignWorkspace

    @classmethod
    def from_config_path(cls, config_opt: Optional[Path], *, allow_dir: bool = False) -> "CampaignAnalysis":
        cfg_path = resolve_campaign_config_path(config_opt, allow_dir=allow_dir)
        cfg = load_config(cfg_path)
        ws = CampaignWorkspace.from_config(cfg, cfg_path)
        return cls(config_path=cfg_path, config=cfg, workspace=ws)

    def records_store(self):
        return records_store_from_config(self.config)

    def read_runs(self) -> pl.DataFrame:
        return read_runs(self.workspace.ledger_runs_path)

    def read_labels(self) -> pl.DataFrame:
        return read_labels(self.workspace.ledger_labels_path)

    def scan_predictions(self) -> pl.LazyFrame:
        return scan_predictions(self.workspace.ledger_predictions_dir)

    def read_predictions(
        self,
        *,
        columns: Optional[Sequence[str]] = None,
        round_selector: Optional[RoundSelector] = None,
        run_id: Optional[str] = None,
        runs_df: Optional[pl.DataFrame] = None,
        allow_missing: bool = False,
        require_run_id: bool = True,
    ) -> pl.DataFrame:
        if runs_df is None:
            runs_df = self.read_runs()
        return read_predictions(
            self.workspace.ledger_predictions_dir,
            columns=columns,
            round_selector=round_selector,
            run_id=run_id,
            runs_df=runs_df,
            allow_missing=allow_missing,
            require_run_id=require_run_id,
        )

    def predictions_with_setpoint(
        self,
        columns: Iterable[str],
        *,
        round_selector: Optional[RoundSelector] = None,
        run_id: Optional[str] = None,
        require_run_id: bool = True,
    ) -> pl.DataFrame:
        return load_predictions_with_setpoint(
            self.workspace.outputs_dir,
            columns,
            round_selector=round_selector,
            run_id=run_id,
            require_run_id=require_run_id,
        )
