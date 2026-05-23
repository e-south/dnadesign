"""Round selector parsing and run-ledger round helpers."""

from __future__ import annotations

import polars as pl

from ...core.utils import ExitCodes, OpalError

RoundSelector = str | list[int]

_ROUND_SELECTOR_HELP = "Use 'latest', 'all', '3', '1,3', or '2-5'."


def parse_round_selector(sel: str | None) -> RoundSelector:
    if not sel:
        return "unspecified"
    s = str(sel).strip().lower()
    if s in {"latest", "all"}:
        return s
    if "-" in s:
        a, b = s.split("-", 1)
        if not a or not b:
            raise _invalid_round_selector(sel)
        try:
            return list(range(int(a), int(b) + 1))
        except Exception as exc:
            raise _invalid_round_selector(sel) from exc
    if "," in s:
        try:
            return [int(x) for x in s.split(",") if x]
        except Exception as exc:
            raise _invalid_round_selector(sel) from exc
    try:
        return [int(s)]
    except Exception as exc:
        raise _invalid_round_selector(sel) from exc


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


def available_rounds(runs_df: pl.DataFrame) -> list[int]:
    if runs_df.is_empty():
        return []
    return sorted({int(x) for x in runs_df["as_of_round"].to_list()})


def latest_round(runs_df: pl.DataFrame) -> int:
    if runs_df.is_empty():
        raise OpalError("No runs available. Run `opal run ...` first.", ExitCodes.BAD_ARGS)
    return int(max(runs_df["as_of_round"].to_list()))


def latest_run_id(runs_df: pl.DataFrame, *, round_k: int | None = None) -> str:
    if runs_df.is_empty():
        raise OpalError("No runs available. Run `opal run ...` first.", ExitCodes.BAD_ARGS)
    df = runs_df
    if round_k is not None:
        df = df.filter(pl.col("as_of_round") == int(round_k))
        if df.is_empty():
            raise OpalError(f"No runs found for round {round_k}.", ExitCodes.BAD_ARGS)
    return str(df.sort("run_id").tail(1)["run_id"][0])


def _invalid_round_selector(sel: str | None) -> OpalError:
    return OpalError(f"Invalid round selector '{sel}'. {_ROUND_SELECTOR_HELP}", ExitCodes.BAD_ARGS)
