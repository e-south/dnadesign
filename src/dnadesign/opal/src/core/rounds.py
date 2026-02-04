# ABOUTME: Resolves round selectors from state and ledger metadata.
# ABOUTME: Validates round selection inputs for OPAL commands.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/core/rounds.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from ..storage.state import CampaignState
from .utils import ExitCodes, OpalError


def _series_to_list(values) -> list:
    if hasattr(values, "to_list"):
        return values.to_list()
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def _df_is_empty(df) -> bool:
    if hasattr(df, "is_empty"):
        return bool(df.is_empty())
    if hasattr(df, "empty"):
        return bool(df.empty)
    try:
        return len(df) == 0
    except Exception:
        return True


def resolve_round_index(
    round_sel: Optional[str],
    *,
    rounds: Sequence[int],
    allow_none: bool = False,
    empty_message: str,
    param_label: str = "--round",
) -> Optional[int]:
    """
    Resolve a single round index from a selector string and known rounds.
    If allow_none=True, a None selector returns None (no filtering).
    """
    if round_sel is None and allow_none:
        return None
    if round_sel is None:
        round_sel = "latest"
    sel = str(round_sel).strip().lower()
    if not rounds:
        raise OpalError(empty_message, ExitCodes.BAD_ARGS)
    if sel in ("", "latest", "unspecified"):
        return int(max(int(r) for r in rounds))
    try:
        val = int(sel)
    except Exception as exc:
        raise OpalError(
            f"Invalid {param_label}: must be an integer or 'latest'.",
            ExitCodes.BAD_ARGS,
        ) from exc
    if val not in {int(r) for r in rounds}:
        raise OpalError(
            f"{param_label} {val} not found. Available rounds: {sorted(set(rounds))}",
            ExitCodes.BAD_ARGS,
        )
    return int(val)


def resolve_round_index_from_state(state_path: Path, round_sel: Optional[str]) -> int:
    sel = (round_sel or "latest").strip().lower()
    if sel not in ("", "latest", "unspecified"):
        try:
            val = int(sel)
        except Exception as exc:
            raise OpalError("Invalid --round: must be an integer or 'latest'.", ExitCodes.BAD_ARGS) from exc
        if state_path.exists():
            st = CampaignState.load(state_path)
            rounds = [int(r.round_index) for r in st.rounds]
            if rounds and val not in set(rounds):
                raise OpalError(
                    f"--round {val} not found in state.json. Available rounds: {sorted(set(rounds))}",
                    ExitCodes.BAD_ARGS,
                )
        return val

    if not state_path.exists():
        raise OpalError(f"state.json not found: {state_path}", ExitCodes.BAD_ARGS)
    st = CampaignState.load(state_path)
    rounds = [int(r.round_index) for r in st.rounds]
    return resolve_round_index(
        sel,
        rounds=rounds,
        allow_none=False,
        empty_message="state.json has no recorded rounds.",
        param_label="--round",
    )


def resolve_round_index_from_runs(runs_df, round_sel: Optional[str], *, allow_none: bool = False) -> Optional[int]:
    if _df_is_empty(runs_df):
        rounds: List[int] = []
    else:
        if "as_of_round" not in getattr(runs_df, "columns", []):
            raise OpalError("outputs/ledger/runs.parquet is missing as_of_round.", ExitCodes.CONTRACT_VIOLATION)
        rounds = [int(x) for x in _series_to_list(runs_df["as_of_round"]) if x is not None]
    return resolve_round_index(
        round_sel,
        rounds=rounds,
        allow_none=allow_none,
        empty_message="No runs found in outputs/ledger/runs.parquet. Run `opal run ...` first.",
        param_label="--round",
    )
