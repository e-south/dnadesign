"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/preflight.py

Preflight validation for run and related commands.

Runs assertive checks before training/scoring:
- essential columns present,
- representation column exists, numeric, fixed width,
- labels available ≤ round k,
- labeled rows have X; candidate universe is well-formed,
- optional uniformity enforcement for bio_type/alphabet.

Returns a compact PreflightReport (warnings, x_dim, counts). Any violation
raises OpalError with a specific exit code and actionable message.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from .data_access import ESSENTIAL_COLS, RecordsStore
from .utils import ExitCodes, OpalError


@dataclass
class PreflightReport:
    warnings: List[str] = field(default_factory=list)
    x_dim: int = 0
    n_labels: int = 0
    n_candidates: int = 0


def ensure_required_on_init(df: pd.DataFrame, require_bio_alphabet: bool) -> None:
    missing = [c for c in ESSENTIAL_COLS if c not in df.columns]
    if missing:
        raise OpalError(
            f"Missing essential columns in records: {missing}",
            ExitCodes.CONTRACT_VIOLATION,
        )
    if require_bio_alphabet:
        if df["bio_type"].isna().any() or df["alphabet"].isna().any():
            raise OpalError(
                "bio_type/alphabet must be present for all rows.",
                ExitCodes.CONTRACT_VIOLATION,
            )


def validate_x_column_fixed_dim(
    store: RecordsStore, df: pd.DataFrame, ids: List[str]
) -> int:
    sub = df.loc[df["id"].isin(ids), store.x_col]
    # coerce + check in small pass
    from .transforms.registry import get_transform

    t = get_transform(store.transform_name, store.transform_params)
    X, d = t.transform(sub)
    # confirm all have same dim:
    if X.shape[1] != d:
        raise OpalError(
            "Internal error: dim mismatch after transform", ExitCodes.INTERNAL_ERROR
        )
    return d


def preflight_run(
    store: RecordsStore,
    df: pd.DataFrame,
    round_k: int,
    fail_on_mixed_bio_alphabet: bool = True,
) -> PreflightReport:
    rep = PreflightReport()

    # essential validation
    ensure_required_on_init(df, require_bio_alphabet=True)

    # effective labels
    labels_eff = store.effective_labels_latest_only(df, round_k)
    rep.n_labels = int(len(labels_eff))
    if rep.n_labels == 0:
        raise OpalError(
            f"No labels available at or before round {round_k}.",
            ExitCodes.CONTRACT_VIOLATION,
        )

    # X present for all labeled
    miss_x = df.set_index("id").loc[labels_eff["id"], store.x_col].isna()
    if miss_x.any():
        missing_ids = labels_eff["id"][miss_x.values].head(10).tolist()
        raise OpalError(
            f"Some labeled ids missing {store.x_col}: {missing_ids}",
            ExitCodes.CONTRACT_VIOLATION,
        )

    # candidate universe
    cand = store.candidate_universe(df, round_k)
    rep.n_candidates = int(len(cand))
    if rep.n_candidates == 0:
        rep.warnings.append("Candidate universe is empty at this round.")

    # check bio/alphabet uniformity
    if fail_on_mixed_bio_alphabet:
        store.check_biotype_alphabet_uniformity(df, labels_eff["id"].tolist())
        store.check_biotype_alphabet_uniformity(df, cand["id"].tolist())

    # fixed dimension checks (labeled + candidates)
    ids_to_check = labels_eff["id"].tolist() + cand["id"].tolist()
    if ids_to_check:
        rep.x_dim = validate_x_column_fixed_dim(store, df, ids_to_check)

    return rep
