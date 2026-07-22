"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/multistate_behavior_rows.py

Identity and numeric validation for behavior scoring rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .multistate_behavior_cohort import behavior_component_columns
from .multistate_behavior_protocol import MultistateBehaviorShadowProtocol


def validated_behavior_score_rows(
    rows: pd.DataFrame,
    *,
    protocol: MultistateBehaviorShadowProtocol,
    evidence_kind: str,
) -> pd.DataFrame:
    """Validate one observed, bootstrap, or fixed-prediction score matrix."""

    components = behavior_component_columns(protocol)
    required = {"id", *components}
    if evidence_kind == "reader_joint_bootstrap":
        required.add("draw_index")
    if evidence_kind == "prediction":
        required.update({"prediction_run_id", "prediction_source_sha256"})
    if missing := sorted(required - set(rows.columns)):
        raise ValueError(f"{evidence_kind} behavior rows missing columns: {missing}")
    if rows.empty:
        raise ValueError(f"{evidence_kind} behavior rows cannot be empty.")
    result = rows.copy()
    duplicate_keys = ["id", "draw_index"] if evidence_kind == "reader_joint_bootstrap" else ["id"]
    if result.duplicated(subset=duplicate_keys).any():
        raise ValueError(f"{evidence_kind} behavior row identities must be unique.")
    if not np.isfinite(result.loc[:, list(components)].to_numpy(dtype=float)).all():
        raise ValueError(f"{evidence_kind} behavior components must be finite.")
    if evidence_kind == "prediction":
        run_ids = tuple(result["prediction_run_id"].astype(str).unique())
        if len(run_ids) != 1 or not run_ids[0].strip() or run_ids[0] != run_ids[0].strip():
            raise ValueError("prediction behavior rows must bind exactly one nonempty prediction_run_id.")
        digests = tuple(result["prediction_source_sha256"].astype(str).unique())
        if len(digests) != 1 or not _is_canonical_sha256(digests[0]):
            raise ValueError("prediction rows must bind one canonical prediction_source_sha256.")
    return result


def bootstrap_rows_with_identity(
    bootstrap_draws: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    protocol: MultistateBehaviorShadowProtocol,
) -> pd.DataFrame:
    """Attach candidate-experiment identity to validated joint draws."""

    draws = validated_behavior_score_rows(
        bootstrap_draws,
        protocol=protocol,
        evidence_kind="reader_joint_bootstrap",
    )
    identity_columns = ["candidate_id", "reader_experiment_id"]
    if missing := sorted({"id", *identity_columns} - set(observed.columns)):
        raise ValueError(f"observed behavior rows lack bootstrap identity fields: {missing}")
    identity = observed.loc[:, ["id", *identity_columns]].drop_duplicates()
    if identity["id"].astype(str).duplicated().any():
        raise ValueError("observed behavior rows map one unit id to multiple identities.")
    supplied = [column for column in identity_columns if column in draws]
    if supplied:
        checked = draws.loc[:, ["id", *supplied]].merge(
            identity.loc[:, ["id", *supplied]],
            on="id",
            how="left",
            suffixes=("_draw", "_observed"),
            validate="many_to_one",
        )
        if any(
            checked[f"{column}_observed"].isna().any()
            or not checked[f"{column}_draw"].astype(str).eq(checked[f"{column}_observed"].astype(str)).all()
            for column in supplied
        ):
            raise ValueError("bootstrap candidate-experiment identity disagrees with observed rows.")
        draws = draws.drop(columns=supplied)
    draws = draws.merge(identity, on="id", how="left", validate="many_to_one")
    if draws[identity_columns].isna().any().any():
        raise ValueError("bootstrap rows could not resolve observed candidate-experiment identity.")
    return draws


def _is_canonical_sha256(value: str) -> bool:
    digest = value.removeprefix("sha256:")
    return (
        value.startswith("sha256:")
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


__all__ = ["bootstrap_rows_with_identity", "validated_behavior_score_rows"]
