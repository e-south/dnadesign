"""Positive oracle builder for the DenseGen TFBS learnability probe v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .contracts import parse_densegen_tfbs_row
from .manifests import label_manifest, row_universe_manifest, source_hash_manifest
from .schema import (
    TFBS_LEARNABILITY_ACTIVE_LABEL_NAMES,
    TFBS_LEARNABILITY_REQUIRED_LABEL_COLUMNS,
)

OBSERVED_LABEL_RATE_EXPECTATIONS = {
    "lexA_present": 0.488,
    "cpxR_present": 0.392,
    "baeR_present": 0.370,
    "cpxR_or_baeR_present": 0.491,
    "lexA_in_slot0": 0.204,
    "lexA_in_slot1": 0.188,
    "lexA_in_slot2": 0.193,
    "cpxR_or_baeR_in_slot0": 0.316,
    "cpxR_or_baeR_in_slot1": 0.323,
    "cpxR_or_baeR_in_slot2": 0.306,
}


@dataclass(frozen=True)
class TfbsLearnabilityOracleBuild:
    """In-memory positive-oracle build plus replay manifests."""

    labels: pd.DataFrame
    row_universe_manifest: dict[str, Any]
    label_manifest: dict[str, Any]
    source_hash_manifest: dict[str, Any]


def build_tfbs_learnability_oracle(
    candidates: pd.DataFrame,
    *,
    densegen_sidecar: pd.DataFrame | None = None,
) -> TfbsLearnabilityOracleBuild:
    """Build the v1 positive oracle label table from candidate rows and DenseGen metadata."""

    _require_columns(candidates, ("id", "sequence"), surface="candidate records")
    _reject_duplicate_ids(candidates, surface="candidate records")
    frame, row_universe = _join_candidate_sidecar(candidates, densegen_sidecar)

    label_rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        try:
            label_rows.append(parse_densegen_tfbs_row(record).to_label_row())
        except ValueError as exc:
            candidate_id = str(record.get("id") or "<missing id>")
            raise ValueError(f"{candidate_id}: TFBS learnability source contract failed: {exc}") from exc

    labels = pd.DataFrame(label_rows, columns=list(TFBS_LEARNABILITY_REQUIRED_LABEL_COLUMNS))
    _validate_label_schema(labels)
    algebra = validate_tfbs_label_algebra(labels)
    rates = observed_label_rate_summary(labels)
    return TfbsLearnabilityOracleBuild(
        labels=labels,
        row_universe_manifest=row_universe_manifest(
            candidates=candidates,
            densegen_sidecar=densegen_sidecar,
            row_universe=row_universe,
            active_row_count=len(labels),
        ),
        label_manifest=label_manifest(labels, algebra=algebra, rates=rates),
        source_hash_manifest=source_hash_manifest(candidates=candidates, densegen_sidecar=densegen_sidecar),
    )


def validate_tfbs_label_algebra(labels: pd.DataFrame) -> dict[str, Any]:
    """Validate count, presence, count-fraction, and composite-label consistency."""

    _validate_label_schema(labels)
    problems: list[str] = []
    checks = {
        "lexA_present_matches_count": labels["lexA_present"].to_numpy() == (labels["lexA_count"].to_numpy() > 0),
        "cpxR_present_matches_count": labels["cpxR_present"].to_numpy() == (labels["cpxR_count"].to_numpy() > 0),
        "baeR_present_matches_count": labels["baeR_present"].to_numpy() == (labels["baeR_count"].to_numpy() > 0),
        "cpxR_or_baeR_count_matches_sum": labels["cpxR_or_baeR_count"].to_numpy()
        == (labels["cpxR_count"].to_numpy() + labels["baeR_count"].to_numpy()),
        "cpxR_or_baeR_present_matches_count": labels["cpxR_or_baeR_present"].to_numpy()
        == (labels["cpxR_or_baeR_count"].to_numpy() > 0),
    }
    for name, values in checks.items():
        if not bool(np.all(values)):
            problems.append(name)
    fraction_specs = (
        ("lexA_count_fraction", "lexA_count"),
        ("cpxR_count_fraction", "cpxR_count"),
        ("baeR_count_fraction", "baeR_count"),
        ("cpxR_or_baeR_count_fraction", "cpxR_or_baeR_count"),
    )
    for fraction_column, count_column in fraction_specs:
        expected = labels[count_column].to_numpy(dtype=float) / 3.0
        if not bool(np.allclose(labels[fraction_column].to_numpy(dtype=float), expected)):
            problems.append(f"{fraction_column}_matches_{count_column}_over_3")
    if problems:
        raise ValueError(f"TFBS learnability label algebra failed: {problems}")
    return {
        "status": "PASS",
        "checks": sorted([*checks, *(f"{frac}_matches_{count}_over_3" for frac, count in fraction_specs)]),
    }


def observed_label_rate_summary(labels: pd.DataFrame) -> dict[str, Any]:
    """Return label-rate means for the v1 active scalar labels."""

    _validate_label_schema(labels)
    rates = {
        column: float(pd.to_numeric(labels[column], errors="raise").mean())
        for column in TFBS_LEARNABILITY_ACTIVE_LABEL_NAMES
        if column in labels.columns
    }
    return {"row_count": int(len(labels)), "rates": rates}


def validate_observed_label_rates(
    labels: pd.DataFrame,
    *,
    expected: Mapping[str, float] = OBSERVED_LABEL_RATE_EXPECTATIONS,
    tolerance: float = 0.005,
) -> dict[str, Any]:
    """Check real-data label rates against the current source-snapshot sanity ranges."""

    rates = observed_label_rate_summary(labels)["rates"]
    checks = []
    failures = []
    for label_name, expected_rate in expected.items():
        observed = float(rates[label_name])
        delta = abs(observed - float(expected_rate))
        row = {
            "label_name": label_name,
            "observed": observed,
            "expected": float(expected_rate),
            "abs_delta": delta,
            "tolerance": float(tolerance),
            "status": "PASS" if delta <= tolerance else "FAIL",
        }
        checks.append(row)
        if row["status"] != "PASS":
            failures.append(row)
    if failures:
        raise ValueError(f"observed TFBS label-rate sanity check failed: {failures}")
    return {"status": "PASS", "checks": checks}


def _join_candidate_sidecar(
    candidates: pd.DataFrame,
    densegen_sidecar: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if densegen_sidecar is None:
        _require_columns(candidates, ("densegen__used_tfbs_detail",), surface="candidate records")
        return candidates.copy(), {
            "candidate_ids": set(candidates["id"].astype(str)),
            "sidecar_ids": set(),
            "candidate_only_ids": set(),
            "sidecar_only_ids": set(),
            "source_mode": "candidate_embedded_densegen_detail",
        }

    _require_columns(densegen_sidecar, ("id", "densegen__used_tfbs_detail"), surface="DenseGen sidecar")
    _reject_duplicate_ids(densegen_sidecar, surface="DenseGen sidecar")
    candidate_ids = set(candidates["id"].astype(str))
    sidecar_ids = set(densegen_sidecar["id"].astype(str))
    candidate_only_ids = candidate_ids - sidecar_ids
    if candidate_only_ids:
        preview = sorted(candidate_only_ids)[:5]
        raise ValueError(f"candidate records missing required DenseGen sidecar metadata: {preview}")
    overlay_columns = [
        column
        for column in (
            "id",
            "densegen__used_tfbs_detail",
            "densegen__plan",
            "densegen__required_regulators",
            "densegen__sampling_library_hash",
        )
        if column in densegen_sidecar.columns
    ]
    merged = candidates.merge(
        densegen_sidecar[overlay_columns],
        on="id",
        how="left",
        suffixes=("", "__sidecar"),
    )
    for column in overlay_columns:
        if column == "id":
            continue
        sidecar_column = f"{column}__sidecar"
        if sidecar_column in merged.columns:
            if column in candidates.columns:
                conflicts = [
                    not _metadata_equal(left, right)
                    for left, right in zip(merged[column], merged[sidecar_column], strict=True)
                ]
                if any(conflicts):
                    conflict_ids = merged.loc[conflicts, "id"].astype(str).head(5).tolist()
                    raise ValueError(f"candidate records conflict with DenseGen sidecar for {column}: {conflict_ids}")
                merged[column] = merged[sidecar_column].where(merged[sidecar_column].notna(), merged[column])
            else:
                merged[column] = merged[sidecar_column]
            merged = merged.drop(columns=[sidecar_column])
    return merged, {
        "candidate_ids": candidate_ids,
        "sidecar_ids": sidecar_ids,
        "candidate_only_ids": candidate_only_ids,
        "sidecar_only_ids": sidecar_ids - candidate_ids,
        "source_mode": "densegen_sidecar_overlay",
    }


def _validate_label_schema(labels: pd.DataFrame) -> None:
    missing = [column for column in TFBS_LEARNABILITY_REQUIRED_LABEL_COLUMNS if column not in labels.columns]
    if missing:
        raise ValueError(f"TFBS learnability labels missing required column(s): {missing}")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], *, surface: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{surface} missing required column(s): {missing}")


def _reject_duplicate_ids(frame: pd.DataFrame, *, surface: str) -> None:
    if "id" not in frame.columns:
        raise ValueError(f"{surface} missing id column")
    if frame["id"].astype(str).duplicated().any():
        duplicate_ids = frame.loc[frame["id"].astype(str).duplicated(), "id"].astype(str).head(5).tolist()
        raise ValueError(f"{surface} contains duplicate id values: {duplicate_ids}")


def _metadata_equal(left: Any, right: Any) -> bool:
    if _is_missing(left) or _is_missing(right):
        return True
    if hasattr(left, "as_py"):
        left = left.as_py()
    if hasattr(right, "as_py"):
        right = right.as_py()
    if isinstance(left, np.ndarray):
        left = left.tolist()
    if isinstance(right, np.ndarray):
        right = right.tolist()
    return left == right


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if value is pd.NA:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False
