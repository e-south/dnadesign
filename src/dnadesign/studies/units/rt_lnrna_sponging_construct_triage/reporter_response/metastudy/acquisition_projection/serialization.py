"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/acquisition_projection/serialization.py

Exact JSON serialization and parsing for acquisition projections.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from ..contracts._values import MetastudyContractError
from ._values import exact_object
from .contracts import (
    AcquisitionContribution,
    AcquisitionCoordinate,
    AcquisitionMetricProjection,
    AcquisitionProjection,
    coordinate_payload,
)

_RAW_FIELDS = {"rfp", "od600", "rfp_over_od600"}


def acquisition_projection_payload(
    projection: AcquisitionProjection,
    include_digest: bool = True,
) -> dict[str, object]:
    if not isinstance(projection, AcquisitionProjection):
        raise MetastudyContractError("acquisition projection must be typed")
    payload: dict[str, object] = {
        "contract_id": projection.contract_id,
        "selected_reduction": list(projection.selected_reduction),
        "coordinates": json.loads(
            json.dumps([coordinate_payload(row) for row in projection.coordinates], allow_nan=False)
        ),
        "projection_digest": projection.projection_digest,
    }
    if not include_digest:
        payload.pop("projection_digest", None)
    return payload


def validate_acquisition_projection_payload(value: object) -> AcquisitionProjection:
    root = exact_object(
        value,
        {"contract_id", "selected_reduction", "coordinates", "projection_digest"},
        "acquisition projection",
    )
    coordinate_rows = root["coordinates"]
    if not isinstance(coordinate_rows, list) or not coordinate_rows:
        raise MetastudyContractError("acquisition projection coordinates must be a non-empty array")
    reduction = root["selected_reduction"]
    if not isinstance(reduction, list):
        raise MetastudyContractError("acquisition selected_reduction must be an array")
    projection = AcquisitionProjection(
        contract_id=root["contract_id"],
        selected_reduction=tuple(reduction),
        coordinates=tuple(_parse_coordinate(row) for row in coordinate_rows),
    )
    if root["projection_digest"] != projection.projection_digest:
        raise MetastudyContractError("acquisition projection digest changed")
    return projection


def _parse_coordinate(value: object) -> AcquisitionCoordinate:
    fields = set(AcquisitionCoordinate.__dataclass_fields__)
    payload = value if isinstance(value, Mapping) else {}
    if set(payload) == fields:
        raw_available = True
    elif set(payload) == fields - _RAW_FIELDS:
        raw_available = False
    else:
        raise MetastudyContractError("acquisition coordinate fields do not match the exact contract")
    row = dict(payload)
    contributions = row["contributions"]
    if not isinstance(contributions, list):
        raise MetastudyContractError("acquisition contributions must be an array")
    contribution_fields = set(AcquisitionContribution.__dataclass_fields__)
    expected_fields = contribution_fields if raw_available else contribution_fields - _RAW_FIELDS
    return AcquisitionCoordinate(
        **{
            **{
                key: row[key]
                for key in fields
                - _RAW_FIELDS
                - {"acquisition_ids", "contributions", "normalized_reporter_response", "relative_od"}
            },
            "acquisition_ids": tuple(row["acquisition_ids"]),
            "contributions": tuple(
                AcquisitionContribution(
                    **{
                        **exact_object(item, expected_fields, "acquisition contribution"),
                        "declared_biological_replicate_ids": tuple(item["declared_biological_replicate_ids"]),
                        **({} if raw_available else {"rfp": None, "od600": None, "rfp_over_od600": None}),
                    }
                )
                for item in contributions
            ),
            "rfp": _parse_metric(row["rfp"]) if raw_available else None,
            "od600": _parse_metric(row["od600"]) if raw_available else None,
            "rfp_over_od600": _parse_metric(row["rfp_over_od600"]) if raw_available else None,
            "normalized_reporter_response": _parse_optional_metric(row["normalized_reporter_response"]),
            "relative_od": _parse_optional_metric(row["relative_od"]),
        }
    )


def _parse_metric(value: object) -> AcquisitionMetricProjection:
    payload = exact_object(value, set(AcquisitionMetricProjection.__dataclass_fields__), "acquisition metric")
    payload["leave_one_acquisition_out_estimates"] = tuple(payload["leave_one_acquisition_out_estimates"])
    return AcquisitionMetricProjection(**payload)


def _parse_optional_metric(value: object) -> AcquisitionMetricProjection | None:
    return None if value is None else _parse_metric(value)


__all__ = ["acquisition_projection_payload", "validate_acquisition_projection_payload"]
