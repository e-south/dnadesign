"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/multistate_behavior_protocol.py

Study-owned contract for shadow evaluation of multistate response behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from yaml.constructor import ConstructorError

from ..core.contracts import StressTargetView
from .multistate_behavior_gate_protocol import (
    BehaviorCompletionGateProtocol,
    parse_behavior_completion_gate,
)
from .multistate_behavior_normalization_protocol import (
    BehaviorNormalizationProtocol,
    parse_behavior_normalization_protocol,
)
from .multistate_behavior_protocol_fields import (
    BehaviorProtocolError,
    BehaviorTargetView,
)
from .multistate_behavior_protocol_fields import (
    nonempty_string as _nonempty_string,
)
from .multistate_behavior_protocol_fields import (
    parse_state_ids as _state_ids,
)
from .multistate_behavior_protocol_fields import (
    parse_target_views as _target_views,
)
from .multistate_behavior_protocol_fields import (
    positive_float as _positive_float,
)
from .multistate_behavior_protocol_fields import (
    require_exact_fields as _require_exact_fields,
)
from .multistate_behavior_protocol_fields import (
    require_literal as _require_literal,
)
from .multistate_behavior_protocol_fields import (
    require_mapping as _mapping,
)
from .multistate_behavior_source_protocol import (
    BehaviorSourceEquivalenceProtocol,
    parse_behavior_source_equivalence,
)

SCHEMA_ID = "stress_ethanol_cipro_growth.multistate_response_behavior_shadow.v1"
SCHEMA_VERSION = "1"


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class MultistateBehaviorShadowProtocol:
    schema_id: str
    schema_version: str
    protocol_id: str
    study_id: str
    status: Literal["shadow_only"]
    source_equivalence: BehaviorSourceEquivalenceProtocol
    objective_name: str
    family_weighting: Literal["equal_one_third"]
    selector_output: Literal["behavior_score"]
    hard_bottleneck_role: Literal["diagnostic_only"]
    state_ids: tuple[str, ...]
    primary_reduction_id: str
    response_semantics: str
    fluorescence_semantics: str
    fluorescence_reference: str
    off_claim_boundary: str
    target_views: tuple[BehaviorTargetView, ...]
    normalization: BehaviorNormalizationProtocol
    campaign_activation: Literal["prohibited"]
    synthesis_authorization: Literal["prohibited"]
    promotion_gate: str
    prediction_raw_top_k: int
    ranking_method: str
    tie_semantics: str
    comparison_role: str
    comparator_objective_name: str
    comparator_score_channel: str
    comparator_direction: str
    completion_gate: BehaviorCompletionGateProtocol
    source_path: Path
    source_sha256: str

    @property
    def target_masks(self) -> dict[str, tuple[float, ...]]:
        return {view.id: view.target_mask for view in self.target_views}

    def assert_target_views(self, target_views: tuple[StressTargetView, ...]) -> None:
        observed = tuple((view.id, tuple(float(value) for value in view.target_mask)) for view in target_views)
        expected = tuple((view.id, view.target_mask) for view in self.target_views)
        if observed != expected:
            raise BehaviorProtocolError(
                "multistate behavior protocol target masks disagree with the study target views: "
                f"protocol={expected}, observed={observed}."
            )


def load_multistate_behavior_protocol(path: Path) -> MultistateBehaviorShadowProtocol:
    """Load and fully validate the persisted shadow protocol."""

    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Multistate behavior shadow protocol is missing: {source_path}")
    try:
        payload = yaml.load(source_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise BehaviorProtocolError(f"multistate behavior shadow protocol YAML is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise BehaviorProtocolError("multistate behavior shadow protocol must be a mapping.")
    _require_exact_fields(
        payload,
        {
            "schema_id",
            "schema_version",
            "protocol_id",
            "study_id",
            "status",
            "source_equivalence",
            "objective",
            "assay",
            "target_views",
            "normalization",
            "evidence_roles",
            "ranking",
            "comparator",
            "completion_gate",
            "activation",
        },
        context="protocol",
    )
    _require_literal(payload, "schema_id", SCHEMA_ID, context="protocol")
    _require_literal(payload, "schema_version", SCHEMA_VERSION, context="protocol")
    _require_literal(payload, "study_id", "stress_ethanol_cipro_growth", context="protocol")
    _require_literal(payload, "status", "shadow_only", context="protocol")
    source_equivalence = parse_behavior_source_equivalence(payload["source_equivalence"])
    protocol_id = _nonempty_string(payload["protocol_id"], field="protocol.protocol_id")
    if protocol_id != "secg_multistate_response_behavior_shadow_v1":
        raise BehaviorProtocolError("protocol.protocol_id must be 'secg_multistate_response_behavior_shadow_v1'.")

    objective = _mapping(payload["objective"], context="objective")
    _require_exact_fields(
        objective,
        {
            "name",
            "family_weighting",
            "normalized_temperature",
            "selector_output",
            "hard_bottleneck_role",
        },
        context="objective",
    )
    _require_literal(objective, "name", "multistate_response_behavior_v1", context="objective")
    _require_literal(objective, "family_weighting", "equal_one_third", context="objective")
    _require_literal(objective, "selector_output", "behavior_score", context="objective")
    _require_literal(objective, "hard_bottleneck_role", "diagnostic_only", context="objective")
    normalized_temperature = _positive_float(objective["normalized_temperature"], field="normalized_temperature")
    if normalized_temperature != 1.0:
        raise BehaviorProtocolError("normalized_temperature must be fixed at 1.0 resolution unit.")

    assay = _mapping(payload["assay"], context="assay")
    _require_exact_fields(
        assay,
        {
            "state_ids",
            "primary_reduction_id",
            "response_semantics",
            "fluorescence_semantics",
            "fluorescence_reference",
            "off_claim_boundary",
        },
        context="assay",
    )
    state_ids = _state_ids(assay["state_ids"])
    if state_ids != ("00", "10", "01", "11"):
        raise BehaviorProtocolError("assay.state_ids must be exactly ('00', '10', '01', '11') for this study.")
    target_views = _target_views(payload["target_views"], state_count=len(state_ids))
    expected_views = (
        BehaviorTargetView("ethanol", (0.0, 1.0, 0.0, 1.0)),
        BehaviorTargetView("ciprofloxacin", (0.0, 0.0, 1.0, 1.0)),
        BehaviorTargetView("and", (0.0, 0.0, 0.0, 1.0)),
    )
    if target_views != expected_views:
        raise BehaviorProtocolError(f"target_views must be exactly {expected_views}.")
    _require_literal(assay, "primary_reduction_id", "event_logmean_4_8h_post", context="assay")
    _require_literal(assay, "response_semantics", "reduced_log2_yfp_over_cfp", context="assay")
    _require_literal(
        assay,
        "fluorescence_semantics",
        "same_state_reference_relative_log2_yfp_over_od600",
        context="assay",
    )
    _require_literal(assay, "fluorescence_reference", "pDual-10", context="assay")
    _require_literal(
        assay,
        "off_claim_boundary",
        "suppression_relative_to_same_state_pDual-10_not_absolute_off",
        context="assay",
    )

    normalization_payload = _mapping(payload["normalization"], context="normalization")
    evidence = _mapping(payload["evidence_roles"], context="evidence_roles")
    normalization = parse_behavior_normalization_protocol(
        normalization_payload,
        evidence,
        normalized_temperature=normalized_temperature,
    )
    activation = _mapping(payload["activation"], context="activation")
    _require_exact_fields(activation, {"campaign", "synthesis", "promotion_gate"}, context="activation")
    _require_literal(activation, "campaign", "prohibited", context="activation")
    _require_literal(activation, "synthesis", "prohibited", context="activation")
    _require_literal(
        activation,
        "promotion_gate",
        "explicit_study_adjudication_after_shadow_evidence",
        context="activation",
    )
    ranking = _mapping(payload["ranking"], context="ranking")
    _require_exact_fields(
        ranking,
        {"prediction_raw_top_k", "method", "tie_semantics", "comparison_role"},
        context="ranking",
    )
    if ranking["prediction_raw_top_k"] != 6:
        raise BehaviorProtocolError("ranking.prediction_raw_top_k must be 6.")
    _require_literal(
        ranking,
        "method",
        "descending_score_then_ascending_candidate_id",
        context="ranking",
    )
    _require_literal(ranking, "tie_semantics", "ordinal_rank_with_id_tiebreak", context="ranking")
    _require_literal(
        ranking,
        "comparison_role",
        "fixed_prediction_raw_candidate_ranking_no_sequence_allocation",
        context="ranking",
    )
    comparator = _mapping(payload["comparator"], context="comparator")
    _require_exact_fields(
        comparator,
        {"objective_name", "score_channel", "direction"},
        context="comparator",
    )
    _require_literal(
        comparator,
        "objective_name",
        "response_magnitude_feasibility_v1",
        context="comparator",
    )
    _require_literal(comparator, "score_channel", "feasibility_margin", context="comparator")
    _require_literal(comparator, "direction", "maximize", context="comparator")

    completion_gate = parse_behavior_completion_gate(payload["completion_gate"])
    return MultistateBehaviorShadowProtocol(
        schema_id=SCHEMA_ID,
        schema_version=SCHEMA_VERSION,
        protocol_id=protocol_id,
        study_id="stress_ethanol_cipro_growth",
        status="shadow_only",
        source_equivalence=source_equivalence,
        objective_name="multistate_response_behavior_v1",
        family_weighting="equal_one_third",
        selector_output="behavior_score",
        hard_bottleneck_role="diagnostic_only",
        state_ids=state_ids,
        primary_reduction_id=_nonempty_string(assay["primary_reduction_id"], field="primary_reduction_id"),
        response_semantics=_nonempty_string(assay["response_semantics"], field="response_semantics"),
        fluorescence_semantics=_nonempty_string(assay["fluorescence_semantics"], field="fluorescence_semantics"),
        fluorescence_reference=_nonempty_string(assay["fluorescence_reference"], field="fluorescence_reference"),
        off_claim_boundary=_nonempty_string(assay["off_claim_boundary"], field="off_claim_boundary"),
        target_views=target_views,
        normalization=normalization,
        campaign_activation="prohibited",
        synthesis_authorization="prohibited",
        promotion_gate=_nonempty_string(activation["promotion_gate"], field="promotion_gate"),
        prediction_raw_top_k=6,
        ranking_method="descending_score_then_ascending_candidate_id",
        tie_semantics="ordinal_rank_with_id_tiebreak",
        comparison_role="fixed_prediction_raw_candidate_ranking_no_sequence_allocation",
        comparator_objective_name="response_magnitude_feasibility_v1",
        comparator_score_channel="feasibility_margin",
        comparator_direction="maximize",
        completion_gate=completion_gate,
        source_path=source_path,
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
    )


__all__ = [
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "BehaviorNormalizationProtocol",
    "BehaviorProtocolError",
    "BehaviorTargetView",
    "MultistateBehaviorShadowProtocol",
    "load_multistate_behavior_protocol",
]
