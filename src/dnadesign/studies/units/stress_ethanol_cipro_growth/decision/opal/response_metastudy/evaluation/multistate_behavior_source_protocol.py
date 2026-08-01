"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/multistate_behavior_source_protocol.py

Corrected-Reader source-equivalence contract for the behavior shadow.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from .multistate_behavior_protocol_fields import (
    BehaviorProtocolError,
    require_exact_fields,
    require_literal,
    require_mapping,
)


@dataclass(frozen=True)
class BehaviorSourceEquivalenceProtocol:
    current_reader_bundle_sha256: str
    prior_observation_reader_bundle_sha256: str
    prior_observation_bundle_repo_path: str
    central_label_requirement: Literal["exact_candidate_source_experiment_vector_equality"]
    reference_signal_requirement: Literal["central_b_descriptive_resampling_sd_and_all_joint_draws_exactly_zero"]


def parse_behavior_source_equivalence(value: object) -> BehaviorSourceEquivalenceProtocol:
    """Parse the source correction without rewriting immutable label artifacts."""

    payload = require_mapping(value, context="source_equivalence")
    require_exact_fields(
        payload,
        {
            "current_reader_bundle_sha256",
            "prior_observation_reader_bundle_sha256",
            "prior_observation_bundle_repo_path",
            "central_label_requirement",
            "reference_signal_requirement",
        },
        context="source_equivalence",
    )
    require_literal(
        payload,
        "central_label_requirement",
        "exact_candidate_source_experiment_vector_equality",
        context="source_equivalence",
    )
    require_literal(
        payload,
        "reference_signal_requirement",
        "central_b_descriptive_resampling_sd_and_all_joint_draws_exactly_zero",
        context="source_equivalence",
    )
    current = _digest(payload["current_reader_bundle_sha256"], field="current_reader_bundle_sha256")
    prior = _digest(
        payload["prior_observation_reader_bundle_sha256"],
        field="prior_observation_reader_bundle_sha256",
    )
    if current == prior:
        raise BehaviorProtocolError("source-equivalence Reader digests must identify distinct bundle versions.")
    bundle_path = payload["prior_observation_bundle_repo_path"]
    expected_prefix = PurePosixPath(
        "src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/response_window_observations"
    )
    if not isinstance(bundle_path, str):
        raise BehaviorProtocolError("source_equivalence.prior_observation_bundle_repo_path must be a string.")
    parsed_path = PurePosixPath(bundle_path)
    if parsed_path.is_absolute() or ".." in parsed_path.parts or parsed_path.parent != expected_prefix:
        raise BehaviorProtocolError(
            "source_equivalence.prior_observation_bundle_repo_path must name one study-owned observation bundle."
        )
    return BehaviorSourceEquivalenceProtocol(
        current_reader_bundle_sha256=current,
        prior_observation_reader_bundle_sha256=prior,
        prior_observation_bundle_repo_path=parsed_path.as_posix(),
        central_label_requirement="exact_candidate_source_experiment_vector_equality",
        reference_signal_requirement="central_b_descriptive_resampling_sd_and_all_joint_draws_exactly_zero",
    )


def _digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise BehaviorProtocolError(f"source_equivalence.{field} must be a lowercase SHA-256 digest.")
    return value


__all__ = ["BehaviorSourceEquivalenceProtocol", "parse_behavior_source_equivalence"]
