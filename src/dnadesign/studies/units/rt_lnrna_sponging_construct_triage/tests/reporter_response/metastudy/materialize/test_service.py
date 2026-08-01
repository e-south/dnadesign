"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/materialize/test_service.py

Owner-aligned materialize contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import yaml

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import DEFAULT_PROTOCOL
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.condition_ontology import (
    DEFAULT_CONDITION_ONTOLOGY,
    ReporterResponseConditionOntology,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.materialize.service import (
    materialize_record_evidence,
)

from ._support import (
    _ontology,
    _policy,
    _rehash,
    _source_closed_inputs,
)


def test_materializer_blocks_records_without_exact_quality_provenance(tmp_path: Path) -> None:
    record, bindings = _source_closed_inputs(tmp_path, quality_columns=False)

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=_ontology(),
        observation_policy=_policy(),
        protocol=DEFAULT_PROTOCOL,
    )

    assert result.status == "blocked"
    assert result.blockers == ("required_quality_columns_missing",)


def test_materializer_rejects_rehashed_noncanonical_ontology(tmp_path: Path) -> None:
    record, bindings = _source_closed_inputs(tmp_path)
    forged = ReporterResponseConditionOntology(
        ontology_id="rt_lnrna_reporter_response_conditions.v1",
        conditions=tuple(
            replace(row, treatment_label="forged baseline") if row.role == "baseline" else row
            for row in DEFAULT_CONDITION_ONTOLOGY.conditions
        ),
        sample_type_value=DEFAULT_CONDITION_ONTOLOGY.sample_type_value,
        reporter_channel=DEFAULT_CONDITION_ONTOLOGY.reporter_channel,
        normalizer_channel=DEFAULT_CONDITION_ONTOLOGY.normalizer_channel,
        ratio_channel=DEFAULT_CONDITION_ONTOLOGY.ratio_channel,
    )

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=forged,
        observation_policy=_policy(),
    )

    assert result.blockers == ("condition_ontology_digest_mismatch",)


def test_materializer_rejects_noncanonical_observation_policy(tmp_path: Path) -> None:
    record, bindings = _source_closed_inputs(tmp_path)
    changed_policy = replace(_policy(), policy_id="caller-selected-policy")

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=DEFAULT_CONDITION_ONTOLOGY,
        observation_policy=changed_policy,
    )

    assert result.blockers == ("observation_policy_digest_mismatch",)


def test_materializer_rejects_unknown_sample_condition_without_inference(tmp_path: Path) -> None:
    record, bindings = _source_closed_inputs(tmp_path)
    frame = pd.read_parquet(record.path)
    frame.loc[frame.index[0], "treatment"] = "undeclared-label"
    frame.to_parquet(record.path, index=False)
    _rehash(record, bindings)

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=_ontology(),
        observation_policy=_policy(),
        protocol=DEFAULT_PROTOCOL,
    )

    assert result.status == "blocked"
    assert result.blockers == ("sample_condition_not_declared",)


def test_checked_in_condition_ontology_matches_typed_default() -> None:
    repo_root = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
    source = (
        repo_root
        / "docs/studies/rt_lnrna_sponging_construct_triage/contexts/reporter-response-metastudy"
        / "condition-ontology.yaml"
    )

    expected = json.loads(json.dumps(asdict(DEFAULT_CONDITION_ONTOLOGY)))
    assert yaml.safe_load(source.read_text(encoding="utf-8")) == expected
