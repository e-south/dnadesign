"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/promoter_candidate_bindings/test_study_alias_registry.py

Append-only study alias registry tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings import (
    PromoterCandidateBindingsError,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings.source_registry import (
    load_source_registry,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings.study_alias_registry import (
    REGISTRY_PATH,
    REGISTRY_SCHEMA_ID,
    REGISTRY_SCHEMA_VERSION,
    STUDY_ALIAS_NAMESPACE,
    load_study_promoter_alias_registry,
    plan_study_aliases,
)


def _sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def _write_registry(repo_root: Path, assignments: list[dict[str, object]]) -> Path:
    records_path = repo_root / "data/candidates.parquet"
    records_path.parent.mkdir(parents=True)
    records = [
        {
            "id": str(row["candidate_id"]),
            "sequence": str(row.pop("sequence")),
        }
        for row in assignments
    ]
    pd.DataFrame(records).to_parquet(records_path, index=False)
    registry_path = repo_root / "record/promoter_aliases.yaml"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        yaml.safe_dump(
            {
                "schema_id": "dnadesign.study.promoter_alias_registry.v1",
                "schema_version": "1",
                "study_id": "stress_ethanol_cipro_growth",
                "alias_namespace": "study.promoter_alias",
                "format": {"prefix": "SECG", "zero_pad_width": 3},
                "candidate_table": {
                    "dataset_id": "candidate-table",
                    "records_path": "data/candidates.parquet",
                },
                "assignments": assignments,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return registry_path


def _assignment(
    ordinal: int,
    candidate_id: str,
    sequence: str,
    *,
    source_aliases: list[str] | None = None,
) -> dict[str, object]:
    return {
        "ordinal": ordinal,
        "alias": f"SECG-{ordinal:03d}",
        "candidate_id": candidate_id,
        "sequence_sha256": _sha256(sequence),
        "first_assignment": {
            "source_authority": "test_source",
            "source_id": "source-batch",
            "nomination_batch_index": 0,
            "model_as_of_round": None,
        },
        "source_aliases": source_aliases or [],
        "sequence": sequence,
    }


def _checked_in_registry_payload(repo_root: Path) -> dict[str, object]:
    payload = yaml.safe_load((repo_root / REGISTRY_PATH).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _require_repository_alias_candidate_table(path: Path) -> None:
    if path.is_file():
        return
    pytest.skip(f"requires local promoter candidate table; missing {path}")


def test_registry_loads_exact_append_only_aliases(tmp_path: Path) -> None:
    sequence_a = "ACGT" * 15
    sequence_b = "TGCA" * 15
    path = _write_registry(
        tmp_path,
        [
            _assignment(1, "candidate-a", sequence_a, source_aliases=["SECG-B0-ETH-01"]),
            _assignment(2, "candidate-b", sequence_b),
        ],
    )

    registry = load_study_promoter_alias_registry(tmp_path, registry_path=path)

    assert STUDY_ALIAS_NAMESPACE == "study.promoter_alias"
    assert registry.alias_for(candidate_id="candidate-a", sequence=sequence_a) == "SECG-001"
    assert registry.alias_for(candidate_id="candidate-b", sequence=sequence_b) == "SECG-002"
    assert registry.assignments[0].source_aliases == ("SECG-B0-ETH-01",)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda rows: rows.__setitem__(1, {**rows[1], "ordinal": 1, "alias": "SECG-001"}), "ordinals"),
        (lambda rows: rows.__setitem__(1, {**rows[1], "candidate_id": "candidate-a"}), "candidate IDs"),
        (
            lambda rows: rows.__setitem__(
                1,
                {
                    **rows[1],
                    "sequence_sha256": rows[0]["sequence_sha256"],
                    "sequence": rows[0]["sequence"],
                },
            ),
            "sequences",
        ),
        (lambda rows: rows.__setitem__(1, {**rows[1], "alias": "SECG-099"}), "does not match"),
    ],
)
def test_registry_rejects_identity_reuse(
    tmp_path: Path,
    mutator: object,
    match: str,
) -> None:
    rows = [
        _assignment(1, "candidate-a", "ACGT" * 15),
        _assignment(2, "candidate-b", "TGCA" * 15),
    ]
    mutator(rows)  # type: ignore[operator]
    path = _write_registry(tmp_path, rows)

    with pytest.raises(PromoterCandidateBindingsError, match=match):
        load_study_promoter_alias_registry(tmp_path, registry_path=path)


def test_registry_rejects_sequence_digest_drift(tmp_path: Path) -> None:
    row = _assignment(1, "candidate-a", "ACGT" * 15)
    row["sequence_sha256"] = "0" * 64
    path = _write_registry(tmp_path, [row])

    with pytest.raises(PromoterCandidateBindingsError, match="sequence digest mismatch"):
        load_study_promoter_alias_registry(tmp_path, registry_path=path)


def test_alias_plan_reuses_existing_aliases_and_appends_new_ordinals(tmp_path: Path) -> None:
    sequence_a = "ACGT" * 15
    sequence_b = "TGCA" * 15
    sequence_c = "GATC" * 15
    path = _write_registry(
        tmp_path,
        [
            _assignment(1, "candidate-a", sequence_a),
            _assignment(2, "candidate-b", sequence_b),
        ],
    )
    registry = load_study_promoter_alias_registry(tmp_path, registry_path=path)

    planned = plan_study_aliases(
        registry,
        [
            ("candidate-b", sequence_b),
            ("candidate-c", sequence_c),
        ],
    )

    assert [(row.candidate_id, row.alias, row.is_new) for row in planned] == [
        ("candidate-b", "SECG-002", False),
        ("candidate-c", "SECG-003", True),
    ]


def test_alias_plan_rejects_registered_sequence_under_new_candidate_id(tmp_path: Path) -> None:
    sequence = "ACGT" * 15
    path = _write_registry(tmp_path, [_assignment(1, "candidate-a", sequence)])
    registry = load_study_promoter_alias_registry(tmp_path, registry_path=path)

    with pytest.raises(PromoterCandidateBindingsError, match="already assigned"):
        plan_study_aliases(registry, [("candidate-b", sequence)])


def test_checked_in_registry_declares_prior_and_current_batches() -> None:
    repo_root = Path(__file__).resolve().parents[7]
    payload = _checked_in_registry_payload(repo_root)
    assignments = payload["assignments"]
    assert isinstance(assignments, list)

    assert payload["schema_id"] == REGISTRY_SCHEMA_ID
    assert payload["schema_version"] == REGISTRY_SCHEMA_VERSION
    assert payload["study_id"] == "stress_ethanol_cipro_growth"
    assert payload["alias_namespace"] == STUDY_ALIAS_NAMESPACE
    assert payload["candidate_table"] == {
        "dataset_id": "usr_prom_eth_cip_opal_candidates",
        "records_path": "src/dnadesign/usr/datasets/usr_prom_eth_cip_opal_candidates/records.parquet",
    }
    assert len(assignments) == 36
    assert [row["alias"] for row in assignments] == [f"SECG-{index:03d}" for index in range(1, 37)]
    assert f"SECG-{int(assignments[-1]['ordinal']) + 1:03d}" == "SECG-037"
    assert assignments[0]["source_aliases"] == ["SECG-B0-ETH-01"]
    assert assignments[17]["source_aliases"] == ["SECG-B0-AND-06"]
    assert assignments[18]["candidate_id"] == "0a1649a2577534c7b29604ed50cd6c8435e5caea"
    assert assignments[-1]["candidate_id"] == "7f79342eebe5afcbd32be843a9ef24fbb54d9a71"
    assert assignments[0]["first_assignment"] == {
        "source_authority": "study_batch0_selector",
        "source_id": "stress-opal-batch0-sfxi-v1",
        "nomination_batch_index": 0,
        "model_as_of_round": None,
    }
    assert assignments[18]["first_assignment"] == {
        "source_authority": "opal_selection_batch",
        "source_id": ("r0-2026-07-19T22:21:41+00:00-01784499701298508000-24e5927eb1ce4d0daf013dc0c352c584"),
        "nomination_batch_index": 1,
        "model_as_of_round": 0,
    }


def test_local_checked_in_registry_verifies_candidate_table() -> None:
    repo_root = Path(__file__).resolve().parents[7]
    payload = _checked_in_registry_payload(repo_root)
    candidate_table = payload["candidate_table"]
    assert isinstance(candidate_table, dict)
    candidate_table_path = Path(str(candidate_table["records_path"]))
    _require_repository_alias_candidate_table(repo_root / candidate_table_path)

    registry = load_study_promoter_alias_registry(repo_root)

    assert len(registry.assignments) == 36
    assert registry.next_ordinal == 37


def test_repository_alias_candidate_table_gate_skips_unmaterialized_input(tmp_path: Path) -> None:
    with pytest.raises(pytest.skip.Exception, match="requires local promoter candidate table"):
        _require_repository_alias_candidate_table(tmp_path / "records.parquet")


def test_checked_in_registry_preserves_immutable_assignment_prefixes() -> None:
    repo_root = Path(__file__).resolve().parents[7]
    assignments = _checked_in_registry_payload(repo_root)["assignments"]
    assert isinstance(assignments, list)

    expected_prefix_sha256 = {
        18: "707ac26a7cadb356e5b074fef7ccec74ad7f43ee1df7ab1d37d7c58dd4382498",  # pragma: allowlist secret
        36: "0bfb2302fb0d705ed4ebb33bcb9b95ab19b4557bdb9797a980926910ddda8037",  # pragma: allowlist secret
    }
    for prefix_length, expected_sha256 in expected_prefix_sha256.items():
        payload = json.dumps(assignments[:prefix_length], sort_keys=True, separators=(",", ":")).encode("utf-8")
        assert hashlib.sha256(payload).hexdigest() == expected_sha256


def test_checked_in_binding_registry_projects_canonical_aliases_for_reader_and_synthesis() -> None:
    repo_root = Path(__file__).resolve().parents[7]
    assignments = _checked_in_registry_payload(repo_root)["assignments"]
    assert isinstance(assignments, list)
    registry = load_source_registry(repo_root)
    study_sources = [source for source in registry.alias_sources if source.source_id == "study_promoter_aliases"]

    assert len(study_sources) == 1
    source = study_sources[0]
    assert source.adapter == "study_promoter_alias_registry.v1"
    assert source.config["registry_path"] == REGISTRY_PATH.as_posix()
    assert source.config["aliases"] == [
        {"namespace": "study.promoter_alias", "template": "{study_alias}"},
        {"namespace": "synthesis.name", "template": "{study_alias}"},
        {"namespace": "reader.design_id", "template": "pDual-10-{study_alias}"},
    ]
    assert assignments[18]["alias"] == "SECG-019"
    assert assignments[18]["candidate_id"] == "0a1649a2577534c7b29604ed50cd6c8435e5caea"
