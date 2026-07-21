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
    preview_promoter_candidate_bindings_from_repo,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings.study_alias_registry import (
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


def test_checked_in_registry_covers_prior_and_current_batches() -> None:
    repo_root = Path(__file__).resolve().parents[7]
    registry = load_study_promoter_alias_registry(repo_root)

    assert len(registry.assignments) == 36
    assert [row.alias for row in registry.assignments] == [f"SECG-{index:03d}" for index in range(1, 37)]
    assert registry.assignments[0].source_aliases == ("SECG-B0-ETH-01",)
    assert registry.assignments[17].source_aliases == ("SECG-B0-AND-06",)
    assert registry.assignments[18].candidate_id == "0a1649a2577534c7b29604ed50cd6c8435e5caea"
    assert registry.assignments[-1].candidate_id == "7f79342eebe5afcbd32be843a9ef24fbb54d9a71"


def test_checked_in_registry_preserves_immutable_assignment_prefixes() -> None:
    repo_root = Path(__file__).resolve().parents[7]
    registry = load_study_promoter_alias_registry(repo_root)

    expected_prefix_sha256 = {
        18: "707ac26a7cadb356e5b074fef7ccec74ad7f43ee1df7ab1d37d7c58dd4382498",  # pragma: allowlist secret
        36: "0bfb2302fb0d705ed4ebb33bcb9b95ab19b4557bdb9797a980926910ddda8037",  # pragma: allowlist secret
    }
    for prefix_length, expected_sha256 in expected_prefix_sha256.items():
        payload = json.dumps(
            [
                {
                    "ordinal": row.ordinal,
                    "alias": row.alias,
                    "candidate_id": row.candidate_id,
                    "sequence_sha256": row.sequence_sha256,
                    "first_assignment": {
                        "source_authority": row.first_assignment.source_authority,
                        "source_id": row.first_assignment.source_id,
                        "nomination_batch_index": row.first_assignment.nomination_batch_index,
                        "model_as_of_round": row.first_assignment.model_as_of_round,
                    },
                    "source_aliases": list(row.source_aliases),
                }
                for row in registry.assignments[:prefix_length]
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        assert hashlib.sha256(payload).hexdigest() == expected_sha256


def test_checked_in_bindings_project_canonical_aliases_for_reader_and_synthesis() -> None:
    repo_root = Path(__file__).resolve().parents[7]

    preview = preview_promoter_candidate_bindings_from_repo(repo_root)
    aliases = preview.bindings.loc[
        preview.bindings["candidate_id"].eq("0a1649a2577534c7b29604ed50cd6c8435e5caea"),
        ["alias_namespace", "alias"],
    ]

    assert set(map(tuple, aliases.to_records(index=False))) == {
        ("study.promoter_alias", "SECG-019"),
        ("synthesis.name", "SECG-019"),
        ("reader.design_id", "pDual-10-SECG-019"),
    }
