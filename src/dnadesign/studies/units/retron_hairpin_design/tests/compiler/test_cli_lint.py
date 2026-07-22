"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/compiler/test_cli_lint.py

Lint command and compiler-spec boundary tests for the Retron MSD compiler.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dnadesign.studies.units.retron_hairpin_design.catalog.compiler_spec import RankedPrimitiveSelectorSpec
from dnadesign.studies.units.retron_hairpin_design.interfaces.cli.app import app

from ..support.cli import RUNNER
from ..support.registry import write_minimal_retron_msd_registry


def test_retron_msd_lint_cli_rejects_duplicate_registry_keys(tmp_path: Path) -> None:
    study_dir = tmp_path / "study"
    compiler_dir = study_dir / "compiler" / "catalog"
    compiler_dir.mkdir(parents=True)
    (compiler_dir / "msd_design_registry.yaml").write_text(
        """
contract: retron_msd_design_registry_v1
schema_version: 1
payloads:
  TetR:
    display_name: msd[teto]
  TetR:
    display_name: overwritten
caps:
  C172:
    source_construct: retron-172
constructs:
  pES-retron-177: {}
""",
        encoding="utf-8",
    )

    result = RUNNER.invoke(
        app,
        [
            "lint",
            "--id",
            "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM",
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert "Retron MSD registry contains duplicate mapping key: 'TetR'" in payload["error"]


def test_retron_msd_lint_cli_reports_reference_json(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)

    result = RUNNER.invoke(
        app,
        [
            "lint",
            "--id",
            "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM",
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["reference"]["msd_design_id"] == "msd-tetr-C172-LCGGT-RACAG-MXMM"
    assert payload["reference"]["scar_nick"]["route_status"] == "note_only"
    assert payload["next_step"].startswith("Input is complete")


def test_retron_msd_lint_cli_fails_fast_on_wrong_profile(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)

    result = RUNNER.invoke(
        app,
        [
            "lint",
            "--id",
            "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MMMM",
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["error_type"] == "MsdIdError"
    assert "provided profile" in payload["error"]
    assert "Correct the declared -MWX profile" in payload["next_step"]


def test_retron_msd_lint_cli_fails_fast_on_unknown_registry_part(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)

    result = RUNNER.invoke(
        app,
        [
            "lint",
            "--id",
            "pES-retron-177-msd[TetR]; C999-LCGGT-RACAG-MXMM",
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "Unknown cap 'C999'" in payload["error"]
    assert "source handles" in payload["error"]
    assert "not inferred from de033 by pattern" in payload["error"]
    assert "Route missing cap or shortening constraints to Snapback" in payload["next_step"]


def test_retron_msd_lint_cli_fails_fast_on_unknown_construct_label(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)

    result = RUNNER.invoke(
        app,
        [
            "lint",
            "--id",
            "pES-retron-typo-msd[TetR]; C172-LCGGT-RACAG-MXMM",
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "Unknown construct 'pES-retron-typo'" in payload["error"]
    assert "Plain labels must reference a registered construct" in payload["error"]
    assert "typed compiler spec with explicit payload and cap sequences" in payload["next_step"]


def test_retron_msd_lint_spec_accepts_explicit_design_parts(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    spec_path = tmp_path / "compiler_spec.yaml"
    spec_path.write_text(
        """
contract: retron_msd_compiler_spec_v1
schema_version: 1
designs:
  - construct_id: pES-retron-177
    payload_id: TetR
    cap_id: C172
    left_base: CGGT
    right_base: ACAG
payload_sequences:
  TetR: tccctatcagtgatagaga
cap_sequences:
  C172: GAGAGACTC
""",
        encoding="utf-8",
    )

    result = RUNNER.invoke(
        app,
        [
            "lint",
            "--spec",
            spec_path.as_posix(),
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["record_count"] == 1
    record = payload["records"][0]
    assert record["construct_label"] == "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM"
    assert record["source_notes"] is None
    assert record["scar_nick"]["route_status"] == "unresolved"
    assert record["scar_nick"]["nick_orientation"] is None


def test_retron_msd_lint_spec_rejects_mixed_labels_and_designs(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    spec_path = tmp_path / "compiler_spec.yaml"
    spec_path.write_text(
        """
contract: retron_msd_compiler_spec_v1
schema_version: 1
labels:
  - pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM
designs:
  - construct_id: pES-retron-177
    payload_id: TetR
    cap_id: C172
    left_base: CGGT
    right_base: ACAG
payload_sequences:
  TetR: tccctatcagtgatagaga
cap_sequences:
  C172: GAGAGACTC
""",
        encoding="utf-8",
    )

    result = RUNNER.invoke(
        app,
        [
            "lint",
            "--spec",
            spec_path.as_posix(),
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert "compiler spec must use labels or designs, not both" in payload["error"]


def test_retron_msd_lint_spec_allows_non_ligatable_s0_when_declared(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    spec_path = tmp_path / "compiler_spec.yaml"
    spec_path.write_text(
        """
contract: retron_msd_compiler_spec_v1
schema_version: 1
allow_non_ligatable_s0: true
labels:
  - pES-retron-177-msd[TetR]; C172-LCGGG-RACAG-MXMX
payload_sequences:
  TetR: tccctatcagtgatagaga
cap_sequences:
  C172: GAGAGACTC
""",
        encoding="utf-8",
    )

    result = RUNNER.invoke(
        app,
        [
            "lint",
            "--spec",
            spec_path.as_posix(),
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    record = json.loads(result.stdout)["records"][0]
    assert record["msd_design_id"] == "msd-tetr-C172-LCGGG-RACAG-MXMX"
    assert record["scar_nick"]["s0_match_required"] is False


def test_retron_msd_lint_spec_resolves_public_primitive_sources(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    snapback_run = tmp_path / "snapback_run"
    snapback_report = snapback_run / "analysis" / "solve_report.json"
    snapback_report.parent.mkdir(parents=True)
    snapback_report.write_text(
        json.dumps(
            {
                "workflow": "snapback_released_solve",
                "hits": [
                    {
                        "rank": 1,
                        "hit_kind": "exact",
                        "nickase_variant_id": "Nb.BtsI",
                        "release_variant_id": "BspQI",
                        "target_search_hit": {
                            "final_candidate": {"designed_sequence": "GAGAGACTC", "paired_bp": 3, "cap_nt": 3}
                        },
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scar_run = tmp_path / "scar_run"
    scar_table = scar_run / "export" / "table__scar_nick_candidates.csv"
    scar_table.parent.mkdir(parents=True)
    scar_table.write_text(
        "\n".join(
            [
                "rank,candidate_id,left_base,right_base,profile_s3s2s1s0,nickase_variant_id,nicked_strand,surviving_strand",
                "1,scar-rank-01,CGGT,ACAG,MXMM,Nb.BtsI,bottom,top",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spec_path = tmp_path / "compiler_spec.yaml"
    spec_path.write_text(
        f"""
contract: retron_msd_compiler_spec_v1
schema_version: 1
designs:
  - construct_id: pES-retron-public-source
    payload_id: TetR
    cap_id: C999
    stem_base_source:
      kind: scar_nick_stem_bases
      run_dir: {scar_run.as_posix()}
      selector:
        mode: rank
        rank: 1
payload_sequences:
  TetR: tccctatcagtgatagaga
cap_sequences:
  C999:
    source:
      kind: snapback_released_solve_cap
      run_dir: {snapback_run.as_posix()}
      selector:
        mode: rank
        rank: 1
""",
        encoding="utf-8",
    )

    result = RUNNER.invoke(
        app,
        [
            "lint",
            "--spec",
            spec_path.as_posix(),
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    record = json.loads(result.stdout)["records"][0]
    assert record["cap"]["id"] == "C999"
    assert record["cap"]["source_construct"] == "snapback-rank-01"
    assert record["cap"]["snapback_topology"] == {
        "kind": "snapback_foldback_geometry_v1",
        "retained_stem_span": {"start": 0, "end": 3},
        "cap_span": {"start": 3, "end": 6},
        "foldback_return_span": {"start": 6, "end": 9},
        "source": "snapback_released_solve.final_candidate",
    }


def test_retron_msd_lint_spec_accepts_unknown_manual_payload_and_cap_sequences(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    spec_path = tmp_path / "compiler_spec.yaml"
    spec_path.write_text(
        """
contract: retron_msd_compiler_spec_v1
schema_version: 1
designs:
  - construct_id: pES-retron-manual
    payload_id: UserPayload
    cap_id: Cmanual
    left_base: CGGT
    right_base: ACAG
payload_sequences:
  UserPayload: AACCGGTTAACC
cap_sequences:
  Cmanual: GAGAGACTC
""",
        encoding="utf-8",
    )

    result = RUNNER.invoke(
        app,
        [
            "lint",
            "--spec",
            spec_path.as_posix(),
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    record = json.loads(result.stdout)["records"][0]
    assert record["payload_or_target"]["id"] == "UserPayload"
    assert record["payload_or_target"]["display_name"] is None
    assert record["cap"] == {
        "id": "Cmanual",
        "source_construct": None,
        "display_name": None,
        "snapback_topology": None,
    }
    assert record["scar_nick"]["route_status"] == "unresolved"


def test_retron_msd_lint_spec_accepts_unknown_manual_label_parts_when_sequences_are_explicit(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    spec_path = tmp_path / "compiler_spec.yaml"
    spec_path.write_text(
        """
contract: retron_msd_compiler_spec_v1
schema_version: 1
labels:
  - pES-retron-manual-msd[UserPayload]; Cmanual-LCGGT-RACAG-MXMM
payload_sequences:
  UserPayload: AACCGGTTAACC
cap_sequences:
  Cmanual: GAGAGACTC
""",
        encoding="utf-8",
    )

    result = RUNNER.invoke(
        app,
        [
            "lint",
            "--spec",
            spec_path.as_posix(),
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    record = json.loads(result.stdout)["records"][0]
    assert record["construct_label"] == "pES-retron-manual-msd[UserPayload]; Cmanual-LCGGT-RACAG-MXMM"
    assert record["msd_design_id"] == "msd-userpayload-Cmanual-LCGGT-RACAG-MXMM"


def test_retron_msd_lint_spec_rejects_unknown_manual_parts_without_sequences(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    spec_path = tmp_path / "compiler_spec.yaml"
    spec_path.write_text(
        """
contract: retron_msd_compiler_spec_v1
schema_version: 1
designs:
  - construct_id: pES-retron-manual
    payload_id: UserPayload
    cap_id: Cmanual
    left_base: CGGT
    right_base: ACAG
""",
        encoding="utf-8",
    )

    result = RUNNER.invoke(
        app,
        [
            "lint",
            "--spec",
            spec_path.as_posix(),
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert "Unknown payload 'UserPayload'" in payload["error"]
    assert "explicit payload and cap sequences" in payload["next_step"]


def test_retron_msd_lint_spec_rejects_payload_primitive_source_without_public_contract(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    spec_path = tmp_path / "compiler_spec.yaml"
    spec_path.write_text(
        f"""
contract: retron_msd_compiler_spec_v1
schema_version: 1
labels:
  - pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM
payload_sequences:
  TetR:
    source:
      kind: snapback_released_solve_cap
      run_dir: {tmp_path.as_posix()}
      selector:
        mode: rank
        rank: 1
cap_sequences:
  C172: GAGAGACTC
""",
        encoding="utf-8",
    )

    result = RUNNER.invoke(
        app,
        [
            "lint",
            "--spec",
            spec_path.as_posix(),
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert "payload_sequences.TetR accepts only literal sequence" in payload["error"]
    assert "payload primitive sources need a dedicated public contract" in payload["error"]


def test_retron_msd_lint_spec_rejects_duplicate_yaml_mapping_keys(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    spec_path = tmp_path / "compiler_spec.yaml"
    spec_path.write_text(
        """
contract: retron_msd_compiler_spec_v1
schema_version: 1
labels:
  - pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM
payload_sequences:
  TetR: tccctatcagtgatagaga
payload_sequences:
  TetR: aaaa
cap_sequences:
  C172: GAGAGACTC
""",
        encoding="utf-8",
    )

    result = RUNNER.invoke(
        app,
        [
            "lint",
            "--spec",
            spec_path.as_posix(),
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert "duplicate mapping key: 'payload_sequences'" in payload["error"]


def test_retron_msd_lint_spec_rejects_duplicate_json_mapping_keys(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    spec_path = tmp_path / "compiler_spec.json"
    spec_path.write_text(
        """
{
  "contract": "retron_msd_compiler_spec_v1",
  "schema_version": 1,
  "labels": ["pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM"],
  "payload_sequences": {
    "TetR": "tccctatcagtgatagaga",
    "TetR": "aaaa"
  },
  "cap_sequences": {"C172": "GAGAGACTC"}
}
""",
        encoding="utf-8",
    )

    result = RUNNER.invoke(
        app,
        [
            "lint",
            "--spec",
            spec_path.as_posix(),
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert "duplicate mapping key: 'TetR'" in payload["error"]


def test_retron_msd_lint_spec_rejects_sequence_keys_that_collide_after_trimming(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    spec_path = tmp_path / "compiler_spec.yaml"
    spec_path.write_text(
        """
contract: retron_msd_compiler_spec_v1
schema_version: 1
labels:
  - pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM
payload_sequences:
  TetR: tccctatcagtgatagaga
  " TetR ": aaaa
cap_sequences:
  C172: GAGAGACTC
""",
        encoding="utf-8",
    )

    result = RUNNER.invoke(
        app,
        [
            "lint",
            "--spec",
            spec_path.as_posix(),
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert "payload_sequences contains duplicate key after trimming: TetR" in payload["error"]


def test_retron_msd_lint_spec_rejects_literal_cap_shorter_than_supplied_topology(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    spec_path = tmp_path / "compiler_spec.yaml"
    spec_path.write_text(
        """
contract: retron_msd_compiler_spec_v1
schema_version: 1
labels:
  - pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM
payload_sequences:
  TetR: tccctatcagtgatagaga
cap_sequences:
  C172: AGGC
""",
        encoding="utf-8",
    )

    result = RUNNER.invoke(
        app,
        [
            "lint",
            "--spec",
            spec_path.as_posix(),
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert "cap_sequences.C172.sequence is 4 nt but supplied topology ends at 9" in payload["error"]


def test_retron_msd_lint_spec_rejects_literal_cap_longer_than_supplied_topology(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    spec_path = tmp_path / "compiler_spec.yaml"
    spec_path.write_text(
        """
contract: retron_msd_compiler_spec_v1
schema_version: 1
labels:
  - pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM
payload_sequences:
  TetR: tccctatcagtgatagaga
cap_sequences:
  C172: GAGAGACTCA
""",
        encoding="utf-8",
    )

    result = RUNNER.invoke(
        app,
        [
            "lint",
            "--spec",
            spec_path.as_posix(),
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert "cap_sequences.C172.sequence is 10 nt but supplied topology ends at 9" in payload["error"]


def test_retron_msd_lint_spec_rejects_scar_nick_primitive_profile_drift(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    scar_run = tmp_path / "scar_run"
    scar_table = scar_run / "export" / "table__scar_nick_candidates.csv"
    scar_table.parent.mkdir(parents=True)
    scar_table.write_text(
        "\n".join(
            [
                "rank,candidate_id,left_base,right_base,profile_s3s2s1s0,nickase_variant_id,nicked_strand,surviving_strand",
                "1,scar-rank-01,CGGT,ACAG,MMMM,Nb.BtsI,bottom,top",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spec_path = tmp_path / "compiler_spec.yaml"
    spec_path.write_text(
        f"""
contract: retron_msd_compiler_spec_v1
schema_version: 1
designs:
  - construct_id: pES-retron-public-source
    payload_id: TetR
    cap_id: C172
    stem_base_source:
      kind: scar_nick_stem_bases
      run_dir: {scar_run.as_posix()}
      selector:
        mode: rank
        rank: 1
payload_sequences:
  TetR: tccctatcagtgatagaga
cap_sequences:
  C172: GAGAGACTC
""",
        encoding="utf-8",
    )

    result = RUNNER.invoke(
        app,
        [
            "lint",
            "--spec",
            spec_path.as_posix(),
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert "profile MMMM does not match left/right bases CGGT/ACAG" in payload["error"]


def test_retron_msd_lint_spec_refuses_non_rank_selector_before_primitive_combinatorics(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    snapback_run = tmp_path / "snapback_run"
    snapback_report = snapback_run / "analysis" / "solve_report.json"
    snapback_report.parent.mkdir(parents=True)
    snapback_report.write_text(
        json.dumps(
            {
                "workflow": "snapback_released_solve",
                "hits": [
                    {
                        "rank": 1,
                        "hit_kind": "exact",
                        "nickase_variant_id": "Nb.BtsI",
                        "release_variant_id": "BspQI",
                        "target_search_hit": {
                            "final_candidate": {"designed_sequence": "GAGAGACTC", "paired_bp": 3, "cap_nt": 3}
                        },
                    },
                    {
                        "rank": 2,
                        "hit_kind": "exact",
                        "nickase_variant_id": "Nb.BsrDI",
                        "release_variant_id": "BspQI",
                        "target_search_hit": {
                            "final_candidate": {"designed_sequence": "GGAAGATCC", "paired_bp": 3, "cap_nt": 3}
                        },
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec_path = tmp_path / "compiler_spec.yaml"
    spec_path.write_text(
        f"""
contract: retron_msd_compiler_spec_v1
schema_version: 1
labels:
  - pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM
payload_sequences:
  TetR: tccctatcagtgatagaga
cap_sequences:
  C172:
    source:
      kind: snapback_released_solve_cap
      run_dir: {snapback_run.as_posix()}
      selector:
        mode: all
""",
        encoding="utf-8",
    )

    result = RUNNER.invoke(
        app,
        [
            "lint",
            "--spec",
            spec_path.as_posix(),
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert "Input should be 'rank'" in payload["error"]
    assert "cap_sequences.C172.source.selector.mode" in payload["error"]
    assert "use selector mode=rank" in payload["next_step"]


@pytest.mark.parametrize(
    ("selector_yaml", "selector_mode"),
    [
        ("mode: ranks\n        ranks: [1]", "ranks"),
        ("mode: range\n        start_rank: 1\n        end_rank: 1", "range"),
        ("mode: all", "all"),
    ],
)
def test_retron_msd_lint_spec_refuses_non_rank_selector_modes_for_single_option(
    tmp_path: Path,
    selector_yaml: str,
    selector_mode: str,
) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    snapback_run = tmp_path / "snapback_run"
    snapback_report = snapback_run / "analysis" / "solve_report.json"
    snapback_report.parent.mkdir(parents=True)
    snapback_report.write_text(
        json.dumps(
            {
                "workflow": "snapback_released_solve",
                "hits": [
                    {
                        "rank": 1,
                        "hit_kind": "exact",
                        "nickase_variant_id": "Nb.BtsI",
                        "release_variant_id": "BspQI",
                        "target_search_hit": {
                            "final_candidate": {"designed_sequence": "GAGAGACTC", "paired_bp": 3, "cap_nt": 3}
                        },
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec_path = tmp_path / "compiler_spec.yaml"
    spec_path.write_text(
        f"""
contract: retron_msd_compiler_spec_v1
schema_version: 1
labels:
  - pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM
payload_sequences:
  TetR: tccctatcagtgatagaga
cap_sequences:
  C172:
    source:
      kind: snapback_released_solve_cap
      run_dir: {snapback_run.as_posix()}
      selector:
        {selector_yaml}
""",
        encoding="utf-8",
    )

    result = RUNNER.invoke(
        app,
        [
            "lint",
            "--spec",
            spec_path.as_posix(),
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert "Input should be 'rank'" in payload["error"]
    assert selector_mode in payload["error"]
    assert "use selector mode=rank" in payload["next_step"]


@pytest.mark.parametrize(
    ("selector_payload", "selector_mode"),
    [
        ({"mode": "ranks", "ranks": [1]}, "ranks"),
        ({"mode": "range", "start_rank": 1, "end_rank": 1}, "range"),
        ({"mode": "all"}, "all"),
    ],
)
def test_ranked_primitive_selector_spec_refuses_unsupported_modes_before_selection(
    selector_payload: dict[str, object],
    selector_mode: str,
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        RankedPrimitiveSelectorSpec.model_validate(selector_payload)

    message = str(exc_info.value)
    assert "mode" in message
    assert selector_mode in message
