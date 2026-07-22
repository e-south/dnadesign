"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/cli/test_cli_selection_set.py

Public selection-view and selection-batch inspection contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from dnadesign.opal import load_selection_batch, load_selection_set
from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.core.utils import OpalError, file_sha256
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_ledger, write_records, write_state


def _setup_workspace(tmp_path: Path, *, run_ids: tuple[str, ...] = ("run-0",)) -> tuple[Path, Path]:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)
    selections, batch = _write_selection_artifacts(workdir)
    artifacts = {
        "selection/selections.parquet": (file_sha256(selections), str(selections)),
        "selection/selection_batch.parquet": (file_sha256(batch), str(batch)),
    }
    for run_id in run_ids:
        write_state(workdir, records_path=records, run_id=run_id, round_index=0)
        write_ledger(
            workdir,
            run_id=run_id,
            round_index=0,
            artifact_paths_and_hashes=artifacts,
        )
    return workdir, campaign


def _write_selection_artifacts(workdir: Path) -> tuple[Path, Path]:
    selection_dir = workdir / "outputs" / "rounds" / "round_0" / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    selections = selection_dir / "selections.parquet"
    batch = selection_dir / "selection_batch.parquet"
    pd.DataFrame(
        [
            {
                "selection_view_id": "primary",
                "id": "a",
                "sequence": "AAA",
                "selection_batch_key": "a",
                "deduplicate_by": "id",
                "score": 0.1,
                "selection_score": 0.1,
                "rank_competition": 1,
                "rank_ordinal": 1,
                "score_ref": "primary/sfxi",
                "allocation_slot": None,
                "selection_origin": "preferred_top_k",
                "run_id": "run-0",
                "as_of_round": 0,
                "campaign_slug": "demo",
            }
        ]
    ).to_parquet(selections, index=False)
    pd.DataFrame([_logical_union_row(candidate_id="a", rank=1)]).to_parquet(batch, index=False)
    return selections, batch


def _logical_union_row(*, candidate_id: str, rank: int) -> dict[str, Any]:
    return {
        "run_id": "run-0",
        "as_of_round": 0,
        "campaign_slug": "demo",
        "id": candidate_id,
        "selection_batch_key": candidate_id,
        "deduplicate_by": "id",
        "selection_view_ids": ["primary"],
        "selection_memberships": [
            {
                "selection_view_id": "primary",
                "rank": rank,
                "rank_ordinal": rank,
                "score": float(rank) / 10.0,
                "selection_score": float(rank) / 10.0,
                "score_ref": "primary/sfxi",
                "allocation_slot": None,
                "selection_origin": "preferred_top_k",
            }
        ],
        "preferred_view_ids": ["primary"],
        "allocation_view_id": None,
        "allocation_slot": None,
    }


def _allocated_row(*, candidate_id: str, rank: int, slot: int) -> dict[str, Any]:
    row = _logical_union_row(candidate_id=candidate_id, rank=rank)
    row["allocation_view_id"] = "primary"
    row["allocation_slot"] = slot
    row["selection_memberships"][0]["allocation_slot"] = slot
    return row


def _write_batch_rows(batch_path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_parquet(batch_path, index=False)


def _write_verified_batch_rows(batch_path: Path, rows: list[dict[str, Any]]) -> None:
    """Write mutually bound batch/selection fixtures and refresh their run-ledger digests."""

    _write_batch_rows(batch_path, rows)
    selection_rows: list[dict[str, Any]] = []
    allocation_trace_rows: list[dict[str, Any]] = []
    decision_order = 0
    for row in rows:
        candidate_id = str(row["id"])
        for membership in row["selection_memberships"]:
            selection_rows.append(
                {
                    "run_id": row["run_id"],
                    "as_of_round": row["as_of_round"],
                    "campaign_slug": row["campaign_slug"],
                    "selection_view_id": membership["selection_view_id"],
                    "id": candidate_id,
                    "sequence": {"a": "AAA", "b": "BBB"}.get(candidate_id, candidate_id),
                    "selection_batch_key": row["selection_batch_key"],
                    "deduplicate_by": row["deduplicate_by"],
                    "rank_competition": membership["rank"],
                    "rank_ordinal": membership["rank_ordinal"],
                    "score": membership["score"],
                    "selection_score": membership["selection_score"],
                    "score_ref": membership["score_ref"],
                    "allocation_slot": membership["allocation_slot"],
                    "selection_origin": membership["selection_origin"],
                }
            )
            if row["allocation_view_id"] is not None:
                decision_order += 1
                allocation_trace_rows.append(
                    {
                        "run_id": row["run_id"],
                        "as_of_round": row["as_of_round"],
                        "campaign_slug": row["campaign_slug"],
                        "decision_order": decision_order,
                        "selection_view_id": membership["selection_view_id"],
                        "allocation_slot": membership["allocation_slot"],
                        "decision": "allocated",
                        "id": candidate_id,
                        "selection_batch_key": row["selection_batch_key"],
                        "deduplicate_by": row["deduplicate_by"],
                        "rank_ordinal": membership["rank_ordinal"],
                        "rank_competition": membership["rank"],
                        "score": membership["score"],
                        "selection_score": membership["selection_score"],
                        "score_ref": membership["score_ref"],
                        "selection_origin": membership["selection_origin"],
                        "conflicting_selection_view_id": None,
                        "conflicting_allocation_slot": None,
                    }
                )
    selections_path = batch_path.with_name("selections.parquet")
    pd.DataFrame(selection_rows).to_parquet(selections_path, index=False)
    artifacts = {
        "selection/selections.parquet": (file_sha256(selections_path), str(selections_path)),
        "selection/selection_batch.parquet": (file_sha256(batch_path), str(batch_path)),
    }
    if allocation_trace_rows:
        allocation_trace_path = batch_path.with_name("allocation_trace.parquet")
        pd.DataFrame(allocation_trace_rows).to_parquet(allocation_trace_path, index=False)
        artifacts["selection/allocation_trace.parquet"] = (
            file_sha256(allocation_trace_path),
            str(allocation_trace_path),
        )
    workdir = batch_path.parents[4]
    shutil.rmtree(workdir / "outputs" / "ledger")
    write_ledger(
        workdir,
        run_id=str(rows[0]["run_id"]),
        round_index=int(rows[0]["as_of_round"]),
        artifact_paths_and_hashes=artifacts,
    )


def _enable_single_view_allocation(campaign: Path) -> None:
    payload = yaml.safe_load(campaign.read_text())
    selection_params = payload["selection_views"][0]["selection"]["params"]
    selection_params.update(
        {
            "top_k": 2,
            "tie_handling": "ordinal",
            "require_exact_top_k": True,
        }
    )
    payload["selection_batch"] = {
        "deduplicate_by": "id",
        "expected_unique_count": 2,
        "allocation": {
            "strategy": "round_robin_next_best_unallocated",
            "view_priority": ["primary"],
        },
    }
    campaign.write_text(yaml.safe_dump(payload, sort_keys=False))


def _enable_second_view(campaign: Path) -> None:
    payload = yaml.safe_load(campaign.read_text())
    second_view = json.loads(json.dumps(payload["selection_views"][0]))
    second_view["id"] = "secondary"
    payload["selection_views"].append(second_view)
    campaign.write_text(yaml.safe_dump(payload, sort_keys=False))


def _enable_two_view_allocation(campaign: Path) -> None:
    _enable_second_view(campaign)
    payload = yaml.safe_load(campaign.read_text())
    for view in payload["selection_views"]:
        view["selection"]["params"].update(
            {
                "top_k": 1,
                "tie_handling": "ordinal",
                "require_exact_top_k": True,
            }
        )
    payload["selection_batch"] = {
        "deduplicate_by": "id",
        "expected_unique_count": 2,
        "allocation": {
            "strategy": "round_robin_next_best_unallocated",
            "view_priority": ["primary", "secondary"],
        },
    }
    campaign.write_text(yaml.safe_dump(payload, sort_keys=False))


def _enable_sequence_deduplication(campaign: Path) -> None:
    payload = yaml.safe_load(campaign.read_text())
    payload["selection_batch"] = {"deduplicate_by": "sequence"}
    campaign.write_text(yaml.safe_dump(payload, sort_keys=False))


def test_load_selection_set_requires_and_projects_named_view(tmp_path: Path) -> None:
    workdir, campaign = _setup_workspace(tmp_path)
    selection_path, _ = _write_selection_artifacts(workdir)

    payload = load_selection_set(campaign, selection_view_id="primary", round_selector="latest")

    assert payload["schema_version"] == "opal.selection_set.v2"
    assert payload["selection_view_id"] == "primary"
    assert payload["selection_path"] == str(selection_path)
    assert payload["verification"]["status"] == "pass"
    assert payload["rows"] == [
        {
            "id": "a",
            "sequence": "AAA",
            "selection_rank": 1,
            "rank_competition": 1,
            "score": 0.1,
            "selection_score": 0.1,
            "run_id": "run-0",
            "as_of_round": 0,
        }
    ]


def test_load_selection_batch_returns_validated_v3_logical_union_in_competition_rank_order(tmp_path: Path) -> None:
    workdir, campaign = _setup_workspace(tmp_path)
    _, batch_path = _write_selection_artifacts(workdir)
    _write_verified_batch_rows(
        batch_path,
        [
            _logical_union_row(candidate_id="a", rank=2),
            _logical_union_row(candidate_id="b", rank=1),
        ],
    )

    payload = load_selection_batch(campaign, round_selector="latest")

    assert payload["schema_version"] == "opal.selection_batch.v3"
    assert payload["selection_batch_path"] == str(batch_path)
    assert payload["deduplicate_by"] == "id"
    assert payload["allocation_strategy"] == "logical_union"
    assert payload["unique_count"] == 2
    assert [row["id"] for row in payload["rows"]] == ["b", "a"]
    assert payload["rows"][0]["selection_view_ids"] == ["primary"]


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("missing_campaign_slug", "missing required columns"),
        ("wrong_run_id", "mixed or unexpected run_id"),
        ("unknown_selection_view", "unknown selection_view_ids"),
        ("membership_view_mismatch", "must exactly match selection_view_ids"),
        ("duplicate_membership_view", "duplicate membership view IDs"),
        ("zero_competition_rank", "rank.*must be a positive integer"),
        ("missing_score_ref", "missing required fields"),
        ("invalid_origin", "unsupported selection_origin"),
        ("unknown_preferred_view", "unknown preferred_view_ids"),
        ("logical_union_allocation_slot", "logical-union rows cannot declare allocation"),
    ],
)
def test_load_selection_batch_rejects_corrupt_nested_contracts(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    workdir, campaign = _setup_workspace(tmp_path)
    _, batch_path = _write_selection_artifacts(workdir)
    row = _logical_union_row(candidate_id="a", rank=1)
    if corruption == "missing_campaign_slug":
        row.pop("campaign_slug")
    elif corruption == "wrong_run_id":
        row["run_id"] = "run-other"
    elif corruption == "unknown_selection_view":
        row["selection_view_ids"] = ["unknown"]
        row["selection_memberships"][0]["selection_view_id"] = "unknown"
    elif corruption == "membership_view_mismatch":
        row["selection_memberships"][0]["selection_view_id"] = "other"
    elif corruption == "duplicate_membership_view":
        row["selection_memberships"].append(dict(row["selection_memberships"][0]))
    elif corruption == "zero_competition_rank":
        row["selection_memberships"][0]["rank"] = 0
    elif corruption == "missing_score_ref":
        row["selection_memberships"][0].pop("score_ref")
    elif corruption == "invalid_origin":
        row["selection_memberships"][0]["selection_origin"] = "implicit_fill"
    elif corruption == "unknown_preferred_view":
        row["preferred_view_ids"] = ["unknown"]
    elif corruption == "logical_union_allocation_slot":
        row["allocation_slot"] = 1
        row["selection_memberships"][0]["allocation_slot"] = 1
    else:  # pragma: no cover - parametrization contract
        raise AssertionError(corruption)
    _write_batch_rows(batch_path, [row])

    with pytest.raises(OpalError, match=message):
        load_selection_batch(campaign, round_selector="latest", selection_batch_path=batch_path)


def test_load_selection_batch_treats_membership_view_order_as_set_identity(tmp_path: Path) -> None:
    workdir, campaign = _setup_workspace(tmp_path)
    _, batch_path = _write_selection_artifacts(workdir)
    _enable_second_view(campaign)
    row = _logical_union_row(candidate_id="a", rank=1)
    secondary_membership = dict(row["selection_memberships"][0])
    secondary_membership.update(
        {
            "selection_view_id": "secondary",
            "rank": 2,
            "rank_ordinal": 2,
            "score_ref": "secondary/sfxi",
        }
    )
    row["selection_view_ids"] = ["secondary", "primary"]
    row["selection_memberships"] = [row["selection_memberships"][0], secondary_membership]
    row["preferred_view_ids"] = ["primary", "secondary"]
    _write_verified_batch_rows(batch_path, [row])

    payload = load_selection_batch(campaign, round_selector="latest")

    assert set(payload["rows"][0]["selection_view_ids"]) == {"primary", "secondary"}
    assert {item["selection_view_id"] for item in payload["rows"][0]["selection_memberships"]} == {
        "primary",
        "secondary",
    }


def test_load_selection_batch_rejects_duplicate_candidate_ids_independently_of_batch_keys(tmp_path: Path) -> None:
    workdir, campaign = _setup_workspace(tmp_path)
    _, batch_path = _write_selection_artifacts(workdir)
    _enable_sequence_deduplication(campaign)
    first = _logical_union_row(candidate_id="a", rank=1)
    second = _logical_union_row(candidate_id="a", rank=2)
    for row, key in ((first, "AAA"), (second, "BBB")):
        row["deduplicate_by"] = "sequence"
        row["selection_batch_key"] = key
    _write_batch_rows(batch_path, [first, second])

    with pytest.raises(OpalError, match="duplicate candidate IDs"):
        load_selection_batch(campaign, round_selector="latest", selection_batch_path=batch_path)


def test_load_selection_batch_validates_explicit_allocation_slots_and_order(tmp_path: Path) -> None:
    workdir, campaign = _setup_workspace(tmp_path)
    _, batch_path = _write_selection_artifacts(workdir)
    _enable_single_view_allocation(campaign)
    _write_verified_batch_rows(
        batch_path,
        [
            _allocated_row(candidate_id="a", rank=3, slot=2),
            _allocated_row(candidate_id="b", rank=1, slot=1),
        ],
    )

    payload = load_selection_batch(campaign, round_selector="latest")

    assert payload["allocation_strategy"] == "round_robin_next_best_unallocated"
    assert [row["id"] for row in payload["rows"]] == ["b", "a"]

    duplicate_slot_rows = [
        _allocated_row(candidate_id="a", rank=1, slot=1),
        _allocated_row(candidate_id="b", rank=2, slot=1),
    ]
    _write_batch_rows(batch_path, duplicate_slot_rows)
    with pytest.raises(OpalError, match="duplicate allocation slot"):
        load_selection_batch(campaign, round_selector="latest", selection_batch_path=batch_path)


def test_load_selection_batch_rejects_run_ledger_digest_drift(tmp_path: Path) -> None:
    workdir, campaign = _setup_workspace(tmp_path)
    batch_path = workdir / "outputs" / "rounds" / "round_0" / "selection" / "selection_batch.parquet"
    drifted = pd.read_parquet(batch_path)
    drifted.loc[0, "selection_batch_key"] = "altered"
    drifted.to_parquet(batch_path, index=False)

    with pytest.raises(OpalError, match="selection/selection_batch.parquet.*SHA-256 mismatch"):
        load_selection_batch(campaign, round_selector="latest")


def test_load_selection_batch_explicit_override_still_requires_selection_provenance(tmp_path: Path) -> None:
    _, campaign = _setup_workspace(tmp_path)
    override_path = tmp_path / "selection_batch_override.parquet"
    row = _logical_union_row(candidate_id="a", rank=1)
    row["selection_memberships"][0]["score"] = 9.0
    row["selection_memberships"][0]["selection_score"] = 8.0
    _write_batch_rows(override_path, [row])

    with pytest.raises(OpalError, match="membership provenance disagrees"):
        load_selection_batch(
            campaign,
            round_selector="latest",
            selection_batch_path=override_path,
        )


def test_load_selection_batch_explicit_override_cannot_fabricate_deduplication_key(tmp_path: Path) -> None:
    workdir, campaign = _setup_workspace(tmp_path)
    _, batch_path = _write_selection_artifacts(workdir)
    _enable_sequence_deduplication(campaign)
    row = _logical_union_row(candidate_id="a", rank=1)
    row["deduplicate_by"] = "sequence"
    row["selection_batch_key"] = "AAA"
    _write_verified_batch_rows(batch_path, [row])
    row["selection_batch_key"] = "FABRICATED"
    override_path = tmp_path / "selection_batch_key_override.parquet"
    _write_batch_rows(override_path, [row])

    with pytest.raises(OpalError, match="keys disagree with the run selection artifact"):
        load_selection_batch(
            campaign,
            round_selector="latest",
            selection_batch_path=override_path,
        )


def test_load_selection_batch_explicit_override_cannot_fabricate_preferred_views(tmp_path: Path) -> None:
    workdir, campaign = _setup_workspace(tmp_path)
    _, batch_path = _write_selection_artifacts(workdir)
    _enable_two_view_allocation(campaign)
    primary = _allocated_row(candidate_id="a", rank=1, slot=1)
    secondary = _allocated_row(candidate_id="b", rank=1, slot=1)
    secondary["selection_view_ids"] = ["secondary"]
    secondary["selection_memberships"][0]["selection_view_id"] = "secondary"
    secondary["selection_memberships"][0]["score_ref"] = "secondary/sfxi"
    secondary["preferred_view_ids"] = ["secondary"]
    secondary["allocation_view_id"] = "secondary"
    _write_verified_batch_rows(batch_path, [primary, secondary])
    primary["preferred_view_ids"] = ["primary", "secondary"]
    override_path = tmp_path / "selection_batch_preference_override.parquet"
    _write_batch_rows(override_path, [primary, secondary])

    with pytest.raises(OpalError, match="preferred_view_ids disagree with the run allocation trace"):
        load_selection_batch(
            campaign,
            round_selector="latest",
            selection_batch_path=override_path,
        )


def test_load_selection_batch_rejects_selection_artifact_digest_drift(tmp_path: Path) -> None:
    workdir, campaign = _setup_workspace(tmp_path)
    selections_path = workdir / "outputs" / "rounds" / "round_0" / "selection" / "selections.parquet"
    drifted = pd.read_parquet(selections_path)
    drifted.loc[0, "score"] = 9.0
    drifted.to_parquet(selections_path, index=False)

    with pytest.raises(OpalError, match="selection/selections.parquet.*SHA-256 mismatch"):
        load_selection_batch(campaign, round_selector="latest")


def test_selection_set_and_batch_cli_show_and_export(tmp_path: Path) -> None:
    workdir, campaign = _setup_workspace(tmp_path)
    _write_selection_artifacts(workdir)
    output_csv = tmp_path / "selection_set.csv"
    runner = CliRunner()
    app = _build()

    shown = runner.invoke(
        app,
        [
            "--no-color",
            "selection-set",
            "show",
            "-c",
            str(campaign),
            "--round",
            "latest",
            "--view",
            "primary",
            "--json",
        ],
    )
    assert shown.exit_code == 0, shown.stdout
    assert json.loads(shown.stdout)["rows"][0]["id"] == "a"

    exported = runner.invoke(
        app,
        [
            "--no-color",
            "selection-set",
            "export",
            "-c",
            str(campaign),
            "--view",
            "primary",
            "--out",
            str(output_csv),
            "--json",
        ],
    )
    assert exported.exit_code == 0, exported.stdout
    assert pd.read_csv(output_csv)["id"].tolist() == ["a"]

    batch = runner.invoke(
        app,
        ["--no-color", "selection-batch", "show", "-c", str(campaign), "--round", "latest", "--json"],
    )
    assert batch.exit_code == 0, batch.stdout
    assert json.loads(batch.stdout)["unique_count"] == 1


def test_selection_set_rejects_ambiguous_reruns_without_run_id(tmp_path: Path) -> None:
    workdir, campaign = _setup_workspace(tmp_path, run_ids=("run-a", "run-b"))
    _write_selection_artifacts(workdir)

    result = CliRunner().invoke(
        _build(),
        ["--no-color", "selection-set", "show", "-c", str(campaign), "--view", "primary", "--round", "0", "--json"],
    )

    assert result.exit_code != 0
    assert "Multiple run_id values found for round 0" in json.loads(result.stdout)["error"]["message"]


def test_selection_set_does_not_infer_missing_artifact_reference(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)
    write_state(workdir, records_path=records, run_id="run-0", round_index=0)
    write_ledger(workdir, run_id="run-0", round_index=0)
    _write_selection_artifacts(workdir)

    result = CliRunner().invoke(
        _build(),
        ["--no-color", "selection-set", "show", "-c", str(campaign), "--view", "primary", "--json"],
    )

    assert result.exit_code != 0
    assert (
        "missing the selection/selections.parquet artifact reference" in json.loads(result.stdout)["error"]["message"]
    )
