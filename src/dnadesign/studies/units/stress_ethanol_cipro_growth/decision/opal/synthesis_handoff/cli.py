"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff/cli.py

CLI for stress-study OPAL synthesis handoff fixture flows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings import (
    PROMOTER_ALIAS_REGISTRY_PATH,
)

from .batch0_source import (
    DEFAULT_BATCH0_BATCH_ID,
    DEFAULT_BATCH0_SELECTION_CONFIG,
    build_batch0_selected_candidates,
)
from .campaigns import DEFAULT_STRESS_OPAL_CAMPAIGN_CONFIG
from .contracts import (
    SelectedCandidate,
    SelectionMembership,
    optional_nonnegative_integer,
    require_nonnegative_integer,
    require_positive_integer,
)
from .exports import (
    campaign_synthesis_artifact_paths,
    campaign_synthesis_output_dir,
    render_campaign_scoped_exports,
    source_evidence_synthesis_output_dir,
)
from .manifest import build_synthesis_manifest
from .opal_round_source import selected_candidates_from_opal_round
from .records import (
    DEFAULT_SYNTHESIS_HANDOFF_RECORD,
    SynthesisHandoffRecord,
    apply_handoff_record_lifecycle,
    get_synthesis_handoff_record,
    handoff_record_payload,
    source_mode_from_handoff_record,
    validate_manifest_against_handoff_record,
)
from .strategy import load_cloning_strategy


def _selected_from_csv(path: Path) -> list[SelectedCandidate]:
    rows = pd.read_csv(path, dtype=str).fillna("")
    required = [
        "campaign_slug",
        "selection_memberships",
        "as_of_round",
        "run_id",
        "selection_rank",
        "id",
        "sequence",
        "synthesis_name",
    ]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError("selected-csv missing required columns: " + ", ".join(missing))
    selected: list[SelectedCandidate] = []
    for idx, row in rows.iterrows():
        try:
            memberships_raw = json.loads(str(row["selection_memberships"]))
        except json.JSONDecodeError as exc:
            raise ValueError(f"selected-csv row {idx} has invalid selection_memberships JSON") from exc
        if not isinstance(memberships_raw, list):
            raise ValueError(f"selected-csv row {idx} selection_memberships must be a JSON list")
        selected.append(
            SelectedCandidate(
                campaign_slug=str(row["campaign_slug"]),
                selection_memberships=tuple(SelectionMembership.from_mapping(item) for item in memberships_raw),
                as_of_round=require_nonnegative_integer(row["as_of_round"], field="as_of_round"),
                run_id=str(row["run_id"]),
                selection_rank=require_positive_integer(row["selection_rank"], field="selection_rank"),
                id=str(row["id"]),
                sequence=str(row["sequence"]),
                synthesis_name=str(row["synthesis_name"]),
                selection_source=str(row.get("selection_source", "selected_csv")),
                selection_epoch=str(row.get("selection_epoch", "external_selected_csv") or "external_selected_csv"),
                assay_batch_index=optional_nonnegative_integer(
                    row.get("assay_batch_index"),
                    field="assay_batch_index",
                ),
                model_as_of_round=optional_nonnegative_integer(
                    row.get("model_as_of_round"),
                    field="model_as_of_round",
                ),
            )
        )
    return selected


def _repo_root_from(path: Path) -> Path:
    for parent in [path.resolve(), *path.resolve().parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"could not resolve repo root from {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and validate stress-study OPAL synthesis handoff artifacts.",
    )
    parser.add_argument(
        "--source",
        choices=("selected-csv", "batch0", "opal-round"),
        help="Selected-candidate source. Omit for a no-write strategy dry run.",
    )
    parser.add_argument(
        "--handoff-id",
        help="Checked-in synthesis handoff lifecycle id. Resolves source, batch id, and record checks.",
    )
    parser.add_argument(
        "--record-yaml",
        type=Path,
        help="Checked-in synthesis handoff lifecycle record. Defaults to the stress-study record plane.",
    )
    parser.add_argument("--selected-csv", type=Path, help="CSV with selected candidate rows.")
    parser.add_argument(
        "--batch0-config",
        type=Path,
        default=DEFAULT_BATCH0_SELECTION_CONFIG,
        help="Batch-0 sampling config used when --source batch0.",
    )
    parser.add_argument(
        "--campaign-config",
        type=Path,
        help=("Unified OPAL campaign config used when --source opal-round. Defaults to the active stress campaign."),
    )
    parser.add_argument(
        "--promoter-alias-registry",
        type=Path,
        default=PROMOTER_ALIAS_REGISTRY_PATH,
        help="Study-owned append-only promoter alias registry used by OPAL-round handoffs.",
    )
    parser.add_argument(
        "--round",
        "--as-of-round",
        dest="as_of_round",
        type=int,
        help="OPAL as_of_round used when --source opal-round.",
    )
    parser.add_argument(
        "--run-id",
        help=(
            "Optional unified OPAL run_id for raw --source opal-round. "
            "For --handoff-id, record run IDs in the lifecycle record instead."
        ),
    )
    parser.add_argument(
        "--strategy-yaml",
        type=Path,
        default=Path(__file__).parent / "configs" / "stress_promoter_insert_v1.yaml",
        help="YAML cloning strategy config.",
    )
    parser.add_argument("--batch-id", help="Synthesis batch identifier.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        help=(
            "Repository root for campaign-scoped outputs. Defaults from the selected source path or working directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for generated artifacts. Campaign-scoped sources write campaign subdirectories here.",
    )
    parser.add_argument("--write", action="store_true", help="Write manifest CSV and Azenta workbook outputs.")
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    return parser


def _summary(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        print(f"{key}: {value}")


def _parser_error(parser: argparse.ArgumentParser, args: argparse.Namespace, message: str) -> None:
    if bool(getattr(args, "json", False)):
        print(
            json.dumps(
                {
                    "status": "error",
                    "context": "synthesis_handoff",
                    "error": {"message": message},
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)
    parser.error(message)


def _record_path_for(repo_root: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        if explicit_path.is_absolute():
            return explicit_path
        return repo_root / explicit_path
    return repo_root / DEFAULT_SYNTHESIS_HANDOFF_RECORD


def _batch0_candidate_records_path(*, config_path: Path, repo_root: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"batch-0 config must be a mapping: {config_path}")
    candidate_table = raw.get("candidate_feature_table")
    if not isinstance(candidate_table, dict):
        raise ValueError(f"batch-0 config missing candidate_feature_table mapping: {config_path}")
    records_path = candidate_table.get("records_path")
    if records_path is None or not str(records_path).strip():
        raise ValueError(f"batch-0 config missing candidate_feature_table.records_path: {config_path}")
    path = Path(str(records_path))
    if path.is_absolute():
        return path
    return repo_root / path


def _resolve_handoff_record(
    *,
    handoff_id: str | None,
    record_yaml: Path | None,
    repo_root: Path,
) -> SynthesisHandoffRecord | None:
    if handoff_id is None:
        return None
    return get_synthesis_handoff_record(_record_path_for(repo_root, record_yaml), handoff_id)


def _source_from_args_and_record(
    *,
    source: str | None,
    selected_csv: Path | None,
    handoff_record: SynthesisHandoffRecord | None,
) -> tuple[str | None, int | None]:
    inferred_round: int | None = None
    if source is None and selected_csv is not None:
        source = "selected-csv"
    if handoff_record is None:
        return source, inferred_round
    if selected_csv is not None:
        raise ValueError("--handoff-id cannot be combined with --selected-csv")
    record_source, inferred_round = source_mode_from_handoff_record(handoff_record)
    if source is None:
        return record_source, inferred_round
    if source != record_source:
        raise ValueError(
            f"--source {source!r} conflicts with handoff record source {record_source!r} "
            f"for {handoff_record.handoff_id}"
        )
    return source, inferred_round


def _batch_id_from_source(
    *,
    source: str,
    args_batch_id: str | None,
    handoff_record: SynthesisHandoffRecord | None,
    as_of_round: int | None,
) -> str:
    if handoff_record is not None:
        if args_batch_id is not None and args_batch_id != handoff_record.handoff_id:
            raise ValueError(
                f"--batch-id {args_batch_id!r} conflicts with handoff record id {handoff_record.handoff_id!r}"
            )
        return handoff_record.handoff_id
    if source == "batch0":
        return args_batch_id or DEFAULT_BATCH0_BATCH_ID
    if source == "opal-round":
        if as_of_round is None:
            raise ValueError("--round is required when --source opal-round")
        if args_batch_id is None:
            raise ValueError(
                "A raw OPAL-round preview requires an explicit --batch-id; model round and physical assay batch "
                "are separate identities."
            )
        return args_batch_id
    return args_batch_id or "stress-opal-synthesis-batch"


def _validate_record_manifest(
    *,
    manifest: pd.DataFrame,
    handoff_record: SynthesisHandoffRecord | None,
    strategy_id: str,
) -> dict[str, Any] | None:
    if handoff_record is None:
        return None
    return validate_manifest_against_handoff_record(
        manifest,
        handoff_record,
        strategy_id=strategy_id,
    )


def _validate_record_write_paths(
    *,
    handoff_record: SynthesisHandoffRecord,
    repo_root: Path,
    batch_id: str,
    output_dir: Path | None,
) -> None:
    if output_dir is not None:
        raise ValueError(
            "A lifecycle-bound OPAL handoff writes only to its recorded artifact paths; omit --output-dir."
        )
    assert handoff_record.campaign_slug is not None
    assert handoff_record.expected_artifact is not None
    export_dir = campaign_synthesis_output_dir(
        repo_root,
        campaign_slug=handoff_record.campaign_slug,
        batch_id=batch_id,
    )
    generated = campaign_synthesis_artifact_paths(
        export_dir,
        batch_id=batch_id,
        campaign_slug=handoff_record.campaign_slug,
    )
    expected = handoff_record.expected_artifact
    declared = {
        "manifest_path": expected.manifest_path,
        "vendor_workbook_path": expected.vendor_workbook_path,
        "genbank_dir_path": expected.genbank_dir_path,
        "genbank_feature_table_path": expected.genbank_feature_table_path,
    }
    generated_by_field = {
        "manifest_path": generated["manifest"],
        "vendor_workbook_path": generated["azenta_workbook"],
        "genbank_dir_path": generated["genbank_dir"],
        "genbank_feature_table_path": generated["genbank_feature_table"],
    }
    mismatches: list[str] = []
    for field, declared_value in declared.items():
        declared_path = Path(declared_value)
        if not declared_path.is_absolute():
            declared_path = repo_root / declared_path
        generated_path = generated_by_field[field]
        if declared_path.resolve() != generated_path.resolve():
            mismatches.append(f"{field} declared={declared_path} generated={generated_path}")
    if mismatches:
        raise ValueError(
            "handoff record artifact paths do not match the campaign output contract: " + "; ".join(mismatches)
        )
    existing = [str(path) for path in generated_by_field.values() if path.exists()]
    if existing:
        raise ValueError("authorized handoff artifact paths must not already exist: " + ", ".join(existing))


def _run_id_for_opal_round_source(
    *,
    args_run_id: str | None,
    handoff_record: SynthesisHandoffRecord | None,
) -> str | None:
    if handoff_record is None:
        return args_run_id
    if args_run_id:
        raise ValueError("--handoff-id records OPAL run IDs; do not also pass --run-id")
    return handoff_record.run_id


def _is_handoff_record_only_preview(
    *,
    args: argparse.Namespace,
    handoff_record: SynthesisHandoffRecord | None,
) -> bool:
    return (
        handoff_record is not None
        and not bool(args.write)
        and args.source is None
        and args.selected_csv is None
        and args.campaign_config is None
        and not args.run_id
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    fallback_repo_root = args.repo_root or _repo_root_from(Path.cwd())
    try:
        handoff_record = _resolve_handoff_record(
            handoff_id=args.handoff_id,
            record_yaml=args.record_yaml,
            repo_root=fallback_repo_root,
        )
        source, inferred_as_of_round = _source_from_args_and_record(
            source=args.source,
            selected_csv=args.selected_csv,
            handoff_record=handoff_record,
        )
    except ValueError as exc:
        _parser_error(parser, args, str(exc))
    as_of_round = args.as_of_round if args.as_of_round is not None else inferred_as_of_round
    if handoff_record is not None and inferred_as_of_round is not None and args.as_of_round is not None:
        if int(args.as_of_round) != int(inferred_as_of_round):
            _parser_error(
                parser,
                args,
                f"--round {args.as_of_round} conflicts with handoff record model_as_of_round {inferred_as_of_round}",
            )

    if source is None:
        if args.write:
            _parser_error(
                parser,
                args,
                "--source batch0, --source opal-round, or --selected-csv is required when --write is set",
            )
        strategy = load_cloning_strategy(args.strategy_yaml)
        _summary(
            {
                "status": "ok",
                "mode": "dry_run_helpful_noop",
                "strategy_id": strategy.strategy_id,
                "batch0_default_batch_id": DEFAULT_BATCH0_BATCH_ID,
                "opal_round_default_campaign_config": str(DEFAULT_STRESS_OPAL_CAMPAIGN_CONFIG),
                "note": "provide --source batch0, --source opal-round, or --selected-csv to build a manifest",
            },
            as_json=bool(args.json),
        )
        return 0

    if args.write and source == "batch0":
        _parser_error(
            parser,
            args,
            "Batch-zero synthesis artifacts are frozen source evidence and cannot be rewritten.",
        )

    if args.write and source == "selected-csv":
        _parser_error(
            parser,
            args,
            "selected-csv is preview-only and cannot write synthesis artifacts; materialization requires "
            "an authorized checked-in OPAL handoff record.",
        )

    if args.write and source == "opal-round":
        if handoff_record is None:
            _parser_error(
                parser,
                args,
                "Writing an OPAL-round handoff requires --handoff-id from the checked-in study lifecycle record.",
            )
        if handoff_record.lifecycle_status != "authorized_for_materialization":
            _parser_error(
                parser,
                args,
                "An OPAL-round handoff may be materialized only from lifecycle_status "
                "authorized_for_materialization. Generated, accepted, or later artifacts are immutable.",
            )

    try:
        batch_id = _batch_id_from_source(
            source=source,
            args_batch_id=args.batch_id,
            handoff_record=handoff_record,
            as_of_round=as_of_round,
        )
    except ValueError as exc:
        _parser_error(parser, args, str(exc))

    if args.write and source == "opal-round":
        assert handoff_record is not None
        try:
            _validate_record_write_paths(
                handoff_record=handoff_record,
                repo_root=fallback_repo_root,
                batch_id=batch_id,
                output_dir=args.output_dir,
            )
        except ValueError as exc:
            _parser_error(parser, args, str(exc))

    if _is_handoff_record_only_preview(args=args, handoff_record=handoff_record):
        root = args.repo_root or fallback_repo_root
        _summary(
            {
                "status": "ok",
                "mode": "handoff_record_preview",
                "source": source,
                "batch_id": batch_id,
                "strategy_id": handoff_record.strategy_id,
                "handoff_record": handoff_record_payload(handoff_record, repo_root=root),
            },
            as_json=bool(args.json),
        )
        return 0

    strategy = load_cloning_strategy(args.strategy_yaml)
    source_report: dict[str, Any] = {}
    candidate_records_path: Path | None = None
    if source == "batch0":
        repo_root = args.repo_root or _repo_root_from(args.batch0_config)
        try:
            candidate_records_path = _batch0_candidate_records_path(
                config_path=args.batch0_config,
                repo_root=repo_root,
            )
            selected, source_report = build_batch0_selected_candidates(
                config_path=args.batch0_config,
                repo_root=repo_root,
            )
        except ValueError as exc:
            _parser_error(parser, args, str(exc))
    elif source == "opal-round":
        repo_root = args.repo_root or _repo_root_from(Path.cwd())
        campaign_config = args.campaign_config or repo_root / DEFAULT_STRESS_OPAL_CAMPAIGN_CONFIG
        try:
            run_id = _run_id_for_opal_round_source(
                args_run_id=args.run_id,
                handoff_record=handoff_record,
            )
        except ValueError as exc:
            _parser_error(parser, args, str(exc))
        try:
            selected, source_report = selected_candidates_from_opal_round(
                campaign_config,
                as_of_round=int(as_of_round),
                run_id=run_id,
                repo_root=repo_root,
                alias_registry_path=args.promoter_alias_registry,
            )
            candidate_records_path = Path(str(source_report["candidate_records_path"]))
        except ValueError as exc:
            _parser_error(parser, args, str(exc))
    else:
        if args.selected_csv is None:
            _parser_error(parser, args, "--selected-csv is required when --source selected-csv")
        repo_root = args.repo_root
        try:
            selected = _selected_from_csv(args.selected_csv)
        except ValueError as exc:
            _parser_error(parser, args, str(exc))

    if handoff_record is not None:
        selected = apply_handoff_record_lifecycle(selected, handoff_record)

    try:
        manifest = build_synthesis_manifest(selected=selected, strategy=strategy, batch_id=batch_id)
        record_manifest_validation = _validate_record_manifest(
            manifest=manifest,
            handoff_record=handoff_record,
            strategy_id=strategy.strategy_id,
        )
    except ValueError as exc:
        _parser_error(parser, args, str(exc))
    campaign_counts = manifest.groupby("campaign_slug", sort=False).size().astype(int).to_dict()
    payload: dict[str, Any] = {
        "status": "ok",
        "mode": "validated",
        "source": source,
        "batch_id": batch_id,
        "row_count": int(len(manifest)),
        "campaign_counts": campaign_counts,
        "strategy_id": strategy.strategy_id,
    }
    if source_report:
        payload["source_report"] = source_report

    if args.write:
        assert source == "opal-round"
        campaign_exports = render_campaign_scoped_exports(
            manifest,
            batch_id=batch_id,
            output_owner="campaign",
            repo_root=repo_root,
            output_root=args.output_dir,
            candidate_records_path=candidate_records_path,
        )
        payload["mode"] = "written"
        payload["campaign_exports"] = campaign_exports.to_dict("records")
    elif source in {"batch0", "opal-round"}:
        root = repo_root or args.repo_root or _repo_root_from(Path.cwd())
        output_dir_for = source_evidence_synthesis_output_dir if source == "batch0" else campaign_synthesis_output_dir
        payload["default_campaign_output_dirs"] = {
            campaign: str(output_dir_for(root, campaign_slug=campaign, batch_id=batch_id))
            for campaign in campaign_counts
        }
    if handoff_record is not None:
        root = repo_root or fallback_repo_root
        payload["handoff_record"] = handoff_record_payload(handoff_record, repo_root=root)
        payload["handoff_record"]["manifest_validation"] = record_manifest_validation

    _summary(payload, as_json=bool(args.json))
    return 0
