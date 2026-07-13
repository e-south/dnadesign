"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff/records.py

Checked-in synthesis handoff lifecycle record helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .azenta import validate_azenta_workbook
from .contracts import SelectedCandidate
from .genbank import validate_genbank_record_set

DEFAULT_SYNTHESIS_HANDOFF_RECORD = Path("docs/studies/stress_ethanol_cipro_growth/record/synthesis_handoffs.yaml")


@dataclass(frozen=True)
class ExpectedHandoffArtifact:
    """Expected campaign-scoped generated artifacts for one handoff."""

    campaign_slug: str
    expected_rows: int
    manifest_path: str
    vendor_workbook_path: str
    genbank_dir_path: str
    genbank_feature_table_path: str
    run_id: str | None = None
    manifest_sha256: str | None = None
    vendor_workbook_sha256: str | None = None
    genbank_dir_sha256: str | None = None
    genbank_feature_table_sha256: str | None = None
    workbook_readback_status: str | None = None
    genbank_readback_status: str | None = None

    def __post_init__(self) -> None:
        if not str(self.campaign_slug).strip():
            raise ValueError("expected campaign_slug must be non-empty")
        if int(self.expected_rows) <= 0:
            raise ValueError(f"expected_rows must be positive for campaign={self.campaign_slug}")
        if not str(self.manifest_path).strip():
            raise ValueError(f"manifest_path must be non-empty for campaign={self.campaign_slug}")
        if not str(self.vendor_workbook_path).strip():
            raise ValueError(f"vendor_workbook_path must be non-empty for campaign={self.campaign_slug}")
        if not str(self.genbank_dir_path).strip():
            raise ValueError(f"genbank_dir_path must be non-empty for campaign={self.campaign_slug}")
        if not str(self.genbank_feature_table_path).strip():
            raise ValueError(f"genbank_feature_table_path must be non-empty for campaign={self.campaign_slug}")

    def to_json(self) -> dict[str, Any]:
        return {
            "campaign_slug": self.campaign_slug,
            "expected_rows": int(self.expected_rows),
            "manifest_path": self.manifest_path,
            "vendor_workbook_path": self.vendor_workbook_path,
            "genbank_dir_path": self.genbank_dir_path,
            "genbank_feature_table_path": self.genbank_feature_table_path,
            "run_id": self.run_id,
            "manifest_sha256": self.manifest_sha256,
            "vendor_workbook_sha256": self.vendor_workbook_sha256,
            "genbank_dir_sha256": self.genbank_dir_sha256,
            "genbank_feature_table_sha256": self.genbank_feature_table_sha256,
            "workbook_readback_status": self.workbook_readback_status,
            "genbank_readback_status": self.genbank_readback_status,
        }


@dataclass(frozen=True)
class ExpectedSelectionView:
    """Expected logical membership count for one OPAL selection view."""

    selection_view_id: str
    expected_rows: int

    def __post_init__(self) -> None:
        if not str(self.selection_view_id).strip():
            raise ValueError("expected selection_view_id must be non-empty")
        if int(self.expected_rows) <= 0:
            raise ValueError(f"expected_rows must be positive for selection_view_id={self.selection_view_id}")

    def to_json(self) -> dict[str, Any]:
        return {
            "selection_view_id": self.selection_view_id,
            "expected_rows": int(self.expected_rows),
        }


@dataclass(frozen=True)
class SynthesisHandoffRecord:
    """One lifecycle record from ``synthesis_handoffs.yaml``."""

    handoff_id: str
    lifecycle_status: str
    source_authority: str
    selection_epoch: str
    assay_batch_index: int | None
    model_as_of_round: int | None
    run_id: str | None
    strategy_id: str
    expected_campaigns: tuple[ExpectedHandoffArtifact, ...] = ()
    campaign_slug: str | None = None
    expected_selection_views: tuple[ExpectedSelectionView, ...] = ()
    expected_artifact: ExpectedHandoffArtifact | None = None

    def __post_init__(self) -> None:
        for field in (
            "handoff_id",
            "lifecycle_status",
            "source_authority",
            "selection_epoch",
            "strategy_id",
        ):
            if not str(getattr(self, field)).strip():
                raise ValueError(f"{field} must be non-empty in synthesis handoff record")
        if self.source_authority == "study_batch0_selector":
            if not self.expected_campaigns:
                raise ValueError(f"batch-0 handoff record {self.handoff_id} requires expected_campaigns")
            if self.run_id is None:
                raise ValueError(f"handoff record {self.handoff_id} requires explicit run_id for batch-0 source")
            if self.campaign_slug is not None or self.expected_selection_views or self.expected_artifact is not None:
                raise ValueError(f"batch-0 handoff record {self.handoff_id} cannot declare measured-round fields")
            return
        if self.source_authority == "opal_selection_batch":
            if self.expected_campaigns:
                raise ValueError(f"measured-round handoff record {self.handoff_id} cannot declare expected_campaigns")
            if self.campaign_slug is None or not str(self.campaign_slug).strip():
                raise ValueError(f"measured-round handoff record {self.handoff_id} requires campaign_slug")
            if self.run_id is None:
                raise ValueError(f"measured-round handoff record {self.handoff_id} requires explicit run_id")
            if not self.expected_selection_views:
                raise ValueError(f"measured-round handoff record {self.handoff_id} requires expected_selection_views")
            if self.expected_artifact is None:
                raise ValueError(f"measured-round handoff record {self.handoff_id} requires expected_artifact")
            if self.expected_artifact.campaign_slug != self.campaign_slug:
                raise ValueError(
                    f"measured-round handoff record {self.handoff_id} artifact campaign does not match campaign_slug"
                )
            view_ids = [row.selection_view_id for row in self.expected_selection_views]
            if len(view_ids) != len(set(view_ids)):
                raise ValueError(f"measured-round handoff record {self.handoff_id} has duplicate selection views")
            return
        raise ValueError(f"unsupported synthesis handoff source_authority: {self.source_authority}")

    @property
    def expected_campaign_counts(self) -> dict[str, int]:
        return {row.campaign_slug: int(row.expected_rows) for row in self.expected_campaigns}

    @property
    def expected_run_ids_by_campaign(self) -> dict[str, str]:
        run_ids: dict[str, str] = {}
        for row in self.expected_campaigns:
            run_id = row.run_id or self.run_id
            if run_id is not None:
                run_ids[row.campaign_slug] = run_id
        return run_ids

    @property
    def expected_selection_view_counts(self) -> dict[str, int]:
        return {row.selection_view_id: int(row.expected_rows) for row in self.expected_selection_views}

    @property
    def artifacts(self) -> tuple[ExpectedHandoffArtifact, ...]:
        if self.source_authority == "study_batch0_selector":
            return self.expected_campaigns
        assert self.expected_artifact is not None
        return (self.expected_artifact,)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if pd.isna(value):
        return None
    return int(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _record_from_raw(raw: dict[str, Any]) -> SynthesisHandoffRecord:
    def artifact_from(value: Any, *, label: str) -> ExpectedHandoffArtifact:
        item = _require_mapping(value, label=label)
        return ExpectedHandoffArtifact(
            campaign_slug=str(item["campaign_slug"]),
            expected_rows=int(item["expected_rows"]),
            manifest_path=str(item["manifest_path"]),
            vendor_workbook_path=str(item["vendor_workbook_path"]),
            genbank_dir_path=str(item["genbank_dir_path"]),
            genbank_feature_table_path=str(item["genbank_feature_table_path"]),
            run_id=_optional_text(item.get("run_id")),
            manifest_sha256=_optional_text(item.get("manifest_sha256")),
            vendor_workbook_sha256=_optional_text(item.get("vendor_workbook_sha256")),
            genbank_dir_sha256=_optional_text(item.get("genbank_dir_sha256")),
            genbank_feature_table_sha256=_optional_text(item.get("genbank_feature_table_sha256")),
            workbook_readback_status=_optional_text(item.get("workbook_readback_status")),
            genbank_readback_status=_optional_text(item.get("genbank_readback_status")),
        )

    source_authority = str(raw["source_authority"])
    expected_campaigns: tuple[ExpectedHandoffArtifact, ...] = ()
    expected_selection_views: tuple[ExpectedSelectionView, ...] = ()
    expected_artifact: ExpectedHandoffArtifact | None = None
    if source_authority == "study_batch0_selector":
        campaigns_raw = raw.get("expected_campaigns")
        if not isinstance(campaigns_raw, list):
            raise ValueError(f"handoff record {raw.get('handoff_id', '<unknown>')} expected_campaigns must be a list")
        expected_campaigns = tuple(artifact_from(item, label="expected campaign") for item in campaigns_raw)
    elif source_authority == "opal_selection_batch":
        views_raw = raw.get("expected_selection_views")
        if not isinstance(views_raw, list):
            raise ValueError(
                f"handoff record {raw.get('handoff_id', '<unknown>')} expected_selection_views must be a list"
            )
        expected_selection_views = tuple(
            ExpectedSelectionView(
                selection_view_id=str(_require_mapping(item, label="expected selection view")["selection_view_id"]),
                expected_rows=int(item["expected_rows"]),
            )
            for item in views_raw
        )
        expected_artifact = artifact_from(raw.get("expected_artifact"), label="expected artifact")
    return SynthesisHandoffRecord(
        handoff_id=str(raw["handoff_id"]),
        lifecycle_status=str(raw["lifecycle_status"]),
        source_authority=source_authority,
        selection_epoch=str(raw["selection_epoch"]),
        assay_batch_index=_optional_int(raw.get("assay_batch_index")),
        model_as_of_round=_optional_int(raw.get("model_as_of_round")),
        run_id=_optional_text(raw.get("run_id")),
        strategy_id=str(raw["strategy_id"]),
        expected_campaigns=expected_campaigns,
        campaign_slug=_optional_text(raw.get("campaign_slug")),
        expected_selection_views=expected_selection_views,
        expected_artifact=expected_artifact,
    )


def load_synthesis_handoff_records(path: str | Path) -> dict[str, SynthesisHandoffRecord]:
    """Load all checked-in synthesis handoff lifecycle records."""

    record_path = Path(path)
    if not record_path.exists():
        raise ValueError(f"synthesis handoff record not found: {record_path}")
    with record_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    root = _require_mapping(raw, label="synthesis handoff record")
    handoffs = root.get("handoffs")
    if not isinstance(handoffs, list):
        raise ValueError(f"synthesis handoff record missing handoffs list: {record_path}")
    records = [_record_from_raw(_require_mapping(item, label="handoff")) for item in handoffs]
    by_id: dict[str, SynthesisHandoffRecord] = {}
    for record in records:
        if record.handoff_id in by_id:
            raise ValueError(f"duplicate handoff_id in synthesis handoff record: {record.handoff_id}")
        by_id[record.handoff_id] = record
    return by_id


def get_synthesis_handoff_record(path: str | Path, handoff_id: str) -> SynthesisHandoffRecord:
    """Return one checked-in synthesis handoff lifecycle record."""

    identifier = str(handoff_id).strip()
    if not identifier:
        raise ValueError("handoff_id must be non-empty")
    records = load_synthesis_handoff_records(path)
    try:
        return records[identifier]
    except KeyError as exc:
        available = ", ".join(sorted(records)) or "<none>"
        raise ValueError(f"unknown synthesis handoff id {identifier!r}; available: {available}") from exc


def source_mode_from_handoff_record(record: SynthesisHandoffRecord) -> tuple[str, int | None]:
    """Resolve a lifecycle record to the synthesis CLI source mode."""

    if record.source_authority == "study_batch0_selector" and record.selection_epoch == "pre_assay_seed":
        return "batch0", None
    if record.source_authority == "opal_selection_batch" and record.selection_epoch == "opal_model_round":
        if record.model_as_of_round is None:
            raise ValueError(f"handoff record {record.handoff_id} requires model_as_of_round for OPAL rounds")
        return "opal-round", int(record.model_as_of_round)
    raise ValueError(
        "unsupported synthesis handoff record source "
        f"{record.source_authority!r}/{record.selection_epoch!r} for handoff_id={record.handoff_id}"
    )


def _resolve_repo_path(repo_root: str | Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(repo_root) / path


def _field_values(manifest: pd.DataFrame, column: str) -> set[Any]:
    values: set[Any] = set()
    for value in manifest[column].tolist():
        if pd.isna(value):
            values.add(None)
        elif isinstance(value, float) and value.is_integer():
            values.add(int(value))
        else:
            values.add(value)
    return values


def _require_manifest_field(manifest: pd.DataFrame, column: str, expected: Any) -> None:
    if column not in manifest.columns:
        raise ValueError(f"synthesis manifest missing lifecycle column required by handoff record: {column}")
    values = _field_values(manifest, column)
    if values != {expected}:
        observed = sorted((repr(value) for value in values))
        raise ValueError(f"handoff record lifecycle mismatch for {column}: expected {expected!r}, observed {observed}")


def _require_batch0_manifest_run_ids(manifest: pd.DataFrame, record: SynthesisHandoffRecord) -> None:
    if "run_id" not in manifest.columns:
        raise ValueError("synthesis manifest missing lifecycle column required by handoff record: run_id")
    if "campaign_slug" not in manifest.columns:
        raise ValueError("synthesis manifest missing campaign_slug required by handoff record")
    expected_by_campaign = record.expected_run_ids_by_campaign
    missing_expected = sorted(set(record.expected_campaign_counts).difference(expected_by_campaign))
    if missing_expected:
        raise ValueError(
            f"handoff record {record.handoff_id} requires explicit run_id for every expected campaign: "
            + ", ".join(missing_expected[:5])
        )
    for campaign_slug, expected_run_id in sorted(expected_by_campaign.items()):
        campaign_manifest = manifest.loc[manifest["campaign_slug"].astype(str) == campaign_slug]
        if campaign_manifest.empty:
            continue
        observed = _field_values(campaign_manifest, "run_id")
        if observed != {expected_run_id}:
            observed_text = sorted((repr(value) for value in observed))
            raise ValueError(
                "handoff record lifecycle mismatch for run_id: "
                f"campaign={campaign_slug} expected {expected_run_id!r}, observed {observed_text}"
            )


def _selection_view_membership_counts(manifest: pd.DataFrame) -> dict[str, int]:
    if "selection_view_ids" not in manifest.columns:
        raise ValueError("synthesis manifest missing selection_view_ids required by measured-round record")
    counts: dict[str, int] = {}
    for raw in manifest["selection_view_ids"].tolist():
        try:
            view_ids = json.loads(str(raw))
        except json.JSONDecodeError as exc:
            raise ValueError("synthesis manifest contains invalid selection_view_ids JSON") from exc
        if not isinstance(view_ids, list) or not view_ids:
            raise ValueError("synthesis manifest selection_view_ids must be a non-empty JSON list")
        for view_id in view_ids:
            key = str(view_id)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _validate_batch0_manifest(manifest: pd.DataFrame, record: SynthesisHandoffRecord) -> dict[str, Any]:
    observed_counts = manifest.groupby("campaign_slug", sort=True).size().astype(int).to_dict()
    expected_counts = record.expected_campaign_counts
    mismatches = []
    for campaign_slug in sorted(set(expected_counts).union(observed_counts)):
        if expected_counts.get(campaign_slug) != observed_counts.get(campaign_slug):
            mismatches.append(
                f"{campaign_slug} expected={expected_counts.get(campaign_slug)} "
                f"observed={observed_counts.get(campaign_slug)}"
            )
    if mismatches:
        raise ValueError("handoff record campaign row mismatch: " + "; ".join(mismatches[:5]))
    _require_batch0_manifest_run_ids(manifest, record)
    return {"campaign_counts": observed_counts}


def _validate_measured_round_manifest(manifest: pd.DataFrame, record: SynthesisHandoffRecord) -> dict[str, Any]:
    assert record.expected_artifact is not None
    _require_manifest_field(manifest, "campaign_slug", record.campaign_slug)
    _require_manifest_field(manifest, "run_id", record.run_id)
    observed_rows = int(len(manifest))
    if observed_rows != int(record.expected_artifact.expected_rows):
        raise ValueError(
            f"handoff record selection batch row mismatch: expected={record.expected_artifact.expected_rows} "
            f"observed={observed_rows}"
        )
    observed_view_counts = _selection_view_membership_counts(manifest)
    expected_view_counts = record.expected_selection_view_counts
    if observed_view_counts != expected_view_counts:
        raise ValueError(
            "handoff record selection-view membership mismatch: "
            f"expected={expected_view_counts} observed={observed_view_counts}"
        )
    return {
        "campaign_slug": record.campaign_slug,
        "selection_view_counts": observed_view_counts,
        "selection_batch_count": observed_rows,
    }


def validate_manifest_against_handoff_record(
    manifest: pd.DataFrame,
    record: SynthesisHandoffRecord,
    *,
    strategy_id: str,
) -> dict[str, Any]:
    """Validate a manifest against the checked-in lifecycle record."""

    if str(strategy_id) != record.strategy_id:
        raise ValueError(f"handoff record strategy mismatch: expected {record.strategy_id}, observed {strategy_id}")
    if "campaign_slug" not in manifest.columns:
        raise ValueError("synthesis manifest missing campaign_slug required by handoff record")
    if record.source_authority == "study_batch0_selector":
        source_validation = _validate_batch0_manifest(manifest, record)
    elif record.source_authority == "opal_selection_batch":
        source_validation = _validate_measured_round_manifest(manifest, record)
    else:
        raise ValueError(f"unsupported synthesis handoff source_authority: {record.source_authority}")

    _require_manifest_field(manifest, "batch_id", record.handoff_id)
    _require_manifest_field(manifest, "selection_epoch", record.selection_epoch)
    _require_manifest_field(manifest, "assay_batch_index", record.assay_batch_index)
    _require_manifest_field(manifest, "model_as_of_round", record.model_as_of_round)
    return {
        "status": "pass",
        "strategy_id": strategy_id,
        **source_validation,
    }


def apply_handoff_record_lifecycle(
    selected: list[SelectedCandidate],
    record: SynthesisHandoffRecord,
) -> list[SelectedCandidate]:
    """Stamp record-owned lifecycle fields onto selected rows before manifest build."""

    expected_run_ids = record.expected_run_ids_by_campaign
    return [
        SelectedCandidate(
            campaign_slug=row.campaign_slug,
            selection_memberships=row.selection_memberships,
            as_of_round=row.as_of_round,
            run_id=(
                str(record.run_id)
                if record.source_authority == "opal_selection_batch"
                else expected_run_ids.get(row.campaign_slug, row.run_id)
            ),
            selection_rank=row.selection_rank,
            id=row.id,
            sequence=row.sequence,
            synthesis_name=row.synthesis_name,
            selection_source=row.selection_source,
            selection_epoch=record.selection_epoch,
            assay_batch_index=record.assay_batch_index,
            model_as_of_round=record.model_as_of_round,
        )
        for row in selected
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_genbank_dir(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(path.glob("*.gb")):
        digest.update(file_path.name.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def artifact_status_for_handoff_record(
    record: SynthesisHandoffRecord,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Inspect generated artifacts referenced by a lifecycle record."""

    rows: list[dict[str, Any]] = []
    present_artifacts = 0
    workbook_readback_pass_count = 0
    genbank_readback_pass_count = 0
    for expected in record.artifacts:
        manifest_path = _resolve_repo_path(repo_root, expected.manifest_path)
        workbook_path = _resolve_repo_path(repo_root, expected.vendor_workbook_path)
        genbank_dir_path = _resolve_repo_path(repo_root, expected.genbank_dir_path)
        genbank_feature_table_path = _resolve_repo_path(repo_root, expected.genbank_feature_table_path)
        row: dict[str, Any] = {
            "campaign_slug": expected.campaign_slug,
            "expected_rows": int(expected.expected_rows),
            "manifest_path": str(manifest_path),
            "vendor_workbook_path": str(workbook_path),
            "genbank_dir_path": str(genbank_dir_path),
            "genbank_feature_table_path": str(genbank_feature_table_path),
            "manifest_exists": manifest_path.exists(),
            "vendor_workbook_exists": workbook_path.exists(),
            "genbank_dir_exists": genbank_dir_path.is_dir(),
            "genbank_feature_table_exists": genbank_feature_table_path.exists(),
            "manifest_sha256": None,
            "vendor_workbook_sha256": None,
            "genbank_dir_sha256": None,
            "genbank_feature_table_sha256": None,
            "manifest_hash_matches_record": None,
            "vendor_workbook_hash_matches_record": None,
            "genbank_dir_hash_matches_record": None,
            "genbank_feature_table_hash_matches_record": None,
            "manifest_row_count": None,
            "manifest_row_count_matches_record": None,
            "workbook_readback_status": None,
            "genbank_readback_status": None,
        }
        manifest: pd.DataFrame | None = None
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path)
            row["manifest_sha256"] = _sha256_file(manifest_path)
            row["manifest_row_count"] = int(len(manifest))
            row["manifest_row_count_matches_record"] = int(len(manifest)) == int(expected.expected_rows)
            if expected.manifest_sha256 is not None:
                row["manifest_hash_matches_record"] = row["manifest_sha256"] == expected.manifest_sha256
        if workbook_path.exists():
            row["vendor_workbook_sha256"] = _sha256_file(workbook_path)
            if expected.vendor_workbook_sha256 is not None:
                row["vendor_workbook_hash_matches_record"] = (
                    row["vendor_workbook_sha256"] == expected.vendor_workbook_sha256
                )
        if genbank_dir_path.is_dir():
            row["genbank_dir_sha256"] = _sha256_genbank_dir(genbank_dir_path)
            if expected.genbank_dir_sha256 is not None:
                row["genbank_dir_hash_matches_record"] = row["genbank_dir_sha256"] == expected.genbank_dir_sha256
        if genbank_feature_table_path.exists():
            row["genbank_feature_table_sha256"] = _sha256_file(genbank_feature_table_path)
            if expected.genbank_feature_table_sha256 is not None:
                row["genbank_feature_table_hash_matches_record"] = (
                    row["genbank_feature_table_sha256"] == expected.genbank_feature_table_sha256
                )
        if manifest is not None and workbook_path.exists():
            try:
                readback = validate_azenta_workbook(manifest, workbook_path)
                row["workbook_readback_status"] = readback["status"]
            except ValueError as exc:
                row["workbook_readback_status"] = "fail"
                row["workbook_readback_error"] = str(exc)
        if manifest is not None and genbank_dir_path.is_dir():
            try:
                genbank_readback = validate_genbank_record_set(manifest, genbank_dir_path)
                row["genbank_readback_status"] = genbank_readback["status"]
            except ValueError as exc:
                row["genbank_readback_status"] = "fail"
                row["genbank_readback_error"] = str(exc)
        if (
            row["manifest_exists"]
            and row["vendor_workbook_exists"]
            and row["genbank_dir_exists"]
            and row["genbank_feature_table_exists"]
        ):
            present_artifacts += 1
        if row["workbook_readback_status"] == "pass":
            workbook_readback_pass_count += 1
        if row["genbank_readback_status"] == "pass":
            genbank_readback_pass_count += 1
        rows.append(row)

    return {
        "summary": {
            "expected_artifact_count": int(len(record.artifacts)),
            "present_artifact_count": int(present_artifacts),
            "readback_pass_count": int(workbook_readback_pass_count),
            "workbook_readback_pass_count": int(workbook_readback_pass_count),
            "genbank_readback_pass_count": int(genbank_readback_pass_count),
        },
        "artifacts": rows,
    }


def handoff_record_payload(
    record: SynthesisHandoffRecord,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Return a JSON-ready record summary for operators and tests."""

    return {
        "handoff_id": record.handoff_id,
        "lifecycle_status": record.lifecycle_status,
        "source_authority": record.source_authority,
        "selection_epoch": record.selection_epoch,
        "assay_batch_index": record.assay_batch_index,
        "model_as_of_round": record.model_as_of_round,
        "run_id": record.run_id,
        "campaign_slug": record.campaign_slug,
        "strategy_id": record.strategy_id,
        "expected_selection_views": [row.to_json() for row in record.expected_selection_views],
        "expected_artifacts": [row.to_json() for row in record.artifacts],
        "artifact_status": artifact_status_for_handoff_record(record, repo_root=repo_root),
    }
