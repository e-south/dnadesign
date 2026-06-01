"""Hard-prune confounded DenseGen TFBS Stage B campaigns from generated artifacts."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..stage_a.manifests import file_sha256

PRUNE_MANIFEST_SCHEMA_VERSION = "stress_ethanol_cipro_growth.tfbs_stage_b_prune.v1"


@dataclass(frozen=True)
class TfbsStageBPruneResult:
    """Summary for a Stage B hard-prune operation."""

    status: str
    config_manifest_path: Path
    prune_manifest_path: Path
    pruned_label_names: tuple[str, ...]
    pruned_campaign_count: int
    retained_campaign_count: int
    deleted_path_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "config_manifest_path": str(self.config_manifest_path),
            "prune_manifest_path": str(self.prune_manifest_path),
            "pruned_label_names": list(self.pruned_label_names),
            "pruned_campaign_count": int(self.pruned_campaign_count),
            "retained_campaign_count": int(self.retained_campaign_count),
            "deleted_path_count": int(self.deleted_path_count),
        }


def prune_tfbs_stage_b_campaigns(
    config_manifest_path: str | Path,
    *,
    prune_label_names: Sequence[str],
    delete_review_artifacts: bool = True,
) -> TfbsStageBPruneResult:
    """Remove declared labels from Stage B config, runtime, validation, review, and notebook artifacts."""

    manifest_path = Path(config_manifest_path)
    manifest = _read_json(manifest_path)
    labels = tuple(dict.fromkeys(str(label) for label in prune_label_names if str(label)))
    if not labels:
        raise ValueError("Stage B prune requires at least one label name")
    stage_b_root = _stage_b_root(manifest_path)
    campaigns = _mapping_list(manifest.get("campaigns"), field="campaigns")
    pairs = _mapping_list(manifest.get("pairs"), field="pairs")
    known_labels = {str(row.get("label_name")) for row in campaigns}
    missing = sorted(set(labels) - known_labels)
    if missing:
        raise ValueError(f"Stage B prune requested unknown label(s): {missing}")

    pruned_campaigns = [row for row in campaigns if str(row.get("label_name")) in labels]
    retained_campaigns = [row for row in campaigns if str(row.get("label_name")) not in labels]
    retained_pairs = [row for row in pairs if str(row.get("label_name")) not in labels]
    deleted_paths: list[str] = []
    for campaign in pruned_campaigns:
        deleted_paths.extend(_delete_campaign_artifacts(campaign))
    validation = _pruned_validation(manifest.get("validation"), labels=labels)
    for report in validation["pruned_reports"]:
        report_path = Path(str(report.get("report_path") or ""))
        if _delete_path(report_path):
            deleted_paths.append(str(report_path))
    if delete_review_artifacts:
        for path in (
            stage_b_root / "review" / "realized_labels",
            stage_b_root / "notebooks" / "collection_visuals",
            stage_b_root / "notebooks" / "tfbs_stage_b_exact_budget_campaign_set.py",
        ):
            if _delete_path(path):
                deleted_paths.append(str(path))

    refreshed = {
        **manifest,
        "campaign_count": int(len(retained_campaigns)),
        "sentinel_labels": [label for label in manifest.get("sentinel_labels", []) if str(label) not in labels],
        "pairs": retained_pairs,
        "campaigns": retained_campaigns,
        "validation": validation["retained_validation"],
        "last_prune_manifest_path": str(stage_b_root / "manifests" / "stage_b_sentinel_prune_manifest.json"),
    }
    _write_json(manifest_path, refreshed)
    prune_manifest = {
        "schema_version": PRUNE_MANIFEST_SCHEMA_VERSION,
        "status": "PASS",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "config_manifest_path": str(manifest_path),
        "config_manifest_hash_after_prune": file_sha256(manifest_path),
        "pruned_label_names": list(labels),
        "pruned_campaign_count": int(len(pruned_campaigns)),
        "retained_campaign_count": int(len(retained_campaigns)),
        "deleted_path_count": int(len(set(deleted_paths))),
        "deleted_paths": sorted(set(deleted_paths)),
        "interpretation_boundary": (
            "These campaigns were hard-pruned from the generated Stage B review surface because their null/control "
            "definition was confounded for the current learnability claim."
        ),
    }
    prune_manifest_path = stage_b_root / "manifests" / "stage_b_sentinel_prune_manifest.json"
    _write_json(prune_manifest_path, prune_manifest)
    return TfbsStageBPruneResult(
        status="PASS",
        config_manifest_path=manifest_path,
        prune_manifest_path=prune_manifest_path,
        pruned_label_names=labels,
        pruned_campaign_count=len(pruned_campaigns),
        retained_campaign_count=len(retained_campaigns),
        deleted_path_count=len(set(deleted_paths)),
    )


def _delete_campaign_artifacts(campaign: Mapping[str, Any]) -> list[str]:
    deleted: list[str] = []
    config_path = Path(str(campaign.get("config_path") or ""))
    if config_path.name == "campaign.yaml" and config_path.parent.name == "configs":
        workdir = config_path.parent.parent
        if _delete_path(workdir):
            deleted.append(str(workdir))
    sidecar_path = Path(str(campaign.get("label_sidecar_path") or ""))
    if _delete_path(sidecar_path):
        deleted.append(str(sidecar_path))
    sidecar_parent = sidecar_path.parent
    if sidecar_parent.name and sidecar_parent.exists() and not any(sidecar_parent.iterdir()):
        sidecar_parent.rmdir()
        deleted.append(str(sidecar_parent))
    return deleted


def _pruned_validation(value: Any, *, labels: Sequence[str]) -> dict[str, Any]:
    validation = dict(value) if isinstance(value, Mapping) else {}
    reports = _mapping_list(validation.get("reports", []), field="validation.reports")
    labels_set = set(labels)
    pruned_reports = [
        row for row in reports if _label_from_campaign_key(str(row.get("campaign_key") or "")) in labels_set
    ]
    retained_reports = [
        row for row in reports if _label_from_campaign_key(str(row.get("campaign_key") or "")) not in labels_set
    ]
    retained = {
        **validation,
        "campaign_count": int(len(retained_reports)),
        "reports": retained_reports,
    }
    return {"retained_validation": retained, "pruned_reports": pruned_reports}


def _label_from_campaign_key(key: str) -> str:
    prefix = "tfbs_"
    suffixes = ("_positive_random_id_seed7", "_matched_null_random_id_seed7")
    if not key.startswith(prefix):
        return ""
    body = key[len(prefix) :]
    for suffix in suffixes:
        if body.endswith(suffix):
            return body[: -len(suffix)]
    return ""


def _stage_b_root(config_manifest_path: Path) -> Path:
    if config_manifest_path.parent.name == "manifests":
        return config_manifest_path.parent.parent
    return config_manifest_path.parent


def _delete_path(path: Path) -> bool:
    if str(path) in {"", "."}:
        return False
    if path.is_dir():
        shutil.rmtree(path)
        return True
    if path.exists() or path.is_symlink():
        path.unlink()
        return True
    return False


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Stage B prune config manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Stage B prune config manifest must be a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mapping_list(value: Any, *, field: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"Stage B prune manifest field {field!r} must be a list")
    if not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"Stage B prune manifest field {field!r} must contain objects")
    return list(value)
