"""Fail-fast validation for replicated Stage B review inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ....profiles import (
    CUSTOM_TFBS_TARGET_PROFILE_ID,
    tfbs_target_profile_for_labels,
    tfbs_target_profile_for_profile_id,
)
from ..io import campaign_rows, pair_rows, read_review_manifest
from .contracts import TFBS_STAGE_B_DETERMINISTIC_REPLICATE_SEEDS, TfbsStageBReplicateManifest


def load_replicate_manifests(
    config_manifest_paths: Sequence[str | Path],
    *,
    expected_seeds: Sequence[int] = TFBS_STAGE_B_DETERMINISTIC_REPLICATE_SEEDS,
) -> list[TfbsStageBReplicateManifest]:
    """Read config manifests and enforce deterministic shared-start replicate contracts."""

    entries = [_load_one(Path(path)) for path in config_manifest_paths]
    expected = tuple(int(seed) for seed in expected_seeds)
    seeds = tuple(sorted(entry.seed for entry in entries))
    if seeds != expected:
        raise ValueError(f"Stage B replicated review requires replicate seeds {list(expected)}; got {list(seeds)}")
    if len({entry.seed for entry in entries}) != len(entries):
        raise ValueError("Stage B replicated review found duplicate replicate seed manifests")
    _validate_common_contracts(entries)
    for entry in entries:
        _validate_supported_label_profile(entry.manifest)
        _validate_shared_initial_ids(entry)
    return sorted(entries, key=lambda entry: entry.seed)


def _load_one(path: Path) -> TfbsStageBReplicateManifest:
    manifest = read_review_manifest(path)
    if manifest.get("status") != "PASS":
        raise ValueError(f"Stage B replicated review requires PASS config manifest: {path}")
    seed = int(manifest.get("seed"))
    campaigns = campaign_rows(manifest)
    pairs = pair_rows(manifest)
    if len(campaigns) != 2 * len(pairs):
        raise ValueError(f"Stage B replicate manifest must contain two campaigns per pair: {path}")
    for row in [*campaigns, *pairs]:
        if int(row.get("seed")) != seed:
            raise ValueError(f"Stage B replicate manifest seed mismatch: manifest={seed}, row={row}")
    return TfbsStageBReplicateManifest(path=path, seed=seed, manifest=manifest)


def _validate_common_contracts(entries: Sequence[TfbsStageBReplicateManifest]) -> None:
    if not entries:
        raise ValueError("Stage B replicated review requires at least one config manifest")
    keys = (
        "split_id",
        "rounds",
        "selection_k",
        "initial_label_count",
        "initial_seed_policy",
        "selection_tie_handling",
        "records_hash",
        "candidate_scope_hash",
    )
    baseline = entries[0].manifest
    baseline_labels = tuple(str(label) for label in baseline.get("sentinel_labels", []))
    baseline_profile_id = _target_profile_id(baseline)
    if not baseline_labels:
        raise ValueError("Stage B replicated review requires non-empty sentinel_labels")
    for entry in entries[1:]:
        labels = tuple(str(label) for label in entry.manifest.get("sentinel_labels", []))
        if labels != baseline_labels:
            raise ValueError(
                "Stage B replicated review requires identical sentinel_labels across seeds "
                f"(seed={entry.seed}, labels={list(labels)}, expected={list(baseline_labels)})"
            )
        profile_id = _target_profile_id(entry.manifest)
        if profile_id != baseline_profile_id:
            raise ValueError(
                "Stage B replicated review requires identical target_profile ids across seeds "
                f"(seed={entry.seed}, got={profile_id!r}, expected={baseline_profile_id!r})"
            )
        for key in keys:
            left = baseline.get(key)
            right = entry.manifest.get(key)
            if left is not None and right is not None and str(left) != str(right):
                raise ValueError(
                    f"Stage B replicated review requires identical {key} across seeds "
                    f"(seed={entry.seed}, got={right!r}, expected={left!r})"
                )


def _target_profile_id(manifest: Mapping[str, Any]) -> str:
    profile = manifest.get("target_profile")
    if isinstance(profile, Mapping) and profile.get("profile_id"):
        return str(profile["profile_id"])
    labels = tuple(str(label) for label in manifest.get("sentinel_labels", []))
    return tfbs_target_profile_for_labels(labels).profile_id


def _validate_supported_label_profile(manifest: Mapping[str, Any]) -> None:
    labels = [str(label) for label in manifest.get("sentinel_labels", [])]
    profile = manifest.get("target_profile")
    if isinstance(profile, Mapping) and profile.get("profile_id"):
        if str(profile["profile_id"]) == CUSTOM_TFBS_TARGET_PROFILE_ID:
            tfbs_target_profile_for_labels(tuple(labels))
            return
        resolved = tfbs_target_profile_for_profile_id(str(profile["profile_id"]))
        if not set(labels).issubset(set(resolved.label_names)):
            raise ValueError(
                "Stage B replicated review target_profile label mismatch "
                f"(profile_id={resolved.profile_id!r}, labels={labels}, expected={list(resolved.label_names)})"
            )
        return
    tfbs_target_profile_for_labels(tuple(labels))


def _validate_shared_initial_ids(entry: TfbsStageBReplicateManifest) -> None:
    campaign_by_key = {str(row["campaign_key"]): row for row in campaign_rows(entry.manifest)}
    for pair in pair_rows(entry.manifest):
        if str(pair.get("initial_seed_pairing")) != "shared_positive_null_starting_ids":
            raise ValueError(f"Stage B replicate pair is not shared-start paired: {pair}")
        positive = _campaign_for_pair(campaign_by_key, pair, field="positive_campaign_key")
        null = _campaign_for_pair(campaign_by_key, pair, field="null_campaign_key")
        positive_hash = str(positive.get("initial_label_ids_hash") or "")
        null_hash = str(null.get("initial_label_ids_hash") or "")
        if positive_hash and null_hash and positive_hash != null_hash:
            raise ValueError(
                "Stage B replicate positive/null initial label IDs differ "
                f"for label={pair.get('label_name')} seed={entry.seed}"
            )
        positive_ids = _initial_label_ids(Path(str(positive["initial_label_input_path"])))
        null_ids = _initial_label_ids(Path(str(null["initial_label_input_path"])))
        if positive_ids != null_ids:
            raise ValueError(
                "Stage B replicate positive/null initial label IDs differ "
                f"for label={pair.get('label_name')} seed={entry.seed}"
            )


def _campaign_for_pair(
    campaign_by_key: Mapping[str, Mapping[str, Any]],
    pair: Mapping[str, Any],
    *,
    field: str,
) -> Mapping[str, Any]:
    key = str(pair.get(field) or "")
    try:
        return campaign_by_key[key]
    except KeyError as exc:
        raise ValueError(f"Stage B replicate pair references unknown campaign key: {key}") from exc


def _initial_label_ids(path: Path) -> tuple[str, ...]:
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"Stage B replicate initial label input missing: {path}")
    frame = pd.read_parquet(path, columns=["id"])
    if "id" not in frame.columns:
        raise ValueError(f"Stage B replicate initial label input missing id column: {path}")
    ids = tuple(sorted(frame["id"].astype(str).tolist()))
    if len(set(ids)) != len(ids):
        raise ValueError(f"Stage B replicate initial label input contains duplicate IDs: {path}")
    return ids
