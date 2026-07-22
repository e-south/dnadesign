"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/stage_b/learning_loop_baselines/sources.py

Source artifact loaders for TFBS learning-loop baselines.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from ..review.io import campaign_workdir, label_table, read_review_manifest, selection_table
from .contracts import COUNT_FRACTION_LEARNING_LOOP_SPEC, LearningLoopBaselineSpec


def load_learning_loop_manifest(path: Path, *, spec: LearningLoopBaselineSpec) -> dict[str, Any]:
    """Read one Stage B config manifest and fail on surfaces outside the review spec."""

    manifest = read_review_manifest(path)
    target_profile = manifest.get("target_profile")
    if not isinstance(target_profile, Mapping):
        raise ValueError(f"Learning-loop baseline manifest is missing target_profile: {path}")
    profile_id = str(target_profile.get("profile_id") or "")
    if profile_id not in spec.accepted_profile_ids:
        raise ValueError(
            "Learning-loop baseline received a config manifest outside the requested review surface: "
            f"review_id={spec.review_id!r} accepted={sorted(spec.accepted_profile_ids)} "
            f"got={profile_id!r} from {path}"
        )
    if str(manifest.get("status") or "") != "PASS":
        raise ValueError(f"Learning-loop baseline requires a PASS config manifest: {path}")
    return manifest


def load_learning_loop_manifests(paths: Iterable[Path], *, spec: LearningLoopBaselineSpec) -> list[dict[str, Any]]:
    """Return validated config manifests sorted by profile and seed."""

    manifests = [load_learning_loop_manifest(Path(path), spec=spec) for path in paths]
    if not manifests:
        raise ValueError("Learning-loop baseline requires at least one config manifest")
    keys = [(str(manifest["target_profile"]["profile_id"]), int(manifest["seed"])) for manifest in manifests]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Learning-loop baseline received duplicate profile/seed manifests: {keys}")
    return sorted(manifests, key=lambda item: (str(item["target_profile"]["profile_id"]), int(item["seed"])))


def load_count_fraction_manifests(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Return validated count-fraction manifests sorted by seed."""

    return load_learning_loop_manifests(paths, spec=COUNT_FRACTION_LEARNING_LOOP_SPEC)


def campaign_rows(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = manifest.get("campaigns")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Learning-loop baseline manifest requires non-empty campaign rows")
    return [row for row in rows if isinstance(row, Mapping)]


def pair_rows(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = manifest.get("pairs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Learning-loop baseline manifest requires non-empty pair rows")
    return [row for row in rows if isinstance(row, Mapping)]


def validate_shared_pair_contracts(manifest: Mapping[str, Any]) -> None:
    """Fail if positive/control pairs do not share starts, scope, and budget."""

    campaigns = {str(row["campaign_key"]): row for row in campaign_rows(manifest)}
    for pair in pair_rows(manifest):
        positive = campaigns[str(pair["positive_campaign_key"])]
        control = campaigns[str(pair["null_campaign_key"])]
        fields = ("label_name", "seed", "selection_k", "rounds", "candidate_scope_hash", "initial_label_ids_hash")
        for field in fields:
            if str(positive.get(field)) != str(control.get(field)):
                raise ValueError(
                    "Learning-loop baseline requires positive/control parity for "
                    f"{field}: positive={positive.get(field)!r}, control={control.get(field)!r}"
                )
        positive_ids = initial_seed_ids(Path(str(positive["initial_label_input_path"])))
        control_ids = initial_seed_ids(Path(str(control["initial_label_input_path"])))
        if positive_ids != control_ids:
            raise ValueError(f"Learning-loop baseline positive/control initial IDs differ for {pair.get('label_name')}")


def campaign_label_table(campaign: Mapping[str, Any]) -> pd.DataFrame:
    """Load the label table for one campaign."""

    return label_table(Path(str(campaign["label_table_path"])), label_name=str(campaign["label_name"]))


def initial_seed_ids(path: Path) -> list[str]:
    """Return ordered initial labeled IDs from a round-0 label input parquet."""

    if not path.exists():
        raise FileNotFoundError(f"Learning-loop baseline initial label input missing: {path}")
    frame = pd.read_parquet(path, columns=["id"])
    ids = frame["id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError(f"Learning-loop baseline initial label input contains duplicate id(s): {path}")
    return ids


def active_selection_frame(campaign: Mapping[str, Any], *, rounds: int) -> pd.DataFrame:
    """Return active retraining selections for a completed campaign."""

    workdir = campaign_workdir(Path(str(campaign["config_path"])))
    rows: list[pd.DataFrame] = []
    selection_k = int(campaign["selection_k"])
    for round_index in range(int(rounds)):
        frame = selection_table(workdir, round_index=round_index)
        if len(frame) != selection_k:
            raise ValueError(
                f"Learning-loop baseline expected {selection_k} active selections but found {len(frame)} "
                f"for {campaign.get('campaign_key')} round {round_index}"
            )
        out = frame.loc[:, ["id"]].copy()
        out["id"] = out["id"].astype(str)
        out["round"] = int(round_index)
        out["selection_source"] = "active_retraining"
        rows.append(out[["round", "id", "selection_source"]])
    selected = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["round", "id", "selection_source"])
    if selected["id"].duplicated().any():
        duplicates = selected.loc[selected["id"].duplicated(), "id"].drop_duplicates().head(5).tolist()
        raise ValueError(f"Learning-loop baseline active selections contain duplicate selected id(s): {duplicates}")
    return selected
