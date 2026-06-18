"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/batch0/select.py

Batch-0 selector for the stress / ethanol / ciprofloxacin OPAL campaigns.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from dnadesign.opal import CandidateEligibilityBlock, PluginRef, apply_candidate_eligibility

from ..synthesis_handoff.strategy import load_cloning_strategy
from .candidate_table import (
    validate_configured_candidate_feature_table,
    validate_selected_ids_against_candidate_feature_table,
)

REQUIRED_REVIEW_COLUMNS: tuple[str, ...] = (
    "campaign",
    "slot",
    "id",
    "sequence",
    "setpoint",
    "canonical_densegen_plan",
    "regulator_composition",
    "sigma35_variant",
    "spacer_length",
    "target_margin",
    "off_target_margins",
    "tfbs_summary",
    "motif_score_summary",
    "x_provenance",
)

_SIGMA35_STRENGTH_RANK = {"f": 0, "e": 1, "d": 2, "c": 3, "b": 4}
_EXPLORATORY_SIGMA35_RANK = {"d": 0, "c": 1, "e": 2, "f": 3, "b": 4}
_REGULATOR_ORDER = {"baeR": 0, "cpxR": 1, "lexA": 2}


def load_sampling_config(path: str | Path) -> dict[str, Any]:
    """Load the batch-0 sampling config with a small structural check."""

    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if "campaigns" not in config or not isinstance(config["campaigns"], list):
        raise ValueError("sampling config must define a campaigns list")
    return config


def _repo_root_from(path: Path) -> Path:
    for parent in [path.resolve(), *path.resolve().parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"could not resolve repo root from {path}")


def _resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def _strategy_eligibility_params(
    *,
    strategy_path: Path,
    sequence_column: str,
    min_remaining_candidates: int | None,
) -> dict[str, Any]:
    strategy = load_cloning_strategy(strategy_path)
    if not strategy.restriction_sites:
        raise ValueError(f"synthesis eligibility strategy has no restriction_sites: {strategy_path}")
    params: dict[str, Any] = {
        "sequence_column": sequence_column,
        "scan_space": "final_assembled_insert",
        "assembly_strategy_ref": strategy.strategy_id,
        "left_flank": strategy.left_flank,
        "right_flank": strategy.right_flank,
        "expected_core_length": strategy.expected_core_length,
        "forbidden_sites": [site.to_json() for site in strategy.restriction_sites],
    }
    if min_remaining_candidates is not None:
        params["min_remaining_candidates"] = int(min_remaining_candidates)
    return params


def _apply_synthesis_eligibility(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> pd.DataFrame:
    raw = config.get("synthesis_eligibility")
    if raw is None:
        return frame
    if not isinstance(raw, Mapping):
        raise ValueError("synthesis_eligibility must be a mapping")
    strategy_yaml = raw.get("strategy_yaml")
    if strategy_yaml is None or not str(strategy_yaml).strip():
        raise ValueError("synthesis_eligibility.strategy_yaml must be non-empty")
    params = _strategy_eligibility_params(
        strategy_path=_resolve_repo_path(repo_root, str(strategy_yaml)),
        sequence_column=str(raw.get("sequence_column", "sequence")),
        min_remaining_candidates=(
            None if raw.get("min_remaining_candidates") is None else int(raw["min_remaining_candidates"])
        ),
    )
    result = apply_candidate_eligibility(
        frame,
        CandidateEligibilityBlock(
            rules=[
                PluginRef(
                    name="restriction_site_exclusion",
                    params=params,
                )
            ]
        ),
    )
    return result.frame


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def _normal_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _canonical_plan(value: Any) -> str:
    raw = _normal_text(value)
    if not raw:
        return ""
    if raw.startswith("plan_pool__"):
        raw = raw.removeprefix("plan_pool__")
    if "__sig35" in raw:
        raw = raw.split("__sig35", 1)[0]
    return raw


def _regulator_base(value: Any) -> str:
    raw = _normal_text(value)
    if not raw or raw == "background":
        return ""
    for regulator in ("baeR", "cpxR", "lexA"):
        if raw.startswith(regulator):
            return regulator
    return raw.split("_", 1)[0]


def _slot_regulator(value: Any) -> str:
    raw = _normal_text(value)
    if not raw:
        return "background"
    lowered = raw.lower()
    if lowered in {"background", "bg", "none", "null"} or "background" in lowered:
        return "background"
    if lowered.startswith("baer"):
        return "baeR"
    if lowered.startswith("cpxr"):
        return "cpxR"
    if lowered.startswith("lexa"):
        return "lexA"
    regulator = _regulator_base(raw)
    if regulator in _REGULATOR_ORDER:
        return regulator
    raise ValueError(f"unknown TFBS regulator for slot pattern: {value!r}")


def _normalize_regulator_composition(value: Any) -> str:
    raw = _normal_text(value)
    if not raw:
        return ""
    if raw in {"background", "control"}:
        return raw
    parts = [_regulator_base(part) for part in raw.replace(",", "+").split("+")]
    regulators = sorted({part for part in parts if part}, key=lambda item: _REGULATOR_ORDER.get(item, 99))
    return "+".join(regulators)


def _normalize_detail_cell(cell: Any) -> list[dict[str, Any]]:
    if _is_missing(cell):
        return []
    if isinstance(cell, dict):
        return [cell]
    if isinstance(cell, str):
        stripped = cell.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return []
        return _normalize_detail_cell(parsed)
    if isinstance(cell, np.ndarray):
        return _normalize_detail_cell(cell.tolist())
    if isinstance(cell, Sequence) and not isinstance(cell, bytes | bytearray | str):
        return [item for item in cell if isinstance(item, dict)]
    return []


def _strict_detail_entries(cell: Any, *, row_id: str) -> list[dict[str, Any]]:
    if _is_missing(cell):
        raise ValueError(f"{row_id}: missing densegen__used_tfbs_detail")
    if isinstance(cell, str):
        stripped = cell.strip()
        if not stripped:
            raise ValueError(f"{row_id}: missing densegen__used_tfbs_detail")
        try:
            cell = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{row_id}: densegen__used_tfbs_detail is not valid JSON: {exc}") from exc
    if isinstance(cell, np.ndarray):
        cell = cell.tolist()
    if isinstance(cell, tuple):
        cell = list(cell)
    if not isinstance(cell, list):
        raise ValueError(f"{row_id}: densegen__used_tfbs_detail must be a list")
    entries: list[dict[str, Any]] = []
    for item in cell:
        if not isinstance(item, dict):
            raise ValueError(f"{row_id}: densegen__used_tfbs_detail entries must be mappings")
        entries.append(item)
    return entries


def _strict_slot_regulator_pattern(row: Mapping[str, Any]) -> tuple[str, str, str]:
    row_id = _normal_text(row.get("id")) or "<unknown>"
    entries = _strict_detail_entries(row.get("densegen__used_tfbs_detail"), row_id=row_id)
    tfbs_entries = [entry for entry in entries if entry.get("part_kind") == "tfbs"]
    if len(tfbs_entries) != 3:
        raise ValueError(f"{row_id}: expected exactly 3 TFBS entries for slot_regulator_pattern")
    offsets: list[int] = []
    for entry in tfbs_entries:
        if "offset_raw" not in entry or _is_missing(entry.get("offset_raw")):
            raise ValueError(f"{row_id}: offset_raw is required for slot_regulator_pattern")
        offsets.append(int(entry["offset_raw"]))
    if len(set(offsets)) != 3:
        raise ValueError(f"{row_id}: ambiguous slot_regulator_pattern from tied offset_raw values")
    ordered = [entry for _, entry in sorted(zip(offsets, tfbs_entries), key=lambda item: item[0])]
    return tuple(_slot_regulator(entry.get("regulator")) for entry in ordered)  # type: ignore[return-value]


def _signal_tfbs(detail: Any) -> list[dict[str, Any]]:
    entries = []
    for item in _normalize_detail_cell(detail):
        if item.get("part_kind") != "tfbs":
            continue
        regulator = _regulator_base(item.get("regulator"))
        if not regulator:
            continue
        entries.append({**item, "regulator_base": regulator})
    return entries


def _tfbs_summary(detail: Any) -> str:
    entries = _signal_tfbs(detail)
    if not entries:
        return "none"
    parts = []
    for entry in sorted(entries, key=lambda item: (str(item.get("regulator_base")), item.get("offset") or 0)):
        regulator = entry.get("regulator_base")
        offset = entry.get("offset")
        orientation = entry.get("orientation") or "?"
        parts.append(f"{regulator}@{offset}:{orientation}")
    return ";".join(parts)


def _tfbs_offsets(detail: Any) -> str:
    offsets = [str(entry.get("offset")) for entry in _signal_tfbs(detail) if entry.get("offset") is not None]
    return ",".join(offsets) if offsets else "none"


def _tfbs_orientations(detail: Any) -> str:
    orientations = sorted({str(entry.get("orientation")) for entry in _signal_tfbs(detail) if entry.get("orientation")})
    return ",".join(orientations) if orientations else "none"


def _motif_tiers(detail: Any) -> str:
    tiers = sorted({str(int(entry.get("tier"))) for entry in _signal_tfbs(detail) if entry.get("tier") is not None})
    return ",".join(tiers) if tiers else "none"


def _motif_score_summary(detail: Any) -> str:
    entries = _signal_tfbs(detail)
    if not entries:
        return "none"
    parts = []
    for entry in sorted(entries, key=lambda item: (str(item.get("regulator_base")), item.get("offset") or 0)):
        regulator = entry.get("regulator_base")
        tier = entry.get("tier")
        rel = entry.get("score_relative_to_theoretical_max")
        if rel is None:
            parts.append(f"{regulator}:tier={tier}")
        else:
            parts.append(f"{regulator}:tier={tier},rel={float(rel):.3f}")
    return ";".join(parts)


def _tfbs_regulator_set_from_detail(detail: Any) -> str:
    regulators = {_regulator_base(entry.get("regulator_base")) for entry in _signal_tfbs(detail)}
    regulators = {regulator for regulator in regulators if regulator}
    return "+".join(sorted(regulators, key=lambda item: _REGULATOR_ORDER.get(item, 99)))


def _tfbs_regulator_set_from_summary(summary: Any) -> str:
    raw = _normal_text(summary)
    if not raw or raw == "none":
        return ""
    regulators: set[str] = set()
    for part in raw.split(";"):
        token = part.split("@", 1)[0].split(":", 1)[0]
        regulator = _regulator_base(token)
        if regulator:
            regulators.add(regulator)
    return "+".join(sorted(regulators, key=lambda item: _REGULATOR_ORDER.get(item, 99)))


def _slot_signal_tfbs_count(pattern: Sequence[str]) -> int:
    return sum(1 for regulator in pattern if regulator != "background")


def _requested_slot_pattern(slot: Mapping[str, Any]) -> tuple[str, str, str] | None:
    raw = slot.get("slot_regulator_pattern")
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, bytes | bytearray | str) or len(raw) != 3:
        raise ValueError("slot_regulator_pattern must be a 3-item sequence")
    return tuple(_slot_regulator(value) for value in raw)  # type: ignore[return-value]


def _count_constraint_matches(value: int, raw: Any) -> bool:
    if raw is None:
        return True
    if isinstance(raw, Mapping):
        if "eq" in raw and value != int(raw["eq"]):
            return False
        if "min" in raw and value < int(raw["min"]):
            return False
        if "max" in raw and value > int(raw["max"]):
            return False
        return True
    return value == int(raw)


def _ensure_candidate_columns(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    if "id" not in out.columns:
        if "construct__anchor_id" not in out.columns:
            raise ValueError("candidate rows require either id or construct__anchor_id")
        out["id"] = out["construct__anchor_id"].astype(str)
    else:
        out["id"] = out["id"].astype(str)

    if "canonical_densegen_plan" not in out.columns:
        source = out.get("design_family", out.get("densegen__plan", ""))
        out["canonical_densegen_plan"] = pd.Series(source, index=out.index).map(_canonical_plan)
    else:
        out["canonical_densegen_plan"] = out["canonical_densegen_plan"].map(_canonical_plan)

    if "regulator_composition" not in out.columns:
        source = out.get("design_regulator_composition", "")
        out["regulator_composition"] = pd.Series(source, index=out.index).map(_normalize_regulator_composition)
    else:
        out["regulator_composition"] = out["regulator_composition"].map(_normalize_regulator_composition)

    if "sigma35_variant" not in out.columns:
        out["sigma35_variant"] = out.get("sig35_variant", "")
    out["sigma35_variant"] = out["sigma35_variant"].map(lambda value: _normal_text(value).lower())

    if "tfbs_summary" not in out.columns:
        out["tfbs_summary"] = out.get("densegen__used_tfbs_detail", pd.Series([None] * len(out))).map(_tfbs_summary)
    if "tfbs_offset_summary" not in out.columns:
        out["tfbs_offset_summary"] = out.get("densegen__used_tfbs_detail", pd.Series([None] * len(out))).map(
            _tfbs_offsets
        )
    if "tfbs_orientation_summary" not in out.columns:
        out["tfbs_orientation_summary"] = out.get("densegen__used_tfbs_detail", pd.Series([None] * len(out))).map(
            _tfbs_orientations
        )
    if "motif_tier_summary" not in out.columns:
        detail_source = out.get("densegen__used_tfbs_detail", pd.Series([None] * len(out)))
        out["motif_tier_summary"] = detail_source.map(_motif_tiers)
    if "motif_score_summary" not in out.columns:
        out["motif_score_summary"] = out.get("densegen__used_tfbs_detail", pd.Series([None] * len(out))).map(
            _motif_score_summary
        )
    if "tfbs_regulators" not in out.columns:
        if "densegen__used_tfbs_detail" in out.columns:
            out["tfbs_regulators"] = out["densegen__used_tfbs_detail"].map(_tfbs_regulator_set_from_detail)
        else:
            out["tfbs_regulators"] = out["tfbs_summary"].map(_tfbs_regulator_set_from_summary)
    if "x_provenance" not in out.columns:
        out["x_provenance"] = "intermediate_embedding_7b_context_anchor_mean_bidir_concat"
    return out


def build_candidate_frame(config: Mapping[str, Any], *, repo_root: str | Path | None = None) -> pd.DataFrame:
    """Build selector input rows from configured LatentDNA and USR artifacts."""

    root = Path(repo_root) if repo_root is not None else _repo_root_from(Path(__file__))
    artifacts = dict(config.get("source_artifacts", {}) or {})
    margins_path = _resolve_repo_path(root, artifacts["latentdna_margins"])
    anchor_path = _resolve_repo_path(root, artifacts["densegen_anchor_records"])

    margins = pd.read_parquet(margins_path)
    if "construct__anchor_id" not in margins.columns:
        raise ValueError(f"{margins_path} is missing construct__anchor_id")
    margins = margins.copy()
    margins["id"] = margins["construct__anchor_id"].astype(str)

    sigma_path_raw = artifacts.get("latentdna_sigma35_stress_margins")
    if sigma_path_raw:
        sigma_path = _resolve_repo_path(root, sigma_path_raw)
        sigma = pd.read_parquet(sigma_path)
        keep = [column for column in ("construct__anchor_id", "sig35_margin_f_vs_b") if column in sigma.columns]
        if keep:
            sigma = sigma[keep].copy()
            sigma["id"] = sigma["construct__anchor_id"].astype(str)
            margins = margins.merge(
                sigma.drop(columns=["construct__anchor_id"]),
                on="id",
                how="left",
                suffixes=("", "__sigma"),
            )

    anchor = pd.read_parquet(anchor_path)
    anchor = anchor.copy()
    anchor["id"] = anchor["id"].astype(str)
    anchor_cols = [
        column
        for column in (
            "id",
            "bio_type",
            "sequence",
            "alphabet",
            "densegen__plan",
            "densegen__used_tfbs",
            "densegen__used_tfbs_detail",
            "densegen__used_tf_counts",
            "densegen__required_regulators",
        )
        if column in anchor.columns
    ]
    out = margins.merge(anchor[anchor_cols], on="id", how="left", suffixes=("", "__anchor"))
    candidate_table = dict(config.get("candidate_feature_table", {}) or {})
    out["x_provenance"] = candidate_table.get(
        "x_source",
        {},
    ).get("view_id", "intermediate_embedding_7b_context_anchor_mean_bidir_concat")
    return _ensure_candidate_columns(out)


def _slot_matches(frame: pd.DataFrame, slot: Mapping[str, Any]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    plans = slot.get("plans")
    if plans is None and slot.get("plan") is not None:
        plans = [slot["plan"]]
    if plans:
        mask &= frame["canonical_densegen_plan"].isin([_canonical_plan(plan) for plan in plans])

    compositions = slot.get("regulator_compositions")
    if compositions:
        wanted = {_normalize_regulator_composition(item) for item in compositions}
        mask &= frame["regulator_composition"].isin(wanted)

    regulators_all = slot.get("regulators_all") or []
    for regulator in regulators_all:
        reg = _regulator_base(regulator)
        mask &= frame["regulator_composition"].map(lambda value: reg in value.split("+"))

    regulators_any = slot.get("regulators_any") or []
    if regulators_any:
        wanted_any = {_regulator_base(regulator) for regulator in regulators_any}
        mask &= frame["regulator_composition"].map(
            lambda value: bool(wanted_any.intersection(set(str(value).split("+"))))
        )

    tfbs_required = slot.get("require_tfbs_regulators_all") or []
    for regulator in tfbs_required:
        reg = _regulator_base(regulator)
        mask &= frame["tfbs_regulators"].map(lambda value: reg in str(value).split("+"))

    tfbs_excluded = slot.get("exclude_tfbs_regulators") or []
    for regulator in tfbs_excluded:
        reg = _regulator_base(regulator)
        mask &= frame["tfbs_regulators"].map(lambda value: reg not in str(value).split("+"))

    allowed_sigma = slot.get("allowed_sigma35_variants")
    if allowed_sigma:
        mask &= frame["sigma35_variant"].isin([str(value).lower() for value in allowed_sigma])

    requested_pattern = _requested_slot_pattern(slot)
    requested_count = slot.get("signal_tfbs_count")
    if requested_pattern is not None or requested_count is not None:
        pattern_by_index = frame.loc[mask].apply(_strict_slot_regulator_pattern, axis=1)
        if requested_pattern is not None:
            pattern_mask = pattern_by_index == requested_pattern
            mask &= pattern_mask.reindex(frame.index, fill_value=False)
        if requested_count is not None:
            count_mask = pattern_by_index.map(
                lambda pattern: _count_constraint_matches(_slot_signal_tfbs_count(pattern), requested_count)
            )
            mask &= count_mask.reindex(frame.index, fill_value=False)
    return mask


def _ranked_slot_candidates(
    frame: pd.DataFrame,
    *,
    target_margin_column: str,
    sigma35_mode: str,
    diversity_keys: Sequence[str],
    already_selected: pd.DataFrame,
) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["_target_margin"] = pd.to_numeric(ranked[target_margin_column], errors="coerce")
    if sigma35_mode == "exploratory":
        sigma_rank = _EXPLORATORY_SIGMA35_RANK
    else:
        sigma_rank = _SIGMA35_STRENGTH_RANK
    ranked["_sigma_rank"] = ranked["sigma35_variant"].map(lambda value: sigma_rank.get(str(value), 99))

    used_values = {
        key: set(already_selected[key].dropna().astype(str).tolist())
        for key in diversity_keys
        if key in ranked.columns and key in already_selected.columns
    }
    if used_values:
        ranked["_diversity_penalty"] = ranked.apply(
            lambda row: sum(str(row.get(key)) in values for key, values in used_values.items()),
            axis=1,
        )
    else:
        ranked["_diversity_penalty"] = 0

    return ranked.sort_values(
        by=["_diversity_penalty", "_target_margin", "_sigma_rank", "id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )


def _select_for_slot(
    candidates: pd.DataFrame,
    *,
    slot: Mapping[str, Any],
    campaign: Mapping[str, Any],
    selected_for_campaign: pd.DataFrame,
    used_ids: set[str],
    require_positive_target_margin: bool,
    diversity_keys: Sequence[str],
    allow_duplicate_ids: bool,
) -> pd.DataFrame:
    target_col = str(campaign["target_margin_column"])
    count = int(slot.get("count", 1))
    pool = candidates[_slot_matches(candidates, slot)].copy()

    if target_col not in pool.columns:
        raise ValueError(f"missing target margin column {target_col!r}")
    pool[target_col] = pd.to_numeric(pool[target_col], errors="coerce")
    pool = pool[pool[target_col].notna()]
    if require_positive_target_margin:
        pool = pool[pool[target_col] > 0]
    if not allow_duplicate_ids:
        pool = pool[~pool["id"].isin(used_ids)]

    mode = str(slot.get("sigma35_mode", "strong"))
    if mode == "exploratory":
        exploratory = pool[pool["sigma35_variant"].isin(["d", "c"])]
        if not exploratory.empty:
            pool = exploratory

    ranked = _ranked_slot_candidates(
        pool,
        target_margin_column=target_col,
        sigma35_mode=mode,
        diversity_keys=diversity_keys,
        already_selected=selected_for_campaign,
    )
    if len(ranked) < count:
        raise ValueError(
            f"slot {slot.get('name', '<unnamed>')!r} for campaign {campaign['slug']!r} "
            f"requires {count} rows but only {len(ranked)} candidates passed filters"
        )
    return ranked.head(count).copy()


def select_batch0(
    candidates: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> pd.DataFrame:
    """Select reviewed batch-0 rows for all configured OPAL campaigns."""

    root = Path(repo_root) if repo_root is not None else _repo_root_from(Path(__file__))
    frame = _ensure_candidate_columns(candidates)
    filters = dict(config.get("filters", {}) or {})
    diversity = dict(config.get("diversity", {}) or {})
    require_positive = bool(filters.get("require_positive_target_margin", True))
    allow_duplicate_ids = bool(config.get("allow_duplicate_ids", False))
    diversity_keys = list(diversity.get("keys", []))
    include_source_class = set(filters.get("include_source_class", []) or [])
    if include_source_class and "source_class" in frame.columns:
        frame = frame[frame["source_class"].isin(include_source_class)].copy()
    exclude_source_class = set(filters.get("exclude_source_class", []) or [])
    if exclude_source_class and "source_class" in frame.columns:
        frame = frame[~frame["source_class"].isin(exclude_source_class)].copy()
    for column in filters.get("exclude_non_null_columns") or []:
        column_name = str(column)
        if column_name in frame.columns:
            frame = frame[frame[column_name].isna()].copy()
    allowed_spacers = filters.get("allowed_spacer_lengths") or []
    if allowed_spacers:
        wanted_spacers = {int(value) for value in allowed_spacers}
        frame = frame[pd.to_numeric(frame["spacer_length"], errors="coerce").isin(wanted_spacers)].copy()
    frame = _apply_synthesis_eligibility(frame, config, repo_root=root)
    selected_frames: list[pd.DataFrame] = []
    used_ids: set[str] = set()

    for campaign in config["campaigns"]:
        campaign_selected = pd.DataFrame(columns=frame.columns)
        for slot in campaign.get("slots", []):
            chosen = _select_for_slot(
                frame,
                slot=slot,
                campaign=campaign,
                selected_for_campaign=campaign_selected,
                used_ids=used_ids,
                require_positive_target_margin=require_positive,
                diversity_keys=diversity_keys,
                allow_duplicate_ids=allow_duplicate_ids,
            )
            if not allow_duplicate_ids:
                used_ids.update(chosen["id"].astype(str).tolist())
            slot_name = str(slot.get("name", "slot"))
            if len(chosen) == 1:
                chosen["slot"] = slot_name
            else:
                chosen["slot"] = [f"{slot_name}_{idx + 1}" for idx in range(len(chosen))]
            chosen["campaign"] = str(campaign["slug"])
            chosen["setpoint"] = [list(campaign["setpoint_vector"]) for _ in range(len(chosen))]
            chosen["target_margin"] = pd.to_numeric(chosen[campaign["target_margin_column"]], errors="coerce")
            off_target_cols = list(campaign.get("off_target_margin_columns", []))
            chosen["off_target_margins"] = chosen.apply(
                lambda row: json.dumps(
                    {
                        column: float(row[column])
                        for column in off_target_cols
                        if column in row and pd.notna(row[column])
                    },
                    sort_keys=True,
                ),
                axis=1,
            )
            campaign_selected = pd.concat([campaign_selected, chosen], ignore_index=True)
            selected_frames.append(chosen)

    if not selected_frames:
        return pd.DataFrame(columns=REQUIRED_REVIEW_COLUMNS)

    selected = pd.concat(selected_frames, ignore_index=True)
    missing = [column for column in REQUIRED_REVIEW_COLUMNS if column not in selected.columns]
    if missing:
        raise ValueError(f"selected review table is missing required columns: {missing}")
    return selected.loc[:, REQUIRED_REVIEW_COLUMNS].copy()


def _write_selection_outputs(selected: pd.DataFrame, config: Mapping[str, Any], *, repo_root: Path) -> list[Path]:
    outputs = dict(config.get("outputs", {}) or {})
    written: list[Path] = []
    for key in ("review_csv", "review_parquet"):
        raw = outputs.get(key)
        if not raw:
            continue
        path = _resolve_repo_path(repo_root, raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            selected.to_parquet(path, index=False)
        else:
            selected.to_csv(path, index=False)
        written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select pre-assay OPAL batch-0 review rows.")
    parser.add_argument("--config", default=Path(__file__).with_name("sampling.yaml"), type=Path)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--write", action="store_true", help="Write configured review CSV/parquet outputs.")
    args = parser.parse_args(argv)

    config = load_sampling_config(args.config)
    repo_root = args.repo_root or _repo_root_from(args.config)
    candidate_table_report = validate_configured_candidate_feature_table(config, repo_root=repo_root)
    candidates = build_candidate_frame(config, repo_root=repo_root)
    selected = select_batch0(candidates, config, repo_root=repo_root)
    selection_table_report = validate_selected_ids_against_candidate_feature_table(
        selected,
        config,
        repo_root=repo_root,
    )
    summary = selected.groupby("campaign").size().to_dict()
    print(
        json.dumps(
            {
                "candidate_feature_table": candidate_table_report,
                "selected": summary,
                "selection_candidate_table": selection_table_report,
            },
            sort_keys=True,
        )
    )
    if args.write:
        written = _write_selection_outputs(selected, config, repo_root=repo_root)
        print(
            json.dumps(
                {
                    "written": [str(path) for path in written],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
