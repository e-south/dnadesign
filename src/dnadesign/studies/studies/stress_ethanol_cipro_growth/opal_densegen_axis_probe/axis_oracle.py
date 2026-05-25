"""Study-owned DenseGen axis OPAL probe package."""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .artifacts import AxisLabel
from .constants import (
    AXIS_CLASS_TO_LOGIC4,
    DEFAULT_SEED,
    NULL_ORACLE_ID,
    ORACLE_ID,
    PLAN_TO_AXIS_CLASS,
    SFXI_INTENSITY_COLUMNS,
    SFXI_STATE_COLUMNS,
)
from .label_families import densegen_plan_class_from_axis_class, tf_family_columns


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if value is pd.NA:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def _clean_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _base_plan(plan: Any) -> str | None:
    text = _clean_text(plan)
    if not text:
        return None
    return text.split("__sig35=", 1)[0]


def parse_sigma35_variant(plan: Any) -> str | None:
    text = _clean_text(plan)
    if not text:
        return None
    match = re.search(r"__sig35=([^_]+)$", text)
    if not match:
        return None
    variant = match.group(1).strip()
    return variant or None


def _normalize_detail_entries(value: Any) -> list[Mapping[str, Any]] | None:
    if _is_missing(value):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"used_tfbs_detail is not valid JSON: {exc}") from exc
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError(f"used_tfbs_detail must be a list of mappings, got {type(value).__name__}")
    entries: list[Mapping[str, Any]] = []
    for item in value:
        if hasattr(item, "as_py"):
            item = item.as_py()
        if not isinstance(item, Mapping):
            raise ValueError("used_tfbs_detail entries must be mappings")
        entries.append(item)
    return entries


def _regulator_family(value: Any) -> str:
    text = _clean_text(value).lower()
    if not text:
        return ""
    if "lexa" in text:
        return "lexA"
    if "cpxr" in text:
        return "cpxR"
    if "baer" in text:
        return "baeR"
    if "background" in text:
        return "background"
    return text


def _counts_from_detail(detail: Any) -> tuple[int, int, int, int]:
    entries = _normalize_detail_entries(detail)
    if entries is None:
        raise ValueError("missing used_tfbs_detail")
    lex_a = cpx_r = bae_r = background = 0
    for entry in entries:
        part_kind = _clean_text(entry.get("part_kind")).lower()
        if not part_kind:
            raise ValueError("used_tfbs_detail entry missing part_kind")
        if part_kind == "fixed_element":
            continue
        if part_kind != "tfbs":
            raise ValueError(f"unsupported used_tfbs_detail part_kind: {part_kind}")
        family = _regulator_family(entry.get("regulator"))
        if family == "lexA":
            lex_a += 1
        elif family == "cpxR":
            cpx_r += 1
        elif family == "baeR":
            bae_r += 1
        elif family == "background":
            background += 1
        else:
            raise ValueError(f"unsupported tfbs regulator family: {family or 'missing'}")
    return lex_a, cpx_r, bae_r, background


def derive_axis_label(row: Mapping[str, Any]) -> AxisLabel:
    candidate_id = _clean_text(row.get("id"))
    plan = _clean_text(row.get("densegen__plan")) or None
    sigma35_variant = parse_sigma35_variant(plan)
    plan_base = _base_plan(plan)
    expected = PLAN_TO_AXIS_CLASS.get(plan_base or "")
    if plan_base and expected is None:
        unsupported_plan = True
    else:
        unsupported_plan = False

    try:
        lex_a, cpx_r, bae_r, background = _counts_from_detail(row.get("densegen__used_tfbs_detail"))
    except ValueError as exc:
        message = str(exc)
        flag = "missing_used_tfbs_detail" if "missing used_tfbs_detail" in message else "malformed_used_tfbs_detail"
        return AxisLabel(
            id=candidate_id,
            axis_class=None,
            logic4=None,
            effect4=None,
            vec8=None,
            quality_flag=flag,
            sigma35_variant=sigma35_variant,
            densegen_plan=plan,
            expected_axis_class_from_plan=expected,
        )

    cipro_axis = lex_a > 0
    ethanol_axis = (cpx_r + bae_r) > 0
    if cipro_axis and ethanol_axis:
        axis_class = "dual_axis_and"
    elif cipro_axis:
        axis_class = "cipro_only"
    elif ethanol_axis:
        axis_class = "ethanol_only"
    else:
        axis_class = "background_only"
    logic4 = list(AXIS_CLASS_TO_LOGIC4[axis_class])
    effect4 = list(logic4)
    vec8 = [*logic4, *effect4]
    densegen_plan_class = densegen_plan_class_from_axis_class(axis_class)

    if unsupported_plan:
        flag = "unsupported_plan"
    elif not sigma35_variant:
        flag = "missing_sigma35_variant"
    elif expected is not None and expected != axis_class:
        flag = "plan_axis_mismatch"
    else:
        flag = "ok"

    return AxisLabel(
        id=candidate_id,
        axis_class=axis_class,
        logic4=logic4,
        effect4=effect4,
        vec8=vec8,
        quality_flag=flag,
        lexA_count=lex_a,
        cpxR_count=cpx_r,
        baeR_count=bae_r,
        background_count=background,
        cipro_axis=cipro_axis,
        ethanol_axis=ethanol_axis,
        densegen_plan_class=densegen_plan_class,
        sigma35_variant=sigma35_variant,
        densegen_plan=plan,
        expected_axis_class_from_plan=expected,
    )


def _metadata_equal(left: Any, right: Any) -> bool:
    if _is_missing(left) or _is_missing(right):
        return True
    if hasattr(left, "as_py"):
        left = left.as_py()
    if hasattr(right, "as_py"):
        right = right.as_py()
    if isinstance(left, np.ndarray):
        left = left.tolist()
    if isinstance(right, np.ndarray):
        right = right.tolist()
    return left == right


def _overlay_densegen_sidecar(candidates: pd.DataFrame, densegen_sidecar: pd.DataFrame | None) -> pd.DataFrame:
    if densegen_sidecar is None or densegen_sidecar.empty:
        return candidates.copy()
    if "id" not in candidates.columns or "id" not in densegen_sidecar.columns:
        raise ValueError("candidate records and densegen sidecar must both include id")
    if candidates["id"].duplicated().any():
        duplicate_ids = candidates.loc[candidates["id"].duplicated(), "id"].astype(str).head(5).tolist()
        raise ValueError(f"candidate records contain duplicate id values: {duplicate_ids}")
    if densegen_sidecar["id"].duplicated().any():
        duplicate_ids = densegen_sidecar.loc[densegen_sidecar["id"].duplicated(), "id"].astype(str).head(5).tolist()
        raise ValueError(f"DenseGen sidecar contains duplicate id values: {duplicate_ids}")
    overlay_cols = [
        "densegen__used_tfbs_detail",
        "densegen__required_regulators",
        "densegen__sampling_library_hash",
        "densegen__plan",
    ]
    present_overlay_cols = [column for column in overlay_cols if column in densegen_sidecar.columns]
    merged = candidates.merge(
        densegen_sidecar[["id", *present_overlay_cols]],
        on="id",
        how="left",
        suffixes=("", "__sidecar"),
    )
    for column in present_overlay_cols:
        sidecar_col = f"{column}__sidecar"
        if sidecar_col in merged.columns:
            if column in merged.columns:
                conflict_mask = [
                    not _metadata_equal(primary, sidecar)
                    for primary, sidecar in zip(merged[column], merged[sidecar_col], strict=True)
                ]
                if any(conflict_mask):
                    conflict_ids = merged.loc[conflict_mask, "id"].astype(str).head(5).tolist()
                    raise ValueError(f"candidate records conflict with DenseGen sidecar for {column}: {conflict_ids}")
                merged[column] = merged[sidecar_col].where(merged[sidecar_col].notna(), merged[column])
            else:
                merged[column] = merged[sidecar_col]
            merged = merged.drop(columns=[sidecar_col])
    return merged


def build_axis_oracle(candidates: pd.DataFrame, *, densegen_sidecar: pd.DataFrame | None = None) -> pd.DataFrame:
    if "id" not in candidates.columns:
        raise ValueError("candidate records must include id")
    frame = _overlay_densegen_sidecar(candidates, densegen_sidecar)
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        label = derive_axis_label(record)
        logic = label.logic4
        effect = label.effect4
        row = {
            "oracle_id": ORACLE_ID,
            "id": label.id,
            "sequence": record.get("sequence"),
            "axis_class": label.axis_class,
            "logic4": logic,
            "effect4": effect,
            "vec8": label.vec8,
            "quality_flag": label.quality_flag,
            "lexA_count": label.lexA_count,
            "cpxR_count": label.cpxR_count,
            "baeR_count": label.baeR_count,
            "background_count": label.background_count,
            "cipro_axis": label.cipro_axis,
            "ethanol_axis": label.ethanol_axis,
            "densegen_plan_class": label.densegen_plan_class,
            "sigma35_variant": label.sigma35_variant,
            "densegen__plan": label.densegen_plan,
            "expected_axis_class_from_plan": label.expected_axis_class_from_plan,
            "densegen__sampling_library_hash": record.get("densegen__sampling_library_hash"),
        }
        row.update(
            tf_family_columns(
                lex_a=label.lexA_count,
                cpx_r=label.cpxR_count,
                bae_r=label.baeR_count,
            )
        )
        if logic is not None and effect is not None:
            for column, value in zip(SFXI_STATE_COLUMNS, logic, strict=True):
                row[column] = int(value)
            for column, value in zip(SFXI_INTENSITY_COLUMNS, effect, strict=True):
                row[column] = float(value)
            row["intensity_log2_offset_delta"] = 0.0
        else:
            for column in (*SFXI_STATE_COLUMNS, *SFXI_INTENSITY_COLUMNS):
                row[column] = np.nan
            row["intensity_log2_offset_delta"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def class_from_logic4(logic4: Sequence[float]) -> str:
    arr = np.asarray(logic4, dtype=float).ravel()
    if arr.size != 4:
        raise ValueError(f"logic4 must have 4 values, got {arr.size}")
    best_class = None
    best_distance = float("inf")
    for axis_class, canonical in AXIS_CLASS_TO_LOGIC4.items():
        dist = float(np.linalg.norm(arr - np.asarray(canonical, dtype=float)))
        if dist < best_distance:
            best_class = axis_class
            best_distance = dist
    if best_class is None:  # pragma: no cover - defensive
        raise ValueError("no canonical axis classes configured")
    return best_class


def make_permuted_labels(labels: pd.DataFrame, *, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    if "vec8" not in labels.columns or "id" not in labels.columns:
        raise ValueError("labels must include id and vec8")
    rng = np.random.default_rng(int(seed))
    out = labels.copy()
    permutation_mask = (
        out["quality_flag"].astype(str).eq("ok").to_numpy()
        if "quality_flag" in out.columns
        else np.ones(len(out), dtype=bool)
    )
    permutation_positions = np.flatnonzero(permutation_mask)
    n_permuted = int(len(permutation_positions))
    order = rng.permutation(n_permuted)
    if n_permuted > 1 and np.array_equal(order, np.arange(n_permuted)):
        order = np.roll(order, 1)

    original_vec8 = out["vec8"].tolist()
    original_class = out["axis_class"].tolist() if "axis_class" in out.columns else [None] * int(len(out))
    permuted_vec8 = list(original_vec8)
    for dest_position, source_order_position in zip(permutation_positions, order, strict=True):
        source_position = int(permutation_positions[int(source_order_position)])
        permuted_vec8[int(dest_position)] = original_vec8[source_position]
    out["oracle_id"] = NULL_ORACLE_ID
    out["true_vec8"] = original_vec8
    out["true_axis_class"] = original_class
    out["vec8"] = permuted_vec8
    out["axis_class"] = [
        class_from_logic4(vec[:4]) if isinstance(vec, (list, tuple, np.ndarray)) and len(vec) >= 4 else None
        for vec in permuted_vec8
    ]
    for column, idx in zip(SFXI_STATE_COLUMNS, range(4), strict=True):
        out[column] = [
            float(vec[idx]) if isinstance(vec, (list, tuple, np.ndarray)) else np.nan for vec in permuted_vec8
        ]
    for column, idx in zip(SFXI_INTENSITY_COLUMNS, range(4, 8), strict=True):
        out[column] = [
            float(vec[idx]) if isinstance(vec, (list, tuple, np.ndarray)) else np.nan for vec in permuted_vec8
        ]
    out["permutation_seed"] = int(seed)
    return out


def _ok_labels(labels: pd.DataFrame) -> pd.DataFrame:
    if "quality_flag" not in labels.columns:
        raise ValueError("labels must include quality_flag")
    return labels.loc[labels["quality_flag"].astype(str) == "ok"].copy()


def _deterministic_sample(ids: Sequence[str], *, n: int, rng: np.random.Generator) -> list[str]:
    ordered = np.asarray(sorted(map(str, ids)), dtype=object)
    if int(len(ordered)) < int(n):
        raise ValueError(f"not enough ids to sample {int(n)} from pool of {int(len(ordered))}")
    positions = rng.choice(len(ordered), size=int(n), replace=False)
    return sorted(map(str, ordered[positions].tolist()))


def _balanced_class_budgets(*, budget: int, seed: int, require_each_class: bool = True) -> dict[str, int]:
    axis_classes = list(AXIS_CLASS_TO_LOGIC4)
    total = int(budget)
    if require_each_class and total < len(axis_classes):
        raise ValueError(f"budget must be >= {len(axis_classes)} to seed every axis class")
    base = total // len(axis_classes)
    remainder = total % len(axis_classes)
    counts = {axis_class: base for axis_class in axis_classes}
    if remainder:
        rng = np.random.default_rng(int(seed) + 7919)
        extra_classes = rng.choice(np.asarray(axis_classes, dtype=object), size=remainder, replace=False)
        for axis_class in map(str, extra_classes.tolist()):
            counts[axis_class] += 1
    return counts


def build_train_ids(
    labels: pd.DataFrame,
    *,
    budget: int,
    seed: int,
    split_id: Literal["random_id", "leave_sigma35_variant"],
    return_metadata: bool = False,
) -> list[str] | tuple[list[str], dict[str, Any]]:
    class_budgets = _balanced_class_budgets(budget=int(budget), seed=int(seed))
    per_class = int(budget) // len(AXIS_CLASS_TO_LOGIC4)
    rng = np.random.default_rng(int(seed))
    pool = _ok_labels(labels)
    metadata: dict[str, Any] = {
        "split_id": split_id,
        "budget": int(budget),
        "per_class": per_class,
        "class_budget": dict(class_budgets),
        "seed": int(seed),
    }
    if split_id == "leave_sigma35_variant":
        variants = sorted(v for v in pool["sigma35_variant"].dropna().astype(str).unique().tolist() if v)
        if not variants:
            raise ValueError("leave_sigma35_variant split requires sigma35_variant values")
        heldout = str(rng.choice(np.asarray(variants, dtype=object)))
        metadata["heldout_sigma35"] = heldout
        pool = pool.loc[pool["sigma35_variant"].astype(str) != heldout].copy()
    elif split_id != "random_id":
        raise ValueError(f"unsupported split_id: {split_id}")

    train_ids: list[str] = []
    for axis_class in AXIS_CLASS_TO_LOGIC4:
        class_ids = pool.loc[pool["axis_class"].astype(str) == axis_class, "id"].astype(str).tolist()
        sampled = _deterministic_sample(class_ids, n=class_budgets[axis_class], rng=rng)
        train_ids.extend(sampled)
    train_ids = sorted(train_ids)
    metadata["train_count"] = int(len(train_ids))
    if split_id == "leave_sigma35_variant":
        metadata["eval_ids"] = (
            _ok_labels(labels)
            .loc[lambda frame: frame["sigma35_variant"].astype(str) == str(metadata["heldout_sigma35"]), "id"]
            .astype(str)
            .sort_values()
            .tolist()
        )
    else:
        ok_ids = set(_ok_labels(labels)["id"].astype(str).tolist())
        metadata["eval_ids"] = sorted(ok_ids - set(train_ids))
    if return_metadata:
        return train_ids, metadata
    return train_ids
