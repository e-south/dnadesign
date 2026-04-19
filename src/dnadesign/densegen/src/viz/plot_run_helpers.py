"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_run_helpers.py

Shared helper utilities for run-level plotting diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

_PLAN_BASE_LABELS = {
    "background_only": "Background",
    "ciprofloxacin": "Cipro",
    "ethanol": "EtOH",
    "ethanol_ciprofloxacin": "EtOH + Cipro",
}
_PLAN_VARIANT_LABELS = {
    "sig35": "σ70",
    "sigma70": "σ70",
}
_PLAN_MARKER_CYCLE = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h")


def _bin_attempts(values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.array([]), np.array([])
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, num=int(bins) + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return edges, centers


def _usage_category_label(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    label = str(value).strip()
    if not label:
        return ""
    if label.lower() in {"none", "nan"}:
        return ""
    if label.startswith("fixed:"):
        return label
    if "_" in label:
        head, tail = label.split("_", 1)
        tail_upper = tail.upper()
        iupac = set("ACGTURYSWKMBDHVN")
        if len(tail_upper) >= 6 and set(tail_upper).issubset(iupac):
            return head
    return label


def _usage_available_unique(
    *,
    input_name: str,
    plan_name: str,
    pools: dict[str, pd.DataFrame] | None,
    library_members_df: pd.DataFrame | None,
) -> tuple[dict[str, int], int]:
    if library_members_df is not None and not library_members_df.empty:
        required = {"input_name", "plan_name", "tf", "tfbs"}
        missing = required - set(library_members_df.columns)
        if missing:
            raise ValueError(f"library_members.parquet missing required columns: {sorted(missing)}")
        offered = library_members_df[
            (library_members_df["input_name"].astype(str) == str(input_name))
            & (library_members_df["plan_name"].astype(str) == str(plan_name))
        ].copy()
        if offered.empty:
            return {}, 0
        offered["category_label"] = offered["tf"].map(_usage_category_label)
        offered["tfbs"] = offered["tfbs"].astype(str)
        offered = offered[offered["category_label"].astype(str).str.strip() != ""].copy()
        if offered.empty:
            return {}, 0
        unique_pairs = offered[["category_label", "tfbs"]].drop_duplicates()
        by_category = (
            unique_pairs.groupby("category_label")[["tfbs"]].nunique().rename(columns={"tfbs": "unique_available"})
        )
        return by_category["unique_available"].to_dict(), int(len(unique_pairs))

    if pools and input_name in pools:
        pool_df = pools[input_name]
        if pool_df.empty or "tf" not in pool_df.columns:
            return {}, 0
        tfbs_col = "tfbs_sequence" if "tfbs_sequence" in pool_df.columns else "tfbs"
        if tfbs_col not in pool_df.columns:
            return {}, 0
        offered = pool_df.assign(
            category_label=pool_df["tf"].map(_usage_category_label),
            tfbs=pool_df[tfbs_col].astype(str),
        )[["category_label", "tfbs"]]
        offered = offered[offered["category_label"].astype(str).str.strip() != ""].copy()
        if offered.empty:
            return {}, 0
        unique_pairs = offered.drop_duplicates()
        by_category = (
            unique_pairs.groupby("category_label")[["tfbs"]].nunique().rename(columns={"tfbs": "unique_available"})
        )
        return by_category["unique_available"].to_dict(), int(len(unique_pairs))
    return {}, 0


def _first_existing_column(df: pd.DataFrame, candidates: list[str], *, context: str) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"{context} requires one of columns: {', '.join(candidates)}.")


def _normalize_plan_name(value: object) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    label = str(value).strip()
    if not label:
        return None
    if label.lower() in {"nan", "none"}:
        return None
    return label


def _title_case_words(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    words = [word for word in text.replace("-", " ").replace("_", " ").split() if word]
    if not words:
        return ""
    return " ".join(word[:1].upper() + word[1:] for word in words)


def capitalize_first(value: object) -> str:
    token = str(value)
    for idx, char in enumerate(token):
        if char.isalpha():
            return token[:idx] + char.upper() + token[idx + 1 :]
    return token


def plan_markers(plan_names: list[str]) -> dict[str, str]:
    return {plan: _PLAN_MARKER_CYCLE[idx % len(_PLAN_MARKER_CYCLE)] for idx, plan in enumerate(plan_names)}


def compact_regulator_label(value: object) -> str:
    token = _usage_category_label(value) or str(value or "").strip()
    if not token:
        return ""
    if token.lower() == "background":
        return "Background"
    parts = [part for part in token.replace("-", "_").split("_") if part]
    labels: list[str] = []
    for part in parts:
        if part.isupper():
            labels.append(part)
            continue
        labels.append(part[:1].upper() + part[1:])
    return " ".join(labels).strip()


_KNOWN_REGULATOR_DISPLAY_ORDER = {
    "lexa": 0,
    "cpxr": 1,
    "baer": 2,
    "background": 99,
}


def order_regulators_for_display(
    regulators: list[str] | tuple[str, ...],
    *,
    counts_by_regulator: dict[str, int] | None = None,
) -> list[str]:
    ordered_unique = list(dict.fromkeys(str(regulator) for regulator in regulators if str(regulator).strip()))
    if not ordered_unique:
        return []
    counts = {str(key): int(value) for key, value in (counts_by_regulator or {}).items()}

    def _sort_key(regulator: str) -> tuple[int, int, str]:
        label = compact_regulator_label(regulator)
        token = label.lower().replace(" ", "")
        primary = _KNOWN_REGULATOR_DISPLAY_ORDER.get(token, 40)
        secondary = -int(counts.get(regulator, 0))
        return primary, secondary, label.lower()

    return sorted(ordered_unique, key=_sort_key)


def compact_plan_label(plan_name: object) -> str:
    plan_text = _normalize_plan_name(plan_name) or ""
    if not plan_text:
        return "Run-level"
    if plan_text == "unscoped":
        return "Run-level"
    if plan_text == "stage_a":
        return "Stage A"
    parts = [part for part in plan_text.split("__") if str(part).strip()]
    base_token = str(parts[0] if parts else plan_text).strip()
    base_label = _PLAN_BASE_LABELS.get(base_token.lower(), _title_case_words(base_token) or base_token)
    variant_tokens: list[str] = []
    for token in parts[1:]:
        token_text = str(token).strip()
        if not token_text:
            continue
        if "=" in token_text:
            key, value = token_text.split("=", 1)
        elif "_" in token_text:
            key, value = token_text.split("_", 1)
        else:
            key, value = token_text, ""
        key = str(key).strip()
        value = str(value).strip()
        if key and value:
            key_label = _PLAN_VARIANT_LABELS.get(key.lower(), _title_case_words(key) or key)
            variant_tokens.append(f"{key_label} {value}")
    if not variant_tokens:
        return base_label
    return f"{base_label} [{' | '.join(variant_tokens)}]"


def compact_failure_reason_label(reason_label: object) -> str:
    text = str(reason_label or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    compact_map = {
        "duplicate output": "Duplicate output",
        "unknown": "Unknown",
        "sequence validation": "Sequence validation",
        "no solution": "No solution",
        "required regulators": "Required TF set",
        "min by regulator": "Per-regulator minimum",
        "min per tf": "Per-TF minimum",
        "min regulator groups": "Regulator-group minimum",
        "solver failure": "Solver failure",
    }
    if lowered in compact_map:
        return compact_map[lowered]
    if lowered.startswith("forbidden kmer:"):
        token = text.split(":", 1)[1].strip()
        return f"Forbidden k-mer {token}".strip()
    if lowered.startswith("forbidden kmers:"):
        tokens = [item.strip() for item in text.split(":", 1)[1].split(",") if item.strip()]
        if len(tokens) <= 2:
            return f"Forbidden k-mers {', '.join(tokens)}".strip()
        return f"Forbidden k-mers {', '.join(tokens[:2])} ({len(tokens) - 2} more)"
    normalized = text.replace("_", " ").strip()
    if not normalized:
        return ""
    return normalized[:1].upper() + normalized[1:]


def _humanize_scope_label(value: object) -> str:
    label = _normalize_plan_name(value) or ""
    if not label:
        return ""
    if label.startswith("plan_pool__"):
        label = label[len("plan_pool__") :]
    return label.replace("__", "; ").replace("_", " ").strip()


def _ellipsize(label: object, max_len: int = 18) -> str:
    text = str(label or "")
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return f"{text[: max_len - 3]}..."


def _forbidden_kmer_tokens(value: object) -> list[str]:
    tokens: set[str] = set()

    def _collect_from_payload(payload: object) -> None:
        if not isinstance(payload, dict):
            return
        single = payload.get("forbidden_kmer")
        if isinstance(single, str) and single.strip():
            tokens.add(single.strip().upper())
        multi = payload.get("forbidden_kmers")
        if isinstance(multi, list):
            for item in multi:
                if isinstance(item, str) and item.strip():
                    tokens.add(item.strip().upper())
        kmer = payload.get("kmer")
        if isinstance(kmer, str) and kmer.strip():
            tokens.add(kmer.strip().upper())
        kmers = payload.get("kmers")
        if isinstance(kmers, list):
            for item in kmers:
                if isinstance(item, str) and item.strip():
                    tokens.add(item.strip().upper())
        violations = payload.get("violations")
        if isinstance(violations, list):
            for item in violations:
                if not isinstance(item, dict):
                    continue
                constraint = str(item.get("constraint") or "").strip().lower()
                if "forbid" not in constraint:
                    continue
                pattern = item.get("pattern")
                if isinstance(pattern, str) and pattern.strip():
                    tokens.add(pattern.strip().upper())
                matched_seq = item.get("matched_seq")
                if isinstance(matched_seq, str) and matched_seq.strip():
                    tokens.add(matched_seq.strip().upper())

    text = str(value or "").strip()
    if not text:
        return []
    if isinstance(value, dict):
        _collect_from_payload(value)
    elif isinstance(value, list):
        for item in value:
            _collect_from_payload(item)
    json_match = re.search(r"\{.*\}", text)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            _collect_from_payload(payload)
        except Exception:
            pass
    for match in re.findall(r'"forbidden_kmer"\s*:\s*"([acgtun]+)"', text):
        tokens.add(match.upper())
    list_match = re.search(r'"forbidden_kmers"\s*:\s*\[([^\]]*)\]', text)
    if list_match:
        for match in re.findall(r'"([acgtun]+)"', list_match.group(1)):
            tokens.add(match.upper())
    for match in re.findall(r'"kmer"\s*:\s*"([acgtun]+)"', text):
        tokens.add(match.upper())
    for match in re.findall(r"(?:forbidden_)?kmer(?:[:=]|_)?([acgtun]+)", text):
        tokens.add(match.upper())
    return sorted(tokens)


def _reason_family_label(status: str, reason: object, detail_json: object | None = None) -> str:
    reason_text = str(reason or "").strip()
    value = reason_text.lower()
    if status == "duplicate" or value == "output_duplicate":
        return "duplicate output"
    if value in {"", "none", "nan"}:
        return "unknown"
    if "forbidden_kmer" in value or value == "postprocess_forbidden_kmer":
        tokens = sorted(set(_forbidden_kmer_tokens(reason_text)) | set(_forbidden_kmer_tokens(detail_json)))
        if len(tokens) == 1:
            return f"forbidden kmer: {tokens[0]}"
        if len(tokens) > 1:
            return f"forbidden kmers: {', '.join(tokens)}"
        return "forbidden kmer"
    if value == "sequence_validation_failed":
        tokens = sorted(set(_forbidden_kmer_tokens(reason_text)) | set(_forbidden_kmer_tokens(detail_json)))
        if len(tokens) == 1:
            return f"forbidden kmer: {tokens[0]}"
        if len(tokens) > 1:
            return f"forbidden kmers: {', '.join(tokens)}"
        return "sequence validation"
    replacements = {
        "postprocess_forbidden_kmer": "forbidden kmer",
        "stall_no_solution": "no solution",
        "no_solution": "no solution",
        "failed_required_regulators": "required regulators",
        "failed_min_count_by_regulator": "min by regulator",
        "failed_min_count_per_tf": "min per TF",
        "failed_min_required_regulators": "min regulator groups",
    }
    if value in replacements:
        return replacements[value]
    if "no_solution" in value:
        return "no solution"
    if "required_regulator" in value:
        return "required regulators"
    if "min_count_by_regulator" in value:
        return "min by regulator"
    if "min_count_per_tf" in value:
        return "min per TF"
    if "min_required_regulators" in value:
        return "min regulator groups"
    if "solver" in value:
        return "solver failure"
    return value.replace("_", " ")
