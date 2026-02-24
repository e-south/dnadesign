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
    label = str(value or "").strip()
    if not label:
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


def _ellipsize(label: object, max_len: int = 18) -> str:
    text = str(label or "")
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return f"{text[: max_len - 3]}..."


def _forbidden_kmer_tokens(value: object) -> list[str]:
    tokens: set[str] = set()
    text = str(value or "").strip()
    if not text:
        return []
    json_match = re.search(r"\{.*\}", text)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            if isinstance(payload, dict):
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
