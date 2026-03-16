"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/analysis/contracts.py

Typed analysis-request contracts for cluster.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..runs.contracts import AnalysisRun, fit_alias_from_cluster_col, utc_now_iso
from ..runtime_contracts import InputSource

ALLOWED_NUMERIC_MISSING_POLICIES = frozenset({"error", "drop_and_log"})
_OPAL_PREFIXES = ("obj__", "pred__", "sel__")


def _split_values(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        tokens = raw.split(",")
    else:
        tokens = []
        for value in raw:
            if value is None:
                continue
            tokens.extend(str(value).split(","))
    values: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        value = str(token).strip()
        if value and value not in seen:
            values.append(value)
            seen.add(value)
    return tuple(values)


@dataclass(frozen=True, slots=True)
class AnalysisRequest:
    source: InputSource
    cluster_col: str
    group_by: tuple[str, ...]
    out_dir: Path
    composition: bool
    diversity: bool
    difffeat: bool
    plots: bool
    numeric_cols: tuple[str, ...]
    numeric_missing_policy: str
    numeric_plots: bool
    font_scale: float
    fit_alias: str | None
    opal_campaign: str | None = None
    opal_as_of_round: int | None = None
    explicit_opal_fields: tuple[str, ...] = ()
    required_opal_fields: tuple[str, ...] = ()

    @classmethod
    def from_runtime(
        cls,
        *,
        source: InputSource,
        df_columns: Sequence[str],
        cluster_col: str,
        group_by: str | Sequence[str] | None,
        out_dir: str | Path | None,
        results_root: Path | None,
        composition: bool,
        diversity: bool,
        difffeat: bool,
        plots: bool,
        numeric: str | Sequence[str] | None,
        numeric_missing_policy: str,
        numeric_plots: bool,
        font_scale: float,
        opal_campaign: str | None,
        opal_as_of_round: int | None,
        opal_fields: str | Sequence[str] | None,
    ) -> "AnalysisRequest":
        normalized_cluster_col = str(cluster_col).strip()
        if not normalized_cluster_col:
            raise ValueError("Analysis requires a non-empty cluster column.")

        group_bys = _split_values(group_by) or ("source",)
        if numeric_missing_policy not in ALLOWED_NUMERIC_MISSING_POLICIES:
            allowed = ", ".join(sorted(ALLOWED_NUMERIC_MISSING_POLICIES))
            raise ValueError(f"Unsupported numeric missing policy '{numeric_missing_policy}'. Use one of: {allowed}.")

        df_column_set = set(df_columns)
        numeric_cols = list(_split_values(numeric))
        if numeric_cols and "permuter__mut_count" in df_column_set and "permuter__mut_count" not in numeric_cols:
            numeric_cols.append("permuter__mut_count")

        fit_alias = fit_alias_from_cluster_col(normalized_cluster_col)
        if out_dir is None:
            if fit_alias is None:
                raise ValueError(
                    "When --out-dir is omitted, --cluster-col must be a fit label column of the form 'cluster__<NAME>'."
                )
            if results_root is None:
                raise ValueError("Analysis requires an explicit results root when --out-dir is omitted.")
            resolved_out_dir = Path(results_root) / fit_alias / "analysis"
        else:
            resolved_out_dir = Path(out_dir)

        explicit_opal_fields = _split_values(opal_fields)
        missing_numeric_fields = tuple(
            c for c in numeric_cols if c.startswith(_OPAL_PREFIXES) and c not in df_column_set
        )
        required_opal_fields = _split_values((*explicit_opal_fields, *missing_numeric_fields))

        request = cls(
            source=source,
            cluster_col=normalized_cluster_col,
            group_by=group_bys,
            out_dir=resolved_out_dir,
            composition=bool(composition),
            diversity=bool(diversity),
            difffeat=bool(difffeat),
            plots=bool(plots),
            numeric_cols=tuple(numeric_cols),
            numeric_missing_policy=numeric_missing_policy,
            numeric_plots=bool(numeric_plots),
            font_scale=float(font_scale),
            fit_alias=fit_alias,
            opal_campaign=opal_campaign,
            opal_as_of_round=opal_as_of_round,
            explicit_opal_fields=explicit_opal_fields,
            required_opal_fields=required_opal_fields,
        )
        if not request.has_work:
            raise ValueError(
                "Select at least one analysis step via --composition, --diversity, --difffeat, or --numeric."
            )
        return request

    @property
    def has_grouped_analyses(self) -> bool:
        return any((self.composition, self.diversity, self.difffeat))

    @property
    def has_work(self) -> bool:
        return bool(self.numeric_cols) or self.has_grouped_analyses

    def command_payload(self) -> dict[str, Any]:
        return {
            "cluster_col": self.cluster_col,
            "group_by": list(self.group_by),
            "composition": bool(self.composition),
            "diversity": bool(self.diversity),
            "difffeat": bool(self.difffeat),
            "plots": bool(self.plots),
            "numeric_plots": bool(self.numeric_plots),
            "numeric_cols": list(self.numeric_cols),
            "numeric_missing_policy": self.numeric_missing_policy,
            "font_scale": float(self.font_scale),
            "opal_campaign": self.opal_campaign,
            "opal_as_of_round": self.opal_as_of_round,
            "opal_fields": list(self.required_opal_fields),
            "out_dir": str(self.out_dir),
        }

    def to_run(self, *, created_utc: str | None = None) -> AnalysisRun:
        return AnalysisRun(
            cluster_col=self.cluster_col,
            created_utc=created_utc or utc_now_iso(),
            source=self.source,
            group_by=self.group_by,
            out_dir=self.out_dir,
            composition=self.composition,
            diversity=self.diversity,
            difffeat=self.difffeat,
            plots=self.plots,
            numeric_cols=self.numeric_cols,
            numeric_plots=self.numeric_plots,
            font_scale=self.font_scale,
            fit_alias=self.fit_alias,
            opal_fields=self.required_opal_fields,
            opal_campaign=self.opal_campaign,
            opal_as_of_round=self.opal_as_of_round,
        )


__all__ = [
    "ALLOWED_NUMERIC_MISSING_POLICIES",
    "AnalysisRequest",
]
