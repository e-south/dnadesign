"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/runtime_contracts.py

Typed runtime contracts for cluster.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence


@dataclass(frozen=True, slots=True)
class InputSource:
    kind: Literal["usr", "parquet", "csv"]
    source_ref: str
    file: Path
    dataset: str | None = None
    usr_root: Path | None = None

    @classmethod
    def from_context(cls, ctx: dict[str, Any]) -> "InputSource":
        kind = str(ctx["kind"])
        file = Path(ctx["file"])
        dataset = ctx.get("dataset")
        source_ref = str(dataset) if dataset else str(file)
        usr_root = Path(ctx["usr_root"]) if ctx.get("usr_root") is not None else None
        return cls(
            kind=kind,  # type: ignore[arg-type]
            source_ref=source_ref,
            file=file,
            dataset=str(dataset) if dataset is not None else None,
            usr_root=usr_root,
        )

    def source_clause(self) -> dict[str, str]:
        if self.kind == "usr":
            if self.dataset is None:
                raise ValueError("USR input source requires a dataset name.")
            return {"kind": "usr", "dataset": self.dataset}
        return {"kind": self.kind, "file": str(self.file)}


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    mode: Literal["single_col", "multi_col"]
    columns: tuple[str, ...]

    @classmethod
    def from_inputs(cls, *, x_col: str | None, x_cols: str | Sequence[str] | None) -> "FeatureSpec":
        single = str(x_col).strip() if x_col and str(x_col).strip() else None
        if isinstance(x_cols, str):
            multi = tuple(c.strip() for c in x_cols.split(",") if c.strip())
        elif x_cols:
            multi = tuple(str(c).strip() for c in x_cols if str(c).strip())
        else:
            multi = ()
        if bool(single) == bool(multi):
            raise ValueError("Provide exactly one of --x-col or --x-cols.")
        if single:
            return cls(mode="single_col", columns=(single,))
        return cls(mode="multi_col", columns=multi)

    @property
    def primary_label(self) -> str:
        return self.columns[0] if self.mode == "single_col" else "<multi>"


@dataclass(frozen=True, slots=True)
class MethodConfig:
    method_id: str
    params: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FitRequest:
    source: InputSource
    key_col: str
    feature: FeatureSpec
    method: MethodConfig

    def input_signature_payload(
        self,
        *,
        row_ids_hash: str,
        x_dim: int,
        fingerprint: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "source_kind": self.source.kind,
            "source_ref": self.source.source_ref,
            "key_col": self.key_col,
            "row_ids_hash": row_ids_hash,
            "x_spec": {
                "mode": self.feature.mode,
                "cols": list(self.feature.columns),
                "x_dim": int(x_dim),
            },
            "fingerprint": fingerprint,
        }
