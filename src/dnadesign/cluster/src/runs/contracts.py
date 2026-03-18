"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/runs/contracts.py

Typed run-artifact contracts for cluster.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ..runtime_contracts import FeatureSpec, InputSource
from .signatures import InputSignature, MethodSignature, UmapSignature

RUN_INDEX_COLUMNS = [
    "kind",
    "run_slug",
    "alias",
    "created_utc",
    "source_kind",
    "source_ref",
    "x_col",
    "n_rows",
    "n_clusters",
    "method_id",
    "method_params",
    "method_sig_hash",
    "input_sig_hash",
    "labels_path",
    "status",
    "umap_slug",
    "umap_params",
    "coords_path",
    "plot_paths",
    "analysis_path",
    "sweep_path",
]

FIT_REUSE_REQUIRED_COLUMNS = frozenset({"kind", "input_sig_hash", "method_sig_hash"})


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class RunCounts:
    n_rows: int
    n_clusters: int | None = None

    def payload(self) -> dict[str, int | None]:
        return {
            "n_rows": int(self.n_rows),
            "n_clusters": int(self.n_clusters) if self.n_clusters is not None else None,
        }


@dataclass(frozen=True, slots=True)
class RunIndexEntry:
    kind: Literal["fit", "umap", "analysis", "sweep"]
    run_slug: str
    alias: str
    created_utc: str
    source_kind: Literal["usr", "parquet", "csv"]
    source_ref: str
    x_col: str
    n_rows: int
    n_clusters: int | None
    method_id: str | None
    method_params: dict[str, Any] | None
    method_sig_hash: str | None
    input_sig_hash: str | None
    labels_path: str | None
    status: str
    umap_slug: str | None
    umap_params: dict[str, Any] | None
    coords_path: str | None
    plot_paths: str | None
    analysis_path: str | None
    sweep_path: str | None

    @classmethod
    def columns(cls) -> list[str]:
        return list(RUN_INDEX_COLUMNS)

    def payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "run_slug": self.run_slug,
            "alias": self.alias,
            "created_utc": self.created_utc,
            "source_kind": self.source_kind,
            "source_ref": self.source_ref,
            "x_col": self.x_col,
            "n_rows": int(self.n_rows),
            "n_clusters": int(self.n_clusters) if self.n_clusters is not None else None,
            "method_id": self.method_id,
            "method_params": (dict(self.method_params) or None) if self.method_params is not None else None,
            "method_sig_hash": self.method_sig_hash,
            "input_sig_hash": self.input_sig_hash,
            "labels_path": self.labels_path,
            "status": self.status,
            "umap_slug": self.umap_slug,
            "umap_params": (dict(self.umap_params) or None) if self.umap_params is not None else None,
            "coords_path": self.coords_path,
            "plot_paths": self.plot_paths,
            "analysis_path": self.analysis_path,
            "sweep_path": self.sweep_path,
        }


@dataclass(frozen=True, slots=True)
class ClusterRun:
    alias: str
    slug: str
    created_utc: str
    input_signature: InputSignature
    method_signature: MethodSignature
    source: InputSource
    feature: FeatureSpec
    x_dim: int
    counts: RunCounts
    wrote_usr_columns: bool
    attached_columns: tuple[str, ...]

    def meta_payload(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "slug": self.slug,
            "created_utc": self.created_utc,
            "input_signature": self.input_signature.dict(),
            "method_signature": self.method_signature.dict(),
            "io": self.source.source_clause(),
            "x": {"col": self.feature.primary_label, "dim": int(self.x_dim)},
            "counts": self.counts.payload(),
            "attach": {"wrote_usr_columns": bool(self.wrote_usr_columns)},
            "columns": list(self.attached_columns),
        }

    def index_entry(
        self,
        *,
        labels_path: Path,
        method_sig_hash: str | None = None,
        input_sig_hash: str | None = None,
        status: str = "complete",
    ) -> RunIndexEntry:
        return RunIndexEntry(
            kind="fit",
            run_slug=self.slug,
            alias=self.alias,
            created_utc=self.created_utc,
            source_kind=self.source.kind,
            source_ref=self.source.source_ref,
            x_col=self.feature.primary_label,
            n_rows=self.counts.n_rows,
            n_clusters=self.counts.n_clusters,
            method_id=self.method_signature.method_id,
            method_params=self.method_signature.params,
            method_sig_hash=method_sig_hash or self.method_signature.hash(),
            input_sig_hash=input_sig_hash or self.input_signature.hash(),
            labels_path=str(labels_path),
            status=status,
            umap_slug=None,
            umap_params=None,
            coords_path=None,
            plot_paths=None,
            analysis_path=None,
            sweep_path=None,
        )


@dataclass(frozen=True, slots=True)
class EmbeddingRun:
    alias: str
    slug: str
    created_utc: str
    source: InputSource
    feature: FeatureSpec
    counts: RunCounts
    params: dict[str, Any]
    signature: UmapSignature
    embedding_kind: Literal["umap"] = "umap"

    def meta_payload(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "slug": self.slug,
            "embedding_kind": self.embedding_kind,
            "created_utc": self.created_utc,
            "source": self.source.source_clause(),
            "x": {"col": self.feature.primary_label},
            "counts": self.counts.payload(),
            "params": dict(self.params),
            "sig": self.signature.hash(),
        }

    def index_entry(
        self,
        *,
        coords_path: Path,
        plot_root: Path | None,
        status: str = "complete",
        umap_slug: str | None = None,
    ) -> RunIndexEntry:
        return RunIndexEntry(
            kind="umap",
            run_slug=self.slug,
            alias=self.alias,
            created_utc=self.created_utc,
            source_kind=self.source.kind,
            source_ref=self.source.source_ref,
            x_col=self.feature.primary_label,
            n_rows=self.counts.n_rows,
            n_clusters=None,
            method_id=None,
            method_params=None,
            method_sig_hash=None,
            input_sig_hash=None,
            labels_path=None,
            status=status,
            umap_slug=umap_slug or self.slug,
            umap_params=self.params,
            coords_path=str(coords_path),
            plot_paths=str(plot_root) if plot_root is not None else None,
            analysis_path=None,
            sweep_path=None,
        )


def fit_alias_from_cluster_col(cluster_col: str) -> str | None:
    if not cluster_col.startswith("cluster__"):
        return None
    parts = cluster_col.split("__")
    if len(parts) != 2 or not parts[1]:
        return None
    return parts[1]


@dataclass(frozen=True, slots=True)
class AnalysisRun:
    alias: str
    slug: str
    cluster_col: str
    created_utc: str
    source: InputSource
    group_by: tuple[str, ...]
    out_dir: Path
    composition: bool
    diversity: bool
    difffeat: bool
    plots: bool
    numeric_cols: tuple[str, ...]
    numeric_plots: bool
    font_scale: float
    fit_alias: str | None = None
    opal_fields: tuple[str, ...] = ()
    opal_campaign: str | None = None
    opal_as_of_round: int | None = None

    def meta_payload(self) -> dict[str, Any]:
        return {
            "analysis_kind": "cluster_summary",
            "alias": self.alias,
            "slug": self.slug,
            "created_utc": self.created_utc,
            "source": self.source.source_clause(),
            "cluster": {
                "column": self.cluster_col,
                "fit_alias": self.fit_alias,
            },
            "out_dir": str(self.out_dir),
            "group_by": list(self.group_by),
            "steps": {
                "composition": bool(self.composition),
                "diversity": bool(self.diversity),
                "difffeat": bool(self.difffeat),
                "numeric": bool(self.numeric_cols),
            },
            "plots": {
                "enabled": bool(self.plots),
                "numeric": bool(self.numeric_plots),
                "font_scale": float(self.font_scale),
            },
            "numeric_cols": list(self.numeric_cols),
            "opal_join": {
                "campaign": self.opal_campaign,
                "as_of_round": self.opal_as_of_round,
                "fields": list(self.opal_fields),
            },
        }

    def index_entry(self, *, analysis_path: Path, status: str = "complete") -> RunIndexEntry:
        return RunIndexEntry(
            kind="analysis",
            run_slug=self.slug,
            alias=self.alias,
            created_utc=self.created_utc,
            source_kind=self.source.kind,
            source_ref=self.source.source_ref,
            x_col=self.cluster_col,
            n_rows=0,
            n_clusters=None,
            method_id=None,
            method_params=None,
            method_sig_hash=None,
            input_sig_hash=None,
            labels_path=None,
            status=status,
            umap_slug=None,
            umap_params=None,
            coords_path=None,
            plot_paths=None,
            analysis_path=str(analysis_path),
            sweep_path=None,
        )


@dataclass(frozen=True, slots=True)
class SweepRun:
    alias: str
    slug: str
    created_utc: str
    source: InputSource
    feature: FeatureSpec
    method_signature: MethodSignature
    res_min: float
    res_max: float
    step: float
    seeds: tuple[int, ...]

    def meta_payload(self) -> dict[str, Any]:
        return {
            "sweep_kind": "method_resolution",
            "alias": self.alias,
            "slug": self.slug,
            "created_utc": self.created_utc,
            "source": self.source.source_clause(),
            "x": {"col": self.feature.primary_label},
            "method": self.method_signature.dict(),
            "resolution": {
                "min": float(self.res_min),
                "max": float(self.res_max),
                "step": float(self.step),
                "seeds": list(self.seeds),
            },
        }

    def index_entry(self, *, sweep_path: Path, status: str = "complete") -> RunIndexEntry:
        return RunIndexEntry(
            kind="sweep",
            run_slug=self.slug,
            alias=self.alias,
            created_utc=self.created_utc,
            source_kind=self.source.kind,
            source_ref=self.source.source_ref,
            x_col=self.feature.primary_label,
            n_rows=0,
            n_clusters=None,
            method_id=self.method_signature.method_id,
            method_params=self.method_signature.params,
            method_sig_hash=self.method_signature.hash(),
            input_sig_hash=None,
            labels_path=None,
            status=status,
            umap_slug=None,
            umap_params=None,
            coords_path=None,
            plot_paths=None,
            analysis_path=None,
            sweep_path=str(sweep_path),
        )


__all__ = [
    "AnalysisRun",
    "ClusterRun",
    "EmbeddingRun",
    "FIT_REUSE_REQUIRED_COLUMNS",
    "RUN_INDEX_COLUMNS",
    "RunCounts",
    "RunIndexEntry",
    "SweepRun",
    "fit_alias_from_cluster_col",
    "utc_now_iso",
]
