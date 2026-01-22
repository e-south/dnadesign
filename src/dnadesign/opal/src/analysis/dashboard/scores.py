"""Score overlay helpers for dashboard views."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from .diagnostics import Diagnostics


@dataclass(frozen=True)
class ScoreOverlayDiagnostics:
    source_key: str
    warnings: list[str]
    scalar_col: str | None
    rank_col: str | None
    top_k_col: str | None
    diagnostics: Diagnostics = field(default_factory=Diagnostics)


def add_provenance_columns(
    df: pl.DataFrame,
    *,
    campaign_slug: str | None,
    run_id: str | None,
    as_of_round: int | None,
    score_source_kind: str,
    prefix: str = "opal__score__",
) -> pl.DataFrame:
    return df.with_columns(
        pl.lit(score_source_kind).alias(f"{prefix}source_kind"),
        pl.lit(campaign_slug).alias(f"{prefix}campaign_slug"),
        pl.lit(run_id).alias(f"{prefix}run_id"),
        pl.lit(as_of_round).alias(f"{prefix}round"),
    )


def apply_score_overlay(
    df: pl.DataFrame,
    *,
    score_source_value: str,
    campaign_slug: str | None = None,
    selected_round: int | None,
    context: object | None = None,
) -> tuple[pl.DataFrame, ScoreOverlayDiagnostics]:
    warnings: list[str] = []
    source_key = "overlay"
    score_scalar_col = None
    score_rank_col = None
    score_top_k_col = None

    if score_source_value.startswith("Ledger"):
        source_key = "ledger"
        score_scalar_col = "opal__ledger__score"
        score_rank_col = "opal__ledger__rank"
        score_top_k_col = "opal__ledger__top_k"
    elif score_source_value.startswith("Records cache"):
        source_key = "cache"
        score_scalar_col = "opal__cache__score"
        score_rank_col = "opal__cache__rank"
        score_top_k_col = "opal__cache__top_k"
    else:
        source_key = "overlay"
        score_scalar_col = "opal__overlay__score"
        score_rank_col = "opal__overlay__rank"
        score_top_k_col = "opal__overlay__top_k"

    if score_scalar_col not in df.columns:
        warnings.append(f"Score source '{score_source_value}' missing '{score_scalar_col}'.")
        score_scalar_col = None
    if score_rank_col not in df.columns:
        warnings.append(f"Score source '{score_source_value}' missing '{score_rank_col}'.")
        score_rank_col = None
    if score_top_k_col not in df.columns:
        warnings.append(f"Score source '{score_source_value}' missing '{score_top_k_col}'.")
        score_top_k_col = None

    if source_key == "cache":
        if "opal__cache__run_id" not in df.columns:
            warnings.append("Cache score source missing opal__cache__run_id; provenance not run-aware.")
        if "opal__cache__round" not in df.columns:
            warnings.append("Cache score source missing opal__cache__round; round provenance unavailable.")

    if source_key == "ledger" and "opal__ledger__run_id" in df.columns:
        run_id_expr = pl.col("opal__ledger__run_id")
    elif source_key == "cache" and "opal__cache__run_id" in df.columns:
        run_id_expr = pl.col("opal__cache__run_id")
    elif source_key == "overlay":
        if "opal__overlay__run_id" in df.columns:
            run_id_expr = pl.col("opal__overlay__run_id")
        else:
            warnings.append("Overlay score source missing opal__overlay__run_id; provenance unavailable.")
            run_id_expr = pl.lit(None).cast(pl.Utf8)
    else:
        run_id_expr = pl.lit(None).cast(pl.Utf8)

    if source_key == "ledger" and "opal__ledger__round" in df.columns:
        round_expr = pl.col("opal__ledger__round")
    elif source_key == "cache" and "opal__cache__round" in df.columns:
        round_expr = pl.col("opal__cache__round")
    elif source_key == "overlay":
        if "opal__overlay__round" in df.columns:
            round_expr = pl.col("opal__overlay__round")
        else:
            warnings.append("Overlay score source missing opal__overlay__round; round provenance unavailable.")
            round_expr = pl.lit(None).cast(pl.Int64)
    else:
        round_expr = pl.lit(None).cast(pl.Int64)

    df_out = df.with_columns(
        [
            (pl.col(score_scalar_col) if score_scalar_col else pl.lit(None)).alias("opal__score__scalar"),
            (pl.col(score_rank_col) if score_rank_col else pl.lit(None)).alias("opal__score__rank"),
            (pl.col(score_top_k_col) if score_top_k_col else pl.lit(None).cast(pl.Boolean)).alias("opal__score__top_k"),
            pl.lit(score_source_value).alias("opal__score__source"),
        ]
    )
    if campaign_slug is None and context is not None:
        campaign_slug = getattr(getattr(context, "campaign_info", None), "slug", None)

    df_out = add_provenance_columns(
        df_out,
        campaign_slug=campaign_slug,
        run_id=None,
        as_of_round=selected_round,
        score_source_kind=source_key,
    ).with_columns(
        run_id_expr.alias("opal__score__run_id"),
        round_expr.alias("opal__score__round"),
    )

    diagnostics = Diagnostics(warnings=warnings)
    diag = ScoreOverlayDiagnostics(
        source_key=source_key,
        warnings=list(diagnostics.warnings),
        scalar_col=score_scalar_col,
        rank_col=score_rank_col,
        top_k_col=score_top_k_col,
        diagnostics=diagnostics,
    )
    return df_out, diag
