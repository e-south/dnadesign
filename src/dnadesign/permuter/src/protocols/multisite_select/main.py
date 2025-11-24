"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/protocols/multisite_select/main.py

Multi-site mutant variant selection.

High-level algorithm (see markdown spec):

  1. Load & validate source dataset (records.parquet from combine_aa + evaluate).
  2. Compute robust z-scores for observed fitness (LLR) and epistasis.
  3. Build composite score(v) = alpha · z_llr(v) + β · z_epi(v).
  4. Apply score-based gating to form a high-score candidate pool.
  5. Summarize clusters and compute medoids in embedding space.
  6. Run greedy score-ordered, diversity-aware selection within the pool:
       • obey per-cluster caps (if configured),
       • enforce minimum angular separation (if enabled),
       • stop when total_variants picks are accepted or pool is exhausted.
  7. Emit artifacts:
       • MULTISITE_SELECT.parquet / .csv
       • CLUSTER_SUMMARY.parquet
       • SELECT_SUMMARY.md
       • diagnostics plots & HEB tables (optional)
  8. Yield selected variants into this run's records.parquet.

All invariants from the spec are enforced:
  • only non-negative epistasis enter the scoring pipeline,
  • higher epistasis → higher composite score when alpha=0, β>0,
  • selection order respects composite score; diversity can only disqualify
    a candidate, never reorder them.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from dnadesign.permuter.src.core.storage import (
    append_record_event,
    atomic_write_parquet,
)
from dnadesign.permuter.src.plots.diagnostics import (
    hist_mut_count,
    plot_mutation_position_counts_selected,
    plot_pairwise_delta_k_selected_vs_random,
    plot_pairwise_levenshtein_selected_vs_random,
)
from dnadesign.permuter.src.plots.edge_bundling import edge_bundle_from_combos
from dnadesign.permuter.src.protocols.base import Protocol

from .geometry import (
    angular_distance,
    l2_normalize_rows,
    medoid_index,
    min_angular_distance_to_set,
    pairwise_angular,
)
from .scoring import compute_scaled_scores
from .utils import (
    MutationWindowSummary,
    _as_int_list,
    build_mutated_aa_sequences,
    extract_embedding_matrix,
    filter_valid_source_rows,
    read_source_records,
    ref_aa_from_dataset_dir,
    summarize_mutation_window,
    uppercase_mutated_codons,
)

_LOG = logging.getLogger("permuter.protocol.multisite_select")


# ---------------------------------------------------------------------------
# Knobs (YAML → dataclass)
# ---------------------------------------------------------------------------


@dataclass
class Knobs:
    # scoring
    normalize_method: str  # "mad" | "none"
    gaussian_consistent: bool
    winsor_mads: float | None
    w_llr: float  # α
    w_epi: float  # β

    # embedding
    embedding_col: str
    l2_normalize_embeddings: bool

    # candidate pool
    total_variants: int
    pool_factor: float  # f_pool ≥ 1

    # clusters
    picks_per_cluster: int | None  # None → no cap
    location_stat: str  # "mean" | "median" | "trimmed_mean"
    trimmed_mean_frac: float
    min_cluster_mean_z_llr: float | None
    min_cluster_pos_epistasis_fraction: float | None

    # diversity
    intracluster_diversity_enabled: bool
    min_angular_distance_deg: float

    # tie‑breakers
    prefer_fewer_mutations: bool
    then_higher_delta: bool
    then_higher_proposal_score: bool

    # diagnostics
    diag_figsize_in: float
    diag_dpi: int
    diag_random_seed: int
    diag_random_repeats: int
    diag_random_sample_factor: float
    diag_random_sample_cap: int

    # HEB
    heb_enabled: bool
    heb_min_cooccur_count: int
    heb_width_scale: str
    heb_figsize_in: float
    heb_out_svg: bool
    heb_edge_cmap: str
    heb_color_by: str

    # reproducibility
    rng_seed: int


# ---------------------------------------------------------------------------
# Selection core: greedy score‑ordered, diversity‑aware
# ---------------------------------------------------------------------------


def _build_score_sort_order(
    df: pd.DataFrame,
    knobs: Knobs,
) -> Tuple[List[str], List[bool]]:
    """
    Build deterministic sort order for score-gated candidate pool:

      1) composite score (desc),
      2) mut_count (asc) if prefer_fewer_mutations,
      3) delta (desc) if then_higher_delta,
      4) proposal_score (desc) if then_higher_proposal_score,
      5) var_id (asc, lexicographic) if present.
    """
    cols: List[str] = ["__score"]
    asc: List[bool] = [False]

    if knobs.prefer_fewer_mutations and "permuter__mut_count" in df.columns:
        cols.append("permuter__mut_count")
        asc.append(True)

    if knobs.then_higher_delta and "__delta" in df.columns:
        cols.append("__delta")
        asc.append(False)

    if knobs.then_higher_proposal_score and "permuter__proposal_score" in df.columns:
        cols.append("permuter__proposal_score")
        asc.append(False)

    if "permuter__var_id" in df.columns:
        cols.append("permuter__var_id")
        asc.append(True)

    return cols, asc


def _greedy_select_with_diversity(
    df_pool: pd.DataFrame,
    U_pool: np.ndarray,
    knobs: Knobs,
) -> List[int]:
    """
    Greedy score-ordered selection within the high-score pool.
    """
    total_variants = int(knobs.total_variants)
    if total_variants <= 0:
        raise ValueError("greedy selection: total_variants must be > 0")

    if len(df_pool) == 0:
        return []

    if U_pool.shape[0] != len(df_pool):
        raise ValueError(
            "greedy selection: embedding matrix and dataframe have incompatible shapes "
            f"(emb rows={U_pool.shape[0]}, df rows={len(df_pool)})"
        )

    clusters = df_pool["cluster__perm_v1"].to_numpy()
    row_indices = df_pool.index.to_numpy()

    picks: List[int] = []
    picks_clusters: Dict[object, int] = {}
    picks_vectors: List[np.ndarray] = []

    skipped_cap = 0
    skipped_diversity = 0

    theta_min_rad = math.radians(knobs.min_angular_distance_deg)

    # --- first pass: strict diversity ------------------------------------
    for pos in range(len(df_pool)):
        if len(picks) >= total_variants:
            break

        idx = int(row_indices[pos])
        cluster_id = clusters[pos]

        # cluster cap
        if knobs.picks_per_cluster is not None:
            if picks_clusters.get(cluster_id, 0) >= knobs.picks_per_cluster:
                skipped_cap += 1
                continue

        # diversity (angular separation, global)
        if knobs.intracluster_diversity_enabled and picks_vectors:
            theta = min_angular_distance_to_set(U_pool[pos], np.stack(picks_vectors))
            if theta < theta_min_rad:
                skipped_diversity += 1
                continue

        # accept
        picks.append(idx)
        picks_clusters[cluster_id] = picks_clusters.get(cluster_id, 0) + 1
        picks_vectors.append(U_pool[pos])

    _LOG.info(
        "[selection] greedy pass:\n"
        "  considered:          %d\n"
        "  picked:              %d\n"
        "  skipped_cluster_cap: %d\n"
        "  skipped_diversity:   %d",
        len(df_pool),
        len(picks),
        skipped_cap,
        skipped_diversity,
    )

    # No automatic backfill: if constraints are too strict, we under-fill.
    if len(picks) < total_variants:
        if knobs.intracluster_diversity_enabled and skipped_diversity > 0:
            _LOG.warning(
                "[selection] budget underfilled after diversity pass "
                "(picks=%d < total_variants=%d); no backfill performed. "
                "If a full budget is required, lower "
                "select.budget.intracluster_diversity.min_angular_distance_deg, "
                "increase select.budget.pool_factor, or disable "
                "select.budget.intracluster_diversity.enabled.",
                len(picks),
                total_variants,
            )
        else:
            _LOG.warning(
                "[selection] budget underfilled (picks=%d < total_variants=%d); "
                "candidate pool exhausted before meeting budget. "
                "Consider increasing select.budget.pool_factor or relaxing "
                "upstream filters.",
                len(picks),
                total_variants,
            )

    if not picks:
        raise RuntimeError(
            "multisite_select: selection produced zero picks; "
            "consider relaxing min_angular_distance_deg, pool_factor, "
            "or upstream filters."
        )

    return picks


# ---------------------------------------------------------------------------
# Protocol implementation
# ---------------------------------------------------------------------------


class MSel(Protocol):
    """
    Multi-site selection protocol driven by job.permute.params.select.

    Artifacts written into params._artifact_dir:

      • MULTISITE_SELECT.parquet / .csv
      • CLUSTER_SUMMARY.parquet
      • SELECT_SUMMARY.md
      • diagnostic PNGs
      • EDGE_BUNDLE_TABLE.parquet (if HEB enabled)

    The yielded variants become rows in this run's records.parquet.
    """

    # -------------------------- Validation ---------------------------------

    def validate_cfg(self, *, params: Dict) -> None:
        if not isinstance(params, dict):
            raise ValueError("multisite_select: params must be a mapping")
        if not params.get("from_dataset"):
            raise ValueError(
                "multisite_select: params.from_dataset is required "
                "(dataset directory or records.parquet path)"
            )

        sel = params.get("select") or {}
        scoring = sel.get("scoring") or {}
        normalize = scoring.get("normalize") or {}
        method = str(normalize.get("method", "mad")).lower()
        if method not in {"mad", "none"}:
            raise ValueError(
                "multisite_select: select.scoring.normalize.method must be 'mad' or 'none'"
            )

        emb_cfg = sel.get("embedding") or {}
        if emb_cfg.get("distance", "angular") != "angular":
            raise ValueError(
                "multisite_select: only angular distance is supported "
                "(select.embedding.distance must be 'angular')"
            )
        if emb_cfg.get("representative", "medoid") != "medoid":
            raise ValueError(
                "multisite_select: only representative='medoid' is implemented"
            )

        budget = sel.get("budget") or {}
        tot = int(budget.get("total_variants") or 0)
        if tot <= 0:
            raise ValueError(
                "multisite_select: select.budget.total_variants must be > 0"
            )

        pool_factor = float(budget.get("pool_factor", 3.0))
        if pool_factor < 1.0:
            raise ValueError(
                "multisite_select: select.budget.pool_factor must be ≥ 1.0"
            )

        intra_cfg = budget.get("intracluster_diversity") or {}
        if intra_cfg.get("enabled", False):
            if "min_angular_distance_deg" not in intra_cfg:
                raise ValueError(
                    "multisite_select: select.budget.intracluster_diversity.enabled "
                    "is true but min_angular_distance_deg is not set; "
                    "provide an explicit angular threshold in degrees."
                )
            try:
                min_ang = float(intra_cfg["min_angular_distance_deg"])
            except Exception as e:
                raise ValueError(
                    "multisite_select: select.budget.intracluster_diversity."
                    "min_angular_distance_deg must be a real number (degrees)"
                ) from e
            if min_ang < 0.0:
                raise ValueError(
                    "multisite_select: select.budget.intracluster_diversity."
                    "min_angular_distance_deg must be ≥ 0"
                )

        cl_cfg = sel.get("clusters") or {}
        picks_per_cluster = cl_cfg.get("picks_per_cluster", None)
        if picks_per_cluster is not None:
            k = int(picks_per_cluster)
            if k <= 0:
                raise ValueError(
                    "multisite_select: select.clusters.picks_per_cluster must be ≥ 1 "
                    "or omitted/Null to disable caps"
                )

        if not params.get("_artifact_dir"):
            raise ValueError(
                "multisite_select: _artifact_dir missing (internal); "
                "invoke via 'permuter run'"
            )

    # -------------------------- Generation ---------------------------------

    def generate(
        self,
        *,
        ref_entry: Dict,
        params: Dict,
        rng: Optional[np.random.Generator] = None,
    ) -> Iterable[Dict]:
        knobs = self._knobs_from(params)
        art_dir = Path(str(params["_artifact_dir"])).expanduser().resolve()
        art_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Load & validate source dataset ------------------------------
        src = read_source_records(params["from_dataset"])
        src_path = Path(str(params["from_dataset"])).expanduser().resolve()
        src_dir = src_path if src_path.is_dir() else src_path.parent

        # Infer the canonical observed/expected metric pair
        exp_cols = [c for c in src.columns if c.startswith("permuter__expected__")]
        if len(exp_cols) != 1:
            raise RuntimeError(
                "multisite_select: expected exactly one 'permuter__expected__*' column "
                f"in source dataset, found {exp_cols!r}. "
                "Ensure your upstream protocol emitted a single expected metric."
            )
        exp_col = exp_cols[0]
        metric_id = exp_col[len("permuter__expected__") :]
        obs_col = f"permuter__observed__{metric_id}"
        if obs_col not in src.columns:
            raise RuntimeError(
                "multisite_select: matching observed column for epistasis is missing.\n"
                f"  expected: {obs_col}\n"
                f"  present expected: {exp_col}\n"
                "Run 'permuter evaluate' with an evaluator id matching the expected metric."
            )

        _LOG.info(
            "[data] metric\n"
            "  metric_id: %s\n"
            "  observed:  %s\n"
            "  expected:  %s\n"
            "  epistasis: 'epistasis' column (obs - exp)",
            metric_id,
            obs_col,
            exp_col,
        )

        if "epistasis" not in src.columns:
            raise ValueError(
                "multisite_select: source dataset is missing required 'epistasis' column.\n"
                "Run 'permuter evaluate' (or equivalent) to attach epistasis "
                "before running multisite_select."
            )

        required_cols = [
            "id",
            "sequence",
            obs_col,
            exp_col,
            "epistasis",
            knobs.embedding_col,
            "permuter__aa_pos_list",
            "permuter__mut_count",
            "cluster__perm_v1",
        ]
        missing = [c for c in required_cols if c not in src.columns]
        if missing:
            raise ValueError(
                f"multisite_select: source dataset missing columns: {missing}"
            )

        df_raw, finfo = filter_valid_source_rows(
            src,
            emb_col=knobs.embedding_col,
            aa_col="permuter__aa_pos_list",
            llr_col=obs_col,
            epi_col="epistasis",
        )
        drops_pretty = json.dumps(finfo["drops_by_cause"], sort_keys=True, indent=2)
        _LOG.info(
            "[data] validation\n"
            "  input_rows: %d\n"
            "  valid_rows: %d\n"
            "  drops_by_cause: %s",
            finfo["n_total"],
            finfo["n_kept"],
            drops_pretty,
        )
        if df_raw.empty:
            raise RuntimeError(
                "multisite_select: no usable rows after validation — "
                f"drops={json.dumps(finfo['drops_by_cause'])}."
            )

        df = df_raw.copy()

        # Aliases
        df["__llr_obs"] = df[obs_col].astype(float).to_numpy()
        df["__llr_exp"] = df[exp_col].astype(float).to_numpy()
        df["__delta"] = df["epistasis"].astype(float).to_numpy()
        if (df["__delta"] < 0).any():
            raise RuntimeError(
                "multisite_select: negative epistasis survived row‑level validation; "
                "this should not happen."
            )

        # Parse AA positions
        df["__aa_list"] = df["permuter__aa_pos_list"].apply(_as_int_list)
        df = df[df["__aa_list"].apply(len) > 0].reset_index(drop=True)
        if df.empty:
            raise RuntimeError(
                "multisite_select: no rows with non‑empty permuter__aa_pos_list after parsing"
            )

        # Embeddings (mean‑pooled logits)
        emb = extract_embedding_matrix(df[knobs.embedding_col])
        if knobs.l2_normalize_embeddings:
            emb = l2_normalize_rows(emb)

        # --- 2. Robust z‑scores & composite score ---------------------------
        if knobs.normalize_method == "none":
            z_llr = df["__llr_obs"].astype(float).to_numpy()
            z_epi = df["__delta"].astype(float).to_numpy()
            score = knobs.w_llr * z_llr + knobs.w_epi * z_epi
            summary = None
        else:
            z_llr, z_epi, score, summary = self._compute_scores(df, knobs)

        df["__z_llr"] = z_llr
        df["__z_epi"] = z_epi
        df["__score"] = score

        if summary is not None:
            _LOG.info(
                "[scaling]\n"
                "  LLR: median=%.3f MAD=%.3f\n"
                "  epi: median=%.3f MAD=%.3f",
                summary.median_llr,
                summary.mad_llr,
                summary.median_epi,
                summary.mad_epi,
            )

        _LOG.info(
            "[scoring]\n" "  weights: α(llr)=%.3f β(epi)=%.3f",
            knobs.w_llr,
            knobs.w_epi,
        )

        # --- 3. Score‑gated high‑score pool --------------------------------
        sort_cols, sort_asc = _build_score_sort_order(df, knobs)
        df_sorted = df.sort_values(by=sort_cols, ascending=sort_asc, kind="mergesort")
        n_valid = len(df_sorted)
        total = knobs.total_variants
        f_pool = float(knobs.pool_factor)
        pool_size = min(n_valid, max(total, int(math.ceil(f_pool * total))))
        _LOG.info(
            "[pool] total_valid=%d total_variants=%d pool_factor=%.2f → pool_size=%d",
            n_valid,
            total,
            f_pool,
            pool_size,
        )

        df_pool = df_sorted.head(pool_size).copy()
        emb_pool = emb[df_pool.index.to_numpy()]
        if knobs.l2_normalize_embeddings:
            U_pool = emb_pool
        else:
            U_pool = l2_normalize_rows(emb_pool)

        # --- 4. Cluster summarization & filters -----------------------------
        clust_df = self._summarize_clusters(df_pool, emb, knobs)
        if clust_df.empty:
            raise RuntimeError(
                "multisite_select: no clusters with valid statistics in df_pool"
            )

        # cluster‑level filters (optional)
        kept_cluster_ids = self._apply_cluster_filters(clust_df, knobs)
        if kept_cluster_ids is not None:
            mask = df_pool["cluster__perm_v1"].isin(kept_cluster_ids)
            df_pool = df_pool[mask].copy()
            U_pool = U_pool[mask.to_numpy()]
            clust_df = clust_df[clust_df["cluster_id"].isin(kept_cluster_ids)].copy()
            _LOG.info(
                "[clusters] after quality filters: clusters=%d pool_rows=%d",
                len(kept_cluster_ids),
                len(df_pool),
            )
            if df_pool.empty:
                raise RuntimeError(
                    "multisite_select: all candidates rejected by cluster filters"
                )

        # --- 5. Diversity‑aware selection within df_pool --------------------
        selected_row_indices = _greedy_select_with_diversity(df_pool, U_pool, knobs)
        if not selected_row_indices:
            raise RuntimeError(
                "multisite_select: selection produced zero picks; "
                "consider relaxing min_angular_distance_deg or pool_factor."
            )

        # --- 6. Build selection frame ---------------------------------------
        picks_df, chosen_cluster_ids = self._build_picks_df(
            df=df,
            emb=emb,
            clust_df=clust_df,
            selected_row_indices=selected_row_indices,
        )

        # --- 7. Cluster summary artifact -----------------------------------
        clust_df_summary = self._augment_cluster_summary_with_angles(
            clust_df=clust_df,
            emb=emb,
            chosen_cluster_ids=chosen_cluster_ids,
        )
        atomic_write_parquet(clust_df_summary, art_dir / "CLUSTER_SUMMARY.parquet")

        # --- 8. Diagnostics & plots -----------------------------------------
        self._run_diagnostics(
            df=df,
            picks_df=picks_df,
            emb=emb,
            src_dir=src_dir,
            art_dir=art_dir,
            knobs=knobs,
        )

        # --- 9. Final artifacts & summary -----------------------------------
        picks_df_raw, picks_df_out = self._emit_selection_artifacts(
            picks_df=picks_df,
            art_dir=art_dir,
        )

        # --- 9b. Mutation-window summary across selected variants ----------
        mut_window_summary: MutationWindowSummary | None = None
        try:
            ref_name, ref_aa = ref_aa_from_dataset_dir(src_dir)
            # Primary nucleotide reference from the CLI ref entry (seq_col)
            ref_nt = ref_entry.get("sequence", None)

            if ref_aa:
                mut_window_summary = summarize_mutation_window(
                    ref_seq=ref_aa,
                    aa_pos_lists=picks_df_out["aa_pos_list"].tolist(),
                    flank=10,
                    ref_nt_seq=ref_nt,
                )
                m = mut_window_summary

                # Compute codon count defensively (falls back to AA window length)
                codons = (
                    m.window_length_nt // 3
                    if m.window_length_nt is not None
                    else m.window_length
                )

                _LOG.info(
                    "[span]\n"
                    "  ref_length:           %d aa%s\n"
                    "  window_start_pos:     %d (aa)\n"
                    "  window_end_pos:       %d (aa)\n"
                    "  window_length:        %d aa%s\n"
                    "  left_flank_aa:        %s\n"
                    "  window_seq_aa:        %s\n"
                    "  right_flank_aa:       %s\n"
                    "  left_flank_nt:        %s\n"
                    "  window_seq_nt:        %s\n"
                    "  right_flank_nt:       %s",
                    m.ref_length,
                    (f" ({m.ref_length_nt} nt)" if m.ref_length_nt is not None else ""),
                    m.start_pos,
                    m.end_pos,
                    m.window_length,
                    (
                        f" ({m.window_length_nt} nt; {codons} codons)"
                        if m.window_length_nt is not None
                        else ""
                    ),
                    m.left_flank,
                    m.window_seq,
                    m.right_flank,
                    m.left_flank_nt or "",
                    m.window_seq_nt or "",
                    m.right_flank_nt or "",
                )
            else:
                _LOG.info(
                    "[span] REF_AA.fa missing in %s; skipping mutation-window "
                    "summary for selected variants",
                    src_dir,
                )
        except Exception as e:
            _LOG.warning("[span] mutation-window summary failed: %s", e)
            mut_window_summary = None

        except Exception as e:
            _LOG.warning("[span] mutation-window summary failed: %s", e)
            mut_window_summary = None

        self._append_record_select(
            art_dir=art_dir,
            df=df,
            knobs=knobs,
            chosen_cluster_ids=chosen_cluster_ids,
        )

        self._write_select_summary(
            art_dir=art_dir,
            picks=picks_df_out,
            clust_df=clust_df_summary,
            chosen_cluster_ids=chosen_cluster_ids,
            mut_window_summary=mut_window_summary,
        )

        # --- 10. Emit variants for records.parquet --------------------------
        for _, row in picks_df_raw.iterrows():
            head = (
                f"select score={row['score']:+.4f} "
                f"z_llr={row['z_llr']:+.3f} z_epi={row['z_epi']:+.3f} "
                f"delta={row['delta']:+.4f} k={int(row['k'])} "
                f"cluster={row['cluster_id']}"
            )
            aa_token = f"aa [{row['aa_combo_str']}]" if row["aa_combo_str"] else "aa []"
            mods = [head, aa_token]
            yield {
                "sequence": str(row["sequence"]),
                "modifications": mods,
                # Flattened scalar columns — run.py will namespace with permuter__*
                "source_id": str(row["source_id"]),
                "llr_obs": float(row["llr_obs"]),
                "llr_exp": float(row["llr_exp"]),
                "delta": float(row["delta"]),
                "z_llr": float(row["z_llr"]),
                "z_epi": float(row["z_epi"]),
                "score": float(row["score"]),
                "aa_pos_list": list(row["aa_pos_list"]),
                "aa_combo_str": str(row["aa_combo_str"]),
                "mut_count": int(row["k"]),
                "cluster_id": row["cluster_id"],
                "proposal_score": (
                    float(row["proposal_score"])
                    if np.isfinite(row["proposal_score"])
                    else None
                ),
                "source_var_id": str(row["source_var_id"]),
                "angle_to_cluster_medoid_deg": (
                    float(row["angle_to_cluster_medoid_deg"])
                    if row["angle_to_cluster_medoid_deg"] is not None
                    else None
                ),
                "angle_to_nearest_selected_medoid_deg": (
                    float(row["angle_to_nearest_selected_medoid_deg"])
                    if row["angle_to_nearest_selected_medoid_deg"] is not None
                    else None
                ),
            }

    # ------------------------------ Helpers ---------------------------------

    def _knobs_from(self, params: Dict) -> Knobs:
        sel = params.get("select") or {}
        scoring = sel.get("scoring") or {}
        normalize = scoring.get("normalize") or {}
        weights = scoring.get("weights") or {}
        emb = sel.get("embedding") or {}
        cl = sel.get("clusters") or {}
        filt = cl.get("filters") or {}
        budget = sel.get("budget") or {}
        tie = sel.get("tie_breakers") or {}
        diag = sel.get("diagnostics") or {}
        heb = sel.get("heb") or {}
        rep = sel.get("reproducibility") or {}

        method = str(normalize.get("method", "mad")).lower()
        gaussian = bool(normalize.get("gaussian_consistent", False))
        winsor = normalize.get("winsor_mads", None)
        winsor = float(winsor) if winsor is not None else None

        min_cluster_mean_z_llr = filt.get("min_cluster_mean_z_llr", None)
        if min_cluster_mean_z_llr is not None:
            min_cluster_mean_z_llr = float(min_cluster_mean_z_llr)
        min_cluster_pos_epi = filt.get("min_cluster_pos_epistasis_fraction", None)
        if min_cluster_pos_epi is not None:
            min_cluster_pos_epi = float(min_cluster_pos_epi)

        # budget & pool
        total_variants = int(budget.get("total_variants", 24))
        pool_factor = float(budget.get("pool_factor", 3.0))

        intracluster_diversity = budget.get("intracluster_diversity") or {}
        # Diversity is opt-in; default is disabled and threshold 0.0 (no constraint).
        intra_enabled = bool(intracluster_diversity.get("enabled", False))
        intra_min_angle_deg = float(
            intracluster_diversity.get("min_angular_distance_deg", 0.0)
        )

        picks_per_cluster = cl.get("picks_per_cluster", None)
        if picks_per_cluster is not None:
            picks_per_cluster = int(picks_per_cluster)

        return Knobs(
            normalize_method=method,
            gaussian_consistent=gaussian,
            winsor_mads=winsor,
            w_llr=float(weights.get("llr", 1.0)),
            w_epi=float(weights.get("epi", 1.0)),
            embedding_col=str(emb.get("column", "permuter__observed__logits_mean")),
            l2_normalize_embeddings=bool(emb.get("l2_normalize", True)),
            total_variants=total_variants,
            pool_factor=pool_factor,
            picks_per_cluster=picks_per_cluster,
            location_stat=str(filt.get("location_stat", "mean")).lower(),
            trimmed_mean_frac=float(filt.get("trimmed_mean_frac", 0.10)),
            min_cluster_mean_z_llr=min_cluster_mean_z_llr,
            min_cluster_pos_epistasis_fraction=min_cluster_pos_epi,
            intracluster_diversity_enabled=intra_enabled,
            min_angular_distance_deg=intra_min_angle_deg,
            prefer_fewer_mutations=bool(tie.get("prefer_fewer_mutations", True)),
            then_higher_delta=bool(tie.get("then_higher_delta", False)),
            then_higher_proposal_score=bool(
                tie.get("then_higher_proposal_score", False)
            ),
            diag_figsize_in=float(diag.get("figsize_in", 8.0)),
            diag_dpi=int(diag.get("dpi", 200)),
            diag_random_seed=int(diag.get("random_sample_seed", 20251113)),
            diag_random_repeats=int(diag.get("random_sample_repeats", 1)),
            diag_random_sample_factor=float(diag.get("random_sample_factor", 4.0)),
            diag_random_sample_cap=int(diag.get("random_sample_cap", 512)),
            heb_enabled=bool(heb.get("enabled", True)),
            heb_min_cooccur_count=int(heb.get("min_cooccur_count", 1)),
            heb_width_scale=str(heb.get("width_scale", "sqrt")).lower(),
            heb_figsize_in=float(heb.get("figsize_in", 10.0)),
            heb_out_svg=bool(heb.get("out_svg", True)),
            heb_edge_cmap=str(heb.get("edge_cmap", "viridis")),
            heb_color_by=str(heb.get("color_by", "node_avg_k")).lower(),
            rng_seed=int(rep.get("rng_seed", 20251113)),
        )

    def _compute_scores(self, df: pd.DataFrame, knobs: Knobs):
        z_llr, z_epi, score, summary = compute_scaled_scores(
            llr_obs=df["__llr_obs"],
            delta=df["__delta"],
            gaussian_consistent=knobs.gaussian_consistent,
            winsor_mads=knobs.winsor_mads,
            w_llr=knobs.w_llr,
            w_epi=knobs.w_epi,
        )
        return z_llr, z_epi, score, summary

    def _summarize_clusters(
        self,
        df_pool: pd.DataFrame,
        emb: np.ndarray,
        knobs: Knobs,
    ) -> pd.DataFrame:
        groups = list(df_pool.groupby("cluster__perm_v1", sort=False))
        if not groups:
            raise RuntimeError(
                "multisite_select: no clusters present in 'cluster__perm_v1'"
            )

        cluster_meta: List[Dict] = []
        for lab, sub in groups:
            idx = sub.index.to_numpy()
            U = l2_normalize_rows(emb[idx])
            mi = medoid_index(U)
            medoid_row = int(idx[mi])

            zllr = sub["__z_llr"].to_numpy()
            zepi = sub["__z_epi"].to_numpy()
            comp = sub["__score"].to_numpy()
            pos_frac = float((sub["__delta"] > 0).mean())

            if knobs.location_stat == "median":
                loc = float(np.median(zllr))
            elif knobs.location_stat == "trimmed_mean":
                f = float(knobs.trimmed_mean_frac)
                a = np.sort(zllr)
                m = len(a)
                if m:
                    lo = int(math.floor(f * m))
                    hi = int(math.ceil((1 - f) * m))
                    loc = float(a[lo:hi].mean() if hi > lo else a.mean())
                else:
                    loc = float("nan")
            else:  # mean
                loc = float(zllr.mean())

            cluster_meta.append(
                {
                    "cluster_id": lab,
                    "size": int(len(sub)),
                    "mean_z_llr": float(zllr.mean()),
                    "mean_z_epi": float(zepi.mean()),
                    "mean_composite": float(comp.mean()),
                    "pos_epi_fraction": pos_frac,
                    "loc_stat": loc,
                    "medoid_row": medoid_row,
                }
            )

        return pd.DataFrame(cluster_meta)

    def _apply_cluster_filters(
        self,
        clust_df: pd.DataFrame,
        knobs: Knobs,
    ) -> Optional[List]:
        mask = np.ones(len(clust_df), dtype=bool)

        if knobs.min_cluster_mean_z_llr is not None:
            m = clust_df["mean_z_llr"].to_numpy()
            keep = m >= float(knobs.min_cluster_mean_z_llr)
            mask &= keep
            _LOG.info(
                "[clusters] filter min_cluster_mean_z_llr=%.3f → kept=%d dropped=%d",
                knobs.min_cluster_mean_z_llr,
                int(keep.sum()),
                int((~keep).sum()),
            )

        if knobs.min_cluster_pos_epistasis_fraction is not None:
            f = clust_df["pos_epi_fraction"].to_numpy()
            keep = f >= float(knobs.min_cluster_pos_epistasis_fraction)
            mask &= keep
            _LOG.info(
                "[clusters] filter min_cluster_pos_epistasis_fraction=%.3f → kept=%d dropped=%d",
                knobs.min_cluster_pos_epistasis_fraction,
                int(keep.sum()),
                int((~keep).sum()),
            )

        if mask.all():
            return None

        kept = clust_df.loc[mask, "cluster_id"].tolist()
        return kept

    def _build_picks_df(
        self,
        *,
        df: pd.DataFrame,
        emb: np.ndarray,
        clust_df: pd.DataFrame,
        selected_row_indices: List[int],
    ) -> Tuple[pd.DataFrame, List]:
        med_row_by_cid = {
            row["cluster_id"]: int(row["medoid_row"]) for _, row in clust_df.iterrows()
        }

        sel_rows: List[Dict] = []
        for idx in selected_row_indices:
            r = df.loc[idx]
            cid = r["cluster__perm_v1"]
            aa_list = list(map(int, r["__aa_list"]))

            angle_deg = None
            med_row = med_row_by_cid.get(cid, None)
            if med_row is not None:
                try:
                    angle_deg = float(
                        math.degrees(angular_distance(emb[idx], emb[int(med_row)]))
                    )
                except Exception:
                    angle_deg = None

            sel_rows.append(
                {
                    "_row_idx": int(idx),
                    "source_id": r["id"],
                    "sequence": str(r["sequence"]),
                    "cluster_id": cid,
                    "k": int(r["permuter__mut_count"]),
                    "llr_obs": float(r["__llr_obs"]),
                    "llr_exp": float(r["__llr_exp"]),
                    "delta": float(r["__delta"]),
                    "z_llr": float(r["__z_llr"]),
                    "z_epi": float(r["__z_epi"]),
                    "score": float(r["__score"]),
                    "angle_to_cluster_medoid_deg": angle_deg,
                    "aa_pos_list": aa_list,
                    "aa_combo_str": str(r.get("permuter__aa_combo_str", "")),
                    "proposal_score": float(r.get("permuter__proposal_score", np.nan)),
                    "source_var_id": str(r.get("permuter__var_id", "")),
                }
            )

        picks_df = pd.DataFrame(sel_rows)
        chosen_cluster_ids = sorted(
            picks_df["cluster_id"].astype(object).unique().tolist()
        )

        # angle to nearest selected cluster medoid (across chosen clusters)
        if not picks_df.empty and chosen_cluster_ids:
            sel_c = clust_df[clust_df["cluster_id"].isin(chosen_cluster_ids)].copy()
            sel_c = sel_c.set_index("cluster_id").loc[chosen_cluster_ids].reset_index()
            med_arr_sel = l2_normalize_rows(
                emb[sel_c["medoid_row"].astype(int).tolist()]
            )
            V = l2_normalize_rows(emb[picks_df["_row_idx"].astype(int).to_numpy()])
            ang_nearest = []
            for vec in V:
                dots = np.clip(med_arr_sel @ vec, -1.0, 1.0)
                ang_nearest.append(float(np.degrees(np.arccos(np.max(dots)))))
            picks_df["angle_to_nearest_selected_medoid_deg"] = ang_nearest

        return picks_df, chosen_cluster_ids

    def _augment_cluster_summary_with_angles(
        self,
        *,
        clust_df: pd.DataFrame,
        emb: np.ndarray,
        chosen_cluster_ids: List,
    ) -> pd.DataFrame:
        clust_df_summary = clust_df.copy()
        clust_df_summary["is_selected"] = clust_df_summary["cluster_id"].isin(
            set(chosen_cluster_ids)
        )

        if len(chosen_cluster_ids) >= 2:
            sel_c = clust_df_summary[
                clust_df_summary["cluster_id"].isin(chosen_cluster_ids)
            ].copy()
            sel_c = sel_c.set_index("cluster_id").loc[chosen_cluster_ids].reset_index()
            med_arr = l2_normalize_rows(emb[sel_c["medoid_row"].astype(int).tolist()])
            A = pairwise_angular(med_arr)
            mins = []
            for i in range(A.shape[0]):
                other = np.delete(A[i], i)
                mins.append(float(np.rad2deg(np.min(other))) if other.size else 0.0)

            clust_df_summary = clust_df_summary.set_index("cluster_id")
            for cid, v in zip(chosen_cluster_ids, mins):
                clust_df_summary.loc[cid, "min_inter_medoid_angle_deg"] = v
            clust_df_summary = clust_df_summary.reset_index()

        return clust_df_summary

    # --- Diagnostics and artifacts -----------------------------------------

    def _run_diagnostics(
        self,
        *,
        df: pd.DataFrame,
        picks_df: pd.DataFrame,
        emb: np.ndarray,
        src_dir: Path,
        art_dir: Path,
        knobs: Knobs,
    ) -> None:
        # random comparator indices
        try:
            rng_diag = np.random.default_rng(knobs.diag_random_seed)
            all_idx = df.index.to_numpy()
            sel_idx = picks_df["_row_idx"].astype(int).to_numpy()
            mask_bg = np.ones_like(all_idx, dtype=bool)
            mask_bg[np.searchsorted(all_idx, sel_idx)] = False
            bg_idx = all_idx[mask_bg]
            n_sel = len(sel_idx)
            random_indices_samples: List[np.ndarray] = []
            factor = float(knobs.diag_random_sample_factor)
            cap = int(knobs.diag_random_sample_cap)
            if len(bg_idx) > 0 and n_sel > 0:
                base = n_sel
                target = int(max(base, math.ceil(base * factor)))
                sample_size = min(max(1, target), cap, len(bg_idx))
                for _ in range(max(1, knobs.diag_random_repeats)):
                    rnd = rng_diag.choice(bg_idx, size=sample_size, replace=False)
                    random_indices_samples.append(np.sort(rnd))
            effective_sample_size = (
                int(random_indices_samples[0].shape[0]) if random_indices_samples else 0
            )
            _LOG.info(
                "[diagnostics]\n"
                "  selected:             %d\n"
                "  background:           %d\n"
                "  random_samples:       %d\n"
                "  random_sample_size:   %s\n"
                "  random_sample_factor: %.2f\n"
                "  random_sample_cap:    %d",
                len(sel_idx),
                len(bg_idx),
                len(random_indices_samples),
                effective_sample_size,
                factor,
                cap,
            )
        except Exception as e:
            _LOG.warning("[diagnostics] random comparator generation failed: %s", e)
            random_indices_samples = []

        # 1) mutation‑count histogram
        try:
            k_all = df["permuter__mut_count"].astype(int).to_numpy()
            k_selected = k_all[sel_idx]
            if random_indices_samples:
                rand_idx_concat = np.concatenate(random_indices_samples)
                k_random = k_all[rand_idx_concat]
            else:
                k_random = None
            hist_mut_count(
                k_selected,
                k_random=k_random,
                out_png=art_dir / "fig_mut_count_hist_selected.png",
                title="Mutation-count distribution (selected vs random)",
                figsize_in=knobs.diag_figsize_in,
                dpi=knobs.diag_dpi,
            )
        except Exception as e:
            _LOG.warning("[diagnostics] mutation‑count histogram failed: %s", e)

        # 1b) mutation-position value counts among selected variants
        try:
            if not picks_df.empty and "aa_pos_list" in picks_df.columns:
                plot_mutation_position_counts_selected(
                    picks_df["aa_pos_list"].tolist(),
                    out_png=art_dir / "fig_mut_position_counts_selected.png",
                    title="Mutation position counts among selected variants",
                    figsize_in=knobs.diag_figsize_in,
                    dpi=knobs.diag_dpi,
                )
        except Exception as e:
            _LOG.warning("[diagnostics] mutation-position count plot failed: %s", e)

        # 2) pairwise |Δk|
        try:
            if len(k_selected) >= 2:
                k_random_samples = [
                    k_all[idx] for idx in random_indices_samples if idx.size >= 2
                ]
                plot_pairwise_delta_k_selected_vs_random(
                    k_selected,
                    k_random_samples,
                    out_png=art_dir / "fig_pairwise_delta_k_selected_vs_random.png",
                    title="Pairwise |Δk| (selected vs random)",
                    figsize_in=knobs.diag_figsize_in,
                    dpi=knobs.diag_dpi,
                )
        except Exception as e:
            _LOG.warning("[diagnostics] pairwise Δk histogram failed: %s", e)

        # 3) pairwise Levenshtein (AA)
        try:
            ref_name, ref_aa = ref_aa_from_dataset_dir(src_dir)
            if ref_aa:
                aa_all = build_mutated_aa_sequences(
                    ref_aa,
                    df.get("permuter__aa_combo_str", pd.Series([""] * len(df))),
                )
                seq_sel = aa_all[sel_idx]
                seq_random_samples: List[Sequence[str]] = [
                    aa_all[idx] for idx in random_indices_samples if idx.size >= 2
                ]
                if len(seq_sel) >= 2 and seq_random_samples:
                    plot_pairwise_levenshtein_selected_vs_random(
                        seq_sel,
                        seq_random_samples,
                        out_png=art_dir
                        / "fig_pairwise_levenshtein_selected_vs_random.png",
                        title="Pairwise Levenshtein distance (AA; selected vs random)",
                        figsize_in=knobs.diag_figsize_in,
                        dpi=knobs.diag_dpi,
                    )
            else:
                _LOG.info(
                    "[diagnostics] REF_AA.fa missing in %s; "
                    "skipping Levenshtein AA plot",
                    src_dir,
                )
        except Exception as e:
            _LOG.warning("[diagnostics] pairwise Levenshtein plot failed: %s", e)

        # 4) HEB (selected only)
        if knobs.heb_enabled:
            try:
                edges_df = edge_bundle_from_combos(
                    aa_combo_strs=picks_df["aa_combo_str"].tolist(),
                    k_values=picks_df["k"].tolist(),
                    min_cooccur_count=knobs.heb_min_cooccur_count,
                    width_scale=knobs.heb_width_scale,
                    out_png=art_dir / "fig_edge_bundling_selected.png",
                    out_pdf=(
                        art_dir / "fig_edge_bundling_selected.pdf"
                        if knobs.heb_out_svg
                        else None
                    ),
                    figsize_in=knobs.heb_figsize_in,
                    dpi=knobs.diag_dpi,
                    edge_cmap=knobs.heb_edge_cmap,
                    color_by=knobs.heb_color_by,
                )
                if edges_df is not None and not edges_df.empty:
                    atomic_write_parquet(
                        edges_df, art_dir / "EDGE_BUNDLE_TABLE.parquet"
                    )
                    _LOG.info(
                        "[heb]\n"
                        "  nodes:  %d\n"
                        "  edges:  %d\n"
                        "  weight[min,med,max]: [%.1f, %.1f, %.1f]\n"
                        "  avg_k[min,max]:      [%.1f, %.1f]",
                        len(set(edges_df["token_a"]).union(set(edges_df["token_b"]))),
                        len(edges_df),
                        float(edges_df["weight_count"].min()),
                        float(edges_df["weight_count"].median()),
                        float(edges_df["weight_count"].max()),
                        float(edges_df["avg_k"].min()),
                        float(edges_df["avg_k"].max()),
                    )
            except Exception as e:
                _LOG.warning("[heb] edge bundling plot failed: %s", e)

        # 5) global minimum pairwise angle among selected variants
        try:
            if len(picks_df) >= 2:
                sel_idx_sorted = np.sort(picks_df["_row_idx"].astype(int).to_numpy())
                U_pick = l2_normalize_rows(emb[sel_idx_sorted])
                A = pairwise_angular(U_pick)
                mins = []
                for i in range(A.shape[0]):
                    other = np.delete(A[i], i)
                    if other.size:
                        mins.append(float(np.rad2deg(np.min(other))))
                if mins:
                    _LOG.info(
                        "[picks] global minimum pairwise angle among picks: "
                        "%.2f deg",
                        float(np.min(mins)),
                    )
        except Exception:
            pass

    def _emit_selection_artifacts(
        self,
        *,
        picks_df: pd.DataFrame,
        art_dir: Path,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if "_row_idx" in picks_df.columns:
            picks_df_raw = picks_df.drop(columns=["_row_idx"])
        else:
            picks_df_raw = picks_df.copy()

        picks_df_out = picks_df_raw.copy()
        picks_df_out["sequence"] = [
            uppercase_mutated_codons(seq, aa_pos_list)
            for seq, aa_pos_list in zip(
                picks_df_out["sequence"].astype(str),
                picks_df_out["aa_pos_list"],
            )
        ]

        atomic_write_parquet(picks_df_out, art_dir / "MULTISITE_SELECT.parquet")
        picks_df_out.to_csv(art_dir / "MULTISITE_SELECT.csv", index=False)

        return picks_df_raw, picks_df_out

    def _append_record_select(
        self,
        *,
        art_dir: Path,
        df: pd.DataFrame,
        knobs: Knobs,
        chosen_cluster_ids: List,
    ) -> None:
        k_counts = df["permuter__mut_count"].value_counts().sort_index().to_dict()
        lines = [
            f"variants_total: {len(df):d}",
            f"k_counts: {json.dumps(k_counts)}",
            (
                "normalize: "
                f"method={knobs.normalize_method} "
                f"gaussian_consistent={knobs.gaussian_consistent} "
                f"winsor_mads={knobs.winsor_mads}"
            ),
            f"weights: llr={knobs.w_llr} epi={knobs.w_epi}",
            (
                "embedding: "
                f"col={knobs.embedding_col} "
                f"l2={knobs.l2_normalize_embeddings} "
                f"dist=angular repr=medoid"
            ),
            (
                "clusters: "
                f"unique_clusters={len(chosen_cluster_ids)} "
                f"picks_per_cluster={knobs.picks_per_cluster}"
            ),
            (
                "budget: total_variants="
                f"{knobs.total_variants} pool_factor={knobs.pool_factor:.2f}"
            ),
            (
                "diversity: "
                f"enabled={knobs.intracluster_diversity_enabled} "
                f"min_angular_distance_deg={knobs.min_angular_distance_deg:.2f}"
            ),
            ("tie_breakers: " f"prefer_fewer_mutations={knobs.prefer_fewer_mutations}"),
            "diagnostics: "
            f"figsize_in={knobs.diag_figsize_in} dpi={knobs.diag_dpi} "
            f"random_sample_seed={knobs.diag_random_seed} "
            f"random_sample_repeats={knobs.diag_random_repeats} "
            f"random_sample_factor={knobs.diag_random_sample_factor} "
            f"random_sample_cap={knobs.diag_random_sample_cap}",
            (
                "heb: "
                f"enabled={knobs.heb_enabled} "
                f"min_cooccur_count={knobs.heb_min_cooccur_count} "
                f"width_scale={knobs.heb_width_scale} "
                f"figsize_in={knobs.heb_figsize_in} "
                f"out_svg={knobs.heb_out_svg} "
                f"edge_cmap={knobs.heb_edge_cmap} "
                f"color_by={knobs.heb_color_by}"
            ),
            f"rng_seed: {knobs.rng_seed}",
        ]
        append_record_event(
            art_dir,
            "SELECT",
            lines=lines,
        )

    def _write_select_summary(
        self,
        *,
        art_dir: Path,
        picks: pd.DataFrame,
        clust_df: pd.DataFrame,
        chosen_cluster_ids: List,
        mut_window_summary: MutationWindowSummary | None = None,
    ) -> None:
        """
        Human-readable SELECT_SUMMARY.md capturing:

          • counts & basic score stats,
          • per-cluster pick summary,
          • chosen cluster statistics and medoid angles.
        """
        out = art_dir / "SELECT_SUMMARY.md"
        lines: List[str] = []
        lines.append("# Selection summary\n")
        lines.append(f"**Picks**: {len(picks)} variants\n")

        if mut_window_summary is not None:
            m = mut_window_summary
            lines.append("## Global mutation span\n")

            if m.ref_length_nt is not None and m.window_length_nt is not None:
                codons = m.window_length_nt // 3
                lines.append(
                    f"- reference length: {m.ref_length} aa " f"({m.ref_length_nt} nt)"
                )
                lines.append(
                    f"- window (AA): positions {m.start_pos}-{m.end_pos} "
                    f"(length {m.window_length} aa)"
                )
                lines.append(
                    f"- window (NT): positions {m.nt_start}-{m.nt_end} "
                    f"(length {m.window_length_nt} nt; {codons} codons)"
                )
            else:
                # Fallback: AA-only summary
                lines.append(f"- reference length: {m.ref_length} aa")
                lines.append(
                    f"- window: positions {m.start_pos}-{m.end_pos} "
                    f"(length {m.window_length} aa)"
                )

            lines.append("- amino-acid context:")
            lines.append(f"  - left flank : `{m.left_flank}`")
            lines.append(f"  - window     : `{m.window_seq}`")
            lines.append(f"  - right flank: `{m.right_flank}`")

            if m.window_seq_nt is not None:
                lines.append("- nucleotide context:")
                lines.append(f"  - left flank : `{m.left_flank_nt}`")
                lines.append(f"  - window     : `{m.window_seq_nt}`")
                lines.append(f"  - right flank: `{m.right_flank_nt}`")

            lines.append("")  # blank line

        if not picks.empty:
            grp = picks.groupby("cluster_id").agg(
                n=("sequence", "count"),
                best_score=("score", "max"),
                mean_score=("score", "mean"),
            )
            lines.append("## Per‑cluster picks\n")
            for cid, r in grp.sort_values("best_score", ascending=False).iterrows():
                lines.append(
                    f"- cluster {cid}: n={int(r['n'])} "
                    f"best={r['best_score']:+.3f} "
                    f"mean={r['mean_score']:+.3f}"
                )

        if not clust_df.empty:
            lines.append("\n## Cluster medoid diagnostics\n")
            sel = clust_df[clust_df["cluster_id"].isin(chosen_cluster_ids)].copy()
            if "min_inter_medoid_angle_deg" in sel.columns:
                sel = sel.sort_values("mean_composite", ascending=False)
                for _, r in sel.iterrows():
                    lines.append(
                        f"- cluster {r['cluster_id']}: "
                        f"size={int(r['size'])} "
                        f"mean_z_llr={r['mean_z_llr']:+.3f} "
                        f"mean_z_epi={r['mean_z_epi']:+.3f} "
                        f"pos_epi_fraction={r['pos_epi_fraction']:.2f} "
                        f"min_inter_medoid_angle_deg={r.get('min_inter_medoid_angle_deg', float('nan')):.1f}"
                    )

        if chosen_cluster_ids:
            lines.append("\n## Chosen clusters (order)\n")
            lines.append(", ".join(str(x) for x in chosen_cluster_ids))

        out.write_text("\n".join(lines), encoding="utf-8")
