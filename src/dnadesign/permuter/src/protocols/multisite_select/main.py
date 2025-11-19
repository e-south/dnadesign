"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/protocols/multisite_select/main.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from dnadesign.permuter.src.core.storage import (
    append_record_event,
    atomic_write_parquet,
)
from dnadesign.permuter.src.plots.diagnostics import (
    hist_mut_count,
    hist_pairwise_hamming,
)
from dnadesign.permuter.src.plots.window_score_mass import compute_mass, render_mass
from dnadesign.permuter.src.protocols.base import Protocol

from .geometry import (
    angular_distance,
    farthest_first_on_medoids,
    l2_normalize_rows,
    medoid_index,
    pairwise_angular,
)
from .scoring import detect_k_drift, robust_z, z_by_k
from .utils import (
    _as_int_list,
    extract_embedding_matrix,
    filter_valid_source_rows,
    pairwise_hamming_from_mutmaps,
    parse_aa_combo_to_map,
    read_source_records,
    ref_aa_from_dataset_dir,
)
from .windows import compute_windows_and_scores, greedy_cover

_LOG = logging.getLogger("permuter.protocol.multisite_select")


@dataclass
class Knobs:
    # scoring
    method: str
    gaussian_consistent: bool
    stratify_by_k: str  # "off"|"on"|"auto"
    winsor_mads: float | None
    w_llr: float
    w_epi: float
    # embedding
    embedding_col: str
    l2_normalize: bool
    # clusters
    K_per_cluster: int
    seed_by: str
    # cluster filters
    filters_stage: str
    min_cluster_mean_z_llr: float
    min_cluster_pos_epistasis_fraction: float
    location_stat: str
    trimmed_mean_frac: float
    # windows
    L_aa: int
    stride_aa: int
    num_windows: int
    min_variants_per_window: int
    # budget
    total_variants: int
    intra_enabled: bool
    intra_min_angle_deg: float
    # tie-breakers
    tb_fewer_mut: bool
    tb_higher_delta: bool
    tb_higher_proposal: bool
    # reproducibility
    rng_seed: int


class MSel(Protocol):
    """
    Implements the developer spec as a selection protocol driven by job.permute.params.select.
    It reads a *source dataset* (records.parquet with multi-mutants + metrics/embeddings),
    computes per-variant scores, performs window & cluster selection, then *emits the chosen
    sequences as variants* for this dataset (so `permuter run` writes a compact dataset of picks).

    All additional artifacts are written into params._artifact_dir.
    """

    # Validation
    def validate_cfg(self, *, params: Dict) -> None:
        if not isinstance(params, dict):
            raise ValueError("multisite_select: params must be a mapping")
        if not params.get("from_dataset"):
            raise ValueError(
                "multisite_select: params.from_dataset is required (dir or records.parquet)"
            )
        sel = params.get("select") or {}
        scoring = sel.get("scoring") or {}
        if (scoring.get("normalize", {}) or {}).get("method", "mad") not in {
            "mad",
            "none",
        }:
            raise ValueError(
                "multisite_select: select.scoring.normalize.method must be 'mad' or 'none'"
            )
        if (sel.get("embedding") or {}).get("distance", "angular") != "angular":
            raise ValueError(
                "multisite_select: select.embedding.distance=angular is required"
            )
        if (sel.get("embedding") or {}).get("representative", "medoid") != "medoid":
            raise ValueError(
                "multisite_select: only representative=medoid is implemented in v1"
            )
        if (sel.get("clusters") or {}).get(
            "selection", "farthest_first"
        ) != "farthest_first":
            raise ValueError(
                "multisite_select: clusters.selection must be 'farthest_first'"
            )
        # budget sanity
        budget = sel.get("budget") or {}
        tot = int((budget.get("total_variants") or 0))
        if tot <= 0:
            raise ValueError("multisite_select: budget.total_variants must be > 0")
        K = int(((sel.get("clusters") or {}).get("picks_per_cluster") or 1))
        if K <= 0:
            raise ValueError("multisite_select: clusters.picks_per_cluster must be ≥ 1")

        # internal paths from CLI plumbing
        if not params.get("_artifact_dir"):
            raise ValueError(
                "multisite_select: _artifact_dir missing (internal); invoke via permuter run"
            )

    # Generation
    def generate(
        self,
        *,
        ref_entry: Dict,
        params: Dict,
        rng: Optional[np.random.Generator] = None,
    ) -> Iterable[Dict]:
        """
        The selection itself happens here. We return final picks as emitted variants (sequence + metadata),
        and also write artifacts into params._artifact_dir.
        """
        # Resolve knobs
        knobs = self._knobs_from(params)
        art_dir = Path(str(params["_artifact_dir"])).expanduser().resolve()
        art_dir.mkdir(parents=True, exist_ok=True)

        # Load source dataset
        src = read_source_records(params["from_dataset"])
        src_dir = Path(str(params["from_dataset"])).expanduser().resolve()
        src_dir = src_dir if src_dir.is_dir() else src_dir.parent
        # required columns (strict, assertive)
        req = [
            "id",
            "sequence",
            "permuter__metric__llr_mean",
            "permuter__expected__llr_mean",
            knobs.embedding_col,  # list[512]
            "permuter__aa_pos_list",
            "permuter__mut_count",
            "cluster__perm_v1",
        ]
        miss = [c for c in req if c not in src.columns]
        if miss:
            raise ValueError(f"rt_select: source dataset missing columns: {miss}")

        # Filter valid rows with diagnostics (assertive, no silent relax)
        df, finfo = filter_valid_source_rows(
            src,
            emb_col=knobs.embedding_col,
            aa_col="permuter__aa_pos_list",
            llr_col="permuter__metric__llr_mean",
            exp_col="permuter__expected__llr_mean",
        )
        _LOG.info(
            "[source] loaded=%d kept=%d drops=%s",
            finfo["n_total"],
            finfo["n_kept"],
            json.dumps(finfo["drops_by_cause"]),
        )
        if df.empty:
            raise RuntimeError(
                "multisite_select: no usable rows after validation — "
                f"drops={json.dumps(finfo['drops_by_cause'])}. "
                "Check that embeddings are numeric 1D vectors and AA position lists are present."
            )

        # Epistasis consistency (recompute Δ; check if 'epistasis' present)
        df["__delta"] = df["permuter__metric__llr_mean"].astype(float) - df[
            "permuter__expected__llr_mean"
        ].astype(float)
        if "epistasis" in df.columns:
            diff = np.nanmax(
                np.abs(
                    df["epistasis"].astype(float).to_numpy() - df["__delta"].to_numpy()
                )
            )
            if np.isfinite(diff) and diff > 1e-9:
                _LOG.info(
                    "[check] epistasis column differs from observed-expected (max abs diff=%.3e); using recomputed Δ",
                    diff,
                )

        # Parse AA pos lists (1-indexed)
        df["__aa_list"] = df["permuter__aa_pos_list"].apply(_as_int_list)
        df = df[df["__aa_list"].apply(len) > 0].reset_index(drop=True)
        if df.empty:
            raise RuntimeError(
                "multisite_select: no rows with non-empty permuter__aa_pos_list"
            )

        # Embedding matrix (robust coercion; assert consistent dimensionality)
        emb = extract_embedding_matrix(df[knobs.embedding_col])
        if knobs.l2_normalize:
            emb = l2_normalize_rows(emb)

        # Robust z-scaling
        strat = str(knobs.stratify_by_k).lower()
        k_series = df["permuter__mut_count"].astype(int)
        if strat == "auto":
            enable_k_llr, info_llr = detect_k_drift(
                df["permuter__metric__llr_mean"].astype(float),
                k_series,
                gaussian_consistent=knobs.gaussian_consistent,
            )
            enable_k_delta, info_delta = detect_k_drift(
                df["__delta"].astype(float),
                k_series,
                gaussian_consistent=knobs.gaussian_consistent,
            )
            enable_k = bool(enable_k_llr or enable_k_delta)
            _LOG.info(
                "[scaling] stratify_by_k=auto → %s (llr=%s, delta=%s)",
                "ON" if enable_k else "OFF",
                json.dumps(info_llr),
                json.dumps(info_delta),
            )
        elif strat == "on":
            enable_k = True
        else:
            enable_k = False

        if knobs.method == "none":
            z_llr = df["permuter__metric__llr_mean"].astype(float).to_numpy()
            z_epi = df["__delta"].astype(float).to_numpy()
        else:
            if enable_k:
                z_llr = z_by_k(
                    df["permuter__metric__llr_mean"].astype(float),
                    k_series,
                    gaussian_consistent=knobs.gaussian_consistent,
                    winsor_mads=knobs.winsor_mads,
                )
                z_epi = z_by_k(
                    df["__delta"].astype(float),
                    k_series,
                    gaussian_consistent=knobs.gaussian_consistent,
                    winsor_mads=knobs.winsor_mads,
                )
            else:
                z_llr = robust_z(
                    df["permuter__metric__llr_mean"].astype(float),
                    gaussian_consistent=knobs.gaussian_consistent,
                    winsor_mads=knobs.winsor_mads,
                )
                z_epi = robust_z(
                    df["__delta"].astype(float),
                    gaussian_consistent=knobs.gaussian_consistent,
                    winsor_mads=knobs.winsor_mads,
                )
        df["__z_llr"] = z_llr
        df["__z_epi"] = z_epi
        df["__score"] = knobs.w_llr * z_llr + knobs.w_epi * z_epi
        df["__score_plus"] = np.maximum(0.0, df["__score"].to_numpy())

        # Cluster geometry
        # group by cluster label; compute medoids/stats
        groups = list(df.groupby("cluster__perm_v1", sort=False))
        if not groups:
            raise RuntimeError("multisite_select: no clusters in 'cluster__perm_v1'")

        cluster_meta: List[Dict] = []
        medoid_row_idx: List[int] = []

        for lab, sub in groups:
            idx = sub.index.to_numpy()
            U = emb[idx]
            mi = medoid_index(U)
            medoid_row_idx.append(int(idx[mi]))

            # stats
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
                lo = int(math.floor(f * m))
                hi = int(math.ceil((1 - f) * m))
                loc = float(a[lo:hi].mean() if hi > lo else a.mean())
            else:
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
                    "medoid_row": int(idx[mi]),
                }
            )

        clust_df = pd.DataFrame(cluster_meta)

        # cluster filters (pre or post)
        def _apply_cluster_filters(base_df: pd.DataFrame, label: str) -> pd.DataFrame:
            dfc = base_df.copy()
            before = len(dfc)
            if knobs.min_cluster_mean_z_llr is not None:
                dfc = dfc[dfc["loc_stat"] >= float(knobs.min_cluster_mean_z_llr)]
            if knobs.min_cluster_pos_epistasis_fraction is not None:
                dfc = dfc[
                    dfc["pos_epi_fraction"]
                    >= float(knobs.min_cluster_pos_epistasis_fraction)
                ]
            _LOG.info(
                "[clusters] %s filters: kept %d / %d clusters", label, len(dfc), before
            )
            return dfc

        # Initial window scan uses all variants, but cluster selection is later restricted to windows.
        # Window selection (global)
        # AA length (prefer REF_AA; otherwise infer)
        ref_name, ref_aa = ref_aa_from_dataset_dir(src_dir)
        if ref_aa:
            L_total = len(ref_aa)
            _LOG.info("[ref] REF_AA.fa found (len=%d aa)", L_total)
        else:
            L_total = int(max(max(x) for x in df["__aa_list"]))
            _LOG.info(
                "[ref] REF_AA.fa missing; inferred AA length from data → L=%d", L_total
            )

        windows_all, windows_df = compute_windows_and_scores(
            aa_pos_lists=df["__aa_list"].tolist(),
            score_plus=df["__score_plus"].to_numpy(),
            L_total=L_total,
            length_aa=knobs.L_aa,
            stride_aa=knobs.stride_aa,
        )

        # Assert window feasibility (no fallback changes; tell user what to change)
        has_feasible = any(
            w.covered_count >= int(knobs.min_variants_per_window) for w in windows_all
        )
        if not has_feasible:
            raise RuntimeError(
                "multisite_select: no window meets the feasibility constraint "
                f"(length_aa={knobs.L_aa}, min_variants_per_window={knobs.min_variants_per_window}). "
                "Increase `select.window.length_aa` and/or decrease "
                "`select.window.min_variants_per_window` in the job YAML."
            )

        selected_wins_1, gains_1 = greedy_cover(
            windows_all, df["__score_plus"].to_numpy(), W=knobs.num_windows
        )
        if not selected_wins_1:
            _LOG.info("[window] greedy selected 0 windows (no positive score⁺ mass)")

        # Explainability: fraction of Σ score⁺ explained by the selected windows
        tot_mass = float(df["__score_plus"].sum())
        exp1 = float(sum(gains_1)) if gains_1 else 0.0
        frac1 = (100.0 * exp1 / tot_mass) if tot_mass > 0 else 0.0
        n_nonpos = int((df["__score"] <= 0).sum())
        _LOG.info(
            "[window] global coverage: selected %d/%d windows; ΔF=%.3f (%.1f%% of Σ score⁺=%.3f across %d variants). "
            "Here Σ score⁺ is the sum of positive composite scores; positive-part guard suppresses %d variants (score ≤ 0).",  # noqa
            len(selected_wins_1),
            len(windows_all),
            exp1,
            frac1,
            tot_mass,
            len(df),
            n_nonpos,
        )
        # stepwise window details
        if selected_wins_1:
            for w, g in zip(selected_wins_1, gains_1):
                _LOG.info(
                    "[window] global step: window_id=%d [%d,%d] ΔF=%.3f covers=%d variants",
                    int(w.window_id),
                    int(w.start_aa),
                    int(w.end_aa),
                    g,
                    int(w.covered_count),
                )

        # Cluster selection (within selected windows)
        # Restrict variants to those covered by any selected window
        covered_idx = np.array(
            sorted({i for w in selected_wins_1 for i in w.covered_idx}), dtype=int
        )
        mask_win = np.zeros(len(df), dtype=bool)
        mask_win[covered_idx] = True
        df_win = df[mask_win].copy()
        if df_win.empty:
            _LOG.info("[warn] no variants fall inside the initial selected window(s)")

        # Recompute cluster representatives/stats on the window‑restricted subset
        def _compute_cluster_meta_for_subset(df_subset: pd.DataFrame):
            cluster_meta_w, medoid_rows_w = [], []
            for lab, sub in df_subset.groupby("cluster__perm_v1", sort=False):
                idx = sub.index.to_numpy()
                if idx.size == 0:
                    continue
                U = emb[idx]
                mi = medoid_index(U)
                medoid_rows_w.append(int(idx[mi]))
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
                    lo = int(math.floor(f * m))
                    hi = int(math.ceil((1 - f) * m))
                    loc = float(a[lo:hi].mean() if hi > lo else a.mean())
                else:
                    loc = float(zllr.mean())
                cluster_meta_w.append(
                    {
                        "cluster_id": lab,
                        "size": int(len(sub)),
                        "mean_z_llr": float(zllr.mean()),
                        "mean_z_epi": float(zepi.mean()),
                        "mean_composite": float(comp.mean()),
                        "pos_epi_fraction": pos_frac,
                        "loc_stat": loc,
                        "medoid_row": int(idx[mi]),
                    }
                )
            return pd.DataFrame(cluster_meta_w), medoid_rows_w

        clust_df_win, _ = _compute_cluster_meta_for_subset(df_win)
        if knobs.filters_stage == "pre_selection":
            cf = _apply_cluster_filters(clust_df_win, "pre")
        else:
            cf = clust_df_win.copy()
        cf = cf.reset_index(drop=True)

        # derive M from budget
        K = knobs.K_per_cluster
        M_target = int(max(1, knobs.total_variants // K))

        # Assert sufficient clusters pass the configured filters; no threshold relax.
        if len(cf) < M_target:
            raise RuntimeError(
                "multisite_select: cluster filters too strict for the budget — "
                f"need ≥ {M_target} clusters with window-compliant variants, "
                f"but only {len(cf)} pass (min_cluster_mean_z_llr={knobs.min_cluster_mean_z_llr}, "
                f"min_cluster_pos_epistasis_fraction={knobs.min_cluster_pos_epistasis_fraction}, "
                f"filters_stage={knobs.filters_stage}). "
                "Consider lowering the thresholds, setting filters_stage=post_selection, "
                "or increasing picks_per_cluster / adjusting budget.total_variants."
            )

        # Seed for k-center on window‑restricted medoids
        if cf.empty:
            chosen_cluster_ids: List = []
        else:
            _LOG.info(
                "[clusters] derived M_target=%d from budget (total_variants=%d, picks_per_cluster=%d)",
                M_target,
                knobs.total_variants,
                K,
            )
            if knobs.seed_by == "mean_composite":
                seed_row = cf.sort_values("mean_composite", ascending=False).iloc[0]
            elif knobs.seed_by == "mean_z_llr":
                seed_row = cf.sort_values("mean_z_llr", ascending=False).iloc[0]
            else:
                seed_row = cf.sort_values("mean_z_epi", ascending=False).iloc[0]
            # medoids array in cf order
            medoid_rows = cf["medoid_row"].astype(int).tolist()
            med_arr = emb[medoid_rows]
            med_arr = l2_normalize_rows(med_arr)  # safe

            # index of seed inside cf
            seed_idx = int(cf.index[cf["medoid_row"] == int(seed_row["medoid_row"])][0])
            order_idx = farthest_first_on_medoids(
                med_arr, M=M_target, seed_idx=seed_idx
            )
            chosen_cluster_ids = [cf.iloc[i]["cluster_id"] for i in order_idx]
            _LOG.info(
                "[clusters] k-center (seed_by=%s) seed_cluster=%s; order=%s",
                knobs.seed_by,
                str(seed_row["cluster_id"]),
                ", ".join(str(cf.iloc[i]["cluster_id"]) for i in order_idx),
            )

        # Apply post‑selection filters (if configured) and back‑fill
        if knobs.filters_stage == "post_selection" and chosen_cluster_ids:

            def _pass_filters(row) -> bool:
                return (row["loc_stat"] >= float(knobs.min_cluster_mean_z_llr)) and (
                    row["pos_epi_fraction"]
                    >= float(knobs.min_cluster_pos_epistasis_fraction)
                )

            cf_idx = cf.set_index("cluster_id")
            kept = [cid for cid in chosen_cluster_ids if _pass_filters(cf_idx.loc[cid])]
            if len(kept) < M_target:
                for i in order_idx:
                    cid = cf.iloc[i]["cluster_id"]
                    if cid in kept:
                        continue
                    if _pass_filters(cf.iloc[i]):
                        kept.append(cid)
                    if len(kept) >= M_target:
                        break
            chosen_cluster_ids = kept

        # Coupling swap: re-select windows restricted to chosen clusters
        if chosen_cluster_ids:
            mask_clusters = (
                df["cluster__perm_v1"].isin(set(chosen_cluster_ids)).to_numpy()
            )
            aa_lists_cc = df[mask_clusters]["__aa_list"].tolist()
            score_plus_cc = df[mask_clusters]["__score_plus"].to_numpy()
            wins_cc, wdf_cc = compute_windows_and_scores(
                aa_pos_lists=aa_lists_cc,
                score_plus=score_plus_cc,
                L_total=L_total,
                length_aa=knobs.L_aa,
                stride_aa=knobs.stride_aa,
            )
            selected_wins_2, gains_2 = greedy_cover(
                wins_cc, score_plus_cc, W=knobs.num_windows
            )
            # Map back selected windows to global coordinates (same bounds since identical generator)
            selected_windows = selected_wins_2 if selected_wins_2 else selected_wins_1
            exp2 = float(sum(gains_2)) if gains_2 else 0.0
            frac2 = (100.0 * exp2 / tot_mass) if tot_mass > 0 else 0.0
            _LOG.info(
                "[window] coupling coverage: re-select within chosen clusters; ΔF=%.3f (%.1f%% of Σ score⁺=%.3f).",
                exp2,
                frac2,
                tot_mass,
            )
            if selected_wins_2:
                for w, g in zip(selected_wins_2, gains_2):
                    _LOG.info(
                        "[window] coupling step: window_id=%d [%d,%d] ΔF=%.3f covers=%d variants",
                        int(w.window_id),
                        int(w.start_aa),
                        int(w.end_aa),
                        g,
                        int(w.covered_count),
                    )
        else:
            selected_windows = selected_wins_1

        # Final window ids (global)
        sel_win_ids = sorted({w.window_id for w in selected_windows})
        # Rebuild a concise windows summary with rank
        if windows_df.empty:
            windows_df_sel = pd.DataFrame(
                columns=[
                    "window_id",
                    "start_aa",
                    "end_aa",
                    "F_w",
                    "covered_count",
                    "rank",
                ]
            )
        else:
            tmp = windows_df[windows_df["window_id"].isin(sel_win_ids)].copy()
            # rank by F_w desc
            tmp = tmp.sort_values(
                ["F_w", "covered_count", "start_aa"], ascending=[False, False, True]
            )
            tmp["rank"] = np.arange(1, len(tmp) + 1)
            windows_df_sel = tmp

        # ---------------------------- Final picks ---------------------------- #
        # restrict to (windows) ∩ (clusters)
        # compute window membership mask fast via min/max position bounds
        pos_min = np.array([min(x) for x in df["__aa_list"]], dtype=int)
        pos_max = np.array([max(x) for x in df["__aa_list"]], dtype=int)
        in_any_window = np.zeros(len(df), dtype=bool)
        for w in selected_windows:
            in_any_window |= (pos_min >= int(w.start_aa)) & (pos_max <= int(w.end_aa))

        mask_final = in_any_window
        if chosen_cluster_ids:
            mask_final &= (
                df["cluster__perm_v1"].isin(set(chosen_cluster_ids)).to_numpy()
            )
        cand = df[mask_final].copy()

        # Tie-breaker sort key
        def _tb_key(row) -> tuple:
            key = (-float(row["__score"]),)
            if knobs.tb_fewer_mut:
                key += (int(row["permuter__mut_count"]),)  # ascending ⇒ fewer better
            if knobs.tb_higher_delta:
                key += (-float(row["__delta"]),)
            if knobs.tb_higher_proposal and "permuter__proposal_score" in row.index:
                key += (-float(row["permuter__proposal_score"]),)
            # final stable tiebreaker: var_id if present
            if "permuter__var_id" in row.index:
                key += (str(row["permuter__var_id"]),)
            return key

        # order within each cluster by tie-aware composite score
        picks: List[Dict] = []
        K = knobs.K_per_cluster
        min_intra_rad = (
            math.radians(knobs.intra_min_angle_deg) if knobs.intra_enabled else 0.0
        )

        for cid in chosen_cluster_ids:
            sub = cand[cand["cluster__perm_v1"] == cid].copy()
            if sub.empty:
                continue
            # Deterministic sort: score desc, fewer k, higher Δ, then proposal desc, then stable var_id asc
            sort_cols = ["__score", "permuter__mut_count", "__delta"]
            asc = [False, True, False]
            if "permuter__proposal_score" in sub.columns:
                sort_cols.append("permuter__proposal_score")
                asc.append(False)
            if "permuter__var_id" in sub.columns:
                sort_cols.append("permuter__var_id")
                asc.append(True)
            sub = sub.sort_values(by=sort_cols, ascending=asc, kind="mergesort")
            # pick first by score; if K>1, pick farthest inside cluster
            if sub.empty:
                continue
            first = sub.iloc[0]
            picks.append((cid, first))
            if K > 1 and len(sub) > 1:
                # farthest within cluster from 'first'
                base = emb[int(first.name)]
                best_j = None
                best_ang = -1.0
                for j, r in sub.iloc[1:].iterrows():
                    ang = angular_distance(base, emb[int(j)])
                    if ang > best_ang and (
                        not knobs.intra_enabled or ang >= min_intra_rad
                    ):
                        best_ang = ang
                        best_j = j
                if best_j is not None:
                    picks.append((cid, sub.loc[best_j]))

        # Fill remainder if needed
        need = int(knobs.total_variants - len(picks))
        if need > 0:
            # sweep remaining candidates (sorted)
            already = set(int(r.name) for _, r in picks)
            rest = cand.drop(index=list(already)).copy()
            sort_cols = ["__score", "permuter__mut_count", "__delta"]
            asc = [False, True, False]
            if "permuter__proposal_score" in rest.columns:
                sort_cols.append("permuter__proposal_score")
                asc.append(False)
            if "permuter__var_id" in rest.columns:
                sort_cols.append("permuter__var_id")
                asc.append(True)
            rest = rest.sort_values(by=sort_cols, ascending=asc, kind="mergesort")
            for _, r in rest.iterrows():
                # enforce intracluster diversity if enabled
                if knobs.intra_enabled:
                    cid = r["cluster__perm_v1"]
                    prior = [rr for cc, rr in picks if cc == cid]
                    ok = True
                    for rr in prior:
                        ang = angular_distance(emb[int(rr.name)], emb[int(r.name)])
                        if ang < min_intra_rad:
                            ok = False
                            break
                    if not ok:
                        continue
                picks.append((r["cluster__perm_v1"], r))
                if len(picks) >= knobs.total_variants:
                    break

        # Materialize selection frame
        # Map cluster → medoid row (window‑restricted) for angle reporting
        med_row_by_cid = {}
        if chosen_cluster_ids:
            med_row_by_cid = {
                r["cluster_id"]: int(r["medoid_row"])
                for _, r in cf[cf["cluster_id"].isin(chosen_cluster_ids)].iterrows()
            }
        sel_rows = []
        for cid, r in picks:
            # determine which selected window covers it (first by start asc)
            wid = None
            for w in sorted(selected_windows, key=lambda w: (w.start_aa, w.end_aa)):
                mn, mx = int(min(r["__aa_list"])), int(max(r["__aa_list"]))
                if mn >= w.start_aa and mx <= w.end_aa:
                    wid = w.window_id
                    break
            # angle to cluster medoid (deg), if available
            angle_deg = None
            try:
                med_row = med_row_by_cid.get(cid, None)
                if med_row is not None:
                    angle_deg = float(
                        np.degrees(
                            angular_distance(emb[int(r.name)], emb[int(med_row)])
                        )
                    )
            except Exception:
                angle_deg = None
            sel_rows.append(
                {
                    "_row_idx": int(r.name),  # internal; dropped before save
                    "source_id": r["id"],
                    "sequence": str(r["sequence"]),
                    "cluster_id": cid,
                    "window_id": wid,
                    "k": int(r["permuter__mut_count"]),
                    "metric__llr_mean": float(r["permuter__metric__llr_mean"]),
                    "expected__llr_mean": float(r["permuter__expected__llr_mean"]),
                    "delta": float(r["__delta"]),
                    "z_llr": float(r["__z_llr"]),
                    "z_epi": float(r["__z_epi"]),
                    "score": float(r["__score"]),
                    "angle_to_cluster_medoid_deg": angle_deg,
                    "aa_pos_list": list(map(int, r["__aa_list"])),
                    "aa_combo_str": str(r.get("permuter__aa_combo_str", "")),
                    "proposal_score": float(r.get("permuter__proposal_score", np.nan)),
                    "source_var_id": str(r.get("permuter__var_id", "")),
                }
            )
        picks_df = pd.DataFrame(sel_rows)

        # ---------------------------- Artifacts ---------------------------- #
        # WINDOWS_SUMMARY.parquet
        if not windows_df.empty:
            atomic_write_parquet(windows_df, art_dir / "WINDOWS_SUMMARY.parquet")
        if not windows_df_sel.empty:
            atomic_write_parquet(windows_df_sel, art_dir / "WINDOWS_SELECTED.parquet")
        # CLUSTER_SUMMARY.parquet
        # pairwise medoid angles for selected set (diagnostic)
        if chosen_cluster_ids:
            sel_c = clust_df[clust_df["cluster_id"].isin(chosen_cluster_ids)].copy()
            sel_c = sel_c.set_index("cluster_id").loc[chosen_cluster_ids].reset_index()
            med_arr = emb[sel_c["medoid_row"].astype(int).tolist()]
            med_arr = l2_normalize_rows(med_arr)
            A = pairwise_angular(med_arr)
            sel_c["min_inter_medoid_angle_deg"] = np.rad2deg(
                np.array(
                    [
                        np.min(np.delete(A[i], i)) if A.shape[0] > 1 else 0.0
                        for i in range(A.shape[0])
                    ]
                )
            )
            atomic_write_parquet(sel_c, art_dir / "CLUSTER_SUMMARY.parquet")

        # Attach required angular distances (nearest selected medoid)
        if chosen_cluster_ids and not picks_df.empty:
            sel_c = clust_df[clust_df["cluster_id"].isin(chosen_cluster_ids)].copy()
            sel_c = sel_c.set_index("cluster_id").loc[chosen_cluster_ids].reset_index()
            med_arr = l2_normalize_rows(emb[sel_c["medoid_row"].astype(int).tolist()])
            V = l2_normalize_rows(emb[picks_df["_row_idx"].astype(int).to_numpy()])
            ang_nearest = []
            for vec in V:
                dots = np.clip(med_arr @ vec, -1.0, 1.0)
                ang_nearest.append(float(np.rad2deg(np.arccos(np.max(dots)))))
            picks_df["angle_to_nearest_selected_medoid_deg"] = ang_nearest

        if "_row_idx" in picks_df.columns:
            picks_df = picks_df.drop(columns=["_row_idx"])

        # MULTISITE_VARIANTS.parquet (selection table)
        atomic_write_parquet(picks_df, art_dir / "MULTISITE_VARIANTS.parquet")
        picks_df.to_csv(art_dir / "MULTISITE_VARIANTS.csv", index=False)

        # AA_MASS + plot
        mass_df = compute_mass(
            L_total=L_total,
            aa_pos_lists=df["__aa_list"].tolist(),
            score_plus=df["__score_plus"].to_numpy(),
            normalize_by_k=False,
        )
        atomic_write_parquet(mass_df, art_dir / "AA_MASS.parquet")
        if not windows_df_sel.empty:
            title = "Window score⁺ density (AA axis)"
            if ref_aa:
                render_mass(
                    df_mass=mass_df,
                    windows=windows_df_sel[["start_aa", "end_aa", "rank"]],
                    out_png=str(art_dir / "fig_window_score_mass.png"),
                    title=title,
                    aa_letters=list(ref_aa),
                )
            else:
                _LOG.info(
                    "[plot] REF_AA.fa missing; skipping render_mass (requires aa_letters). "
                    "AA_MASS.parquet is still written."
                )

        # Diagnostics plots for selected variants
        if not picks_df.empty:
            try:
                # 1) mutation-count histogram
                hist_mut_count(
                    picks_df["k"].to_numpy(),
                    out_png=str(art_dir / "fig_mut_count_hist.png"),
                    title="Mutation-count distribution (selected variants)",
                )
                # 2) pairwise Hamming histogram (based on mutations)
                mutmaps = [
                    parse_aa_combo_to_map(s) for s in picks_df["aa_combo_str"].tolist()
                ]
                dists = pairwise_hamming_from_mutmaps(mutmaps)
                hist_pairwise_hamming(
                    dists,
                    out_png=str(art_dir / "fig_pairwise_hamming_hist.png"),
                    title="Pairwise Hamming distance (selected variants)",
                )
            except Exception as e:
                _LOG.warning("[plot] diagnostics histograms failed: %s", str(e))
            else:
                _LOG.warning(
                    "[plot] REF_AA.fa missing; skipping window score mass figure (requires aa_letters)."
                )

        # SELECT summary (compact)
        lines = [
            f"variants_total: {len(df):d}",
            f"k_counts: {json.dumps(df['permuter__mut_count'].value_counts().sort_index().to_dict())}",
            f"normalize: method={knobs.method} gaussian_consistent={knobs.gaussian_consistent} "
            f"stratify_by_k={knobs.stratify_by_k} winsor_mads={knobs.winsor_mads}",
            f"weights: llr={knobs.w_llr} epi={knobs.w_epi}",
            f"embedding: col={knobs.embedding_col} l2={knobs.l2_normalize} dist=angular repr=medoid",
            f"windows: length_aa={knobs.L_aa} stride={knobs.stride_aa} num_windows={knobs.num_windows} "
            f"selected={sel_win_ids}",
            f"clusters: target_M={max(1, knobs.total_variants // knobs.K_per_cluster)} "
            f"chosen_M={len(chosen_cluster_ids)} picks_per_cluster={knobs.K_per_cluster}",
            f"budget: total_variants={knobs.total_variants}",
            f"tie_breakers: fewer_mut={knobs.tb_fewer_mut} higher_delta={knobs.tb_higher_delta} "
            f"higher_proposal={knobs.tb_higher_proposal}",
            f"rng_seed: {knobs.rng_seed}",
        ]
        append_record_event(
            art_dir,
            "SELECT",
            lines=lines,
        )

        # SELECT_SUMMARY.md (human friendly)
        self._write_select_summary(
            art_dir=art_dir,
            picks=picks_df,
            windows_df=windows_df_sel,
            clust_df=clust_df,
            chosen_cluster_ids=chosen_cluster_ids,
        )

        # ---------------------------- Emit variants to build records.parquet ---------------------------- #
        # We emit only the final picks as this dataset's rows. Per run.py, keys will be namespaced as permuter__*
        for _, row in picks_df.iterrows():
            # Compact, transparent header modification
            head = (
                f"select score={row['score']:+.4f} z_llr={row['z_llr']:+.3f} z_epi={row['z_epi']:+.3f} "
                f"delta={row['delta']:+.4f} k={int(row['k'])} cluster={row['cluster_id']} window={row['window_id']}"
            )
            aa_token = f"aa [{row['aa_combo_str']}]" if row["aa_combo_str"] else "aa []"
            mods = [head, aa_token]
            yield {
                "sequence": str(row["sequence"]),
                "modifications": mods,
                # Flattened columns — run.py will prefix permuter__*
                "source_id": str(row["source_id"]),
                "metric__llr_mean": float(row["metric__llr_mean"]),
                "expected__llr_mean": float(row["expected__llr_mean"]),
                "delta": float(row["delta"]),
                "z_llr": float(row["z_llr"]),
                "z_epi": float(row["z_epi"]),
                "score": float(row["score"]),
                "aa_pos_list": list(row["aa_pos_list"]),
                "aa_combo_str": str(row["aa_combo_str"]),
                "mut_count": int(row["k"]),
                "cluster_id": row["cluster_id"],
                "window_id": (
                    int(row["window_id"]) if row["window_id"] is not None else -1
                ),
                "proposal_score": (
                    float(row["proposal_score"])
                    if np.isfinite(row["proposal_score"])
                    else None
                ),
                "source_var_id": str(row["source_var_id"]),
            }

        # Global min pairwise angle among picks (explainability)
        try:
            if not picks_df.empty:
                # rebuild from source indices using the original df index of selected rows
                sel_idx = df.index[df["id"].isin(picks_df["source_id"])].to_numpy()
                U_pick = l2_normalize_rows(emb[sel_idx])
                A = pairwise_angular(U_pick)
                if A.size and A.shape[0] > 1:
                    mins = [
                        float(np.min(np.delete(A[i], i))) for i in range(A.shape[0])
                    ]
                    if mins:
                        _LOG.info(
                            "[picks] global min pairwise angle among picks: %.2f deg",
                            float(np.rad2deg(np.min(mins))),
                        )
        except Exception:
            pass

    # ------------------------------ Helpers ------------------------------ #
    def _knobs_from(self, params: Dict) -> Knobs:
        sel = params.get("select") or {}
        scoring = sel.get("scoring") or {}
        norm = scoring.get("normalize") or {}
        weights = scoring.get("weights") or {}
        emb = sel.get("embedding") or {}
        cl = sel.get("clusters") or {}
        filt = cl.get("filters") or {}
        win = sel.get("window") or {}
        tie = sel.get("tie_breakers") or {}
        rep = sel.get("reproducibility") or {}

        gaussian = bool(norm.get("gaussian_consistent", False))
        winsor = norm.get("winsor_mads", None)
        winsor = float(winsor) if winsor is not None else None
        return Knobs(
            method=str(norm.get("method", "mad")).lower(),
            gaussian_consistent=gaussian,
            stratify_by_k=str(norm.get("stratify_by_k", "auto")).lower(),
            winsor_mads=winsor,
            w_llr=float(weights.get("llr", 1.0)),
            w_epi=float(weights.get("epi", 1.0)),
            embedding_col=str(emb.get("column", "permuter__metric__logits_mean")),
            l2_normalize=bool(emb.get("l2_normalize", True)),
            K_per_cluster=int(cl.get("picks_per_cluster", 1)),
            seed_by=str(cl.get("seed_by", "mean_composite")).lower(),
            filters_stage=str(
                (cl.get("filters") or {}).get("apply_stage", "pre_selection")
            ).lower(),
            min_cluster_mean_z_llr=float(filt.get("min_cluster_mean_z_llr", -0.25)),
            min_cluster_pos_epistasis_fraction=float(
                filt.get("min_cluster_pos_epistasis_fraction", 0.55)
            ),
            location_stat=str(filt.get("location_stat", "mean")).lower(),
            trimmed_mean_frac=float(filt.get("trimmed_mean_frac", 0.10)),
            L_aa=int(win.get("length_aa", 100)),
            stride_aa=int(win.get("stride_aa", 1)),
            num_windows=int(win.get("num_windows", 1)),
            min_variants_per_window=int(win.get("min_variants_per_window", 1)),
            total_variants=int((sel.get("budget") or {}).get("total_variants", 24)),
            intra_enabled=bool(
                ((sel.get("budget") or {}).get("intracluster_diversity") or {}).get(
                    "enabled", True
                )
            ),
            intra_min_angle_deg=float(
                ((sel.get("budget") or {}).get("intracluster_diversity") or {}).get(
                    "min_angular_distance_deg", 12.0
                )
            ),
            tb_fewer_mut=bool((tie.get("prefer_fewer_mutations", True))),
            tb_higher_delta=bool((tie.get("then_higher_delta", True))),
            tb_higher_proposal=bool((tie.get("then_higher_proposal_score", True))),
            rng_seed=int((rep.get("rng_seed", 20251113))),
        )

    def _write_select_summary(
        self,
        *,
        art_dir: Path,
        picks: pd.DataFrame,
        windows_df: pd.DataFrame,
        clust_df: pd.DataFrame,
        chosen_cluster_ids: List,
    ) -> None:
        out = art_dir / "SELECT_SUMMARY.md"
        lines: List[str] = []
        lines.append("# Selection summary\n")
        lines.append(f"**Picks**: {len(picks)} variants\n")
        if not picks.empty:
            # per cluster digest
            grp = picks.groupby("cluster_id").agg(
                n=("sequence", "count"),
                best_score=("score", "max"),
                mean_score=("score", "mean"),
            )
            lines.append("## Per‑cluster picks\n")
            for cid, r in grp.sort_values("best_score", ascending=False).iterrows():
                lines.append(
                    f"- cluster {cid}: n={int(r['n'])} best={r['best_score']:+.3f} mean={r['mean_score']:+.3f}"
                )
        if not windows_df.empty:
            lines.append("\n## Selected windows\n")
            for _, r in windows_df.sort_values("rank").iterrows():
                lines.append(
                    f"- rank {int(r['rank'])}: [{int(r['start_aa'])}, {int(r['end_aa'])}] "
                    f"F_w={float(r['F_w']):.3f} n={int(r['covered_count'])}"
                )
        if chosen_cluster_ids:
            lines.append("\n## Chosen clusters (order)\n")
            lines.append(", ".join(str(x) for x in chosen_cluster_ids))
        out.write_text("\n".join(lines), encoding="utf-8")
