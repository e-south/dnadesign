"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/core/sampler.py

Binding-site sampling utilities for DenseGen.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def compute_tf_weights(
    df: pd.DataFrame,
    *,
    usage_counts: dict[tuple[str, str], int] | None,
    coverage_boost_alpha: float,
    coverage_boost_power: float,
    failure_counts: dict[tuple[str, str], int] | None,
    avoid_failed_motifs: bool,
    failure_penalty_alpha: float,
    failure_penalty_power: float,
) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
    weight_by_tf: dict[str, float] = {}
    usage_count_by_tf: dict[str, int] = {}
    failure_count_by_tf: dict[str, int] = {}
    for row in df.itertuples(index=False):
        tf = str(row.tf)
        tfbs = str(row.tfbs)
        key = (tf, tfbs)
        count = int(usage_counts.get(key, 0)) if usage_counts else 0
        usage_count_by_tf[tf] = usage_count_by_tf.get(tf, 0) + count
        weight = 1.0 + float(coverage_boost_alpha) / ((1.0 + count) ** float(coverage_boost_power))
        if avoid_failed_motifs and failure_counts is not None:
            fails = int(failure_counts.get(key, 0))
            failure_count_by_tf[tf] = failure_count_by_tf.get(tf, 0) + fails
            if fails > 0:
                penalty = 1.0 + float(failure_penalty_alpha) * float(fails)
                weight = weight / (penalty ** float(failure_penalty_power))
        weight_by_tf[tf] = weight_by_tf.get(tf, 0.0) + float(weight)
    return weight_by_tf, usage_count_by_tf, failure_count_by_tf


@dataclass(frozen=True)
class _PreparedSamplingData:
    df: pd.DataFrame
    tf_to_indices: dict[str, list[int]]
    tfbs_to_indices: dict[str, list[int]]
    unique_tfs: list[str]
    total_unique_tfbs: int
    total_unique_cores: int
    has_site_id: bool
    has_source: bool
    has_tfbs_id: bool
    has_motif_id: bool


@dataclass
class _LibraryState:
    sites: list[str]
    meta: list[str]
    labels: list[str]
    reasons: list[str]
    site_ids: list[str | None]
    sources: list[str | None]
    tfbs_ids: list[str | None]
    motif_ids: list[str | None]
    stage_a_best_hit_scores: list[float | None]
    stage_a_ranks_within_regulator: list[int | None]
    stage_a_tiers: list[int | None]
    stage_a_fimo_starts: list[int | None]
    stage_a_fimo_stops: list[int | None]
    stage_a_fimo_strands: list[str | None]
    stage_a_selection_ranks: list[int | None]
    stage_a_selection_score_norms: list[float | None]
    stage_a_tfbs_cores: list[str | None]
    stage_a_score_theoretical_maxes: list[float | None]
    stage_a_selection_policies: list[str | None]
    stage_a_nearest_selected_similarities: list[float | None]
    stage_a_nearest_selected_distances: list[float | None]
    stage_a_nearest_selected_distance_norms: list[float | None]
    seen_tfbs: set[str]
    seen_cores: set[tuple[str, str]]
    used_per_tf: dict[str, int]


class TFSampler:
    """
    Sampler for binding-site tables (DataFrame with 'tf' and 'tfbs').
    """

    def __init__(self, df: pd.DataFrame, rng: np.random.Generator):
        if df is None or df.empty:
            raise ValueError("Input DataFrame for sampling is empty.")
        if "tf" not in df.columns or "tfbs" not in df.columns:
            raise ValueError("DataFrame must contain 'tf' and 'tfbs' columns.")
        self.df = df.copy()
        self.rng = rng
        self._prepared_cache: dict[tuple[bool, bool], _PreparedSamplingData] = {}

    def _prepare_sampling_data(
        self,
        *,
        unique_binding_sites: bool,
        unique_binding_cores: bool,
    ) -> _PreparedSamplingData:
        cache_key = (unique_binding_sites, unique_binding_cores)
        cached = self._prepared_cache.get(cache_key)
        if cached is not None:
            return cached

        df = self.df
        if unique_binding_sites:
            df = df.drop_duplicates(["tfbs"]).reset_index(drop=True)
        if unique_binding_cores and "tfbs_core" not in df.columns:
            raise ValueError(
                "unique_binding_cores=true requires a 'tfbs_core' column in the input data. "
                "Provide tfbs_core in the pool or disable unique_binding_cores."
            )

        tf_to_indices: dict[str, list[int]] = {}
        tfbs_to_indices: dict[str, list[int]] = {}
        index_values = df.index.tolist()
        tf_values = df["tf"].tolist()
        tfbs_values = df["tfbs"].tolist()
        for idx, (tf, tfbs) in zip(index_values, zip(tf_values, tfbs_values)):
            tf = str(tf)
            tfbs = str(tfbs)
            tf_to_indices.setdefault(tf, []).append(idx)
            tfbs_to_indices.setdefault(tfbs, []).append(idx)

        unique_tfs = sorted(tf_to_indices)
        if not unique_tfs:
            raise ValueError("No regulators found in input.")

        total_unique_tfbs = len(df.drop_duplicates(["tfbs"]))
        total_unique_cores = len(df.drop_duplicates(["tf", "tfbs_core"])) if unique_binding_cores else total_unique_tfbs
        prepared = _PreparedSamplingData(
            df=df,
            tf_to_indices=tf_to_indices,
            tfbs_to_indices=tfbs_to_indices,
            unique_tfs=unique_tfs,
            total_unique_tfbs=total_unique_tfbs,
            total_unique_cores=total_unique_cores,
            has_site_id="site_id" in df.columns,
            has_source="source" in df.columns,
            has_tfbs_id="tfbs_id" in df.columns,
            has_motif_id="motif_id" in df.columns,
        )
        self._prepared_cache[cache_key] = prepared
        return prepared

    @staticmethod
    def _normalize_required_tfbs(required_tfbs: list[str] | None, *, available_tfbs: set[str]) -> list[str]:
        required = [str(s).strip().upper() for s in (required_tfbs or []) if str(s).strip()]
        if len(set(required)) != len(required):
            raise ValueError("required_tfbs must be unique")
        if required:
            missing = [m for m in required if m not in available_tfbs]
            if missing:
                preview = ", ".join(missing[:10])
                raise ValueError(f"Required TFBS motifs not found in input: {preview}")
        return required

    @staticmethod
    def _normalize_required_tfs(required_tfs: list[str] | None, *, available_tfs: set[str]) -> list[str]:
        required_tf_list = [str(t).strip() for t in (required_tfs or []) if str(t).strip()]
        if len(set(required_tf_list)) != len(required_tf_list):
            raise ValueError("required_tfs must be unique")
        if required_tf_list:
            missing_tfs = [t for t in required_tf_list if t not in available_tfs]
            if missing_tfs:
                preview = ", ".join(missing_tfs[:10])
                raise ValueError(f"Required regulators not found in input: {preview}")
        return required_tf_list

    @staticmethod
    def _build_sampling_info(
        *,
        state: _LibraryState,
        has_site_id: bool,
        has_source: bool,
        has_tfbs_id: bool,
        has_motif_id: bool,
        has_stage_a_best_hit_score: bool,
        has_stage_a_rank_within_regulator: bool,
        has_stage_a_tier: bool,
        has_stage_a_fimo_start: bool,
        has_stage_a_fimo_stop: bool,
        has_stage_a_fimo_strand: bool,
        has_stage_a_selection_rank: bool,
        has_stage_a_selection_score_norm: bool,
        has_stage_a_tfbs_core: bool,
        has_stage_a_score_theoretical_max: bool,
        has_stage_a_selection_policy: bool,
        has_stage_a_nearest_selected_similarity: bool,
        has_stage_a_nearest_selected_distance: bool,
        has_stage_a_nearest_selected_distance_norm: bool,
        weight_by_tf: dict[str, float] | None,
        weight_fraction_by_tf: dict[str, float] | None,
        usage_count_by_tf: dict[str, int] | None,
        failure_count_by_tf: dict[str, int] | None,
        relaxed_cap: bool,
        final_cap: int | None,
    ) -> dict:
        return {
            "achieved_length": int(sum(len(s) for s in state.sites)),
            "relaxed_cap": bool(relaxed_cap),
            "final_cap": final_cap,
            "site_id_by_index": state.site_ids if has_site_id else None,
            "source_by_index": state.sources if has_source else None,
            "tfbs_id_by_index": state.tfbs_ids if has_tfbs_id else None,
            "motif_id_by_index": state.motif_ids if has_motif_id else None,
            "stage_a_best_hit_score_by_index": (state.stage_a_best_hit_scores if has_stage_a_best_hit_score else None),
            "stage_a_rank_within_regulator_by_index": (
                state.stage_a_ranks_within_regulator if has_stage_a_rank_within_regulator else None
            ),
            "stage_a_tier_by_index": state.stage_a_tiers if has_stage_a_tier else None,
            "stage_a_fimo_start_by_index": state.stage_a_fimo_starts if has_stage_a_fimo_start else None,
            "stage_a_fimo_stop_by_index": state.stage_a_fimo_stops if has_stage_a_fimo_stop else None,
            "stage_a_fimo_strand_by_index": state.stage_a_fimo_strands if has_stage_a_fimo_strand else None,
            "stage_a_selection_rank_by_index": (state.stage_a_selection_ranks if has_stage_a_selection_rank else None),
            "stage_a_selection_score_norm_by_index": (
                state.stage_a_selection_score_norms if has_stage_a_selection_score_norm else None
            ),
            "stage_a_tfbs_core_by_index": state.stage_a_tfbs_cores if has_stage_a_tfbs_core else None,
            "stage_a_score_theoretical_max_by_index": (
                state.stage_a_score_theoretical_maxes if has_stage_a_score_theoretical_max else None
            ),
            "stage_a_selection_policy_by_index": (
                state.stage_a_selection_policies if has_stage_a_selection_policy else None
            ),
            "stage_a_nearest_selected_similarity_by_index": (
                state.stage_a_nearest_selected_similarities if has_stage_a_nearest_selected_similarity else None
            ),
            "stage_a_nearest_selected_distance_by_index": (
                state.stage_a_nearest_selected_distances if has_stage_a_nearest_selected_distance else None
            ),
            "stage_a_nearest_selected_distance_norm_by_index": (
                state.stage_a_nearest_selected_distance_norms if has_stage_a_nearest_selected_distance_norm else None
            ),
            "selection_reason_by_index": state.reasons,
            "sampling_weight_by_tf": weight_by_tf,
            "sampling_weight_fraction_by_tf": weight_fraction_by_tf,
            "sampling_usage_count_by_tf": usage_count_by_tf,
            "sampling_failure_count_by_tf": failure_count_by_tf,
        }

    def sample_unique_tfs(self, required_tf_count: int, allow_replacement: bool = False) -> list:
        if required_tf_count <= 0:
            raise ValueError("required_tf_count must be > 0")
        unique_tfs = self.df["tf"].unique()
        if required_tf_count > len(unique_tfs):
            if not allow_replacement:
                raise ValueError(f"Requested {required_tf_count} unique TFs, but only {len(unique_tfs)} available.")
            allow_replacement = True
        sampled_tfs = self.rng.choice(unique_tfs, size=required_tf_count, replace=allow_replacement)
        return sampled_tfs.tolist()

    def subsample_binding_sites(self, sample_size: int, unique_tf_only: bool = False) -> list:
        if unique_tf_only:
            prepared = self._prepare_sampling_data(unique_binding_sites=False, unique_binding_cores=False)
            sampled_tfs = self.sample_unique_tfs(sample_size, allow_replacement=False)
            binding_sites = []
            for tf in sampled_tfs:
                indices = list(prepared.tf_to_indices.get(tf, []))
                if not indices:
                    raise ValueError(f"No binding sites found for TF '{tf}'.")
                idx = self.rng.choice(indices)
                row = prepared.df.loc[idx]
                binding_sites.append((row["tf"], row["tfbs"], "binding_sites"))
            return binding_sites
        else:
            grouped = self.df.groupby("tf")
            samples = []
            for _, group in grouped:
                n = min(sample_size, len(group))
                samples.append(group.sample(n=n, random_state=int(self.rng.integers(1_000_000))))
            df_sampled = pd.concat(samples, ignore_index=True)
            return list(
                zip(
                    df_sampled["tf"].tolist(),
                    df_sampled["tfbs"].tolist(),
                    ["binding_sites"] * len(df_sampled),
                )
            )

    def generate_binding_site_library(
        self,
        library_size: int,
        *,
        required_tfbs: list[str] | None = None,
        required_tfs: list[str] | None = None,
        cover_all_tfs: bool = False,
        unique_binding_sites: bool = True,
        unique_binding_cores: bool = True,
        max_sites_per_tf: int | None = None,
        relax_on_exhaustion: bool = True,
        sampling_strategy: str = "tf_balanced",
        usage_counts: dict[tuple[str, str], int] | None = None,
        coverage_boost_alpha: float = 0.15,
        coverage_boost_power: float = 1.0,
        failure_counts: dict[tuple[str, str], int] | None = None,
        avoid_failed_motifs: bool = False,
        failure_penalty_alpha: float = 0.5,
        failure_penalty_power: float = 1.0,
    ) -> tuple[list, list, list, dict]:
        """
        Build a motif library with a fixed number of motifs (library_size).

        Alignment notes:
        - (1) library_size is the count of motifs offered to the solver.
        - (4) sampling_strategy selects TF-balanced vs uniform-over-pairs behavior.
        - (2.3) cap relaxation only applies when max_sites_per_tf is set.
        """
        if library_size <= 0:
            raise ValueError("library_size must be > 0")

        prepared = self._prepare_sampling_data(
            unique_binding_sites=unique_binding_sites,
            unique_binding_cores=unique_binding_cores,
        )
        df = prepared.df
        has_site_id = prepared.has_site_id
        has_source = prepared.has_source
        has_tfbs_id = prepared.has_tfbs_id
        has_motif_id = prepared.has_motif_id
        has_stage_a_best_hit_score = "best_hit_score" in df.columns
        has_stage_a_rank_within_regulator = "rank_within_regulator" in df.columns
        has_stage_a_tier = "tier" in df.columns
        has_stage_a_fimo_start = "fimo_start" in df.columns
        has_stage_a_fimo_stop = "fimo_stop" in df.columns
        has_stage_a_fimo_strand = "fimo_strand" in df.columns
        has_stage_a_selection_rank = "selection_rank" in df.columns
        has_stage_a_selection_score_norm = "selection_score_norm" in df.columns
        has_stage_a_tfbs_core = "tfbs_core" in df.columns
        has_stage_a_score_theoretical_max = "score_theoretical_max" in df.columns
        has_stage_a_selection_policy = "selection_policy" in df.columns
        has_stage_a_nearest_selected_similarity = "nearest_selected_similarity" in df.columns
        has_stage_a_nearest_selected_distance = "nearest_selected_distance" in df.columns
        has_stage_a_nearest_selected_distance_norm = "nearest_selected_distance_norm" in df.columns
        total_unique_tfbs = prepared.total_unique_tfbs
        total_unique_cores = prepared.total_unique_cores
        unique_tfs = prepared.unique_tfs

        weight_by_tf: dict[str, float] | None = None
        weight_fraction_by_tf: dict[str, float] | None = None
        usage_count_by_tf: dict[str, int] | None = None
        failure_count_by_tf: dict[str, int] | None = None
        if sampling_strategy == "coverage_weighted":
            weight_by_tf, usage_count_by_tf, failure_count_by_tf = compute_tf_weights(
                df,
                usage_counts=usage_counts,
                coverage_boost_alpha=coverage_boost_alpha,
                coverage_boost_power=coverage_boost_power,
                failure_counts=failure_counts,
                avoid_failed_motifs=avoid_failed_motifs,
                failure_penalty_alpha=failure_penalty_alpha,
                failure_penalty_power=failure_penalty_power,
            )
            total_weight = sum(weight_by_tf.values()) if weight_by_tf else 0.0
            weight_fraction_by_tf = {
                tf: (float(val) / total_weight if total_weight > 0 else 0.0) for tf, val in (weight_by_tf or {}).items()
            }

        required = self._normalize_required_tfbs(required_tfbs, available_tfbs=set(df["tfbs"].tolist()))
        required_tf_list = self._normalize_required_tfs(required_tfs, available_tfs=set(unique_tfs))

        if len(required) > library_size:
            raise ValueError(f"library_size={library_size} is smaller than required_tfbs ({len(required)}).")

        state = _LibraryState(
            sites=[],
            meta=[],
            labels=[],
            reasons=[],
            site_ids=[],
            sources=[],
            tfbs_ids=[],
            motif_ids=[],
            stage_a_best_hit_scores=[],
            stage_a_ranks_within_regulator=[],
            stage_a_tiers=[],
            stage_a_fimo_starts=[],
            stage_a_fimo_stops=[],
            stage_a_fimo_strands=[],
            stage_a_selection_ranks=[],
            stage_a_selection_score_norms=[],
            stage_a_tfbs_cores=[],
            stage_a_score_theoretical_maxes=[],
            stage_a_selection_policies=[],
            stage_a_nearest_selected_similarities=[],
            stage_a_nearest_selected_distances=[],
            stage_a_nearest_selected_distance_norms=[],
            seen_tfbs=set(),
            seen_cores=set(),
            used_per_tf={},
        )

        def _clean_optional(val):
            if val is None:
                return None
            try:
                if pd.isna(val):
                    return None
            except Exception:
                pass
            return val

        def _append_row(row, reason: str) -> bool:
            tf = str(row["tf"])
            tfbs = str(row["tfbs"])
            tfbs_key = tfbs
            if unique_binding_sites and tfbs_key in state.seen_tfbs:
                return False
            if unique_binding_cores:
                core = str(row["tfbs_core"])
                core_key = (tf, core)
                if core_key in state.seen_cores:
                    return False
            state.sites.append(tfbs)
            state.meta.append(f"{tf}:{tfbs}")
            state.labels.append(tf)
            state.reasons.append(reason)
            state.seen_tfbs.add(tfbs_key)
            if unique_binding_cores:
                state.seen_cores.add((tf, str(row["tfbs_core"])))
            state.used_per_tf[tf] = state.used_per_tf.get(tf, 0) + 1
            state.site_ids.append(str(row["site_id"]) if has_site_id else None)
            state.sources.append(str(row["source"]) if has_source else None)
            state.tfbs_ids.append(str(row["tfbs_id"]) if has_tfbs_id else None)
            state.motif_ids.append(str(row["motif_id"]) if has_motif_id else None)
            stage_a_best_hit_score = _clean_optional(row["best_hit_score"]) if has_stage_a_best_hit_score else None
            state.stage_a_best_hit_scores.append(
                float(stage_a_best_hit_score) if stage_a_best_hit_score is not None else None
            )
            stage_a_rank = _clean_optional(row["rank_within_regulator"]) if has_stage_a_rank_within_regulator else None
            state.stage_a_ranks_within_regulator.append(int(stage_a_rank) if stage_a_rank is not None else None)
            stage_a_tier = _clean_optional(row["tier"]) if has_stage_a_tier else None
            state.stage_a_tiers.append(int(stage_a_tier) if stage_a_tier is not None else None)
            stage_a_fimo_start = _clean_optional(row["fimo_start"]) if has_stage_a_fimo_start else None
            state.stage_a_fimo_starts.append(int(stage_a_fimo_start) if stage_a_fimo_start is not None else None)
            stage_a_fimo_stop = _clean_optional(row["fimo_stop"]) if has_stage_a_fimo_stop else None
            state.stage_a_fimo_stops.append(int(stage_a_fimo_stop) if stage_a_fimo_stop is not None else None)
            stage_a_fimo_strand = _clean_optional(row["fimo_strand"]) if has_stage_a_fimo_strand else None
            state.stage_a_fimo_strands.append(str(stage_a_fimo_strand) if stage_a_fimo_strand is not None else None)
            stage_a_selection_rank = _clean_optional(row["selection_rank"]) if has_stage_a_selection_rank else None
            state.stage_a_selection_ranks.append(
                int(stage_a_selection_rank) if stage_a_selection_rank is not None else None
            )
            stage_a_selection_score_norm = (
                _clean_optional(row["selection_score_norm"]) if has_stage_a_selection_score_norm else None
            )
            state.stage_a_selection_score_norms.append(
                float(stage_a_selection_score_norm) if stage_a_selection_score_norm is not None else None
            )
            stage_a_tfbs_core = _clean_optional(row["tfbs_core"]) if has_stage_a_tfbs_core else None
            state.stage_a_tfbs_cores.append(str(stage_a_tfbs_core) if stage_a_tfbs_core is not None else None)
            stage_a_score_theoretical_max = (
                _clean_optional(row["score_theoretical_max"]) if has_stage_a_score_theoretical_max else None
            )
            state.stage_a_score_theoretical_maxes.append(
                float(stage_a_score_theoretical_max) if stage_a_score_theoretical_max is not None else None
            )
            stage_a_selection_policy = (
                _clean_optional(row["selection_policy"]) if has_stage_a_selection_policy else None
            )
            state.stage_a_selection_policies.append(
                str(stage_a_selection_policy) if stage_a_selection_policy is not None else None
            )
            stage_a_nearest_selected_similarity = (
                _clean_optional(row["nearest_selected_similarity"]) if has_stage_a_nearest_selected_similarity else None
            )
            state.stage_a_nearest_selected_similarities.append(
                float(stage_a_nearest_selected_similarity) if stage_a_nearest_selected_similarity is not None else None
            )
            stage_a_nearest_selected_distance = (
                _clean_optional(row["nearest_selected_distance"]) if has_stage_a_nearest_selected_distance else None
            )
            state.stage_a_nearest_selected_distances.append(
                float(stage_a_nearest_selected_distance) if stage_a_nearest_selected_distance is not None else None
            )
            stage_a_nearest_selected_distance_norm = (
                _clean_optional(row["nearest_selected_distance_norm"])
                if has_stage_a_nearest_selected_distance_norm
                else None
            )
            state.stage_a_nearest_selected_distance_norms.append(
                float(stage_a_nearest_selected_distance_norm)
                if stage_a_nearest_selected_distance_norm is not None
                else None
            )
            return True

        def _pick_for_tf(tf: str, *, reason: str, cap_override: int | None = None) -> bool:
            indices = list(prepared.tf_to_indices.get(tf, []))
            if not indices:
                return False
            if cap_override is not None and state.used_per_tf.get(tf, 0) >= cap_override:
                return False

            if sampling_strategy == "coverage_weighted":
                candidates: list[int] = []
                weights: list[float] = []
                for idx in indices:
                    row = df.loc[idx]
                    tfbs_key = str(row["tfbs"])
                    tf_tfbs_key = (str(row["tf"]), tfbs_key)
                    if unique_binding_sites and tfbs_key in state.seen_tfbs:
                        continue
                    if unique_binding_cores:
                        core_key = (str(row["tf"]), str(row["tfbs_core"]))
                        if core_key in state.seen_cores:
                            continue
                    count = int(usage_counts.get(tf_tfbs_key, 0)) if usage_counts else 0
                    weight = 1.0 + float(coverage_boost_alpha) / ((1.0 + count) ** float(coverage_boost_power))
                    if avoid_failed_motifs and failure_counts is not None:
                        fails = int(failure_counts.get(tf_tfbs_key, 0))
                        if fails > 0:
                            penalty = 1.0 + float(failure_penalty_alpha) * float(fails)
                            weight = weight / (penalty ** float(failure_penalty_power))
                    candidates.append(idx)
                    weights.append(weight)
                if not candidates:
                    return False
                weights = np.asarray(weights, dtype=float)
                weights = weights / weights.sum()
                # attempt up to a few draws to satisfy uniqueness
                for _ in range(min(20, len(candidates))):
                    choice = int(self.rng.choice(candidates, p=weights))
                    row = df.loc[choice]
                    if _append_row(row, reason):
                        return True
                    if unique_binding_sites:
                        # drop used candidate and renormalize
                        idx_pos = candidates.index(choice)
                        candidates.pop(idx_pos)
                        weights = np.delete(weights, idx_pos)
                        if not candidates:
                            return False
                        weights = weights / weights.sum()
                return False

            # tf_balanced / uniform strategies use random order
            self.rng.shuffle(indices)
            for idx in indices:
                row = df.loc[idx]
                tfbs_key = str(row["tfbs"])
                if unique_binding_sites and tfbs_key in state.seen_tfbs:
                    continue
                if unique_binding_cores:
                    core_key = (str(row["tf"]), str(row["tfbs_core"]))
                    if core_key in state.seen_cores:
                        continue
                if _append_row(row, reason):
                    return True
            return False

        # Required TFBS first (side-bias motifs included here)
        for motif in required:
            rows = df[df["tfbs"] == motif]
            if rows.empty:
                raise ValueError(f"Required TFBS motif not found in input: {motif}")
            row = rows.sort_values(["tf", "tfbs"]).iloc[0]
            if not _append_row(row, "required_tfbs"):
                raise ValueError(
                    "Required TFBS motifs conflict with uniqueness constraints. "
                    "Disable unique_binding_cores or remove duplicate cores from required_tfbs."
                )

        # Required regulators (at least one per TF)
        for tf in required_tf_list:
            if state.used_per_tf.get(tf, 0) > 0:
                continue
            if not _pick_for_tf(tf, reason="required_tf", cap_override=None):
                raise ValueError(f"Failed to select a motif for required regulator '{tf}'")

        # Coverage pass
        if cover_all_tfs:
            remaining_slots = library_size - len(state.sites)
            missing_tfs = [tf for tf in unique_tfs if state.used_per_tf.get(tf, 0) == 0]
            if remaining_slots < len(missing_tfs):
                raise ValueError(
                    "cover_all_regulators requires at least one site per TF, "
                    f"but library_size={library_size} is too small for {len(unique_tfs)} TFs. "
                    "Increase library_size or disable cover_all_regulators."
                )
            if remaining_slots <= 0:
                if missing_tfs:
                    raise ValueError(
                        "Required coverage cannot be satisfied given required_tfbs/required_tfs. "
                        "Increase library_size or relax coverage constraints."
                    )
            tf_order = unique_tfs.copy()
            self.rng.shuffle(tf_order)
            for tf in tf_order:
                if len(state.sites) >= library_size:
                    break
                if state.used_per_tf.get(tf, 0) > 0:
                    continue
                if _pick_for_tf(tf, reason="coverage", cap_override=None):
                    continue
                raise ValueError(f"Failed to select a motif for TF '{tf}' while cover_all_regulators=true.")

        cap = max_sites_per_tf
        relaxed_cap = False

        def _fill_tf_balanced() -> None:
            nonlocal cap, relaxed_cap
            tf_order = unique_tfs.copy()
            self.rng.shuffle(tf_order)
            while len(state.sites) < library_size:
                progressed = False
                for tf in tf_order:
                    if len(state.sites) >= library_size:
                        break
                    if cap is not None and state.used_per_tf.get(tf, 0) >= cap:
                        continue
                    if _pick_for_tf(tf, reason="filler", cap_override=cap):
                        progressed = True
                if progressed:
                    continue
                if unique_binding_sites and len(state.seen_tfbs) >= total_unique_tfbs:
                    raise ValueError(
                        "Unique TFBS pool exhausted before reaching library_size. "
                        "Reduce library_size or allow duplicates."
                    )
                if unique_binding_cores and len(state.seen_cores) >= total_unique_cores:
                    raise ValueError(
                        "Unique TFBS core pool exhausted before reaching library_size. "
                        "Reduce library_size, disable unique_binding_cores, or add more input sites."
                    )
                if relax_on_exhaustion and cap is not None:
                    cap += 1
                    relaxed_cap = True
                    continue
                if relax_on_exhaustion and cap is None:
                    raise ValueError(
                        "Sampling stalled with max_sites_per_regulator unset. "
                        "Reduce library_size, disable unique_binding_sites, or add more input sites."
                    )
                raise ValueError(
                    "Could not build library_size with current sampling constraints. "
                    "Enable relax_on_exhaustion or loosen caps."
                )

        def _fill_uniform_over_pairs() -> None:
            nonlocal cap, relaxed_cap
            indices = df.index.to_list()
            if not indices:
                raise ValueError("No TFBS entries available for sampling.")
            while len(state.sites) < library_size:
                self.rng.shuffle(indices)
                progressed = False
                for idx in indices:
                    if len(state.sites) >= library_size:
                        break
                    row = df.loc[idx]
                    tf = str(row["tf"])
                    if cap is not None and state.used_per_tf.get(tf, 0) >= cap:
                        continue
                    if _append_row(row, "filler"):
                        progressed = True
                if progressed:
                    continue
                if unique_binding_sites and len(state.seen_tfbs) >= total_unique_tfbs:
                    raise ValueError(
                        "Unique TFBS pool exhausted before reaching library_size. "
                        "Reduce library_size or allow duplicates."
                    )
                if unique_binding_cores and len(state.seen_cores) >= total_unique_cores:
                    raise ValueError(
                        "Unique TFBS core pool exhausted before reaching library_size. "
                        "Reduce library_size, disable unique_binding_cores, or add more input sites."
                    )
                if relax_on_exhaustion and cap is not None:
                    cap += 1
                    relaxed_cap = True
                    continue
                if relax_on_exhaustion and cap is None:
                    raise ValueError(
                        "Sampling stalled with max_sites_per_regulator unset. "
                        "Reduce library_size, disable unique_binding_sites, or add more input sites."
                    )
                raise ValueError(
                    "Could not build library_size with current sampling constraints. "
                    "Enable relax_on_exhaustion or loosen caps."
                )

        if len(state.sites) > library_size:
            raise ValueError(
                "Required motifs exceed library_size. Increase library_size or relax required constraints."
            )
        if len(state.sites) < library_size:
            if sampling_strategy in {"tf_balanced", "coverage_weighted"}:
                _fill_tf_balanced()
            elif sampling_strategy == "uniform_over_pairs":
                _fill_uniform_over_pairs()
            else:
                raise ValueError(f"Unknown library sampling strategy: {sampling_strategy}")

        info = self._build_sampling_info(
            state=state,
            has_site_id=has_site_id,
            has_source=has_source,
            has_tfbs_id=has_tfbs_id,
            has_motif_id=has_motif_id,
            has_stage_a_best_hit_score=has_stage_a_best_hit_score,
            has_stage_a_rank_within_regulator=has_stage_a_rank_within_regulator,
            has_stage_a_tier=has_stage_a_tier,
            has_stage_a_fimo_start=has_stage_a_fimo_start,
            has_stage_a_fimo_stop=has_stage_a_fimo_stop,
            has_stage_a_fimo_strand=has_stage_a_fimo_strand,
            has_stage_a_selection_rank=has_stage_a_selection_rank,
            has_stage_a_selection_score_norm=has_stage_a_selection_score_norm,
            has_stage_a_tfbs_core=has_stage_a_tfbs_core,
            has_stage_a_score_theoretical_max=has_stage_a_score_theoretical_max,
            has_stage_a_selection_policy=has_stage_a_selection_policy,
            has_stage_a_nearest_selected_similarity=has_stage_a_nearest_selected_similarity,
            has_stage_a_nearest_selected_distance=has_stage_a_nearest_selected_distance,
            has_stage_a_nearest_selected_distance_norm=has_stage_a_nearest_selected_distance_norm,
            weight_by_tf=weight_by_tf,
            weight_fraction_by_tf=weight_fraction_by_tf,
            usage_count_by_tf=usage_count_by_tf,
            failure_count_by_tf=failure_count_by_tf,
            relaxed_cap=relaxed_cap,
            final_cap=cap,
        )
        return state.sites, state.meta, state.labels, info
