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

import numpy as np
import pandas as pd


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
            sampled_tfs = self.sample_unique_tfs(sample_size, allow_replacement=False)
            binding_sites = []
            for tf in sampled_tfs:
                group = self.df[self.df["tf"] == tf]
                if group.empty:
                    raise ValueError(f"No binding sites found for TF '{tf}'.")
                chosen = group.sample(n=1, random_state=int(self.rng.integers(1_000_000)))
                binding_sites.append((chosen["tf"].iloc[0], chosen["tfbs"].iloc[0], "binding_sites"))
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
        sequence_length: int,
        budget_overhead: int,
        required_tfbs: list[str] | None = None,
        required_tfs: list[str] | None = None,
        cover_all_tfs: bool = False,
        unique_binding_sites: bool = True,
        unique_binding_cores: bool = True,
        max_sites_per_tf: int | None = None,
        relax_on_exhaustion: bool = True,
        allow_incomplete_coverage: bool = False,
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

        df = self.df
        if unique_binding_sites:
            df = df.drop_duplicates(["tf", "tfbs"]).reset_index(drop=True)
        if unique_binding_cores:
            if "tfbs_core" not in df.columns:
                raise ValueError(
                    "unique_binding_cores=true requires a 'tfbs_core' column in the input data. "
                    "Provide tfbs_core in the pool or disable unique_binding_cores."
                )

        has_site_id = "site_id" in df.columns
        has_source = "source" in df.columns
        has_tfbs_id = "tfbs_id" in df.columns
        has_motif_id = "motif_id" in df.columns
        total_unique_tfbs = len(df.drop_duplicates(["tf", "tfbs"]))
        total_unique_cores = len(df.drop_duplicates(["tf", "tfbs_core"])) if unique_binding_cores else total_unique_tfbs

        unique_tfs = sorted(df["tf"].unique().tolist())
        if not unique_tfs:
            raise ValueError("No regulators found in input.")

        weight_by_tf: dict[str, float] | None = None
        weight_fraction_by_tf: dict[str, float] | None = None
        usage_count_by_tf: dict[str, int] | None = None
        failure_count_by_tf: dict[str, int] | None = None
        if sampling_strategy == "coverage_weighted":
            weight_by_tf = {}
            usage_count_by_tf = {}
            failure_count_by_tf = {}
            for _, row in df.iterrows():
                tf = str(row["tf"])
                tfbs = str(row["tfbs"])
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
            total_weight = sum(weight_by_tf.values()) if weight_by_tf else 0.0
            weight_fraction_by_tf = {
                tf: (float(val) / total_weight if total_weight > 0 else 0.0) for tf, val in (weight_by_tf or {}).items()
            }

        required = [str(s).strip().upper() for s in (required_tfbs or []) if str(s).strip()]
        if len(set(required)) != len(required):
            raise ValueError("required_tfbs must be unique")
        if required:
            available_tfbs = set(df["tfbs"].tolist())
            missing = [m for m in required if m not in available_tfbs]
            if missing:
                preview = ", ".join(missing[:10])
                raise ValueError(f"Required TFBS motifs not found in input: {preview}")

        required_tf_list = [str(t).strip() for t in (required_tfs or []) if str(t).strip()]
        if len(set(required_tf_list)) != len(required_tf_list):
            raise ValueError("required_tfs must be unique")
        if required_tf_list:
            available = set(df["tf"].tolist())
            missing_tfs = [t for t in required_tf_list if t not in available]
            if missing_tfs:
                preview = ", ".join(missing_tfs[:10])
                raise ValueError(f"Required regulators not found in input: {preview}")

        if len(required) > library_size:
            raise ValueError(f"library_size={library_size} is smaller than required_tfbs ({len(required)}).")

        sites: list[str] = []
        meta: list[str] = []
        labels: list[str] = []
        reasons: list[str] = []
        site_ids: list[str | None] = []
        sources: list[str | None] = []
        tfbs_ids: list[str | None] = []
        motif_ids: list[str | None] = []
        seen_tfbs = set()
        seen_cores = set()
        used_per_tf: dict[str, int] = {}

        def _append_row(row, reason: str) -> bool:
            tf = str(row["tf"])
            tfbs = str(row["tfbs"])
            key = (tf, tfbs)
            if unique_binding_sites and key in seen_tfbs:
                return False
            if unique_binding_cores:
                core = str(row["tfbs_core"])
                core_key = (tf, core)
                if core_key in seen_cores:
                    return False
            sites.append(tfbs)
            meta.append(f"{tf}:{tfbs}")
            labels.append(tf)
            reasons.append(reason)
            seen_tfbs.add(key)
            if unique_binding_cores:
                seen_cores.add((tf, str(row["tfbs_core"])))
            used_per_tf[tf] = used_per_tf.get(tf, 0) + 1
            site_ids.append(str(row["site_id"]) if has_site_id else None)
            sources.append(str(row["source"]) if has_source else None)
            tfbs_ids.append(str(row["tfbs_id"]) if has_tfbs_id else None)
            motif_ids.append(str(row["motif_id"]) if has_motif_id else None)
            return True

        def _pick_for_tf(tf: str, *, reason: str, cap_override: int | None = None) -> bool:
            group = df[df["tf"] == tf]
            if group.empty:
                return False
            if cap_override is not None and used_per_tf.get(tf, 0) >= cap_override:
                return False

            indices = group.index.to_list()
            if not indices:
                return False

            if sampling_strategy == "coverage_weighted":
                candidates: list[int] = []
                weights: list[float] = []
                for idx in indices:
                    row = group.loc[idx]
                    key = (str(row["tf"]), str(row["tfbs"]))
                    if unique_binding_sites and key in seen_tfbs:
                        continue
                    if unique_binding_cores:
                        core_key = (str(row["tf"]), str(row["tfbs_core"]))
                        if core_key in seen_cores:
                            continue
                    count = int(usage_counts.get(key, 0)) if usage_counts else 0
                    weight = 1.0 + float(coverage_boost_alpha) / ((1.0 + count) ** float(coverage_boost_power))
                    if avoid_failed_motifs and failure_counts is not None:
                        fails = int(failure_counts.get(key, 0))
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
                    row = group.loc[choice]
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
                row = group.loc[idx]
                key = (str(row["tf"]), str(row["tfbs"]))
                if unique_binding_sites and key in seen_tfbs:
                    continue
                if unique_binding_cores:
                    core_key = (str(row["tf"]), str(row["tfbs_core"]))
                    if core_key in seen_cores:
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
            if used_per_tf.get(tf, 0) > 0:
                continue
            if not _pick_for_tf(tf, reason="required_tf", cap_override=None):
                raise ValueError(f"Failed to select a motif for required regulator '{tf}'")

        # Coverage pass
        if cover_all_tfs:
            remaining_slots = library_size - len(sites)
            missing_tfs = [tf for tf in unique_tfs if used_per_tf.get(tf, 0) == 0]
            if remaining_slots < len(missing_tfs) and not allow_incomplete_coverage:
                raise ValueError(
                    "cover_all_regulators requires at least one site per TF, "
                    f"but library_size={library_size} is too small for {len(unique_tfs)} TFs. "
                    "Increase library_size or set allow_incomplete_coverage=true."
                )
            if remaining_slots <= 0:
                if missing_tfs and not allow_incomplete_coverage:
                    raise ValueError(
                        "Required coverage cannot be satisfied given required_tfbs/required_tfs. "
                        "Increase library_size or relax coverage constraints."
                    )
            tf_order = unique_tfs.copy()
            self.rng.shuffle(tf_order)
            for tf in tf_order:
                if len(sites) >= library_size:
                    break
                if used_per_tf.get(tf, 0) > 0:
                    continue
                if _pick_for_tf(tf, reason="coverage", cap_override=None):
                    continue
                if not allow_incomplete_coverage:
                    raise ValueError(f"Failed to select a motif for TF '{tf}' while cover_all_regulators=true.")

        cap = max_sites_per_tf
        relaxed_cap = False

        def _fill_tf_balanced() -> None:
            nonlocal cap, relaxed_cap
            tf_order = unique_tfs.copy()
            self.rng.shuffle(tf_order)
            while len(sites) < library_size:
                progressed = False
                for tf in tf_order:
                    if len(sites) >= library_size:
                        break
                    if cap is not None and used_per_tf.get(tf, 0) >= cap:
                        continue
                    if _pick_for_tf(tf, reason="filler", cap_override=cap):
                        progressed = True
                if progressed:
                    continue
                if unique_binding_sites and len(seen_tfbs) >= total_unique_tfbs:
                    raise ValueError(
                        "Unique TFBS pool exhausted before reaching library_size. "
                        "Reduce library_size or allow duplicates."
                    )
                if unique_binding_cores and len(seen_cores) >= total_unique_cores:
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
            while len(sites) < library_size:
                self.rng.shuffle(indices)
                progressed = False
                for idx in indices:
                    if len(sites) >= library_size:
                        break
                    row = df.loc[idx]
                    tf = str(row["tf"])
                    if cap is not None and used_per_tf.get(tf, 0) >= cap:
                        continue
                    if _append_row(row, "filler"):
                        progressed = True
                if progressed:
                    continue
                if unique_binding_sites and len(seen_tfbs) >= total_unique_tfbs:
                    raise ValueError(
                        "Unique TFBS pool exhausted before reaching library_size. "
                        "Reduce library_size or allow duplicates."
                    )
                if unique_binding_cores and len(seen_cores) >= total_unique_cores:
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

        if len(sites) > library_size:
            raise ValueError(
                "Required motifs exceed library_size. Increase library_size or relax required constraints."
            )
        if len(sites) < library_size:
            if sampling_strategy in {"tf_balanced", "coverage_weighted"}:
                _fill_tf_balanced()
            elif sampling_strategy == "uniform_over_pairs":
                _fill_uniform_over_pairs()
            else:
                raise ValueError(f"Unknown library sampling strategy: {sampling_strategy}")

        info = {
            "target_length": int(sequence_length + budget_overhead),
            "achieved_length": int(sum(len(s) for s in sites)),
            "relaxed_cap": bool(relaxed_cap),
            "final_cap": cap,
            "site_id_by_index": site_ids if has_site_id else None,
            "source_by_index": sources if has_source else None,
            "tfbs_id_by_index": tfbs_ids if has_tfbs_id else None,
            "motif_id_by_index": motif_ids if has_motif_id else None,
            "selection_reason_by_index": reasons,
            "sampling_weight_by_tf": weight_by_tf,
            "sampling_weight_fraction_by_tf": weight_fraction_by_tf,
            "sampling_usage_count_by_tf": usage_count_by_tf,
            "sampling_failure_count_by_tf": failure_count_by_tf,
        }
        return sites, meta, labels, info
