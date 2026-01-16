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

    def generate_binding_site_subsample(
        self,
        sequence_length: int,
        budget_overhead: int,
        *,
        required_tfbs: list[str] | None = None,
        required_tfs: list[str] | None = None,
        cover_all_tfs: bool = False,
        unique_binding_sites: bool = True,
        max_sites_per_tf: int | None = None,
        relax_on_exhaustion: bool = True,
        allow_incomplete_coverage: bool = False,
    ) -> tuple[list, list, list, dict]:
        """
        Build a motif library whose total length >= sequence_length + budget_overhead.

        Returns
        -------
        sites, tfbs_parts, regulator_labels, info

        - If cover_all_tfs=True, we first ensure >=1 TFBS per every unique TF.
        - unique_binding_sites=True prevents duplicate (TF, TFBS) pairs.
        - max_sites_per_tf caps per-TF TFBS AFTER coverage is satisfied (None = no cap).
        - If we cannot meet the length budget without violating these rules and
          relax_on_exhaustion=True, we gradually relax the cap to avoid stalling.
        - With strict policies, failures raise ValueError (no fallback).
        """
        target = sequence_length + budget_overhead
        sites: list[str] = []
        meta: list[str] = []
        labels: list[str] = []
        site_ids: list[str | None] = []
        sources: list[str | None] = []
        seen_tfbs = set()  # for unique_binding_sites (tf, tfbs)
        used_per_tf: dict[str, int] = {}

        has_site_id = "site_id" in self.df.columns
        has_source = "source" in self.df.columns

        def _append_provenance(row) -> None:
            site_ids.append(str(row["site_id"]) if has_site_id else None)
            sources.append(str(row["source"]) if has_source else None)

        unique_tfs = self.df["tf"].unique().tolist()
        self.rng.shuffle(unique_tfs)
        total_unique_tfbs = len(self.df.drop_duplicates(["tf", "tfbs"]))

        required = [s.strip().upper() for s in (required_tfbs or []) if str(s).strip()]
        if len(set(required)) != len(required):
            raise ValueError("required_tfbs must be unique")
        if required:
            available_tfbs = set(self.df["tfbs"].tolist())
            missing = [m for m in required if m not in available_tfbs]
            if missing:
                preview = ", ".join(missing[:10])
                raise ValueError(f"Required TFBS motifs not found in input: {preview}")

        required_tf_list = [str(t).strip() for t in (required_tfs or []) if str(t).strip()]
        if len(set(required_tf_list)) != len(required_tf_list):
            raise ValueError("required_tfs must be unique")
        if required_tf_list:
            available = set(self.df["tf"].tolist())
            missing_tfs = [t for t in required_tf_list if t not in available]
            if missing_tfs:
                preview = ", ".join(missing_tfs[:10])
                raise ValueError(f"Required regulators not found in input: {preview}")

        def _pick_for_tf(tf: str) -> bool:
            group = self.df[self.df["tf"] == tf]
            # try a few draws to satisfy uniqueness if requested
            for _ in range(min(20, len(group))):
                row = group.sample(n=1, random_state=int(self.rng.integers(1_000_000)))
                tfbs = row["tfbs"].iloc[0]
                key = (tf, tfbs)
                if (not unique_binding_sites) or (key not in seen_tfbs):
                    sites.append(tfbs)
                    meta.append(f"{tf}:{tfbs}")
                    labels.append(tf)
                    _append_provenance(row.iloc[0])
                    seen_tfbs.add(key)
                    used_per_tf[tf] = used_per_tf.get(tf, 0) + 1
                    return True
            return False  # couldn’t find a new TFBS that met uniqueness

        def _add_required_tfbs() -> None:
            if not required:
                return
            for motif in required:
                rows = self.df[self.df["tfbs"] == motif]
                if rows.empty:
                    raise ValueError(f"Required TFBS motif not found in input: {motif}")
                row = rows.iloc[0]
                tf = row["tf"]
                sites.append(motif)
                meta.append(f"{tf}:{motif}")
                labels.append(tf)
                _append_provenance(row)
                seen_tfbs.add((tf, motif))
                used_per_tf[tf] = used_per_tf.get(tf, 0) + 1

        def _add_required_tfs() -> None:
            if not required_tf_list:
                return
            for tf in required_tf_list:
                if used_per_tf.get(tf, 0) > 0:
                    continue
                if not _pick_for_tf(tf):
                    raise ValueError(f"Failed to select a motif for required regulator '{tf}'")

        _add_required_tfbs()
        _add_required_tfs()

        # 1) coverage pass: ensure >=1 TFBS per TF
        if cover_all_tfs:
            missing = []
            for tf in unique_tfs:
                if not _pick_for_tf(tf):
                    missing.append(tf)
            if missing and not allow_incomplete_coverage:
                raise ValueError(
                    f"Failed to cover all TFs (missing {len(missing)}). "
                    "Allow incomplete coverage or relax uniqueness constraints."
                )

        # 2) expand until we meet/exceed target length
        cap = max_sites_per_tf
        relaxed_cap = False
        while sum(len(s) for s in sites) < target:
            progressed = False
            # cycle through TFs to add more sites within per-TF caps
            for tf in unique_tfs:
                if cap is not None and used_per_tf.get(tf, 0) >= cap:
                    continue
                if _pick_for_tf(tf):
                    progressed = True
                if sum(len(s) for s in sites) >= target:
                    break

            if progressed:
                continue

            if unique_binding_sites and len(seen_tfbs) >= total_unique_tfbs:
                raise ValueError(
                    "Unique TFBS pool exhausted before meeting target length. "
                    "Reduce target length/budget or allow duplicates."
                )

            if relax_on_exhaustion:
                cap = (1 if cap is None else cap) + 1
                relaxed_cap = True
                continue

            raise ValueError(
                "Could not meet target length with current sampling constraints. "
                "Enable relax_on_exhaustion or loosen caps."
            )

        info = {
            "target_length": int(target),
            "achieved_length": int(sum(len(s) for s in sites)),
            "relaxed_cap": bool(relaxed_cap),
            "final_cap": cap,
            "site_id_by_index": site_ids if has_site_id else None,
            "source_by_index": sources if has_source else None,
        }

        return sites, meta, labels, info
