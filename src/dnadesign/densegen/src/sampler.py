"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/src/sampler.py

TF/TFBS sampling utilities for DenseGen.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


class TFSampler:
    """
    Sampler for TF/TFBS mappings (DataFrame with 'tf' and 'tfbs').
    """

    def __init__(self, df: pd.DataFrame):
        assert df is not None and not df.empty, "Input DataFrame for sampling is empty."
        assert "tf" in df.columns and "tfbs" in df.columns, "DataFrame must contain 'tf' and 'tfbs' columns."
        self.df = df.copy()

    def sample_unique_tfs(self, required_tf_count: int, allow_replacement: bool = False) -> list:
        unique_tfs = self.df["tf"].unique()
        if required_tf_count > len(unique_tfs):
            warnings.warn(
                f"Req. unique TF count ({required_tf_count}) exceeds TFs ({len(unique_tfs)}). Sampling w/ replacement."
            )
            allow_replacement = True
        sampled_tfs = np.random.choice(unique_tfs, size=required_tf_count, replace=allow_replacement)
        return sampled_tfs.tolist()

    def subsample_binding_sites(self, sample_size: int, unique_tf_only: bool = False) -> list:
        if unique_tf_only:
            sampled_tfs = self.sample_unique_tfs(sample_size, allow_replacement=False)
            binding_sites = []
            for tf in sampled_tfs:
                group = self.df[self.df["tf"] == tf]
                assert not group.empty, f"No binding sites found for TF '{tf}'."
                chosen = group.sample(n=1, random_state=np.random.randint(10000))
                binding_sites.append((chosen["tf"].iloc[0], chosen["tfbs"].iloc[0], "csv_tfbs"))
            return binding_sites
        else:
            grouped = self.df.groupby("tf")
            samples = []
            for _, group in grouped:
                n = min(sample_size, len(group))
                samples.append(group.sample(n=n, random_state=np.random.randint(10000)))
            df_sampled = pd.concat(samples, ignore_index=True)
            return list(
                zip(
                    df_sampled["tf"].tolist(),
                    df_sampled["tfbs"].tolist(),
                    ["csv_tfbs"] * len(df_sampled),
                )
            )

    def generate_binding_site_subsample(
        self,
        sequence_length: int,
        budget_overhead: int,
        *,
        cover_all_tfs: bool = False,
        unique_binding_sites: bool = True,
        max_sites_per_tf: int | None = None,
        relax_on_exhaustion: bool = True,
    ) -> tuple[list, list]:
        """
        Build a motif library whose total length >= sequence_length + budget_overhead.

        - If cover_all_tfs=True, we first ensure >=1 TFBS per every unique TF.
        - unique_binding_sites=True prevents the same TFBS string from appearing twice.
        - max_sites_per_tf caps per-TF TFBS AFTER coverage is satisfied (None = no cap).
        - If we cannot meet the length budget without violating these rules and
          relax_on_exhaustion=True, we gradually relax the cap to avoid stalling.
        """
        target = sequence_length + budget_overhead
        sites: list[str] = []
        meta: list[str] = []
        seen_tfbs = set()  # for unique_binding_sites
        used_per_tf: dict[str, int] = {}

        rng = np.random.default_rng()
        unique_tfs = self.df["tf"].unique().tolist()
        rng.shuffle(unique_tfs)

        def _pick_for_tf(tf: str) -> bool:
            group = self.df[self.df["tf"] == tf]
            # try a few draws to satisfy uniqueness if requested
            for _ in range(min(20, len(group))):
                row = group.sample(n=1, random_state=rng.integers(1_000_000))
                tfbs = row["tfbs"].iloc[0]
                if (not unique_binding_sites) or (tfbs not in seen_tfbs):
                    sites.append(tfbs)
                    meta.append(f"{tf}:{tfbs}")
                    seen_tfbs.add(tfbs)
                    used_per_tf[tf] = used_per_tf.get(tf, 0) + 1
                    return True
            return False  # couldn’t find a new TFBS that met uniqueness

        # 1) coverage pass: ensure >=1 TFBS per TF
        if cover_all_tfs:
            for tf in unique_tfs:
                _ = _pick_for_tf(tf)

        # 2) expand until we meet/exceed target length
        cap = max_sites_per_tf
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

            # If we can’t progress, consider relaxing the cap
            if relax_on_exhaustion:
                cap = (1 if cap is None else cap) + 1
            else:
                break

        return sites, meta
