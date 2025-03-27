"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/sampler.py

This module provides routines for sampling TFs and selecting binding sites.
The new method generate_binding_site_subsample() accumulates binding sites
until the total length reaches (sequence_length + subsample_over_length_budget_by),
ensuring no duplicate binding sites are included in the input to the solver.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import warnings
import random

class TFSampler:
    def __init__(self, df: pd.DataFrame):
        assert not df.empty, "Input DataFrame for sampling is empty."
        assert 'tf' in df.columns and 'tfbs' in df.columns, "DataFrame must contain 'tf' and 'tfbs' columns."
        self.df = df.copy()

    def sample_unique_tfs(self, required_tf_count: int, allow_replacement: bool = False) -> list:
        unique_tfs = self.df['tf'].unique()
        if required_tf_count > len(unique_tfs):
            warnings.warn(
                f"Required unique TF count ({required_tf_count}) exceeds available unique TFs ({len(unique_tfs)}). Sampling with replacement."
            )
            allow_replacement = True
        sampled_tfs = np.random.choice(unique_tfs, size=required_tf_count, replace=allow_replacement)
        return sampled_tfs.tolist()

    def subsample_binding_sites(self, sample_size: int, unique_tf_only: bool = False) -> list:
        # This legacy method is preserved for compatibility.
        if unique_tf_only:
            sampled_tfs = self.sample_unique_tfs(sample_size, allow_replacement=False)
            binding_sites = []
            for tf in sampled_tfs:
                group = self.df[self.df['tf'] == tf]
                assert not group.empty, f"No binding sites found for TF '{tf}'."
                chosen = group.sample(n=1, random_state=np.random.randint(10000))
                tf_val = chosen['tf'].iloc[0]
                tfbs_val = chosen['tfbs'].iloc[0]
                file_source = chosen['deg_source'].iloc[0] if 'deg_source' in chosen.columns else "unknown"
                assert tf_val != "" and tfbs_val != "", f"Empty tf or tfbs encountered for TF '{tf}'."
                binding_sites.append((tf_val, tfbs_val, file_source))
            return binding_sites
        else:
            grouped = self.df.groupby('tf')
            samples = []
            for tf, group in grouped:
                n = min(sample_size, len(group))
                samples.append(group.sample(n=n, random_state=np.random.randint(10000)))
            df_sampled = pd.concat(samples, ignore_index=True)
            assert not (df_sampled['tf'] == "").any(), "Empty TF value found in subsample."
            assert not (df_sampled['tfbs'] == "").any(), "Empty TFBS value found in subsample."
            return list(zip(df_sampled['tf'].tolist(), df_sampled['tfbs'].tolist(),
                            df_sampled.get('deg_source', pd.Series(["unknown"]*len(df_sampled))).tolist()))

    def generate_binding_site_subsample(self, sequence_length: int, budget_overhead: int) -> tuple:
        """
        Generates a subsample of binding sites until the total length of binding sites
        is at least (sequence_length + budget_overhead). Duplicate binding sites are skipped.
        
        Returns:
            A tuple (sampled_binding_sites, meta_tfbs_parts) where:
              - sampled_binding_sites: list of binding site strings.
              - meta_tfbs_parts: list of metadata strings in the format "tf:tfbs".
        """
        target_length = sequence_length + budget_overhead
        unique_binding_sites = set()
        sampled_binding_sites = []
        meta_tfbs_parts = []
        tf_list = self.df['tf'].unique().tolist()
        # Continue sampling until the accumulated length meets or exceeds the target.
        while sum(len(bs) for bs in sampled_binding_sites) < target_length:
            tf = random.choice(tf_list)
            group = self.df[self.df['tf'] == tf]
            chosen = group.sample(n=1, random_state=random.randint(0, 100000))
            tf_val = chosen['tf'].iloc[0]
            tfbs_val = chosen['tfbs'].iloc[0]
            # Skip if this binding site is already in the sample.
            if tfbs_val in unique_binding_sites:
                continue
            unique_binding_sites.add(tfbs_val)
            sampled_binding_sites.append(tfbs_val)
            meta_tfbs_parts.append(f"{tf_val}:{tfbs_val}")
        return sampled_binding_sites, meta_tfbs_parts
