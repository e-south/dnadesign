"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/sampler.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import warnings

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
