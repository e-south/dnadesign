from __future__ import annotations

import numpy as np
import pandas as pd

from dnadesign.densegen.src.core.sampler import TFSampler


def _sample_df() -> pd.DataFrame:
    tfbs = ["AAA", "CCC", "GGG", "TTT", "AAC", "CCA"]
    return pd.DataFrame(
        {
            "tf": ["TF1", "TF1", "TF1", "TF2", "TF2", "TF2"],
            "tfbs": tfbs,
            "tfbs_core": tfbs,
        }
    )


def test_generate_binding_site_library_exact_size() -> None:
    df = _sample_df()
    sampler = TFSampler(df, np.random.default_rng(0))
    sites, meta, labels, info = sampler.generate_binding_site_library(
        4,
        cover_all_tfs=True,
        unique_binding_sites=True,
        relax_on_exhaustion=False,
        sampling_strategy="tf_balanced",
    )
    assert len(sites) == 4
    assert len(meta) == 4
    assert len(labels) == 4
    assert info["achieved_length"] == sum(len(s) for s in sites)


def test_generate_binding_site_library_required_tfs() -> None:
    df = _sample_df()
    sampler = TFSampler(df, np.random.default_rng(1))
    sites, meta, labels, _info = sampler.generate_binding_site_library(
        3,
        required_tfs=["TF2"],
        cover_all_tfs=False,
        unique_binding_sites=True,
        relax_on_exhaustion=False,
        sampling_strategy="tf_balanced",
    )
    assert len(sites) == 3
    assert "TF2" in labels
