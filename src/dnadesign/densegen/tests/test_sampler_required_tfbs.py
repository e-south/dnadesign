from __future__ import annotations

import numpy as np
import pandas as pd

from dnadesign.densegen.src.core.sampler import TFSampler


def test_required_tfbs_are_injected() -> None:
    tfbs = ["AAA", "CCC", "GGG"]
    df = pd.DataFrame(
        {
            "tf": ["TF1", "TF1", "TF2"],
            "tfbs": tfbs,
            "tfbs_core": tfbs,
        }
    )
    rng = np.random.default_rng(0)
    sampler = TFSampler(df, rng)
    sites, meta, labels, _info = sampler.generate_binding_site_library(
        library_size=2,
        required_tfbs=["CCC"],
        cover_all_tfs=False,
        unique_binding_sites=True,
        relax_on_exhaustion=False,
    )
    assert "CCC" in sites
    assert any(item.endswith(":CCC") for item in meta)
    assert "TF1" in labels


def test_required_tfs_are_injected() -> None:
    tfbs = ["AAA", "CCC", "GGG"]
    df = pd.DataFrame(
        {
            "tf": ["TF1", "TF1", "TF2"],
            "tfbs": tfbs,
            "tfbs_core": tfbs,
        }
    )
    rng = np.random.default_rng(1)
    sampler = TFSampler(df, rng)
    sites, meta, labels, _info = sampler.generate_binding_site_library(
        library_size=2,
        required_tfs=["TF2"],
        cover_all_tfs=False,
        unique_binding_sites=True,
        relax_on_exhaustion=False,
    )
    assert "GGG" in sites
    assert any(item.startswith("TF2:") for item in meta)
    assert "TF2" in labels
