"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_sampler_unique_binding_cores.py

Unit tests for Stage-B core-uniqueness sampling behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dnadesign.densegen.src.core.sampler import TFSampler


def test_unique_binding_cores_filters_duplicate_cores() -> None:
    df = pd.DataFrame(
        {
            "tf": ["lexA", "lexA", "lexA"],
            "tfbs": ["AAAATTT", "AAAAAAA", "CCCCGGG"],
            "tfbs_core": ["AAAA", "AAAA", "CCCC"],
        }
    )
    sampler = TFSampler(df, rng=np.random.default_rng(1))
    sites, _meta, _labels, _info = sampler.generate_binding_site_library(
        library_size=2,
        unique_binding_sites=False,
        unique_binding_cores=True,
        sampling_strategy="tf_balanced",
    )
    core_by_site = dict(zip(df["tfbs"].tolist(), df["tfbs_core"].tolist()))
    cores = [core_by_site[site] for site in sites]
    assert len(set(cores)) == len(cores)


def test_unique_binding_cores_requires_core_column() -> None:
    df = pd.DataFrame({"tf": ["lexA"], "tfbs": ["AAAATTT"]})
    sampler = TFSampler(df, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="tfbs_core"):
        sampler.generate_binding_site_library(
            library_size=1,
            unique_binding_sites=True,
            unique_binding_cores=True,
            sampling_strategy="tf_balanced",
        )
