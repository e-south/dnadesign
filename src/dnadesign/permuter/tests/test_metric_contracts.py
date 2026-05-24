"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/tests/test_metric_contracts.py

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dnadesign.permuter.src.cli.plot import _normalize_for_plots
from dnadesign.permuter.src.cli.validate import validate
from dnadesign.permuter.src.contracts.metrics import interaction_metric_column
from dnadesign.permuter.src.protocols.combine.combine_aa import CombineAA

CODON_CSV = """codon,amino_acid,fraction,frequency
AAA,K,0.73,33.2
AAC,N,0.53,24.4
CAG,Q,0.70,27.7
"""


def _write_codon_table(tmp_path: Path) -> Path:
    p = tmp_path / "codon.csv"
    p.write_text(CODON_CSV, encoding="utf-8")
    return p


def _write_singles(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "singles.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


def _params(tmp_path: Path, dms_path: Path) -> dict:
    return {
        "from_dataset": str(dms_path),
        "codon_table": str(_write_codon_table(tmp_path)),
        "singles_metric_id": "llr_mean",
        "select": {"top_global": 2},
        "combine": {
            "k_min": 2,
            "k_max": 2,
            "budget_total": 1,
            "strategy": "random",
            "random": {"samples_per_k": {"2": 1}},
        },
        "codon_choice": "top",
        "rng_seed": 7,
    }


def test_combine_aa_consumes_canonical_observed_metric(tmp_path: Path) -> None:
    ref = "AAAAAA"
    dms = _write_singles(
        tmp_path,
        [
            {
                "sequence": ref,
                "permuter__round": 1,
                "permuter__aa_pos": 1,
                "permuter__aa_wt": "K",
                "permuter__aa_alt": "N",
                "permuter__observed__llr_mean": 1.0,
            },
            {
                "sequence": ref,
                "permuter__round": 1,
                "permuter__aa_pos": 2,
                "permuter__aa_wt": "K",
                "permuter__aa_alt": "Q",
                "permuter__observed__llr_mean": 2.0,
            },
        ],
    )

    records = list(
        CombineAA().generate(
            ref_entry={"ref_name": "toy", "sequence": ref},
            params=_params(tmp_path, dms),
            rng=np.random.default_rng(0),
        )
    )

    assert len(records) == 1
    assert records[0]["expected__llr_mean"] == pytest.approx(3.0)


def test_combine_aa_rejects_legacy_metric_column(tmp_path: Path) -> None:
    ref = "AAAAAA"
    dms = _write_singles(
        tmp_path,
        [
            {
                "sequence": ref,
                "permuter__round": 1,
                "permuter__aa_pos": 1,
                "permuter__aa_wt": "K",
                "permuter__aa_alt": "N",
                "permuter__metric__llr_mean": 1.0,
            },
            {
                "sequence": ref,
                "permuter__round": 1,
                "permuter__aa_pos": 2,
                "permuter__aa_wt": "K",
                "permuter__aa_alt": "Q",
                "permuter__metric__llr_mean": 2.0,
            },
        ],
    )

    with pytest.raises(ValueError, match="permuter__observed__llr_mean"):
        list(
            CombineAA().generate(
                ref_entry={"ref_name": "toy", "sequence": ref},
                params=_params(tmp_path, dms),
                rng=np.random.default_rng(0),
            )
        )


def test_plot_normalization_requires_only_observed_metric_globally() -> None:
    df = pd.DataFrame(
        [
            {
                "sequence": "ACGT",
                "permuter__modifications": [],
                "permuter__round": 1,
                "permuter__observed__llr_mean": 0.25,
            }
        ]
    )

    normalized = _normalize_for_plots(df, "llr_mean")

    assert normalized["permuter__observed__llr_mean"].iloc[0] == pytest.approx(0.25)


def test_validate_strict_accepts_namespaced_interaction_metric(tmp_path: Path) -> None:
    records = _write_contract_dataset(
        tmp_path,
        extra={interaction_metric_column("epistasis", "llr_mean"): 0.2},
    )

    validate(records, strict=True)


def test_validate_strict_rejects_unnamespaced_epistasis(tmp_path: Path) -> None:
    records = _write_contract_dataset(tmp_path, extra={"epistasis": 0.2})

    with pytest.raises(ValueError, match="Non-namespaced"):
        validate(records, strict=True)


def _write_contract_dataset(tmp_path: Path, *, extra: dict) -> Path:
    sequence = "ACGT"
    row = {
        "id": hashlib.sha1(f"dna|{sequence}".encode("utf-8")).hexdigest(),
        "bio_type": "dna",
        "sequence": sequence,
        "alphabet": "dna_4",
        "length": len(sequence),
        "source": "unit",
        "created_at": "2026-05-24T00:00:00Z",
        "permuter__job": "unit",
        "permuter__ref": "toy",
        "permuter__protocol": "scan_dna",
        "permuter__var_id": "v1",
        "permuter__observed__llr_mean": 1.0,
        "permuter__expected__llr_mean": 0.8,
        **extra,
    }
    path = tmp_path / "records.parquet"
    pd.DataFrame([row]).to_parquet(path, index=False)
    return path
