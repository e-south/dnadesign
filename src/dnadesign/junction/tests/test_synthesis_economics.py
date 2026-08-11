from __future__ import annotations

import importlib
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

from dnadesign.junction.economics import (
    SynthesisScenario,
    build_purchase_price_rows,
    load_pricing_snapshot,
    render_purchase_price_comparison,
    stable_oligo_pool_advantage,
)
from dnadesign.junction.economics import render as economics_render
from dnadesign.junction.economics.contracts import default_pricing_snapshot_path


def _scenario() -> SynthesisScenario:
    return SynthesisScenario(
        target_length_nt=1_000,
        oligos_per_target=36,
        minimum_oligo_length_nt=20,
        maximum_oligo_length_nt=120,
    )


def test_pricing_snapshot_covers_all_supplied_count_and_length_tiers() -> None:
    snapshot = load_pricing_snapshot(default_pricing_snapshot_path())

    assert len(snapshot.oligo_length_bands) == 6
    assert len(snapshot.oligo_pool_tiers) == 28
    assert snapshot.maximum_oligos_per_pool == 696_000
    assert snapshot.n_nucleotide_surcharge_fraction == Decimal("0.20")
    assert snapshot.oligo_pool_price(oligo_count=100, length_band_id="20-120nt") == 320
    assert snapshot.oligo_pool_price(oligo_count=696_000, length_band_id="301-350nt") == Decimal("114422.40")


def test_price_rows_span_the_complete_pool_capacity() -> None:
    snapshot = load_pricing_snapshot(default_pricing_snapshot_path())
    rows = build_purchase_price_rows(snapshot, _scenario())

    assert len(rows) == 19_333
    assert rows[0].target_count == 1
    assert rows[0].gene_fragments_usd == 60
    assert rows[0].oligo_pool_usd == 320
    assert rows[-1].oligo_count == 695_988
    assert rows[-1].oligo_pool_usd == 38_976
    assert stable_oligo_pool_advantage(rows) == 17


def test_price_contract_rejects_ranges_that_cross_length_bands() -> None:
    snapshot = load_pricing_snapshot(default_pricing_snapshot_path())

    with pytest.raises(ValueError, match="fit exactly one price band"):
        build_purchase_price_rows(
            snapshot,
            SynthesisScenario(
                target_length_nt=1_000,
                oligos_per_target=36,
                minimum_oligo_length_nt=110,
                maximum_oligo_length_nt=130,
            ),
        )


def test_price_rows_start_at_the_first_complete_priced_pool() -> None:
    snapshot = load_pricing_snapshot(default_pricing_snapshot_path())

    rows = build_purchase_price_rows(snapshot, replace(_scenario(), oligos_per_target=1))

    assert rows[0].target_count == 2
    assert rows[0].oligo_count == 2


def test_snapshot_rejects_duplicate_length_band_ids_and_non_usd_currency() -> None:
    snapshot = load_pricing_snapshot(default_pricing_snapshot_path())
    duplicate_id = replace(snapshot.oligo_length_bands[1], band_id=snapshot.oligo_length_bands[0].band_id)

    with pytest.raises(ValueError, match="ids must be unique"):
        replace(snapshot, oligo_length_bands=(snapshot.oligo_length_bands[0], duplicate_id))
    with pytest.raises(ValueError, match="must use USD"):
        replace(snapshot, currency="EUR")


def test_n_nucleotide_surcharge_is_explicit() -> None:
    snapshot = load_pricing_snapshot(default_pricing_snapshot_path())
    baseline = build_purchase_price_rows(snapshot, _scenario())[0]
    with_n = build_purchase_price_rows(snapshot, replace(_scenario(), uses_n_nucleotide=True))[0]

    assert baseline.oligo_pool_usd == Decimal("320.00")
    assert with_n.oligo_pool_usd == Decimal("384.0000")


def test_purchase_price_figure_is_square_generic_and_deterministic(tmp_path: Path) -> None:
    snapshot = load_pricing_snapshot(default_pricing_snapshot_path())
    first = render_purchase_price_comparison(snapshot, _scenario(), output_stem=tmp_path / "first")
    second = render_purchase_price_comparison(snapshot, _scenario(), output_stem=tmp_path / "second")

    assert first.svg_path.read_bytes() == second.svg_path.read_bytes()
    assert first.png_path.read_bytes() == second.png_path.read_bytes()
    svg = first.svg_path.read_text(encoding="utf-8")
    assert "Gene fragments and Junction oligo pools" in svg
    assert "Eco1" not in svg
    assert "Budget capacity" not in svg
    assert "Portal prices" not in svg
    assert first.png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert plt.get_fignums() == []


def test_importing_renderer_preserves_the_selected_backend() -> None:
    backend = matplotlib.get_backend()

    importlib.reload(economics_render)

    assert matplotlib.get_backend() == backend
