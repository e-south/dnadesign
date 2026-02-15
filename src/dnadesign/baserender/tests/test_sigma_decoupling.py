"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_sigma_decoupling.py

Tests for sigma decoupling from core/render and generic sigma transform output.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.baserender.src.core import Record, Span
from dnadesign.baserender.src.core.record import Display, Feature
from dnadesign.baserender.src.pipeline.sigma70 import Sigma70Transform


def test_core_and_render_do_not_embed_sigma70_semantics() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in ("core", "render"):
        for path in (root / rel).rglob("*.py"):
            text = path.read_text().lower()
            assert "sigma70" not in text
            assert "tf:sigma70_" not in text


def test_sigma_transform_emits_generic_features_effects_and_display_labels() -> None:
    sequence = "TTGACA" + ("A" * 16) + "TATAAT"
    record = Record(
        id="r1",
        alphabet="DNA",
        sequence=sequence,
        features=(),
        effects=(),
        display=Display(),
        meta={},
    ).validate()

    transformed = Sigma70Transform().apply(record)

    sigma_features = [f for f in transformed.features if "sigma" in f.tags]
    assert sigma_features
    assert all(f.kind == "kmer" for f in sigma_features)
    assert all(f.render.get("track") == 0 for f in sigma_features)
    assert all(f.render.get("priority") == 0 for f in sigma_features)

    span_links = [e for e in transformed.effects if e.kind == "span_link"]
    assert span_links
    assert transformed.display.tag_labels.get("sigma", "").startswith("σ70 ")


def test_sigma_transform_links_to_existing_feature_ids_when_features_are_deduped() -> None:
    sequence = "TTGACA" + ("A" * 16) + "TATAAT"
    record = Record(
        id="r1",
        alphabet="DNA",
        sequence=sequence,
        features=(
            Feature(
                id="existing_35",
                kind="kmer",
                span=Span(start=0, end=6, strand="fwd"),
                label="TTGACA",
                tags=("sigma",),
                attrs={"strength": "high", "piece": "-35"},
                render={"track": 0, "priority": 0},
            ),
            Feature(
                id="existing_10",
                kind="kmer",
                span=Span(start=22, end=28, strand="fwd"),
                label="TATAAT",
                tags=("sigma",),
                attrs={"strength": "high", "piece": "-10"},
                render={"track": 0, "priority": 0},
            ),
        ),
        effects=(),
        display=Display(),
        meta={},
    ).validate()

    transformed = Sigma70Transform().apply(record)
    span_links = [e for e in transformed.effects if e.kind == "span_link"]
    assert len(span_links) == 1
    link = span_links[0]

    feature_ids = {f.id for f in transformed.features}
    assert link.target["from_feature_id"] in feature_ids
    assert link.target["to_feature_id"] in feature_ids
