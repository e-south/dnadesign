"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/integrations/test_dense_arrays_resting_frames.py

Check complete resting duplexes and stable placement geometry.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pytest
from dense_arrays.playback import PlaybackDocument, reconstruct_playback

from dnadesign.densegen.src.integrations.dense_arrays.baserender_projection import (
    AnchoredIllustrationPresentation,
    BaseRenderDuplexProjection,
    DuplexPresentation,
)
from dnadesign.densegen.src.integrations.dense_arrays.playback import realized_array_from_densegen_record


def _document():
    sequence = "ACGTTGCAAGTCCTGATCGTACCGATGCTTAGGACGTAAAA"
    details = [
        {
            "part_kind": "tfbs",
            "sequence": sequence[start:end],
            "offset": start,
            "end": end,
            "orientation": "fwd",
            "tfbs_id": f"motif-{index}",
            "regulator": f"motif-{index}",
        }
        for index, (start, end) in enumerate(((0, 16), (7, 23), (21, 37)))
    ]
    realized = realized_array_from_densegen_record(
        {"id": "resting", "sequence": sequence, "densegen__used_tfbs_detail": details},
        source_ref="fixture.parquet",
    )
    plan = reconstruct_playback(realized)
    return PlaybackDocument(plan=plan, title="Resting example")


def test_resting_frame_keeps_every_base_and_feature_in_neutral_gray():
    document = _document()
    projection = BaseRenderDuplexProjection((document,))
    figure = projection._figure(document, None)
    try:
        glyphs = {
            artist.get_gid(): artist
            for artist in figure.axes[0].patches
            if (artist.get_gid() or "").startswith("sequence:")
        }
        assert len(glyphs) == 2 * len(document.plan.realized_sequence)
        assert all(np.allclose(artist.get_facecolor()[:3], [210 / 255] * 3) for artist in glyphs.values())
    finally:
        plt.close(figure)
    baseline = projection.render_rgba(document, None)
    finished = projection.render_rgba(document, len(document.plan.steps) - 1)
    assert baseline.shape == finished.shape
    assert np.array_equal(baseline[:, :, 0], baseline[:, :, 1])
    assert np.array_equal(baseline[:, :, 1], baseline[:, :, 2])
    assert np.count_nonzero(baseline[:, :, 0] < 245) > 100
    assert not np.array_equal(baseline, finished)


def test_pending_constraint_annotations_keep_complete_neutral_geometry():
    sequence = "A" * 60
    details = [
        {
            "part_kind": "fixed_element",
            "sequence": "AAAAAA",
            "offset": start,
            "end": start + 6,
            "orientation": "fwd",
            "constraint_name": "anchor",
            "placement_index": 0,
            "role": role,
            "variant_id": "a",
            "spacer_length": 19,
        }
        for role, start in (("upstream", 20), ("downstream", 45))
    ]
    realized = realized_array_from_densegen_record(
        {"id": "anchored-rest", "sequence": sequence, "densegen__used_tfbs_detail": details},
        source_ref="fixture.parquet",
    )
    document = PlaybackDocument(plan=reconstruct_playback(realized), title="Anchored resting example")
    projection = BaseRenderDuplexProjection(
        (document,),
        realized_arrays={document.plan.realization_digest: realized},
        presentation=DuplexPresentation(
            fixed_element_annotations="variant",
            anchored_illustration=AnchoredIllustrationPresentation("rnap_sigma70", "anchor"),
        ),
    )
    figures = [projection._figure(document, step) for step in (None, 0, 1)]
    try:
        axes = [figure.axes[0] for figure in figures]
        assert all(any(text.get_text() == "19 bp" for text in ax.texts) for ax in axes)
        assert [len(ax.images) for ax in axes] == [1, 1, 1]
        baseline_image = np.asarray(axes[0].images[0].get_array())
        assert np.array_equal(baseline_image[:, :, 0], baseline_image[:, :, 1])
        assert np.array_equal(baseline_image[:, :, 1], baseline_image[:, :, 2])
        assert all(np.allclose(colors.to_rgb(line.get_color()), [210 / 255] * 3) for line in axes[0].lines)
        assert len({tuple(ax.images[0].get_extent()) for ax in axes}) == 1
        assert len({tuple(ax.get_xlim()) for ax in axes}) == 1
    finally:
        for figure in figures:
            plt.close(figure)


@pytest.mark.parametrize("step", [-1, True, 1.5, 3])
def test_projection_rejects_invalid_frame_indices(step):
    document = _document()
    projection = BaseRenderDuplexProjection((document,))
    projection.render_rgba(document, 1)
    with pytest.raises(IndexError, match="step_index"):
        projection.render_rgba(document, step)
