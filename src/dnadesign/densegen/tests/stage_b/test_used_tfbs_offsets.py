from __future__ import annotations

from dnadesign.densegen.src.core.pipeline.sequence_validation import _apply_pad_offsets
from dnadesign.densegen.src.core.pipeline.usage_tracking import _compute_used_tf_info, _update_usage_summary


class _DummySol:
    def __init__(self, sequence: str, library: list[str], indices: list[int]) -> None:
        self.sequence = sequence
        self.library = library
        self._indices = indices

    def offset_indices_in_order(self):
        return [(0, self._indices[0]), (2, self._indices[1])]


def test_used_tfbs_offsets_shift_with_5prime_padding() -> None:
    sol = _DummySol(sequence="AAAA", library=["TT", "GG"], indices=[0, 1])
    used_tfbs, used_detail, used_counts, used_list = _compute_used_tf_info(
        sol,
        ["TT", "GG"],
        ["TF1", "TF2"],
        None,
        None,
        None,
        None,
        None,
    )
    assert used_tfbs == ["TF1:TT", "TF2:GG"]
    assert used_counts == {"TF1": 1, "TF2": 1}
    assert used_list == ["TF1", "TF2"]

    pad_meta = {"used": True, "bases": 3, "end": "5prime"}
    updated = _apply_pad_offsets(used_detail, pad_meta)
    assert updated[0]["offset_raw"] == 0
    assert updated[0]["offset"] == 3
    assert updated[0]["length"] == 2
    assert updated[0]["end"] == 5
    assert updated[0]["pad_left"] == 3
    assert updated[1]["offset_raw"] == 2
    assert updated[1]["offset"] == 5
    assert updated[1]["length"] == 2
    assert updated[1]["end"] == 7
    assert updated[1]["pad_left"] == 3


def test_update_usage_summary_counts_tf_and_tfbs() -> None:
    usage_counts: dict[tuple[str, str], int] = {}
    tf_usage_counts: dict[str, int] = {}
    used_tfbs_detail = [
        {"tf": "TF1", "tfbs": "AAA"},
        {"tf": "TF1", "tfbs": "AAA"},
        {"tf": "TF2", "tfbs": "CCC"},
    ]

    _update_usage_summary(usage_counts, tf_usage_counts, used_tfbs_detail)

    assert usage_counts == {("TF1", "AAA"): 2, ("TF2", "CCC"): 1}
    assert tf_usage_counts == {"TF1": 2, "TF2": 1}


def test_used_tfbs_detail_includes_stage_a_lineage_fields() -> None:
    sol = _DummySol(sequence="AAAA", library=["TT"], indices=[0, 0])
    _used_tfbs, used_detail, _used_counts, _used_list = _compute_used_tf_info(
        sol,
        ["TT"],
        ["TF1"],
        None,
        None,
        None,
        None,
        None,
        stage_a_best_hit_score_by_index=[7.25],
        stage_a_rank_within_regulator_by_index=[3],
        stage_a_tier_by_index=[1],
        stage_a_fimo_start_by_index=[12],
        stage_a_fimo_stop_by_index=[17],
        stage_a_fimo_strand_by_index=["+"],
        stage_a_selection_rank_by_index=[2],
        stage_a_selection_score_norm_by_index=[0.91],
        stage_a_tfbs_core_by_index=["TT"],
    )
    assert len(used_detail) == 2
    for entry in used_detail:
        assert entry["stage_a_best_hit_score"] == 7.25
        assert entry["stage_a_rank_within_regulator"] == 3
        assert entry["stage_a_tier"] == 1
        assert entry["stage_a_fimo_start"] == 12
        assert entry["stage_a_fimo_stop"] == 17
        assert entry["stage_a_fimo_strand"] == "+"
        assert entry["stage_a_selection_rank"] == 2
        assert entry["stage_a_selection_score_norm"] == 0.91
        assert entry["stage_a_tfbs_core"] == "TT"
