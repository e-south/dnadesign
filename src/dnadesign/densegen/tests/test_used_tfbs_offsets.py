from __future__ import annotations

from dnadesign.densegen.src.core.pipeline import _apply_gap_fill_offsets, _compute_used_tf_info


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

    gap_meta = {"used": True, "bases": 3, "end": "5prime"}
    updated = _apply_gap_fill_offsets(used_detail, gap_meta)
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
