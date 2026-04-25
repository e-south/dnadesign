from matplotlib.transforms import Bbox

from dnadesign.latentdna.src.annotation_layout import choose_annotation_placement


def _boxes_overlap(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> bool:
    return not (left[2] <= right[0] or right[2] <= left[0] or left[3] <= right[1] or right[3] <= left[1])


def test_choose_annotation_placement_avoids_existing_boxes() -> None:
    axes_box = Bbox.from_bounds(0.0, 0.0, 240.0, 180.0)
    first = choose_annotation_placement(
        display_x=120.0,
        display_y=90.0,
        label_text="spyP",
        axes_box=axes_box,
        placed_boxes=[],
        x_mid=120.0,
        y_mid=90.0,
        font_size=9.5,
        left_padding_px=8.0,
        right_padding_px=8.0,
    )
    second = choose_annotation_placement(
        display_x=126.0,
        display_y=94.0,
        label_text="J23105",
        axes_box=axes_box,
        placed_boxes=[first.box],
        x_mid=120.0,
        y_mid=90.0,
        font_size=9.5,
        left_padding_px=8.0,
        right_padding_px=8.0,
    )

    assert not _boxes_overlap(first.box, second.box)
    assert second.offset_x != first.offset_x or second.offset_y != first.offset_y


def test_choose_annotation_placement_clamps_inside_axes_box() -> None:
    axes_box = Bbox.from_bounds(0.0, 0.0, 140.0, 90.0)
    placement = choose_annotation_placement(
        display_x=132.0,
        display_y=82.0,
        label_text="sulAp",
        axes_box=axes_box,
        placed_boxes=[],
        x_mid=70.0,
        y_mid=45.0,
        font_size=9.5,
        left_padding_px=10.0,
        right_padding_px=10.0,
        top_padding_px=6.0,
        bottom_padding_px=6.0,
    )

    left, bottom, right, top = placement.box
    assert left >= 10.0
    assert right <= 130.0
    assert bottom >= 6.0
    assert top <= 84.0
