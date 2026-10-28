"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/docs/banners/glyphs.py

Builds the restrained SVG glyphs used by documentation banners.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

INK = "#163047"
MUTED = "#526475"
DIM = "#A5B3BE"
ACCENT = "#2F6EAD"


def _rect(x: float, y: float, width: float, height: float, fill: str = INK) -> str:
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="1" fill="{fill}"/>'


def _line(path: str, stroke: str = INK, width: float = 2) -> str:
    return (
        f'<path d="{path}" stroke="{stroke}" stroke-width="{width}" fill="none" '
        'stroke-linejoin="round" stroke-linecap="round"/>'
    )


def _dot(x: float, y: float, color: str = INK, radius: float = 3) -> str:
    return f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}"/>'


def _sequence_rows(*, aligned: bool = False, variants: bool = False) -> str:
    rows = []
    for row, y in enumerate((12, 29, 46)):
        for col in range(7):
            x = 7 + col * 17
            if aligned and (row, col) in {(0, 3), (2, 1)}:
                rows.append(_line(f"M{x} {y + 4}h11", DIM))
            else:
                color = ACCENT if (col == 3 if variants else col in (2, 3, 5)) else MUTED
                rows.append(_rect(x, y, 11, 8, color))
    return "".join(rows)


def _feature_matrix() -> str:
    return "".join(
        _rect(11 + col * 19, 7 + row * 17, 13, 11, (DIM, MUTED, ACCENT)[(row * 3 + col * 2 + col // 2) % 3])
        for row in range(3)
        for col in range(6)
    )


def glyph(name: str) -> str:
    """Render a capability mark in a shared 128-by-64 coordinate system."""
    selected = {(9, 13), (29, 31), (49, 49)}
    marks = {
        "align": _sequence_rows(aligned=True),
        "render": _line("M7 20H121M7 46H121", DIM)
        + '<path d="M15 13H42L50 20L42 27H15Z" fill="'
        + INK
        + '"/>'
        + '<path d="M103 39H76L68 46L76 53H103Z" fill="'
        + ACCENT
        + '"/>',
        "cluster": "".join(
            _dot(x, y, color)
            for color, points in (
                (INK, ((15, 13), (28, 11), (19, 27), (33, 24))),
                (ACCENT, ((79, 16), (93, 10), (98, 26), (84, 30))),
                (MUTED, ((49, 45), (62, 50), (72, 43))),
            )
            for x, y in points
        ),
        "construct": _rect(4, 23, 29, 18, MUTED)
        + _rect(39, 23, 49, 18, ACCENT)
        + _rect(94, 23, 30, 18)
        + _line("M33 32H39M88 32H94", DIM),
        "contracts": _line("M5 7H51V57H5ZM77 7H123V57H77Z", MUTED)
        + _line("M13 18H42M13 29H36M13 40H42M85 18H114M85 29H108M85 40H114", DIM)
        + _line("M46 32H82", ACCENT, 3),
        "optimize": _sequence_rows(variants=True) + _line("M3 5H126V59H3Z", DIM, 1),
        "generate": "".join(
            _rect(7 + sum(widths[:i]) + i * 4, 10 + row * 18, width, 9, (INK, ACCENT, MUTED)[(i + row) % 3])
            for row, widths in enumerate(((23, 46, 34), (38, 23, 42), (46, 34, 23)))
            for i, width in enumerate(widths)
        ),
        "test": _line("M8 7H51V57H8ZM77 7H120V57H77Z", DIM) + _line("M18 33L27 42L42 23M87 33L96 42L111 23", ACCENT, 3),
        "fold": _line("M24 57H48V27C33 21 33 5 48 4H80C95 5 95 21 80 27V57H104", INK, 3)
        + _line("M48 32H80M48 41H80M48 50H80", ACCENT, 2),
        "infer": _feature_matrix(),
        "latent": _line("M9 5V57H122", DIM, 1.5)
        + "".join(
            _dot(x, y, ACCENT if i % 3 == 0 else MUTED)
            for i, (x, y) in enumerate(
                ((24, 39), (37, 31), (44, 45), (59, 27), (68, 38), (74, 18), (89, 29), (105, 14), (112, 23))
            )
        ),
        "notify": _line("M7 12H99V43H30L17 54V43H7Z", MUTED, 2.5)
        + _line("M20 24H71M20 33H56", DIM)
        + _dot(108, 17, ACCENT, 8),
        "select": "".join(
            _dot(x, y, ACCENT if (x, y) in selected else DIM, 4) for x in (9, 29, 49) for y in (13, 31, 49)
        )
        + _line("M67 31H91M85 25L91 31L85 37", MUTED)
        + "".join(_dot(113, y, ACCENT, 5) for y in (13, 31, 49)),
        "route": "".join(
            _rect(8, y, 88, 9, DIM) + _dot(115, y + 4.5, color, 4) for y, color in ((10, INK), (28, ACCENT), (46, DIM))
        ),
        "permute": _sequence_rows(variants=True),
        "thread": _line("M10 43L32 15L53 47L76 15L98 46L119 24", DIM, 3)
        + "".join(
            _dot(x, y, ACCENT if i in (1, 3) else INK, 5)
            for i, (x, y) in enumerate(((10, 43), (32, 15), (53, 47), (76, 15), (98, 46), (119, 24)))
        ),
        "junction": _line("M5 29H58V5M70 5V29H123M5 43H50M58 43H123", INK, 3)
        + _line("M58 5V29M70 5V29", ACCENT, 3)
        + _line("M58 10H70M58 18H70M16 29V43M35 29V43M88 29V43M108 29V43", DIM)
        + _line("M50 37L58 49", ACCENT),
        "records": _line("M7 7H121V57H7ZM7 23H121M7 40H121M36 7V57", MUTED)
        + _rect(45, 12, 44, 6, ACCENT)
        + _rect(45, 29, 63, 6, DIM)
        + _rect(45, 46, 53, 6, DIM),
        "diversity": _line("M9 7V57H121", DIM, 1.5)
        + "".join(
            _rect(x, 57 - h, 12, h, color)
            for x, h, color in ((22, 17, MUTED), (46, 35, INK), (70, 25, ACCENT), (94, 43, INK))
        ),
        "transform": _line("M7 7H121V57H7ZM7 23H121M7 40H121M36 7V57M77 7V57", DIM)
        + "".join(_rect(83, y, 30, 7, ACCENT) for y in (11, 28, 45)),
        "trace": _line("M23 32H106", DIM)
        + "".join(_rect(x, 18, 22, 28, color) for x, color in ((5, MUTED), (53, ACCENT), (101, INK))),
    }
    try:
        return marks[name]
    except KeyError as error:
        raise ValueError(f"Unknown banner glyph: {name}") from error
