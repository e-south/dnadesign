"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/docs/banners/glyphs.py

Builds the restrained SVG glyphs used by documentation banners.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

INK = "#F3EFE7"
MUTED = "#969087"
DIM = "#5A5650"
ACCENT = "#D97757"


def _rect(x: int, y: int, width: int, height: int, fill: str = INK) -> str:
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}"/>'


def _line(path: str, stroke: str = INK, width: int = 4, dash: str | None = None) -> str:
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<path d="{path}" stroke="{stroke}" stroke-width="{width}" fill="none"{dashed}/>'


def glyph(name: str) -> str:
    """Return one deliberately small, crisp capability mark."""
    marks = {
        "align": _line("M0 18H72M0 42H72", MUTED, 3) + _rect(12, 13, 20, 10) + _rect(38, 37, 20, 10, ACCENT),
        "render": _line("M0 8H66V52H0Z", MUTED, 3) + _line("M12 20H54M12 30H42M12 40H48", INK, 4),
        "diversity": "".join(
            _rect(index * 18, 46 - height, 10, height, color)
            for index, (height, color) in enumerate(((14, MUTED), (30, INK), (20, DIM), (42, ACCENT), (25, INK)))
        ),
        "cluster": (
            '<circle cx="15" cy="18" r="7" fill="#F3EFE7"/>'
            '<circle cx="34" cy="12" r="5" fill="#969087"/>'
            '<circle cx="26" cy="34" r="6" fill="#D97757"/>'
            '<circle cx="68" cy="38" r="7" fill="#F3EFE7"/>'
            '<circle cx="84" cy="27" r="5" fill="#5A5650"/>'
        ),
        "construct": _rect(0, 21, 26, 16, MUTED)
        + _rect(32, 21, 42, 16, ACCENT)
        + _rect(80, 21, 24, 16)
        + _line("M26 29H32M74 29H80", DIM, 3),
        "contracts": _line("M0 12H40V46H0ZM62 12H102V46H62Z", MUTED, 3) + _line("M34 29H68", ACCENT, 5),
        "optimize": _line("M0 8H102L72 30V50H30V30Z", MUTED, 3) + _rect(45, 39, 12, 12, ACCENT),
        "generate": _line("M0 29H24M24 29L48 10M24 29L48 29M24 29L48 48", MUTED, 3)
        + _line("M48 10H92M48 29H102M48 48H82", INK, 4),
        "test": _line("M0 12H42V50H0ZM60 12H102V50H60Z", MUTED, 3)
        + _line("M9 31L18 40L34 20M69 31L78 40L94 20", ACCENT, 4),
        "fold": _line("M0 45C14 45 14 10 32 10C50 10 50 45 66 45C82 45 82 20 98 20", INK, 4)
        + _rect(29, 7, 6, 6, ACCENT),
        "infer": _line("M0 28H30M30 28L48 10M30 28H54M30 28L48 46", MUTED, 3)
        + _rect(62, 9, 34, 8)
        + _rect(62, 24, 48, 8, ACCENT)
        + _rect(62, 39, 26, 8, MUTED),
        "latent": _line("M0 49L28 9L55 49", DIM, 3)
        + '<circle cx="13" cy="34" r="5" fill="#F3EFE7"/>'
        + '<circle cx="33" cy="27" r="5" fill="#D97757"/>'
        + '<circle cx="47" cy="41" r="4" fill="#969087"/>'
        + _line("M70 49L98 9L125 49", DIM, 3),
        "shuffle": _line("M0 12H24C48 12 48 46 72 46H104", MUTED, 3)
        + _line("M0 46H24C48 46 48 12 72 12H104", ACCENT, 3),
        "factor": _rect(0, 8, 44, 44, MUTED)
        + _rect(8, 16, 10, 10, INK)
        + _rect(26, 34, 10, 10, ACCENT)
        + _line("M56 30H72", DIM, 3)
        + _rect(82, 8, 10, 44, INK)
        + _rect(102, 8, 10, 44, ACCENT),
        "notify": _line("M0 33H18L28 12L42 48L54 24L68 33H106", INK, 4) + _rect(102, 29, 8, 8, ACCENT),
        "select": _line("M0 45L25 34L48 38L74 17L108 8", MUTED, 3)
        + '<circle cx="74" cy="17" r="7" fill="#D97757"/>'
        + _line("M98 1L108 8L98 15", INK, 3),
        "route": _line("M0 28H26M26 28L50 10M26 28L50 46M50 10H102M50 46H84", INK, 3) + _rect(96, 6, 10, 8, ACCENT),
        "permute": _line("M0 14H104M0 30H104M0 46H104", DIM, 3)
        + _rect(22, 9, 18, 10)
        + _rect(48, 25, 18, 10, ACCENT)
        + _rect(74, 41, 18, 10, MUTED),
        "study": _line("M0 10H42V48H0ZM58 10H100V48H58Z", MUTED, 3)
        + _rect(8, 19, 26, 5)
        + _rect(66, 19, 26, 5, ACCENT)
        + _rect(8, 31, 18, 5, DIM)
        + _rect(66, 31, 22, 5),
        "knockdown": _rect(0, 8, 10, 42)
        + _rect(22, 19, 10, 31, MUTED)
        + _rect(44, 30, 10, 20, ACCENT)
        + _line("M70 8V50M62 42L70 50L78 42", INK, 4),
        "thread": _line("M0 16C20 16 20 44 40 44S60 16 80 16S100 44 120 44", INK, 4)
        + _rect(37, 39, 7, 10, ACCENT)
        + _rect(77, 11, 7, 10, MUTED),
        "junction": _line("M0 28H58V2M70 2V28H128", INK, 4)
        + _line("M0 42H50M58 42H128", MUTED, 3)
        + _line("M50 28H58M58 42H66", ACCENT, 4)
        + _line("M58 2V28M70 2V28", "#7BB4AE", 4)
        + _line("M58 8H70M58 15H70M58 22H70", DIM, 2)
        + _line("M14 28V42M34 28V42M86 28V42M106 28V42", DIM, 2)
        + _line("M50 36L58 48", INK, 2),
        "records": _line("M0 8H88V22H0ZM0 27H104V41H0ZM0 46H76V60H0Z", MUTED, 3) + _rect(92, 10, 8, 8, ACCENT),
    }
    try:
        return marks[name]
    except KeyError as error:
        raise ValueError(f"Unknown banner glyph: {name}") from error
