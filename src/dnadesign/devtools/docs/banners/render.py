"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/docs/banners/render.py

Renders repository and tool banners from their checked source catalog.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from .catalog import BANNERS, REPOSITORY_BANNER_PATH, BannerSpec
from .glyphs import ACCENT, DIM, INK, MUTED, glyph

BACKGROUND = "#1E1D1A"
FONT_STACK = "Menlo, Monaco, Consolas, monospace"


def _resolve_repo_root(repo_root: Path) -> Path:
    root = repo_root.expanduser().resolve()
    pyproject_path = root / "pyproject.toml"
    package_marker = root / "src" / "dnadesign" / "__init__.py"
    if not pyproject_path.is_file() or not package_marker.is_file():
        raise ValueError(f"Not a dnadesign repository root: {root}")
    try:
        project_name = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"]["name"]
    except (KeyError, OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"Not a dnadesign repository root: {root}") from error
    if project_name != "dnadesign":
        raise ValueError(f"Not a dnadesign repository root: {root}")
    return root


def _resolve_output_path(root: Path, relative_path: str) -> Path:
    declared_path = Path(relative_path)
    if declared_path.is_absolute():
        raise ValueError(f"Banner output path escapes repository root: {relative_path}")
    candidate = root
    for component in declared_path.parts:
        candidate /= component
        if candidate.is_symlink():
            raise ValueError(f"Banner output path contains a symlink component: {relative_path}")
    output_path = (root / declared_path).resolve()
    try:
        output_path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Banner output path escapes repository root: {relative_path}") from error
    return output_path


def repository_svg() -> str:
    return f'''<svg width="1280" height="260" viewBox="0 0 1280 260" fill="none"
  xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc"
  shape-rendering="crispEdges">
  <title id="title">dnadesign sequence-design toolkit</title>
  <desc id="desc">A dark, quiet path through sequence design, construction, modeling, and analysis.</desc>
  <rect width="1280" height="260" fill="{BACKGROUND}"/>
  <g font-family="{FONT_STACK}">
    <text x="52" y="58" fill="{MUTED}" font-size="12" letter-spacing="1.4">MODULAR SEQUENCE TOOLKIT</text>
    <text x="48" y="151" fill="{INK}" font-size="72" font-weight="700" letter-spacing="-2">dnadesign</text>
    <rect x="52" y="177" width="40" height="6" fill="{ACCENT}"/>
  </g>
  <path d="M432 46V214" stroke="#3F3C37" stroke-width="2"/>
  <path d="M520 132H1192" stroke="{DIM}" stroke-width="2"/>
  <g font-family="{FONT_STACK}" font-size="12" font-weight="700" letter-spacing="0.6">
    <g transform="translate(520 0)">
      <path d="M0 82H56M0 98H40M0 114H48" stroke="{INK}" stroke-width="6"/>
      <text x="0" y="174" fill="{INK}">DESIGN</text>
    </g>
    <g transform="translate(696 0)">
      <rect x="0" y="88" width="22" height="22" fill="{INK}"/>
      <rect x="30" y="88" width="38" height="22" fill="{ACCENT}"/>
      <rect x="76" y="88" width="20" height="22" fill="{INK}"/>
      <text x="0" y="174" fill="{INK}">ASSEMBLE</text>
    </g>
    <g transform="translate(892 0)">
      <path d="M0 110L22 84L44 104L68 72" stroke="{INK}" stroke-width="5" fill="none"/>
      <rect x="64" y="68" width="8" height="8" fill="{ACCENT}"/>
      <text x="0" y="174" fill="{INK}">MODEL</text>
    </g>
    <g transform="translate(1064 0)">
      <rect x="0" y="104" width="12" height="12" fill="{MUTED}"/>
      <rect x="22" y="90" width="12" height="26" fill="{INK}"/>
      <rect x="44" y="76" width="12" height="40" fill="{ACCENT}"/>
      <text x="0" y="174" fill="{INK}">ANALYZE</text>
    </g>
  </g>
</svg>
'''


def tool_svg(spec: BannerSpec) -> str:
    title_id = f"{spec.name}-banner-title"
    description_id = f"{spec.name}-banner-description"
    return f'''<svg width="1200" height="180" viewBox="0 0 1200 180" fill="none"
  xmlns="http://www.w3.org/2000/svg" role="img"
  aria-labelledby="{title_id} {description_id}" shape-rendering="crispEdges">
  <title id="{title_id}">{spec.name}: {spec.capability.lower()}</title>
  <desc id="{description_id}">{spec.description}</desc>
  <rect width="1200" height="180" fill="{BACKGROUND}"/>
  <g font-family="{FONT_STACK}">
    <text x="48" y="78" fill="{INK}" font-size="42" font-weight="700" letter-spacing="-1">{spec.name}</text>
    <text x="50" y="114" fill="{MUTED}" font-size="12" font-weight="700" letter-spacing="1.2">{spec.capability}</text>
    <rect x="50" y="132" width="32" height="5" fill="{ACCENT}"/>
  </g>
  <path d="M510 34V146" stroke="#3F3C37" stroke-width="2"/>
  <path d="M584 124H1144" stroke="{DIM}" stroke-width="2"/>
  <g transform="translate(720 48)">
    {glyph(spec.glyph)}
  </g>
</svg>
'''


def expected_banners(repo_root: Path) -> dict[Path, str]:
    root = _resolve_repo_root(repo_root)
    declarations = ((REPOSITORY_BANNER_PATH, repository_svg()),) + tuple(
        (spec.path, tool_svg(spec)) for spec in BANNERS
    )
    expected: dict[Path, str] = {}
    for relative_path, content in declarations:
        output_path = _resolve_output_path(root, relative_path)
        if output_path in expected:
            raise ValueError(f"Duplicate banner output path: {relative_path}")
        expected[output_path] = content
    return expected


def render_banners(repo_root: Path) -> tuple[Path, ...]:
    rendered: list[Path] = []
    for path, content in expected_banners(repo_root).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        rendered.append(path)
    return tuple(rendered)


def check_banners(repo_root: Path) -> tuple[Path, ...]:
    root = _resolve_repo_root(repo_root)
    return tuple(
        path.relative_to(root)
        for path, content in expected_banners(root).items()
        if not path.exists() or path.read_text(encoding="utf-8") != content
    )
