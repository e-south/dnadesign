"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/docs/banners/render.py

Renders repository and tool banners from their checked source catalog.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from .catalog import BANNERS, REPOSITORY_BANNER_PATH, BannerSpec
from .glyphs import INK, MUTED, glyph

BACKGROUND = "#F1F4F6"
FONT_STACK = "Helvetica Neue, Helvetica, Arial, sans-serif"
BANNER_NAME_PATTERN = re.compile(r"[a-z0-9]+(?:[a-z0-9_-]*[a-z0-9])?")


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
    if project_name != "dnadesign-tools":
        raise ValueError(f"Not a dnadesign repository root: {root}")
    return root


def _canonical_relative_path(value: str, *, label: str) -> Path:
    if not value or value.startswith("/") or "\\" in value:
        raise ValueError(f"{label} must be a canonical repository-relative POSIX path: {value!r}")
    components = value.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError(f"{label} must be a canonical repository-relative POSIX path: {value!r}")
    return Path(*components)


def _resolve_repo_path(
    root: Path,
    relative_path: Path,
    *,
    label: str,
    must_exist: bool,
    reject_hardlinks: bool = False,
) -> Path:
    candidate = root
    components = relative_path.parts
    for index, component in enumerate(components):
        candidate /= component
        if candidate.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {relative_path.as_posix()}")
        if not candidate.exists():
            continue
        if index < len(components) - 1 and not candidate.is_dir():
            raise ValueError(f"{label} parent component is not a directory: {relative_path.as_posix()}")
        if index == len(components) - 1 and not candidate.is_file():
            raise ValueError(f"{label} is not a regular file: {relative_path.as_posix()}")
        if index == len(components) - 1 and reject_hardlinks and candidate.stat(follow_symlinks=False).st_nlink != 1:
            raise ValueError(f"{label} has multiple hard links: {relative_path.as_posix()}")
    resolved_path = (root / relative_path).resolve()
    try:
        resolved_path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes repository root: {relative_path.as_posix()}") from error
    if must_exist and not resolved_path.is_file():
        raise ValueError(f"{label} does not exist as a regular file: {relative_path.as_posix()}")
    return resolved_path


def _preflight_banner_binding(
    root: Path,
    *,
    name: str,
    output_value: str,
    readme_value: str,
) -> tuple[Path, Path]:
    if BANNER_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(f"Banner name must be a lowercase slug: {name!r}")

    output_relative = _canonical_relative_path(output_value, label="Banner output path")
    readme_relative = _canonical_relative_path(readme_value, label="Banner README path")
    expected_leaf = f"{name}-banner.svg"
    if output_relative.name != expected_leaf:
        raise ValueError(f"Banner output path for {name!r} must end with {expected_leaf!r}")
    if readme_relative.name != "README.md":
        raise ValueError(f"Banner README path for {name!r} must end with 'README.md'")

    owner_root = readme_relative.parent
    try:
        owner_relative = output_relative.relative_to(owner_root)
    except ValueError as error:
        raise ValueError(
            f"Banner output path for {name!r} must be under its owning README directory: {owner_root.as_posix()}"
        ) from error
    if "assets" not in owner_relative.parts[:-1]:
        raise ValueError(f"Banner output path for {name!r} must be in an assets directory")

    readme_path = _resolve_repo_path(root, readme_relative, label="Banner README path", must_exist=True)
    output_path = _resolve_repo_path(
        root,
        output_relative,
        label="Banner output",
        must_exist=False,
        reject_hardlinks=True,
    )
    return output_path, readme_path


def _preflight_banner_catalog(root: Path) -> tuple[tuple[Path, BannerSpec | None], ...]:
    declarations: list[tuple[Path, BannerSpec | None]] = []
    seen_names: set[str] = set()
    seen_outputs: set[Path] = set()
    seen_readmes: set[Path] = set()

    bindings = (("dnadesign", REPOSITORY_BANNER_PATH, "README.md", None),) + tuple(
        (spec.name, spec.path, spec.readme_path, spec) for spec in BANNERS
    )
    for name, output_value, readme_value, spec in bindings:
        output_path, readme_path = _preflight_banner_binding(
            root,
            name=name,
            output_value=output_value,
            readme_value=readme_value,
        )
        if name in seen_names:
            raise ValueError(f"Duplicate banner name: {name}")
        if output_path in seen_outputs:
            raise ValueError(f"Duplicate banner output path: {output_value}")
        if readme_path in seen_readmes:
            raise ValueError(f"Duplicate banner README path: {readme_value}")
        seen_names.add(name)
        seen_outputs.add(output_path)
        seen_readmes.add(readme_path)
        declarations.append((output_path, spec))
    return tuple(declarations)


def repository_svg() -> str:
    stages = "".join(
        f'<g transform="translate({x - 51.2} 47) scale(.8)">{glyph(mark)}</g>'
        f'<text x="{x}" y="144" text-anchor="middle" fill="{MUTED}" font-size="18">{label}</text>'
        for x, label, mark in zip(
            (602, 774, 946, 1118),
            ("Design", "Construct", "Model", "Analyze"),
            ("generate", "construct", "infer", "latent"),
            strict=True,
        )
    )
    return f'''<svg width="1280" height="200" viewBox="0 0 1280 200" fill="none"
  xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc"
  shape-rendering="geometricPrecision">
  <title id="title">dnadesign sequence-design toolkit</title>
  <desc id="desc">Sequence libraries, assembled parts, model features, and comparisons.</desc>
  <rect width="1280" height="200" rx="18" fill="{BACKGROUND}"/>
  <g font-family="{FONT_STACK}">
    <text x="48" y="101" fill="{INK}" font-size="64" font-weight="500" letter-spacing="-1.1">dnadesign</text>
    <text x="51" y="140" fill="{MUTED}" font-size="20">Modular sequence toolkit</text>
    <path d="M486 44V156" stroke="#CDD6DE" stroke-width="1.2"/>
    {stages}
  </g>
</svg>
'''


def tool_svg(spec: BannerSpec) -> str:
    title_id = f"{spec.name}-banner-title"
    description_id = f"{spec.name}-banner-description"
    capability = spec.capability.capitalize().replace("rna", "RNA").replace("dna", "DNA")
    return f'''<svg width="1200" height="180" viewBox="0 0 1200 180" fill="none"
  xmlns="http://www.w3.org/2000/svg" role="img"
  aria-labelledby="{title_id} {description_id}" shape-rendering="geometricPrecision">
  <title id="{title_id}">{spec.name}: {capability}</title>
  <desc id="{description_id}">{spec.description}</desc>
  <rect width="1200" height="180" rx="16" fill="{BACKGROUND}"/>
  <g font-family="{FONT_STACK}">
    <text x="48" y="89" fill="{INK}" font-size="56" font-weight="500" letter-spacing="-1.1">{spec.name}</text>
    <text x="50" y="126" fill="{MUTED}" font-size="20">{capability}</text>
  </g>
  <path d="M622 43V137" stroke="#CDD6DE" stroke-width="1.2"/>
  <g transform="translate(808.8 32.4) scale(1.8)">
    {glyph(spec.glyph)}
  </g>
</svg>
'''


def expected_banners(repo_root: Path) -> dict[Path, str]:
    root = _resolve_repo_root(repo_root)
    declarations = _preflight_banner_catalog(root)
    return {output_path: repository_svg() if spec is None else tool_svg(spec) for output_path, spec in declarations}


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
