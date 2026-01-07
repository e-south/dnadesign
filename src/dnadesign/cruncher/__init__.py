"""Cruncher package root."""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

# Include the nested src/ tree so imports like dnadesign.cruncher.cli resolve.
__path__ = extend_path(__path__, __name__)
_nested_src = Path(__file__).resolve().parent / "src"
if _nested_src.is_dir():
    __path__.append(str(_nested_src))

del extend_path, Path, _nested_src
