"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/__init__.py

Public distribution version for the DNADesign toolkit.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""


def __getattr__(name: str) -> str:
    if name != "__version__":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib.metadata import version

    return version("dnadesign")
