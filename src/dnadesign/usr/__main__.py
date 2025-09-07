"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/usr/src/__main__.py

Entrypoint for `python -m dnadesign.usr`.

It simply forwards to the CLI defined in `dnadesign.usr.src.cli:main`.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .src.cli import main

if __name__ == "__main__":
    main()
