"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/usr/__main__.py

Entrypoint for `python -m dnadesign.usr`.

It forwards to the Typer CLI defined in `dnadesign.usr.src.app:main`.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .src.app import main

if __name__ == "__main__":
    main()
