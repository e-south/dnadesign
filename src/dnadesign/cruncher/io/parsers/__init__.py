"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cruncher/io/parsers/__init__.py

Built-in PWM parsers.

Importing this package registers built-in parsers via side effects
(@register decorators in each parser module).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

# Import modules for side-effect registration.
from . import meme  # noqa: F401
