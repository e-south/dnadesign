"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/objectives/__init__.py

Package exports for OPAL objectives.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

# The objective registry discovers built-ins on demand. Importing a pure scoring
# module must not initialize campaign plugins or their optional dependencies.
