"""
Cruncher implementation package.

This subpackage exists so setuptools includes cruncher modules (cli, workflows,
services, etc.) during packaging. dnadesign.cruncher extends __path__ so these
modules are also importable as dnadesign.cruncher.<module>.
"""
