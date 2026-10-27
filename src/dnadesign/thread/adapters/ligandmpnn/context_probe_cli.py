"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/ligandmpnn/context_probe_cli.py

Headless CLI entrypoint for the pinned LigandMPNN context probe.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.thread.adapters.ligandmpnn.context_probe import _main

if __name__ == "__main__":
    raise SystemExit(_main())
