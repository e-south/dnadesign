"""CLI entrypoint for Eco1 conservation roster-cache materialization."""

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.roster_cache.pipeline import (
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
