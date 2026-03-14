construct creates derived DNA constructs from anchor sequences, template context, and explicit placement specs, then writes the realized sequences into USR datasets with fail-fast lineage metadata.

## Documentation map

Read in this order:

1. [construct docs overview](docs/README.md)
2. [construct docs index](docs/index.md)
3. [workspaces guide](workspaces/README.md)
4. [developer notes](docs/dev/README.md)
5. [development journal](docs/dev/journal.md)
6. [repository docs index](../../../docs/README.md)

## Entrypoint contract

1. Audience: construct operators and maintainers.
2. Primary entrypoints: `uv run construct ...` and `python -m dnadesign.construct ...`.
3. Prerequisites: valid config YAML, readable template input, and readable/writable USR roots.
4. Verify next: `uv run construct validate config --config <path> --runtime`.

## Boundary reminder

- USR owns the resulting sequence records and construct lineage overlays.
- construct owns realization logic: template loading, part placement, focal-window extraction, and output dataset writes.
- Placement semantics are explicit: `kind=insert` or `kind=replace`, with optional incumbent-sequence checks for replacements.
- infer consumes construct outputs as ordinary USR sequences; it does not own construct assembly semantics.
