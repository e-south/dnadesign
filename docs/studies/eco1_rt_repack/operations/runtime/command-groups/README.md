## Eco1 RT Repack Command Groups

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-19

This is a placeholder route for future machine-readable command groups. No
commands are executable yet.

Use `pipeline.yaml` only as a planning scaffold until `thread` validators or
explicit `infer` execution surfaces are implemented.

The planned lanes mirror the artifact chain: structure authority, residue map,
evidence profiles, mask contract, sampling plan, sample ingest, candidate
table, fold-check QA, assembly feasibility, candidate handoff, and RT-lnRNA
handoff.

Keep these lanes independently runnable when executable commands are
introduced. A future orchestration command may call them in order only after
each lane has its own validator and negative-path fixture. Do not collapse them
into a single hidden pipeline.
