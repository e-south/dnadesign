## Documentation Templates

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-21

### At a glance
Templates provide consistent structure for system-of-record docs, runbooks, ADRs, and execution plans.

### Contents
- [System-of-record template](system-of-record.md)
- [Runbook template](runbook.md)
- [ADR template](adr.md)
- [Execution plan template](exec-plan.md)
- [Promoter study index template](promoter-study-index.yaml)
- [Promoter study datasets template](promoter-study-datasets.yaml)
- [Promoter study status template](promoter-study-status.md)
- [Promoter study OPS contract template](promoter-study-ops.study.yaml)
- [Cruncher study datasets template](cruncher-study-datasets.yaml)
- [Cruncher study status template](cruncher-study-status.md)
- [Cruncher study routes template](cruncher-study-routes.md)
- [Cruncher study pipeline template](cruncher-study-pipeline.yaml)
- [Cruncher study OPS contract template](cruncher-study-ops.study.yaml)

The promoter-study templates define the checked-in record plane. The matching
study status adapter code lives under `src/dnadesign/studies/`, not inside OPS
core.

The Cruncher-study templates do the same for command-centric tracked studies
that need one checked-in route map, one pipeline context, and one native-agent
bootstrap surface in addition to the generic study record files.
