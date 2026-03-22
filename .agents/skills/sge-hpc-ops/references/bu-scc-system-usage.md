## BU SCC System Usage Map

Use this reference when BU SCC policy claims are part of the task.

### Core route pages

| Topic | URL | Use for |
| --- | --- | --- |
| BU system usage index | https://www.bu.edu/tech/support/research/system-usage/ | top-level route to all SCC policy pages |
| Connect to SCC | https://www.bu.edu/tech/support/research/system-usage/connect-scc/ | login path selection and host access routes |
| SCC OnDemand | https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/ | browser-based interactive sessions and OnDemand-specific limits |
| SCC environment | https://www.bu.edu/tech/support/research/system-usage/scc-environment/ | software/module and environment baseline guidance |
| Transferring files | https://www.bu.edu/tech/support/research/system-usage/transferring-files/ | transfer-node and data movement guidance |
| Running jobs | https://www.bu.edu/tech/support/research/system-usage/running-jobs/ | scheduler command surface and job lifecycle vocabulary |

### Running-jobs deep links

| Topic | URL | Use for |
| --- | --- | --- |
| Best practices | https://www.bu.edu/tech/support/research/system-usage/running-jobs/best-practices/ | shared-cluster etiquette, runtime and array guidance |
| File system structure | https://www.bu.edu/tech/support/research/system-usage/using-file-system/file-system-structure/ | `/project` vs `/projectnb` placement and quota checks |
| Advanced batch | https://www.bu.edu/tech/support/research/system-usage/running-jobs/advanced-batch/ | dependencies (`-hold_jid`), arrays, `.sge_request`, job env |
| Tracking jobs | https://www.bu.edu/tech/support/research/system-usage/running-jobs/tracking-jobs/ | `qstat`, `qdel`, `qhold`, `qrls`, `qacct` interpretation |
| Parallel batch | https://www.bu.edu/tech/support/research/system-usage/running-jobs/parallel-batch/ | PE selection and thread-slot alignment patterns |
| Allocating memory | https://www.bu.edu/tech/support/research/system-usage/running-jobs/allocating-memory-for-your-job/ | `maxvmem` evidence path for `mem_per_core` sizing |
| Resources for jobs | https://www.bu.edu/tech/support/research/system-usage/running-jobs/resources-jobs/ | runtime defaults, memory model, scratch behavior |
| Batch script examples | https://www.bu.edu/tech/support/research/system-usage/running-jobs/batch-script-examples/ | submit-ready patterns for CPU, memory, array, MPI, GPU |
| Process reaper | https://www.bu.edu/tech/support/research/system-usage/running-jobs/process-reaper/ | enforced termination conditions and recovery triage |

### High-volatility claim map

| Claim area | Primary URL | Volatility | Current guidance |
| --- | --- | --- | --- |
| OnDemand route and GUI recommendation | https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/ | medium | OnDemand is the recommended SCC web access path for interactive GUI workflows |
| OnDemand session behavior after logout | https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/ | high | logging out of browser does not end running interactive sessions |
| OnDemand interactive policy threshold | https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/ | high | limit of 5 interactive jobs for requests over 12h and/or added resources |
| Batch submit defaults and directives | https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/ | high | `#!/bin/bash -l`, default `h_rt` 12h, array jobs up to 75,000 |
| Runtime optimization guidance | https://www.bu.edu/tech/support/research/system-usage/running-jobs/best-practices/ | medium | prefer `h_rt` 12h or less when possible; run small tests before large fanout |
| Parallel fanout pressure guidance | https://www.bu.edu/tech/support/research/system-usage/running-jobs/best-practices/ | medium | use arrays instead of many individual submits to reduce scheduler load |
| Filesystem quotas and placement | https://www.bu.edu/tech/support/research/system-usage/using-file-system/file-system-structure/ | high | home quota 10 GB; production work should favor `/project` or `/projectnb` |
| Array and dependency mechanics | https://www.bu.edu/tech/support/research/system-usage/running-jobs/advanced-batch/ | medium | use `-hold_jid` for sequencing and `-t` plus `SGE_TASK_ID` for arrays |
| Tracking lifecycle states | https://www.bu.edu/tech/support/research/system-usage/running-jobs/tracking-jobs/ | medium | rely on `qstat` states and `qacct` for post-run diagnostics |
| Parallel PE and thread alignment | https://www.bu.edu/tech/support/research/system-usage/running-jobs/parallel-batch/ | high | request matching PE slots and align thread count with `NSLOTS` |
| Memory sizing workflow | https://www.bu.edu/tech/support/research/system-usage/running-jobs/allocating-memory-for-your-job/ | high | tune `mem_per_core` using observed `maxvmem` from `qstat -j` and `qacct` |
| Resource model and scratch retention | https://www.bu.edu/tech/support/research/system-usage/running-jobs/resources-jobs/ | high | runtime default 12h; scratch data can be removed after 31 days |
| Transfer pathways and limits | https://www.bu.edu/tech/support/research/system-usage/transferring-files/ | high | OnDemand file manager upload limit is 10 GB |
| Transfer node behavior | https://www.bu.edu/tech/support/research/system-usage/transferring-files/cloud-applications/ | high | transfer node is `scc-globus.bu.edu`; `-l download` is batch-oriented (max 24h, one core) |
| Process reaper thresholds | https://www.bu.edu/tech/support/research/system-usage/running-jobs/process-reaper/ | high | login-node CPU reaper threshold is >15 min and >25% lifetime; idle GPU reaper threshold is 2h; unassigned GPU usage can terminate jobs |

### Routing rules

- Start from the system usage index when context is unclear.
- For connectivity questions, consult connect-scc and OnDemand pages first.
- For submit and monitor behavior, consult running-jobs pages first.
- For dependency, array, or advanced qsub options, route to advanced-batch before proposing flags.
- For memory tuning, route to allocating-memory and resources-jobs before setting `mem_per_core` claims.
- For output placement and quota issues, route to file-system-structure and require `groups` plus `pquota` checks.
- For process-reaper incidents, route to process-reaper plus tracking-jobs before proposing retries.
- For transfer workflows, consult transferring-files pages before suggesting download-node jobs.
- Record URL and retrieval date for each claim in `source-evidence.md`.

### Freshness rules

- Treat numeric limits, hostnames, queue/runtime defaults, process-reaper thresholds, and storage quotas as volatile claims.
- Refresh volatile claims when source evidence is older than 45 days.
- If refresh cannot be completed in-session, mark the claim unknown and avoid hard-coded limits.
