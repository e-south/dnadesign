## Source Evidence

Retrieved: 2026-03-12

Update this table whenever external policy claims are added or changed.

| Claim area | Source URL | Retrieved (UTC date) | Volatility | Validation note |
| --- | --- | --- | --- | --- |
| BU SCC documentation index and routing | https://www.bu.edu/tech/support/research/system-usage/ | 2026-02-28 | medium | canonical route page for SCC policy navigation |
| BU SCC connectivity and access routes | https://www.bu.edu/tech/support/research/system-usage/connect-scc/ | 2026-02-28 | medium | OnDemand is the preferred path for many graphical and reconnect workflows |
| BU SCC OnDemand entrypoint and behavior | https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/ | 2026-02-28 | high | recommended web entry; session persists after browser logout |
| BU SCC OnDemand interactive threshold | https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/ | 2026-02-28 | high | limit of 5 interactive jobs for requests >12h and/or added resources |
| BU SCC environment/software baseline | https://www.bu.edu/tech/support/research/system-usage/scc-environment/ | 2026-02-28 | medium | environment and module baseline route |
| BU SCC transfer pathways overview | https://www.bu.edu/tech/support/research/system-usage/transferring-files/ | 2026-02-28 | high | includes OnDemand file manager with 10 GB upload limit |
| BU SCC running-jobs index | https://www.bu.edu/tech/support/research/system-usage/running-jobs/ | 2026-02-28 | medium | lifecycle route to submit, interactive, monitoring, and advanced batch docs |
| BU SCC submitting-jobs directives | https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/ | 2026-02-28 | high | includes `#!/bin/bash -l`, default `h_rt` 12h, arrays up to 75,000 |
| BU SCC interactive-jobs directives | https://www.bu.edu/tech/support/research/system-usage/running-jobs/interactive-jobs/ | 2026-02-28 | high | qsh/qrsh defaults and `qrsh -now n` queue behavior |
| BU SCC best-practices guidance | https://www.bu.edu/tech/support/research/system-usage/running-jobs/best-practices/ | 2026-02-28 | medium | avoid long login-node work, test small before fanout, prefer <=12h, prefer arrays |
| BU SCC parallel submit pressure guidance | https://www.bu.edu/tech/support/research/system-usage/running-jobs/best-practices/ | 2026-02-28 | medium | array jobs are preferred over many individual submissions to reduce scheduler load |
| BU SCC file-system structure and quotas | https://www.bu.edu/tech/support/research/system-usage/using-file-system/file-system-structure/ | 2026-02-28 | high | home quota 10 GB; `/project` and `/projectnb` baseline allocations and `pquota` guidance |
| BU SCC advanced-batch directives | https://www.bu.edu/tech/support/research/system-usage/running-jobs/advanced-batch/ | 2026-02-28 | medium | dependency patterns (`-hold_jid`), arrays (`-t`), `.sge_request`, and env variables |
| BU SCC job tracking and accounting | https://www.bu.edu/tech/support/research/system-usage/running-jobs/tracking-jobs/ | 2026-02-28 | medium | `qstat` state meanings (including `Eqw`) and `qacct` usage |
| BU SCC parallel batch directives | https://www.bu.edu/tech/support/research/system-usage/running-jobs/parallel-batch/ | 2026-02-28 | high | PE options and `OMP_NUM_THREADS=$NSLOTS` alignment guidance |
| BU SCC memory allocation workflow | https://www.bu.edu/tech/support/research/system-usage/running-jobs/allocating-memory-for-your-job/ | 2026-02-28 | high | tune requests using observed `maxvmem` from `qstat -j` and `qacct` |
| BU SCC resource model details | https://www.bu.edu/tech/support/research/system-usage/running-jobs/resources-jobs/ | 2026-02-28 | high | default runtime 12h; memory sharing model; scratch retention up to 31 days |
| BU SCC batch script examples | https://www.bu.edu/tech/support/research/system-usage/running-jobs/batch-script-examples/ | 2026-02-28 | high | practical templates for memory, arrays, PE usage, and module-safe shebang |
| BU SCC process-reaper policy | https://www.bu.edu/tech/support/research/system-usage/running-jobs/process-reaper/ | 2026-02-28 | high | login-node CPU reaper threshold is >15 min and >25% lifetime; idle GPU reaper threshold is 2h; unassigned GPU usage can terminate jobs |
| BU SCC transfer node cloud app detail | https://www.bu.edu/tech/support/research/system-usage/transferring-files/cloud-applications/ | 2026-02-28 | high | transfer node `scc-globus.bu.edu`, `-l download` runtime/interactive constraints |
| Grid Engine submit validation fallback semantics (`-w v`) | https://gridengine.sourceforge.io/SGE/htmlman/htmlman1/submit.html | 2026-02-28 | low | fallback semantics when `-verify` is unavailable |

### Freshness policy

- Volatile entries should be refreshed when older than 45 days.
- If a volatile claim cannot be refreshed, mark it unknown in the execution report and avoid hard-coded thresholds.
