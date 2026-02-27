#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="$PWD/config.yaml"
NOTEBOOK="$PWD/outputs/notebooks/densegen_run_overview.py"
source "$SCRIPT_DIR/../_shared/runbook_lib.sh"

densegen_runbook_main \
  --config "$CONFIG" \
  --notebook "$NOTEBOOK" \
  --runner "uv" \
  --ensure-usr-registry "false" \
  --require-fimo "false"
