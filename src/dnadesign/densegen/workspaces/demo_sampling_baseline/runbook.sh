#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="$PWD/config.yaml"
NOTEBOOK="$PWD/outputs/notebooks/densegen_run_overview.py"
MODE="${DENSEGEN_RUNBOOK_MODE:-fresh}"
source "$SCRIPT_DIR/../_shared/workspace_runbook_flow.sh"

densegen_workspace_runbook_flow \
  --config "$CONFIG" \
  --notebook "$NOTEBOOK" \
  --mode "$MODE" \
  --runner "pixi" \
  --ensure-usr-registry "true" \
  --require-fimo "true" \
  "$@"
