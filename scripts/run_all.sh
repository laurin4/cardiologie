#!/usr/bin/env bash
set -euo pipefail

# End-to-end run of the clinical extraction framework for a single task.
#
# Usage:
#   scripts/run_all.sh [TASK] [REPORTS_SOURCE]
#
# Defaults:
#   TASK            = demo_extraction
#   REPORTS_SOURCE  = examples/demo_reports  (a CSV file or a folder of .txt)
#
# Environment:
#   PYTHON_BIN      python interpreter (default: ./cardiology_venv/bin/python)
#   LLM_PROVIDER    usz_api (default) | ollama

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-"$ROOT_DIR/cardiology_venv/bin/python"}"

TASK="${1:-demo_extraction}"
REPORTS="${2:-$ROOT_DIR/examples/demo_reports}"

echo "=== Clinical Extraction Framework: full run ==="
echo "Project root: $ROOT_DIR"
echo "Python:       $PYTHON_BIN"
echo "Task:         $TASK"
echo "Reports:      $REPORTS"
echo

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python not executable at $PYTHON_BIN"
  echo "Hint: create a venv or set PYTHON_BIN explicitly."
  exit 1
fi

cd "$ROOT_DIR"

echo "[1/2] Validate input reports"
"$PYTHON_BIN" -m src.validation.validate_inputs --reports "$REPORTS" || true
echo

echo "[2/2] Run extraction pipeline"
"$PYTHON_BIN" -m src.pipeline.pipeline --task "$TASK" --reports "$REPORTS"
echo

echo "Full run completed. Outputs:"
echo "  - outputs/extractions/${TASK}_results.csv"
echo "  - outputs/extractions/${TASK}_results.jsonl"
