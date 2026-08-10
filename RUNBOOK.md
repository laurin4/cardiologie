# Runbook — Clinical Extraction Framework

Project root: `clinical_extraction_framework/`.

## Setup

1. Python 3.9+ and a virtualenv (recommended).
2. `pip install -r requirements.txt`
3. Provide report inputs (never commit sensitive data):
   - a CSV under `data/raw/` with id + text columns, or
   - a directory of `.txt` files.
4. Start the LLM backend so the provider URL is reachable.

## Environment — LLM (USZ default)

```bash
export LLM_PROVIDER=usz_api
export USZ_LLM_URL=http://localhost:8100/generate
export LLM_MODEL_LABEL=gemma4_31B
export LLM_TEMPERATURE=0
export LLM_TOP_P=1
export LLM_MAX_TOKENS=1000
export LLM_TIMEOUT=300
export LLM_MAX_RETRIES=2
```

Optional Ollama comparison:

```bash
export LLM_PROVIDER=ollama
export OLLAMA_URL=http://127.0.0.1:11500
export OLLAMA_MODEL=qwen2.5:7b
```

## Evidence extraction tuning (optional)

```bash
export EVIDENCE_MAX_SNIPPETS=12
export EVIDENCE_MAX_LLM_CHARS=8000
export EVIDENCE_WINDOW_SENTENCES=1
export EVIDENCE_MAX_SNIPPET_CHARS=400
export DEBUG_LLM_OUTPUT=false
```

## Smoke test the LLM endpoint

```bash
python scripts/test_usz_llm_api.py
```

## Run

```bash
# Preflight: tasks import + input validation
scripts/preflight_check.sh examples/demo_reports

# Full run for a task
python -m src.pipeline.pipeline --task demo_extraction --reports examples/demo_reports
# or
scripts/run_all.sh <task> <reports_source>
```

Run controls:

- `--max-reports N` (or `MAX_REPORTS=N` / `all`) caps the number of reports.
- `PIPELINE_RESUME=true` resumes from a checkpoint CSV in the output dir.
- `ENABLE_SQLITE_LOGGING=true` also writes rows to `outputs/logs/extractions.sqlite`.

## Outputs

| Artifact | Location |
|----------|----------|
| Flattened results (CSV) | `outputs/extractions/<task>_results.csv` |
| Structured objects (JSONL) | `outputs/extractions/<task>_results.jsonl` |
| LLM debug dumps | `outputs/logs/llm_debug/` |
| Optional SQLite log | `outputs/logs/extractions.sqlite` |

## Troubleshooting

- **No report input found** — pass `--reports <csv|dir>` or place files under `data/raw/`.
- **LLM unavailable** — extraction rows get `status=failed` and a debug JSON under
  `outputs/logs/llm_debug/`; restore the backend or switch providers.
- **All rows `skipped_no_evidence`** — the task's `evidence_groups` did not match;
  check keyword phrases and section markers against real report wording.
- **Schema errors in output** — the model returned invalid/missing fields; inspect
  `schema_errors` column and the debug dumps.

## Cardiology smoke run (server)

Place the sensitive HER Diagnose export under `data/raw/` as CSV (preferred). Then:

```bash
# Both (or all) HER_Diagnose* files under data/raw/ — preferred:
python3 -m src.pipeline.pipeline \
  --task cardiology_smoke \
  --max-reports 2

# Or explicit paths (CSV and/or Excel), merged by PatientID:
python3 -m src.pipeline.pipeline \
  --task cardiology_smoke \
  --reports data/raw/HER_Diagnose_202601_202606.csv data/raw/HER_Diagnose_vor2026.xlsx \
  --max-reports 2
```

If you omit `--reports`, **all** `HER_Diagnose*` files in `data/raw/` are loaded
and merged (one Diagnoseliste per PatientID).

Interpretability artifacts:
- CSV columns `reasoning`, `evidence_quotes`, `stage_path`, plus one-hot `field__value` columns
- JSONL `audit.stages` (preprocessing → rule_evidence → llm_extraction → …; **per variable** for cardiology_smoke)
- JSONL `audit.llm.per_variable` (system/user prompts + raw model output for each variable)

Cardiology smoke labels are mostly binary presence: `Nein | Ja | Unbekannt`
(CVA: `Keine | TIA | Schlaganfall | Unbekannt`; Re-Op-/Re-Thorakotomie-Kontext: Freitext).
Policy: **V.a. / Verdacht zählt nicht als Ja**; steht sonst nichts Bestätigtes
→ **Unbekannt** (gilt für alle Klassifikations-Variablen, nicht nur Zirrhose).
**10 LLM calls per patient** (one per variable). Start with `--max-reports 2`
when testing.

Prompts live under `prompts/cardiology_smoke_*.txt` and `prompts/cardiology_var_*.txt`.
Keywords: `configs/tasks/cardiology_smoke/task.py`.

### Retry only failed rows (empty LLM / timeout)

If some patients have `status=failed` and `raw_output: ""`, re-run **only those** with a longer timeout:

```bash
python3 -m src.pipeline.pipeline \
  --task cardiology_smoke \
  --retry-failed \
  --llm-timeout 300
```

Only patients with ``status=failed`` or ``partial`` in the existing results CSV
are re-run (not the rest of the dataset). Successful rows stay untouched.

Technical robustness: empty/timeout/non-JSON LLM answers are retried
(``LLM_MAX_RETRIES``, default 2). Mixed success across variables → ``partial``,
not whole-row ``failed``.

Large Diagnoselisten (> ``EVIDENCE_MAX_FULL_TEXT_CHARS``, default 12000) are
automatically shortened on full-text fallback only: entry headers + keyword
windows are kept. Short texts are unchanged.

### Manual consistency review sheet

Default is **CSV** (semicolon-separated, UTF-8 BOM) so it opens cleanly in Excel on USZ Windows:

```bash
python scripts/export_review_sheet.py --task cardiology_smoke --max-rows 20
# -> outputs/evaluation/cardiology_smoke_review.csv

# optional Excel:
python scripts/export_review_sheet.py --task cardiology_smoke --max-rows 20 --format xlsx
```

Open the CSV in Excel, fill `correct_swi` / `correct_reop` / `notes`.

## Tests

```bash
python -m compileall src configs scripts
python -m pytest -q
```
