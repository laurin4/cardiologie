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

Place sensitive HER exports under `data/raw/` as CSV/Excel (preferred):

- `HER_Diagnose*` — Diagnoseliste (merged by PatientID)
- `HER_Verlegungsbericht*` / `HER_IPS_Verlegungsbericht*` — latest Verlegungsbericht
  per **FallNummer** (`berdat`); sections `diag` / `epikrise` /
  `jetziges_leiden`|`jetztleid` / `prozedere`|`procedere`; joined onto
  Diagnoseliste patients via matching FallNummer
- `HER_OPBericht*` — Operationsbericht (for Re-Thor / reasons; loader TBD)
- Dendrite postop gold (`Dendrite postop data set_LLM_v1*`) — validation labels
  (join via `FallNummer FID`)

Then:

```bash
# Auto-discover Diagnose + Verlegung under data/raw/ — preferred:
python3 -m src.pipeline.pipeline \
  --task cardiology_smoke \
  --max-reports 2

# Or explicit paths (CSV and/or Excel), Diagnose merged by PatientID;
# Verlegung auto-attached from data/raw/ if not passed:
python3 -m src.pipeline.pipeline \
  --task cardiology_smoke \
  --reports data/raw/HER_Diagnose_202601_202606.csv data/raw/HER_Diagnose_vor2026.xlsx \
  --max-reports 2
```

If you omit `--reports`, **all** `HER_Diagnose*` files in `data/raw/` are loaded
and merged (one Diagnoseliste per PatientID), and matching `HER_Verlegungsbericht*`
are attached.

Interpretability artifacts:
- CSV columns `reasoning`, `evidence_quotes`, `stage_path`, plus one-hot `field__value` columns
- JSONL `audit.stages` (preprocessing → rule_evidence → llm_extraction → …; **per variable** for cardiology_smoke; includes `text_source`)
- JSONL `audit.llm.per_variable` (system/user prompts + raw model output for each variable)

### Variables and labels (`cardiology_smoke`)

Nine LLM calls per patient (one per variable). Text source is per variable
(Verlegung only, Diagnoseliste only, or both). Details:
`configs/tasks/cardiology_smoke/task.py` and `schema.json`.

| Field | Enum / type | Source |
|---|---|---|
| `pacemaker` | `Neu` \| `Schon vorhanden` \| `Kein` \| `Unbekannt` \| `k.A.` | Verlegung |
| `atrial_fibrillation` | `Neu` \| `Vorbestehend` \| `Kein` \| `Unbekannt` \| `k.A.` | both |
| `cerebrovascular_event` | `Keine` \| `TIA` \| `Schlaganfall` \| `Unbekannt` \| `k.A.` | both |
| `reoperation_required` | `Nein` \| `Ja` \| `Unbekannt` \| `k.A.` | both (interim) |
| `reoperation_context` | Freitext | both (interim) |
| `multi_system_failure` | `Nein` \| `Ja` \| `Unbekannt` \| `k.A.` | Verlegung |
| `rethoracotomy` | `Nein` \| `Ja` \| `Unbekannt` \| `k.A.` | Verlegung (interim) |
| `rethoracotomy_context` | Freitext | Verlegung (interim) |
| `liver_cirrhosis` | `Nein` \| `Ja` \| `Unbekannt` \| `k.A.` (no Child-Pugh yet) | both (interim) |

**Shared label policy**
- Confirmed status → `Neu` / `TIA` / `Schlaganfall` / `Ja` / `Schon vorhanden` / `Vorbestehend` (field-specific)
- Explicit negation → `Kein` / `Keine` / `Nein`
- Mentioned but unclear / V.a. alone → `Unbekannt` (temporary pacemaker alone → `Unbekannt`, not `Neu`)
- Topic not in text → `k.A.` (keine Angabe)

**Out of scope / deferred (no enum yet)**
- SWI (sternal wound infection) — not extracted by LLM
- Child-Pugh for cirrhosis — when Eintrittsbericht / criteria exist
- Structured Re-Op reason enums — interim stays Freitext context
- Final Re-Op / Re-Thor from structured OP / Opsbericht when available

Prompts: `prompts/cardiology_smoke_*.txt`, `prompts/cardiology_var_*.txt`.
Keywords: `configs/tasks/cardiology_smoke/task.py`.
Start with `--max-reports 2` when testing.

### Next: new Verlegung data → run → validate

Generic scoring: `src/evaluation/compare_groundtruth.py`, `metrics.py`.
Dendrite gold format is known (`Label (Code)`); join key **`FallNummer FID`**.

Sequence on the server:

1. Place `HER_IPS_Verlegungsbericht*`, Diagnose, OP, Dendrite under `data/raw/`
2. Confirm FallNummer join: `python3 scripts/check_verlegung_join.py`
3. Run pipeline (`cardiology_smoke`), export review sheet
4. Score against Dendrite (start with `pacemaker` + `atrial_fibrillation` + CVA)

**Dendrite → our fields (score mapping)**

| Dendrite column | Our field | Gold values → score class |
|---|---|---|
| `PM/ICD Implant` | `pacemaker` | `Ja (1)` / `Nein (0)`; collapse pred: `Neu`→Ja, `Kein`→Nein |
| `Vorhofsarrhythmie postop` | `atrial_fibrillation` | same Ja/Nein collapse (`Neu`→Ja, `Kein`→Nein) |
| `Neue post-OP neurol. Funktionsstörung` | `cerebrovascular_event` | `Keine (0)`→Keine; `TIA … (1)`→TIA; `Dauerhafter Schlaganfall (2)`→Schlaganfall |
| `Reoperationen` | `reoperation_required` (+ reasons later) | `(0)` keine → Nein; any other code → Ja (multi-label reasons) |
| `Multisystem failure` | `multi_system_failure` | `Yes (1)` / `No (0)` / `Unknown (99)` |
| `Sternale Wundinfektion` | — | gold only; LLM out of scope (`Oberflächlich`/`Tief`) |

**CVA score exclude:** `Paraparese (3)` / `Paraplegie (4)` (and combinations) — not classic CVA; skip row for CVA metrics.
Empty Dendrite cells → missing (not Nein).
Finer extraction enums (`Schon vorhanden`, `Vorbestehend`, `Unbekannt`, `k.A.`) stay in the pipeline; collapse only when scoring against Dendrite Ja/Nein.

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

By default the export **enriches FallNummer** from `data/raw/` HER files
(no LLM re-run). Use `--no-enrich-fall` to skip.

Open the CSV in Excel, fill `correct_reop` / `notes`.
Review sheet includes `patient_id`, `fall_nummers`, `verlegung_fallnr`, and
`verlegung_matched` so you can look up Diagnoseliste by patient and Verlegung by FallNummer.

### Check Diagnoseliste ↔ Verlegung join

Before a run (or if SM/MOV look empty), verify **FallNummer** overlap:

```bash
python3 scripts/check_verlegung_join.py
```

Prints FallNummer match rate (primary) and PatientID/`patnr` for reference.
Exit code `2` if FallNummer overlap is 0 (merge would be silent empty).
Excel-style IDs (`12345.0`) are normalized to `12345`.

## Tests

```bash
python -m compileall src configs scripts
python -m pytest -q
```
