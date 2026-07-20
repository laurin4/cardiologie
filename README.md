# Clinical Extraction Framework

A reusable **Hybrid NLP–LLM** framework for structured information extraction from
free-text clinical reports. Rule-based evidence extraction narrows each report to
the relevant snippets; an LLM then produces a structured JSON object that is
schema-validated and passed through deterministic clinical guardrails.

The framework is **task-agnostic**: a new extraction problem is a new *task*
(schema + keywords + prompt + rules), not new pipeline code. The first planned
application is cardiology extraction (e.g. sternal wound infection, re-operation,
re-thoracotomy, liver cirrhosis); a small neutral `demo_extraction` task ships so
the pipeline runs and tests pass out of the box.

> This project was refactored from a previous delirium-detection prototype. It no
> longer contains the delirium reviewer cascade, delirium-specific logic, or
> thesis analysis code.

## Architecture

```
Clinical Report
      |
      v
Preprocessing (clean text, section split)
      |
      v
Rule-based Evidence Extraction  -->  Evidence Bundle
      |
      v
LLM Extraction (schema-guided JSON)
      |
      v
Schema Validation (types, enums, required)
      |
      v
Clinical Guardrails / Consistency Checks
      |
      v
Final Structured JSON
```

There is a single pipeline and **no reviewer cascade**.

## Layout

```
clinical_extraction_framework/
  src/
    preprocessing/   clean_text, section_splitter, report_loader, report_identity, report_filters
    extraction/      rule_evidence.py            (rule-based evidence engine)
    llm/             llm_interface, model_config, json_parsing, llm_debug, llm_extraction
    prompts/         registry.py                 (prompt loading + schema block)
    validation/      schema_validation.py, validate_inputs.py
    guardrails/      clinical_guardrails.py      (generic consistency engine)
    pipeline/        pipeline.py (ClinicalExtractionPipeline), records_io.py
    evaluation/      metrics.py, compare_groundtruth.py
    utils/           tabular_io, sqlite_logging, id_normalize
  configs/
    config.py                                    (paths + run settings)
    tasks/           base.py (task spec) + <task>/task.py, schema.json
  prompts/           <task>.txt prompt templates
  examples/          demo_reports/*.txt          (runnable sample input)
  data/ raw/ processed/
  outputs/
  scripts/           run_all.sh, preflight_check.sh, checks, USZ smoke test
  docs/              architecture.md
  tests/
```

## Install

Python 3.9+.

```bash
pip install -r requirements.txt
```

## Run the demo (mocked-free)

The `demo_extraction` task (neutral fever example) runs the full pipeline on the
bundled sample reports. It calls the configured LLM backend, so point it at a
running provider first (see below), or use the tests for an offline check.

```bash
python -m src.pipeline.pipeline --task demo_extraction --reports examples/demo_reports
```

Outputs:

- `outputs/extractions/demo_extraction_results.csv`  (flattened, one row per report)
- `outputs/extractions/demo_extraction_results.jsonl` (full structured objects)

Or run everything (validate + extract):

```bash
scripts/run_all.sh demo_extraction examples/demo_reports
```

## LLM providers

Provider-agnostic; configured via environment (see `src/llm/model_config.py`).

Primary (USZ local HTTP API):

```bash
export LLM_PROVIDER=usz_api
export USZ_LLM_URL=http://localhost:8100/generate
export LLM_MODEL_LABEL=gemma4_26b_usz
```

Optional comparison (Ollama):

```bash
export LLM_PROVIDER=ollama
export OLLAMA_URL=http://127.0.0.1:11500
export OLLAMA_MODEL=qwen2.5:7b
```

Shared generation settings: `LLM_TEMPERATURE`, `LLM_TOP_P`, `LLM_MAX_TOKENS`,
`LLM_TIMEOUT`, `LLM_LONG_INPUT_WARNING_CHARS`.

Smoke test the USZ endpoint:

```bash
python scripts/test_usz_llm_api.py
```

## Input reports

A report is `{report_id, report_text, <metadata...>}`. Two input shapes are
supported:

- A CSV with configurable id/text columns (`REPORT_ID_COLUMN`, `REPORT_TEXT_COLUMN`).
- A directory of `.txt` files (one report per file; `report_id` = file stem).

Sensitive data is never committed (see `.gitignore`). Put real inputs under
`data/raw/` and pass `--reports <path>`.

## Adding a new extraction task

See `docs/architecture.md`. In short:

1. Create `configs/tasks/<name>/task.py` exposing a `TASK` (`ExtractionTask`) with
   `fields`, `evidence_groups`, `section_markers`, optional `negation_patterns`,
   `prompt_name`, and `consistency_rules`.
2. Add a prompt template `prompts/<name>.txt`.
3. Register the name in `configs/tasks/__init__.py` (`_REGISTERED_TASKS`).
4. Run `python -m src.pipeline.pipeline --task <name> --reports <path>`.

`configs/tasks/demo_extraction/` is a complete, copy-pasteable template.

## Testing

```bash
python -m pytest -q
```

Tests are deterministic and do not require a network/LLM (the LLM call is mocked).

## Further reading

- `docs/architecture.md` — pipeline stages and how to add tasks.
- `RUNBOOK.md` — operational setup and troubleshooting.
