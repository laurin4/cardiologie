"""
ClinicalExtractionPipeline: the single Hybrid NLP-LLM extraction pipeline.

Per report the stages are:

    Clinical Report / Diagnoseliste
        -> Preprocessing (clean text)
        -> Rule-based Evidence Extraction (evidence bundle)
        -> LLM Extraction (schema-guided JSON + reasoning)
        -> Schema Validation (inside the LLM stage)
        -> Clinical Guardrails / Consistency Checks
        -> Final Structured JSON (+ stage audit trail)

There is no reviewer cascade. Everything task-specific comes from the loaded
:class:`ExtractionTask`, so a new extraction problem is a new task, not new code.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from configs.config import (
    MAX_REPORTS,
    PREDICTIONS_DIR,
    SQLITE_PREDICTIONS_DB_PATH,
)
from configs.tasks import load_task
from configs.tasks.base import ExtractionTask
from src.extraction.rule_evidence import (
    extract_rule_evidence,
    snippets_json_for_csv,
)
from src.guardrails.clinical_guardrails import apply_guardrails
from src.llm.llm_extraction import extract_entities
from src.pipeline import records_io
from src.preprocessing.clean_text import clean_report_text
from src.preprocessing.report_identity import SOURCE_ROW_ID_COL
from src.preprocessing.report_loader import (
    REPORT_ID_KEY,
    REPORT_TEXT_KEY,
    load_reports,
)

LOGGER = logging.getLogger(__name__)

STATUS_EXTRACTED = "extracted"
STATUS_SKIPPED = "skipped_no_evidence"
STATUS_FAILED = "failed"

BASE_FIELDS = [
    "report_id",
    "source_row_id",
    "report_name",
    "status",
    "llm_called",
    "reduction_method",
    "original_report_text_length",
    "llm_report_text_length",
    "keyword_hits_count",
    "has_positive_evidence",
    "has_negation",
    "manual_review_candidate",
    "schema_errors",
    "guardrail_contradictions",
    "guardrail_missing_fields",
    "guardrail_normalized",
    "evidence_snippets",
    "reasoning",
    "evidence_quotes",
    "stage_path",
]


def _bool_csv(value: Any) -> str:
    return "True" if bool(value) else "False"


def _join(items) -> str:
    return " | ".join(str(x) for x in (items or []))


def _one_hot_columns(task: ExtractionTask, fields: Dict[str, Any]) -> Dict[str, int]:
    """Expand enum fields into one-hot indicator columns for tabular evaluation."""
    out: Dict[str, int] = {}
    for f in task.fields:
        if f.type != "enum" or not f.enum:
            continue
        current = fields.get(f.name)
        for option in f.enum:
            col = f"{f.name}__{option}".replace(" ", "_").replace(";", "")
            out[col] = 1 if str(current) == str(option) else 0
    return out


class ClinicalExtractionPipeline:
    """Runs the six-stage extraction pipeline for a given task."""

    def __init__(self, task: ExtractionTask) -> None:
        self.task = task

    def fieldnames(self) -> List[str]:
        one_hot_names: List[str] = []
        for f in self.task.fields:
            if f.type == "enum":
                for option in f.enum:
                    one_hot_names.append(
                        f"{f.name}__{option}".replace(" ", "_").replace(";", "")
                    )
        clinical = [f.name for f in self.task.fields]
        # Keep audit helpers after clinical fields; one-hot last for evaluation.
        return BASE_FIELDS + clinical + one_hot_names

    def run_report(self, report: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run all stages for one report. Returns ``(csv_row, structured_record)``."""
        report_id = str(report.get(REPORT_ID_KEY, "") or "").strip()
        source_row_id = str(report.get(SOURCE_ROW_ID_COL, "") or "").strip()
        report_name = str(report.get("report_name", "") or report_id).strip()

        stages: List[Dict[str, Any]] = []

        # --- Stage 1: preprocessing ---
        text = clean_report_text(report.get(REPORT_TEXT_KEY, ""))
        stages.append(
            {
                "stage": "preprocessing",
                "action": "clean_report_text",
                "input_chars": len(str(report.get(REPORT_TEXT_KEY, "") or "")),
                "output_chars": len(text),
                "notes": "Unicode NFC + whitespace normalization; wording preserved.",
            }
        )

        # --- Stage 2: rule evidence ---
        evidence = extract_rule_evidence(text, self.task)
        stages.append(
            {
                "stage": "rule_evidence",
                "action": "extract_rule_evidence",
                "reduction_method": evidence["reduction_method"],
                "keyword_hits_count": evidence["keyword_hits_count"],
                "has_positive_evidence": evidence["has_positive_evidence"],
                "has_negation": evidence["has_negation"],
                "evidence_flags": evidence["evidence_flags"],
                "snippet_count": len(evidence["evidence_snippets"]),
                "llm_input_chars": evidence["llm_report_text_length"],
            }
        )

        default_fields = {f.name: f.default for f in self.task.fields}
        schema_errors: List[str] = []
        llm_called = False
        reasoning = ""
        evidence_quotes: Any = []
        raw_output = ""
        llm_audit: Dict[str, Any] = {}
        llm_text_used = evidence.get("llm_report_text", "")
        used_full_text_fallback = False

        # --- Stages 3-4: LLM extraction + schema validation ---
        if evidence["has_positive_evidence"]:
            result = extract_entities(
                evidence, self.task, record_id=report_id, report_name=report_name
            )
            fields = result["fields"]
            schema_errors = result["schema_errors"]
            llm_called = result["llm_called"]
            reasoning = str(result.get("reasoning", "") or "")
            evidence_quotes = result.get("evidence_quotes", [])
            raw_output = result.get("raw_output", "")
            llm_audit = {
                "system_prompt": result.get("system_prompt", ""),
                "user_prompt": result.get("user_prompt", ""),
                "raw_output": raw_output,
                "debug_path": result.get("debug_path"),
            }
            status = (
                STATUS_FAILED
                if any(str(e).startswith("llm_extraction_failed") for e in schema_errors)
                else STATUS_EXTRACTED
            )
            stages.append(
                {
                    "stage": "llm_extraction",
                    "action": "extract_entities",
                    "llm_called": True,
                    "input_mode": "evidence_snippets",
                    "schema_errors": schema_errors,
                    "reasoning": reasoning,
                    "evidence_quotes": evidence_quotes,
                }
            )
        elif self.task.send_full_text_when_no_evidence and text.strip():
            used_full_text_fallback = True
            llm_text_used = (
                "No rule-based evidence snippets matched. Full cleaned Diagnoseliste "
                "/ report text follows:\n\n" + text
            )
            result = extract_entities(
                evidence,
                self.task,
                record_id=report_id,
                report_name=report_name,
                llm_text_override=llm_text_used,
            )
            fields = result["fields"]
            schema_errors = result["schema_errors"]
            llm_called = result["llm_called"]
            reasoning = str(result.get("reasoning", "") or "")
            evidence_quotes = result.get("evidence_quotes", [])
            raw_output = result.get("raw_output", "")
            llm_audit = {
                "system_prompt": result.get("system_prompt", ""),
                "user_prompt": result.get("user_prompt", ""),
                "raw_output": raw_output,
                "debug_path": result.get("debug_path"),
            }
            status = (
                STATUS_FAILED
                if any(str(e).startswith("llm_extraction_failed") for e in schema_errors)
                else STATUS_EXTRACTED
            )
            stages.append(
                {
                    "stage": "llm_extraction",
                    "action": "extract_entities",
                    "llm_called": True,
                    "input_mode": "full_text_fallback",
                    "schema_errors": schema_errors,
                    "reasoning": reasoning,
                    "evidence_quotes": evidence_quotes,
                }
            )
        else:
            status = STATUS_SKIPPED
            fields = dict(default_fields)
            stages.append(
                {
                    "stage": "llm_extraction",
                    "action": "skipped",
                    "llm_called": False,
                    "reason": "no_positive_rule_evidence",
                }
            )

        stages.append(
            {
                "stage": "schema_validation",
                "action": "validate_against_schema",
                "errors": schema_errors,
                "ok": not any(str(e).startswith("llm_extraction_failed") for e in schema_errors),
            }
        )

        # --- Stage 5: guardrails ---
        final_fields, guard = apply_guardrails(fields, self.task, evidence)
        stages.append(
            {
                "stage": "guardrails",
                "action": "apply_guardrails",
                "normalized_fields": guard["normalized_fields"],
                "contradictions": guard["contradictions"],
                "missing_dependent_fields": guard["missing_dependent_fields"],
                "manual_review_candidate": guard["manual_review_candidate"],
            }
        )

        stage_path = " -> ".join(s["stage"] for s in stages)
        one_hot = _one_hot_columns(self.task, final_fields)

        quotes_csv = (
            json.dumps(evidence_quotes, ensure_ascii=False)
            if not isinstance(evidence_quotes, str)
            else evidence_quotes
        )

        row: Dict[str, Any] = {
            "report_id": report_id,
            "source_row_id": source_row_id,
            "report_name": report_name,
            "status": status,
            "llm_called": _bool_csv(llm_called),
            "reduction_method": evidence["reduction_method"],
            "original_report_text_length": evidence["original_report_text_length"],
            "llm_report_text_length": len(llm_text_used) if llm_called else evidence["llm_report_text_length"],
            "keyword_hits_count": evidence["keyword_hits_count"],
            "has_positive_evidence": _bool_csv(evidence["has_positive_evidence"]),
            "has_negation": _bool_csv(evidence["has_negation"]),
            "manual_review_candidate": _bool_csv(guard["manual_review_candidate"]),
            "schema_errors": _join(schema_errors),
            "guardrail_contradictions": _join(guard["contradictions"]),
            "guardrail_missing_fields": _join(guard["missing_dependent_fields"]),
            "guardrail_normalized": _join(guard["normalized_fields"]),
            "evidence_snippets": snippets_json_for_csv(evidence["evidence_snippets"]),
            "reasoning": reasoning or final_fields.get("reasoning", ""),
            "evidence_quotes": quotes_csv,
            "stage_path": stage_path,
        }
        for f in self.task.fields:
            row[f.name] = final_fields.get(f.name)
        row.update(one_hot)

        structured = {
            "report_id": report_id,
            "source_row_id": source_row_id,
            "task": self.task.name,
            "status": status,
            "fields": final_fields,
            "one_hot": one_hot,
            "manual_review_candidate": guard["manual_review_candidate"],
            "schema_errors": schema_errors,
            "guardrail": guard,
            "reasoning": reasoning or final_fields.get("reasoning", ""),
            "evidence_quotes": evidence_quotes,
            "stage_path": stage_path,
            "audit": {
                "stages": stages,
                "llm": llm_audit,
                "used_full_text_fallback": used_full_text_fallback,
                "input_metadata": {
                    k: report.get(k)
                    for k in ("patient_id", "n_diagnosis_entries", "input_kind")
                    if k in report
                },
            },
            "evidence": {
                "reduction_method": evidence["reduction_method"],
                "keyword_hits_count": evidence["keyword_hits_count"],
                "evidence_flags": evidence["evidence_flags"],
                "has_positive_evidence": evidence["has_positive_evidence"],
                "has_negation": evidence["has_negation"],
                "snippets": evidence["evidence_snippets"],
            },
        }
        return row, structured


# --------------------------------------------------------------------------- #
# CLI runner
# --------------------------------------------------------------------------- #
def _resume_enabled() -> bool:
    return os.environ.get("PIPELINE_RESUME", "").strip().lower() in ("1", "true", "yes", "auto")


def run(
    task_name: str,
    *,
    source: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    max_reports: Optional[int] = MAX_REPORTS,
) -> Path:
    """Run the pipeline for *task_name* and write CSV + JSONL results."""
    task = load_task(task_name)
    pipeline = ClinicalExtractionPipeline(task)

    reports = load_reports(source)
    if max_reports is not None:
        reports = reports[:max_reports]

    out_dir = Path(output_dir) if output_dir else PREDICTIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{task.name}_results.csv"
    jsonl_path = out_dir / f"{task.name}_results.jsonl"
    checkpoint_path = out_dir / f"{task.name}_results.checkpoint.csv"

    rows: List[Dict[str, Any]] = []
    if _resume_enabled():
        rows = records_io.load_checkpoint_rows(checkpoint_path)
        if rows:
            print(f"RESUME loaded {len(rows)} rows from {checkpoint_path}")
    done = records_io.done_keys(rows)
    todo = records_io.filter_unprocessed(reports, done)

    print(f"=== {task.name}: Hybrid NLP-LLM extraction ===")
    print(f"reports_total={len(reports)} resumed={len(rows)} remaining={len(todo)}")

    structured_records: List[Dict[str, Any]] = []
    n_extracted = n_skipped = n_failed = 0
    for i, report in enumerate(todo, start=len(rows) + 1):
        row, structured = pipeline.run_report(report)
        rows.append(row)
        structured_records.append(structured)
        status = row["status"]
        n_extracted += status == STATUS_EXTRACTED
        n_skipped += status == STATUS_SKIPPED
        n_failed += status == STATUS_FAILED
        print(
            f"[{i}/{len(reports)}] report_id={row['report_id']} status={status} "
            f"evidence={row['keyword_hits_count']} method={row['reduction_method']} "
            f"path={row['stage_path']}"
        )

    records_io.write_csv(rows, csv_path, pipeline.fieldnames())
    records_io.write_jsonl(structured_records, jsonl_path)
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except OSError:
            LOGGER.warning("Could not remove checkpoint: %s", checkpoint_path)

    print("\n=== Run summary ===")
    print(f"total={len(rows)} extracted={n_extracted} skipped={n_skipped} failed={n_failed}")
    print(f"CSV:   {csv_path}")
    print(f"JSONL: {jsonl_path}  (includes full stage audit + LLM reasoning)")

    if os.environ.get("ENABLE_SQLITE_LOGGING", "").strip().lower() in ("1", "true", "yes"):
        from src.utils.sqlite_logging import init_extraction_db, log_extraction_row

        init_extraction_db(SQLITE_PREDICTIONS_DB_PATH)
        for row in rows:
            log_extraction_row(SQLITE_PREDICTIONS_DB_PATH, row, id_field="report_id")
        print(f"SQLite log: {SQLITE_PREDICTIONS_DB_PATH}")

    return csv_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run the clinical extraction pipeline.")
    parser.add_argument(
        "--task",
        default=os.environ.get("EXTRACTION_TASK", "demo_extraction"),
        help="Registered extraction task name (default: demo_extraction).",
    )
    parser.add_argument(
        "--reports",
        default=None,
        help="Path to a HER Diagnose CSV (preferred), Excel, reports table, or .txt directory.",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory for results.")
    parser.add_argument(
        "--max-reports",
        default=None,
        help="Cap number of reports/patients (int or 'all'). Overrides MAX_REPORTS env.",
    )
    args = parser.parse_args()

    from configs.config import parse_max_reports_env

    max_reports = (
        parse_max_reports_env(args.max_reports) if args.max_reports is not None else MAX_REPORTS
    )
    source = Path(args.reports) if args.reports else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    run(args.task, source=source, output_dir=output_dir, max_reports=max_reports)


if __name__ == "__main__":
    main()
