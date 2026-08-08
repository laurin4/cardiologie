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
from typing import Any, Dict, List, Optional, Sequence, Tuple

from configs.config import (
    MAX_REPORTS,
    PREDICTIONS_DIR,
    SQLITE_PREDICTIONS_DB_PATH,
)
from configs.tasks import load_task
from configs.tasks.base import ExtractionTask, VariableSpec
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


def _variable_as_task(parent: ExtractionTask, var: VariableSpec) -> ExtractionTask:
    """Build a mini-task for one variable (own prompt, schema, keyword subset)."""
    allowed = set(var.evidence_group_names)
    groups = tuple(g for g in parent.evidence_groups if g.name in allowed)
    return ExtractionTask(
        name=f"{parent.name}:{var.name}",
        description=var.label or var.name,
        fields=var.fields,
        evidence_groups=groups,
        section_markers=parent.section_markers,
        negation_patterns=parent.negation_patterns,
        prompt_name=var.prompt_name,
        consistency_rules=var.consistency_rules,
        language=parent.language,
        send_full_text_when_no_evidence=parent.send_full_text_when_no_evidence,
    )


def _merge_quotes(*parts: Any) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for part in parts:
        items = part if isinstance(part, list) else ([] if part in (None, "") else [part])
        for item in items:
            key = str(item).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item)
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

    def _llm_once(
        self,
        *,
        text: str,
        evidence: Dict[str, Any],
        task: ExtractionTask,
        report_id: str,
        report_name: str,
        stages: List[Dict[str, Any]],
        stage_label: str,
    ) -> Dict[str, Any]:
        """Run one LLM extraction (snippets or full-text fallback)."""
        used_full_text_fallback = False
        llm_text_used = evidence.get("llm_report_text", "")
        if evidence["has_positive_evidence"]:
            result = extract_entities(
                evidence, task, record_id=report_id, report_name=report_name
            )
            input_mode = "evidence_snippets"
        elif task.send_full_text_when_no_evidence and text.strip():
            used_full_text_fallback = True
            from src.extraction.text_budget import fit_text_for_llm

            capped_text, was_truncated = fit_text_for_llm(text, task)
            if (task.language or "en").lower().startswith("de"):
                prefix = (
                    "Keine regelbasierten Evidenz-Snippets gefunden. Es folgt die "
                    "bereinigte Diagnoseliste / der Berichtstext"
                    + (" (wegen Länge gekürzt)" if was_truncated else "")
                    + ":\n\n"
                )
            else:
                prefix = (
                    "No rule-based evidence snippets matched. Cleaned Diagnoseliste "
                    "/ report text follows"
                    + (" (truncated for length)" if was_truncated else "")
                    + ":\n\n"
                )
            llm_text_used = prefix + capped_text
            result = extract_entities(
                evidence,
                task,
                record_id=report_id,
                report_name=report_name,
                llm_text_override=llm_text_used,
            )
            input_mode = (
                "full_text_fallback_truncated" if was_truncated else "full_text_fallback"
            )
            result = dict(result)
            result["full_text_truncated"] = was_truncated
            result["full_text_chars_original"] = len(text)
            result["full_text_chars_sent"] = len(capped_text)
        else:
            stages.append(
                {
                    "stage": "llm_extraction",
                    "variable": stage_label,
                    "action": "skipped",
                    "llm_called": False,
                    "reason": "no_positive_rule_evidence",
                }
            )
            return {
                "fields": {f.name: f.default for f in task.fields},
                "schema_errors": [],
                "llm_called": False,
                "reasoning": "",
                "evidence_quotes": [],
                "raw_output": "",
                "llm_audit": {},
                "llm_text_used": llm_text_used,
                "used_full_text_fallback": False,
                "status": STATUS_SKIPPED,
                "evidence": evidence,
            }

        schema_errors = result["schema_errors"]
        status = (
            STATUS_FAILED
            if any(str(e).startswith("llm_extraction_failed") for e in schema_errors)
            else STATUS_EXTRACTED
        )
        llm_audit = {
            "variable": stage_label,
            "system_prompt": result.get("system_prompt", ""),
            "user_prompt": result.get("user_prompt", ""),
            "raw_output": result.get("raw_output", ""),
            "debug_path": result.get("debug_path"),
        }
        if used_full_text_fallback:
            llm_audit["full_text_truncated"] = result.get("full_text_truncated")
            llm_audit["full_text_chars_original"] = result.get("full_text_chars_original")
            llm_audit["full_text_chars_sent"] = result.get("full_text_chars_sent")

        stage_entry: Dict[str, Any] = {
            "stage": "llm_extraction",
            "variable": stage_label,
            "action": "extract_entities",
            "llm_called": True,
            "input_mode": input_mode,
            "schema_errors": schema_errors,
            "reasoning": result.get("reasoning", ""),
            "evidence_quotes": result.get("evidence_quotes", []),
        }
        if used_full_text_fallback:
            stage_entry["full_text_truncated"] = result.get("full_text_truncated")
            stage_entry["full_text_chars_original"] = result.get("full_text_chars_original")
            stage_entry["full_text_chars_sent"] = result.get("full_text_chars_sent")
        stages.append(stage_entry)

        return {
            "fields": result["fields"],
            "schema_errors": schema_errors,
            "llm_called": True,
            "reasoning": str(result.get("reasoning", "") or ""),
            "evidence_quotes": result.get("evidence_quotes", []),
            "raw_output": result.get("raw_output", ""),
            "llm_audit": llm_audit,
            "llm_text_used": llm_text_used,
            "used_full_text_fallback": used_full_text_fallback,
            "status": status,
            "evidence": evidence,
        }

    def _run_variable_extractions(
        self,
        *,
        text: str,
        report_id: str,
        report_name: str,
        stages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """One LLM call per VariableSpec; merge into a single field dict."""
        merged = {f.name: f.default for f in self.task.fields}
        schema_errors: List[str] = []
        reason_parts: List[str] = []
        quotes: List[Any] = []
        llm_audits: List[Dict[str, Any]] = []
        evidence_snippets: List[Dict[str, Any]] = []
        keyword_hits = 0
        has_positive = False
        has_negation = False
        any_llm = False
        any_failed = False
        any_extracted = False
        used_full_text_fallback = False
        llm_text_len = 0
        evidence_flags: Dict[str, bool] = {}
        reduction_methods: List[str] = []
        original_len = len(text)

        for var in self.task.variables:
            mini = _variable_as_task(self.task, var)
            evidence = extract_rule_evidence(text, mini)
            stages.append(
                {
                    "stage": "rule_evidence",
                    "variable": var.name,
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
            keyword_hits += int(evidence["keyword_hits_count"] or 0)
            has_positive = has_positive or bool(evidence["has_positive_evidence"])
            has_negation = has_negation or bool(evidence["has_negation"])
            evidence_snippets.extend(evidence["evidence_snippets"] or [])
            evidence_flags.update(evidence.get("evidence_flags") or {})
            reduction_methods.append(f"{var.name}:{evidence['reduction_method']}")
            original_len = evidence["original_report_text_length"]

            one = self._llm_once(
                text=text,
                evidence=evidence,
                task=mini,
                report_id=report_id,
                report_name=report_name,
                stages=stages,
                stage_label=var.name,
            )
            if one["llm_called"]:
                any_llm = True
                llm_text_len = max(llm_text_len, len(str(one.get("llm_text_used") or "")))
            used_full_text_fallback = used_full_text_fallback or bool(
                one.get("used_full_text_fallback")
            )
            if one["status"] == STATUS_FAILED:
                any_failed = True
            if one["status"] == STATUS_EXTRACTED:
                any_extracted = True
            schema_errors.extend(
                f"{var.name}: {err}" for err in (one.get("schema_errors") or [])
            )
            if one.get("llm_audit"):
                llm_audits.append(one["llm_audit"])

            for f in var.fields:
                if f.name in ("reasoning", "evidence_quotes", "information_sufficient"):
                    continue
                merged[f.name] = one["fields"].get(f.name, f.default)

            var_reason = str(one.get("reasoning") or one["fields"].get("reasoning") or "").strip()
            if var_reason:
                reason_parts.append(f"### {var.label}\n{var_reason}")
            quotes = _merge_quotes(quotes, one.get("evidence_quotes"))

            # Per-variable sufficiency: overall True only if every called variable was sufficient.
            # Store last; recomputed below.
            merged[f"_suff_{var.name}"] = bool(one["fields"].get("information_sufficient"))

        suff_keys = [k for k in list(merged.keys()) if k.startswith("_suff_")]
        if suff_keys:
            merged["information_sufficient"] = all(bool(merged.pop(k)) for k in suff_keys)
        else:
            merged["information_sufficient"] = False
        merged["reasoning"] = "\n\n".join(reason_parts)
        merged["evidence_quotes"] = quotes

        if any_failed:
            status = STATUS_FAILED
        elif any_extracted or any_llm:
            status = STATUS_EXTRACTED
        else:
            status = STATUS_SKIPPED

        return {
            "fields": merged,
            "schema_errors": schema_errors,
            "llm_called": any_llm,
            "reasoning": merged["reasoning"],
            "evidence_quotes": quotes,
            "llm_audit": {"per_variable": llm_audits},
            "llm_text_used_len": llm_text_len,
            "used_full_text_fallback": used_full_text_fallback,
            "status": status,
            "evidence_summary": {
                "reduction_method": " | ".join(reduction_methods),
                "keyword_hits_count": keyword_hits,
                "has_positive_evidence": has_positive,
                "has_negation": has_negation,
                "evidence_flags": evidence_flags,
                "snippets": evidence_snippets,
                "original_report_text_length": original_len,
                "llm_report_text_length": llm_text_len,
            },
        }

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

        default_fields = {f.name: f.default for f in self.task.fields}

        if self.task.variables:
            multi = self._run_variable_extractions(
                text=text,
                report_id=report_id,
                report_name=report_name,
                stages=stages,
            )
            fields = multi["fields"]
            schema_errors = multi["schema_errors"]
            llm_called = multi["llm_called"]
            reasoning = multi["reasoning"]
            evidence_quotes = multi["evidence_quotes"]
            llm_audit = multi["llm_audit"]
            used_full_text_fallback = multi["used_full_text_fallback"]
            status = multi["status"]
            ev = multi["evidence_summary"]
            evidence = {
                "reduction_method": ev["reduction_method"],
                "keyword_hits_count": ev["keyword_hits_count"],
                "has_positive_evidence": ev["has_positive_evidence"],
                "has_negation": ev["has_negation"],
                "evidence_flags": ev["evidence_flags"],
                "evidence_snippets": ev["snippets"],
                "original_report_text_length": ev["original_report_text_length"],
                "llm_report_text_length": ev["llm_report_text_length"],
            }
            llm_text_used = ""
            llm_text_len_for_row = ev["llm_report_text_length"]
        else:
            # --- Stage 2: rule evidence (single-call tasks) ---
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
            one = self._llm_once(
                text=text,
                evidence=evidence,
                task=self.task,
                report_id=report_id,
                report_name=report_name,
                stages=stages,
                stage_label=self.task.name,
            )
            fields = one["fields"] if one["llm_called"] or one["status"] != STATUS_SKIPPED else dict(default_fields)
            if one["status"] == STATUS_SKIPPED and not one["llm_called"]:
                fields = dict(default_fields)
            schema_errors = one["schema_errors"]
            llm_called = one["llm_called"]
            reasoning = one["reasoning"]
            evidence_quotes = one["evidence_quotes"]
            llm_audit = one["llm_audit"]
            used_full_text_fallback = one["used_full_text_fallback"]
            status = one["status"]
            llm_text_used = one.get("llm_text_used", "")
            llm_text_len_for_row = (
                len(llm_text_used) if llm_called else evidence["llm_report_text_length"]
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
            "llm_report_text_length": llm_text_len_for_row,
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
                "per_variable_llm": bool(self.task.variables),
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


def _apply_llm_timeout(seconds: Optional[int]) -> None:
    """Raise LLM HTTP timeout for this process (used for failed-row retries)."""
    if seconds is None:
        return
    if seconds <= 0:
        raise ValueError("llm_timeout must be a positive integer (seconds).")
    os.environ["LLM_TIMEOUT"] = str(seconds)
    # model_config.TIMEOUT is read at import time; patch the live module values.
    import src.llm.llm_interface as llm_interface
    import src.llm.model_config as model_config

    model_config.TIMEOUT = seconds
    llm_interface.TIMEOUT = seconds
    print(f"LLM_TIMEOUT set to {seconds}s for this run")


def _count_statuses(rows: Sequence[Dict[str, Any]]) -> Tuple[int, int, int]:
    n_extracted = n_skipped = n_failed = 0
    for row in rows:
        status = str(row.get("status", ""))
        n_extracted += status == STATUS_EXTRACTED
        n_skipped += status == STATUS_SKIPPED
        n_failed += status == STATUS_FAILED
    return n_extracted, n_skipped, n_failed


def run(
    task_name: str,
    *,
    source: Optional[Path | Sequence[Path]] = None,
    output_dir: Optional[Path] = None,
    max_reports: Optional[int] = MAX_REPORTS,
    retry_failed: bool = False,
    llm_timeout: Optional[int] = None,
) -> Path:
    """
    Run the pipeline for *task_name* and write CSV + JSONL results.

    ``source`` may be one path or several HER Diagnose files (merged by PatientID).
    ``retry_failed=True`` re-runs only rows with ``status=failed`` from an existing
    results CSV and merges them back (successful rows are kept).
    """
    _apply_llm_timeout(llm_timeout)

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

    existing_rows: List[Dict[str, Any]] = []
    existing_structured: List[Dict[str, Any]] = []

    if retry_failed:
        if not csv_path.exists():
            raise FileNotFoundError(
                f"--retry-failed requires existing results at {csv_path}"
            )
        all_existing = records_io.load_checkpoint_rows(csv_path)
        all_structured = records_io.load_jsonl(jsonl_path)
        failed_rows = [
            row
            for row in all_existing
            if str(row.get("status", "")).strip().lower() == "failed"
        ]
        n_failed_existing = len(failed_rows)
        if n_failed_existing == 0:
            print(f"No failed rows in {csv_path}; nothing to retry.")
            return csv_path

        # Keep only non-failed rows; re-run ONLY patients that were failed in the CSV
        # (never the whole remaining dataset).
        existing_rows = [
            row
            for row in all_existing
            if str(row.get("status", "")).strip().lower() != "failed"
        ]
        existing_structured = [
            rec
            for rec in all_structured
            if str(rec.get("status", "")).strip().lower() != "failed"
        ]
        failed_key_set = records_io.failed_keys(all_existing)
        failed_report_ids = {
            str(row.get("report_id", "")).strip()
            for row in failed_rows
            if str(row.get("report_id", "")).strip()
        }
        todo = [
            r
            for r in reports
            if records_io.resume_key(r) in failed_key_set
            or str(r.get(REPORT_ID_KEY, "")).strip() in failed_report_ids
        ]
        print(f"=== {task.name}: RETRY failed rows ===")
        print(
            f"failed_in_csv={n_failed_existing} kept_ok={len(existing_rows)} "
            f"retrying={len(todo)}"
        )
        if not todo:
            print(
                "WARNING: failed rows found in CSV but no matching input reports. "
                "Check report_id / source_row_id alignment."
            )
            return csv_path
    else:
        rows: List[Dict[str, Any]] = []
        if _resume_enabled():
            rows = records_io.load_checkpoint_rows(checkpoint_path)
            if not rows and csv_path.exists():
                rows = records_io.load_checkpoint_rows(csv_path)
            if rows:
                print(f"RESUME loaded {len(rows)} rows")
        done = records_io.done_keys(rows)
        # Do not treat prior failures as done when resuming a crashed run that
        # already wrote a partial CSV — only skip non-failed completed rows.
        done_ok = {
            records_io.resume_key(row)
            for row in rows
            if str(row.get("status", "")).strip().lower() != "failed"
        }
        todo = records_io.filter_unprocessed(reports, done_ok if rows else done)
        existing_rows = [
            row
            for row in rows
            if str(row.get("status", "")).strip().lower() != "failed"
        ]
        existing_structured = [
            rec
            for rec in records_io.load_jsonl(jsonl_path)
            if str(rec.get("status", "")).strip().lower() != "failed"
        ]
        print(f"=== {task.name}: Hybrid NLP-LLM extraction ===")
        print(
            f"reports_total={len(reports)} kept={len(existing_rows)} remaining={len(todo)}"
        )

    new_rows: List[Dict[str, Any]] = []
    new_structured: List[Dict[str, Any]] = []
    for i, report in enumerate(todo, start=1):
        row, structured = pipeline.run_report(report)
        new_rows.append(row)
        new_structured.append(structured)
        print(
            f"[{i}/{len(todo)}] report_id={row['report_id']} status={row['status']} "
            f"evidence={row['keyword_hits_count']} method={row['reduction_method']} "
            f"path={row['stage_path']}"
        )

    if retry_failed or existing_rows:
        rows = records_io.merge_rows_by_key(existing_rows, new_rows)
        structured_records = records_io.merge_rows_by_key(existing_structured, new_structured)
    else:
        rows = new_rows
        structured_records = new_structured

    records_io.write_csv(rows, csv_path, pipeline.fieldnames())
    records_io.write_jsonl(structured_records, jsonl_path)
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except OSError:
            LOGGER.warning("Could not remove checkpoint: %s", checkpoint_path)

    n_extracted, n_skipped, n_failed = _count_statuses(rows)
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
        nargs="*",
        default=None,
        help=(
            "One or more HER Diagnose CSV/Excel paths (merged by PatientID), "
            "a reports table, or a .txt directory. "
            "If omitted, all HER_Diagnose* files under data/raw/ are used."
        ),
    )
    parser.add_argument("--output-dir", default=None, help="Output directory for results.")
    parser.add_argument(
        "--max-reports",
        default=None,
        help="Cap number of reports/patients (int or 'all'). Overrides MAX_REPORTS env.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run only rows with status=failed from the existing results CSV and merge.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=None,
        help="Override LLM_TIMEOUT seconds for this run (useful with --retry-failed).",
    )
    args = parser.parse_args()

    from configs.config import parse_max_reports_env

    max_reports = (
        parse_max_reports_env(args.max_reports) if args.max_reports is not None else MAX_REPORTS
    )
    if args.reports is None:
        source = None
    elif len(args.reports) == 0:
        source = None
    elif len(args.reports) == 1:
        source = Path(args.reports[0])
    else:
        source = [Path(p) for p in args.reports]
    output_dir = Path(args.output_dir) if args.output_dir else None
    run(
        args.task,
        source=source,
        output_dir=output_dir,
        max_reports=max_reports,
        retry_failed=args.retry_failed,
        llm_timeout=args.llm_timeout,
    )


if __name__ == "__main__":
    main()