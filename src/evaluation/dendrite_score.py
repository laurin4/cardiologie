"""
Score cardiology pipeline predictions against Dendrite postop gold labels.

Join key: Dendrite ``FallNummer FID`` ↔ pipeline ``verlegung_fallnr``
(or any FallNummer listed in ``fall_nummers``).

Score v1 fields: pacemaker, atrial_fibrillation, cerebrovascular_event,
reoperation_required, multi_system_failure.

Collapse policy (pred → score class):
  - Neu → Ja; Kein → Nein; Schon vorhanden / Vorbestehend → Nein
  - Unbekannt / k.A. → exclude (missing)
Gold blanks → exclude. Paraparese/Paraplegie → exclude from CVA metrics.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from src.evaluation.evidence_format import FIELD_TEXT_SOURCE
from src.evaluation.metrics import binary_metrics, categorical_accuracy
from src.preprocessing.report_identity import normalize_str
from src.utils.table_io import read_table

PathLike = Union[str, Path]

# Dendrite column → our field
DENDRITE_COLUMNS: Dict[str, str] = {
    "PM/ICD Implant": "pacemaker",
    "Vorhofsarrhythmie postop": "atrial_fibrillation",
    "Neue post-OP neurol. Funktionsstörung": "cerebrovascular_event",
    "Reoperationen": "reoperation_required",
    "Multisystem failure": "multi_system_failure",
}

PAIR_COLUMNS = [
    "fall",
    "patient_id",
    "field",
    "dendrite_column",
    "gold_raw",
    "gold",
    "pred_raw",
    "pred",
    "match",
    "source_report",
    "source_columns",
    "evidence_quotes",
    "reasoning",
]

FALL_ALIASES = (
    "FallNummer FID",
    "FallNummer",
    "Fallnummer FID",
    "fallnummer fid",
    "Fall Nr",
)

_CODE_SUFFIX = re.compile(r"\s*\(\d+\)\s*$")
_EXCLUDE_PRED = frozenset({"unbekannt", "k.a.", "k.a", "ka", ""})


def _strip_code(label: str) -> str:
    return _CODE_SUFFIX.sub("", normalize_str(label)).strip()


def _find_column(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    lower_map = {str(c).strip().lower(): str(c) for c in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def parse_yn_gold(raw: Any) -> Optional[str]:
    """Dendrite Ja/Nein / Yes/No (+ code) → Ja|Nein, else None."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = _strip_code(str(raw)).lower()
    if not text:
        return None
    if text in ("ja", "yes", "y", "1"):
        return "Ja"
    if text in ("nein", "no", "n", "0"):
        return "Nein"
    return None


def parse_mov_gold(raw: Any) -> Optional[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = _strip_code(str(raw)).lower()
    if not text:
        return None
    if text in ("yes", "ja", "1"):
        return "Ja"
    if text in ("no", "nein", "0"):
        return "Nein"
    if "unknown" in text or text in ("99", "unbekannt"):
        return None
    return None


def parse_cva_gold(raw: Any) -> Optional[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = normalize_str(str(raw))
    if not text:
        return None
    lower = text.lower()
    if "paraparese" in lower or "paraplegie" in lower:
        return None
    stripped = _strip_code(text).lower()
    if stripped.startswith("keine") or stripped == "0":
        return "Keine"
    if stripped.startswith("tia") or "vorübergehend" in stripped or "voruebergehend" in stripped:
        return "TIA"
    if "schlaganfall" in stripped or stripped.startswith("dauerhaft"):
        return "Schlaganfall"
    return None


def parse_reop_gold(raw: Any) -> Optional[str]:
    """Any non-(0) reason (incl. multi-label) → Ja; none required → Nein."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = normalize_str(str(raw))
    if not text:
        return None
    lower = text.lower()
    if "keine erneute" in lower or lower.strip().startswith("keine erneute"):
        return "Nein"
    # bare (0) only
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return None
    codes = []
    for part in parts:
        m = re.search(r"\((\d+)\)\s*$", part)
        if m:
            codes.append(m.group(1))
    if codes and all(c == "0" for c in codes):
        return "Nein"
    if codes and any(c != "0" for c in codes):
        return "Ja"
    if "re-operation" in lower or "reoperation" in lower or "wiederhol" in lower:
        return "Ja"
    return None


def collapse_pacemaker_pred(raw: Any) -> Optional[str]:
    text = normalize_str(str(raw) if raw is not None else "")
    if text.lower() in _EXCLUDE_PRED:
        return None
    if text == "Neu":
        return "Ja"
    if text in ("Kein", "Schon vorhanden"):
        return "Nein"
    if text in ("Ja", "Nein"):
        return text
    return None


def collapse_af_pred(raw: Any) -> Optional[str]:
    text = normalize_str(str(raw) if raw is not None else "")
    if text.lower() in _EXCLUDE_PRED:
        return None
    if text == "Neu":
        return "Ja"
    if text in ("Kein", "Vorbestehend"):
        return "Nein"
    if text in ("Ja", "Nein"):
        return text
    return None


def collapse_yn_pred(raw: Any) -> Optional[str]:
    text = normalize_str(str(raw) if raw is not None else "")
    if text.lower() in _EXCLUDE_PRED:
        return None
    if text in ("Ja", "Nein"):
        return text
    return None


def collapse_cva_pred(raw: Any) -> Optional[str]:
    text = normalize_str(str(raw) if raw is not None else "")
    if text.lower() in _EXCLUDE_PRED:
        return None
    if text in ("Keine", "TIA", "Schlaganfall"):
        return text
    return None


GOLD_PARSERS = {
    "pacemaker": parse_yn_gold,
    "atrial_fibrillation": parse_yn_gold,
    "cerebrovascular_event": parse_cva_gold,
    "reoperation_required": parse_reop_gold,
    "multi_system_failure": parse_mov_gold,
}

PRED_COLLAPSE = {
    "pacemaker": collapse_pacemaker_pred,
    "atrial_fibrillation": collapse_af_pred,
    "cerebrovascular_event": collapse_cva_pred,
    "reoperation_required": collapse_yn_pred,
    "multi_system_failure": collapse_yn_pred,
}

BINARY_FIELDS = frozenset(
    {
        "pacemaker",
        "atrial_fibrillation",
        "reoperation_required",
        "multi_system_failure",
    }
)


def _fall_keys_from_pred_row(row: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    primary = normalize_str(row.get("verlegung_fallnr", ""))
    if primary:
        keys.append(primary)
    falls = row.get("fall_nummers", "")
    if isinstance(falls, list):
        parts = [normalize_str(x) for x in falls]
    else:
        parts = [normalize_str(p) for p in re.split(r"[|;,]", str(falls or ""))]
    for p in parts:
        if p and p not in keys:
            keys.append(p)
    return keys


def load_dendrite_gold(path: PathLike) -> pd.DataFrame:
    df = read_table(path)
    fall_col = _find_column(df, FALL_ALIASES)
    if fall_col is None:
        raise ValueError(
            f"Dendrite file missing FallNummer column. Tried {FALL_ALIASES}. "
            f"Found: {list(df.columns)}"
        )
    missing = [c for c in DENDRITE_COLUMNS if _find_column(df, (c,)) is None]
    if missing:
        # allow partial: warn via KeyError only if none found
        found_any = any(_find_column(df, (c,)) for c in DENDRITE_COLUMNS)
        if not found_any:
            raise ValueError(
                f"No Dendrite score columns found. Expected one of {list(DENDRITE_COLUMNS)}. "
                f"Found: {list(df.columns)}"
            )
    out = df.copy()
    out["_fall"] = out[fall_col].map(normalize_str)
    out = out[out["_fall"] != ""].copy()
    return out


def load_predictions(path: PathLike) -> pd.DataFrame:
    df = read_table(path)
    return df


def align_predictions_to_dendrite(
    preds: pd.DataFrame, gold: pd.DataFrame
) -> List[Dict[str, Any]]:
    """One aligned pair per Dendrite FallNummer that matches a prediction."""
    pred_by_fall: Dict[str, Dict[str, Any]] = {}
    for row in preds.to_dict(orient="records"):
        for key in _fall_keys_from_pred_row(row):
            # Prefer row that has verlegung_fallnr == key
            existing = pred_by_fall.get(key)
            if existing is None:
                pred_by_fall[key] = row
            elif normalize_str(row.get("verlegung_fallnr", "")) == key:
                pred_by_fall[key] = row

    aligned: List[Dict[str, Any]] = []
    for grow in gold.to_dict(orient="records"):
        fall = normalize_str(grow.get("_fall", ""))
        if not fall:
            continue
        prow = pred_by_fall.get(fall)
        if prow is None:
            continue
        aligned.append({"fall": fall, "pred": prow, "gt": grow})
    return aligned


def _flatten_quotes(raw: Any) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return ""
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return " | ".join(str(x).strip() for x in parsed if str(x).strip())
    except Exception:
        pass
    return s.replace("\n", " | ")


def provenance_from_pred_row(
    pred: Dict[str, Any],
    field: str,
    *,
    text_by_fall: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Berichtstyp / Spalten / getaggte Snippets / Reasoning nur für diese Variable."""
    from src.evaluation.evidence_format import (
        format_tagged_evidence,
        parse_quotes_list,
        reasoning_for_field,
        source_text_for_field,
    )
    from src.preprocessing.verlegung_loader import provenance_for_text_source

    src_report = str(pred.get(f"{field}_source_report") or "").strip()
    src_cols = str(pred.get(f"{field}_source_columns") or "").strip()
    if src_report.lower() in ("nan", "none", "null"):
        src_report = ""
    if src_cols.lower() in ("nan", "none", "null"):
        src_cols = ""
    if not src_report:
        src_report, src_cols = provenance_for_text_source(
            FIELD_TEXT_SOURCE.get(field, "report")
        )

    raw_quotes = pred.get(f"{field}_evidence_quotes")
    quotes = parse_quotes_list(raw_quotes)
    if not quotes:
        # Do not fall back to global evidence_quotes (mixed variables).
        quotes = []

    source_text = source_text_for_field(pred, field, text_by_fall=text_by_fall)
    tagged = format_tagged_evidence(
        quotes,
        source_text=source_text,
        fallback_column=src_cols.split(",")[0].strip() if src_cols else "quelle",
    )
    reasoning = reasoning_for_field(pred, field)
    return {
        "source_report": src_report,
        "source_columns": src_cols,
        "evidence_quotes": tagged,
        "reasoning": reasoning,
    }


def _pair_row(
    *,
    item: Dict[str, Any],
    field: str,
    dend_col: str,
    gt_raw: Any,
    gold: Optional[str],
    pred_raw: Any,
    pred: Optional[str],
    scored: bool,
    exclude_reason: str = "",
    text_by_fall: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    prow = item["pred"]
    prov = provenance_from_pred_row(prow, field, text_by_fall=text_by_fall)
    match: Any = ""
    if scored and gold is not None and pred is not None:
        match = gold == pred
    row = {
        "fall": item["fall"],
        "patient_id": normalize_str(prow.get("patient_id") or prow.get("report_id") or ""),
        "field": field,
        "dendrite_column": dend_col,
        "gold_raw": gt_raw if gt_raw is not None else "",
        "gold": gold if gold is not None else "",
        "pred_raw": pred_raw if pred_raw is not None else "",
        "pred": pred if pred is not None else "",
        "match": match,
        "scored": scored,
        "exclude_reason": exclude_reason,
        **prov,
    }
    return row


def _score_binary_pairs(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    # Map Ja→True for binary_metrics helper
    yt = [t == "Ja" for t in y_true]
    yp = [p == "Ja" for p in y_pred]
    metrics = binary_metrics(yt, yp)
    metrics["type"] = "binary"
    metrics["confusion"] = categorical_accuracy(y_true, y_pred)["confusion"]
    return metrics


def score_aligned(
    aligned: List[Dict[str, Any]],
    *,
    text_by_fall: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    per_field: Dict[str, Any] = {}
    pair_rows: List[Dict[str, Any]] = {}

    for dend_col, field in DENDRITE_COLUMNS.items():
        gold_parser = GOLD_PARSERS[field]
        pred_collapse = PRED_COLLAPSE[field]
        y_true: List[str] = []
        y_pred: List[str] = []
        n_missing_gold = 0
        n_missing_pred = 0
        n_excluded = 0

        actual_col = None
        if aligned:
            sample_gt = aligned[0]["gt"]
            for c in sample_gt.keys():
                if str(c).strip().lower() == dend_col.lower():
                    actual_col = c
                    break
            if actual_col is None and field == "cerebrovascular_event":
                for c in sample_gt.keys():
                    if "neurol" in str(c).lower():
                        actual_col = c
                        break

        field_pairs: List[Dict[str, Any]] = []
        for item in aligned:
            gt_raw = item["gt"].get(actual_col) if actual_col else None
            if gt_raw is None:
                for c, v in item["gt"].items():
                    if str(c).strip().lower() == dend_col.lower():
                        gt_raw = v
                        actual_col = c
                        break

            g = gold_parser(gt_raw)
            pred_raw = item["pred"].get(field)
            p = pred_collapse(pred_raw)
            if g is None:
                n_missing_gold += 1
                n_excluded += 1
                field_pairs.append(
                    _pair_row(
                        item=item,
                        field=field,
                        dend_col=dend_col,
                        gt_raw=gt_raw,
                        gold=None,
                        pred_raw=pred_raw,
                        pred=p,
                        scored=False,
                        exclude_reason="missing_or_excluded_gold",
                        text_by_fall=text_by_fall,
                    )
                )
                continue
            if p is None:
                n_missing_pred += 1
                n_excluded += 1
                field_pairs.append(
                    _pair_row(
                        item=item,
                        field=field,
                        dend_col=dend_col,
                        gt_raw=gt_raw,
                        gold=g,
                        pred_raw=pred_raw,
                        pred=None,
                        scored=False,
                        exclude_reason="missing_pred_Unbekannt_or_k.A.",
                        text_by_fall=text_by_fall,
                    )
                )
                continue
            y_true.append(g)
            y_pred.append(p)
            field_pairs.append(
                _pair_row(
                    item=item,
                    field=field,
                    dend_col=dend_col,
                    gt_raw=gt_raw,
                    gold=g,
                    pred_raw=pred_raw,
                    pred=p,
                    scored=True,
                    text_by_fall=text_by_fall,
                )
            )

        if field in BINARY_FIELDS:
            field_metrics: Dict[str, Any] = (
                _score_binary_pairs(y_true, y_pred)
                if y_true
                else {
                    "type": "binary",
                    "support": 0,
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "tn": 0,
                    "confusion": {},
                }
            )
        else:
            field_metrics = (
                {"type": "categorical", **categorical_accuracy(y_true, y_pred)}
                if y_true
                else {
                    "type": "categorical",
                    "support": 0,
                    "accuracy": 0.0,
                    "confusion": {},
                }
            )
        field_metrics["dendrite_column"] = dend_col
        field_metrics["n_aligned_fall"] = len(aligned)
        field_metrics["n_scored"] = len(y_true)
        field_metrics["n_excluded"] = n_excluded
        field_metrics["n_missing_gold"] = n_missing_gold
        field_metrics["n_missing_pred"] = n_missing_pred
        per_field[field] = field_metrics
        pair_rows[field] = field_pairs

    return {
        "n_aligned_fall": len(aligned),
        "per_field": per_field,
        "pairs": pair_rows,
    }


def load_dendrite_fall_ids(path: PathLike) -> set[str]:
    """Unique normalized FallNummer FID values from a Dendrite gold file."""
    df = load_dendrite_gold(path)
    return {normalize_str(x) for x in df["_fall"].tolist() if normalize_str(x)}


def filter_reports_by_dendrite_falls(
    reports: Sequence[Dict[str, Any]], fall_ids: set[str]
) -> List[Dict[str, Any]]:
    """
    Keep patients whose ``verlegung_fallnr`` or any ``fall_nummers`` entry is in
    *fall_ids* (Dendrite FallNummer FID set).
    """
    if not fall_ids:
        return []
    kept: List[Dict[str, Any]] = []
    for rec in reports:
        keys: List[str] = []
        vfall = normalize_str(rec.get("verlegung_fallnr", ""))
        if vfall:
            keys.append(vfall)
        falls = rec.get("fall_nummers") or []
        if isinstance(falls, str):
            falls = re.split(r"[|;,]", falls)
        for f in falls:
            k = normalize_str(f)
            if k:
                keys.append(k)
        if any(k in fall_ids for k in keys):
            kept.append(rec)
    return kept


def discover_dendrite_paths(raw_dir: Optional[Path] = None) -> List[Path]:
    from configs.config import RAW_DATA_DIR

    root = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR
    if not root.exists():
        return []
    suffixes = {".xlsx", ".xls", ".xlsm", ".csv"}
    out: List[Path] = []
    for pattern in ("Dendrite*", "*dendrite*", "*Dendrite*"):
        for p in root.glob(pattern):
            if p.is_file() and p.suffix.lower() in suffixes:
                out.append(p)
    seen = set()
    uniq: List[Path] = []
    for p in sorted(out, key=lambda x: x.name.lower()):
        if p.resolve() not in seen:
            seen.add(p.resolve())
            uniq.append(p)
    return uniq


def format_score_report(result: Dict[str, Any]) -> str:
    lines = [
        f"Aligned FallNummer (pred∩Dendrite): {result['n_aligned_fall']}",
        "",
    ]
    for field, m in result["per_field"].items():
        lines.append(f"## {field}  ←  {m.get('dendrite_column')}")
        lines.append(
            f"  scored={m.get('n_scored')}  excluded={m.get('n_excluded')} "
            f"(missing_gold={m.get('n_missing_gold')}, missing_pred={m.get('n_missing_pred')})"
        )
        lines.append(f"  accuracy={m.get('accuracy')}")
        if m.get("type") == "binary":
            lines.append(
                f"  precision={m.get('precision')}  recall={m.get('recall')}  "
                f"f1={m.get('f1')}  "
                f"tp={m.get('tp')} fp={m.get('fp')} fn={m.get('fn')} tn={m.get('tn')}"
            )
        conf = m.get("confusion") or {}
        if conf:
            lines.append(f"  confusion={json.dumps(conf, ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines)


def select_complete_pairs_for_export(
    pairs_by_field: Dict[str, List[Dict[str, Any]]],
    *,
    max_per_field: Optional[int] = 25,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Keep only rows with both gold and pred (scored=True), then sample up to
    *max_per_field* per variable.
    """
    import random

    rng = random.Random(seed)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for field, rows in pairs_by_field.items():
        complete = [r for r in rows if r.get("scored") is True]
        if max_per_field is not None and max_per_field > 0 and len(complete) > max_per_field:
            complete = rng.sample(complete, max_per_field)
            complete = sorted(complete, key=lambda r: str(r.get("fall") or ""))
        cleaned: List[Dict[str, Any]] = []
        for r in complete:
            cleaned.append({k: r.get(k, "") for k in PAIR_COLUMNS})
        out[field] = cleaned
    return out


def run_dendrite_score(
    predictions_path: PathLike,
    dendrite_path: PathLike,
    *,
    max_patients: Optional[int] = None,
    seed: int = 42,
    complete_only: bool = True,
) -> Dict[str, Any]:
    """
    Score predictions vs Dendrite gold.

    Metrics use the full overlap. Export pairs default to complete rows only
    (gold + pred both present), sampled to ``max_patients`` **per field**.
    """
    from src.evaluation.evidence_format import load_verlegung_text_by_fall

    preds = load_predictions(predictions_path)
    gold = load_dendrite_gold(dendrite_path)
    aligned = align_predictions_to_dendrite(preds, gold)
    text_by_fall = load_verlegung_text_by_fall()
    result = score_aligned(aligned, text_by_fall=text_by_fall)

    export_pairs = result.get("pairs") or {}
    if complete_only:
        export_pairs = select_complete_pairs_for_export(
            export_pairs,
            max_per_field=max_patients,
            seed=seed,
        )
    result["pairs"] = export_pairs
    result["n_patients_in_export"] = {f: len(rows) for f, rows in export_pairs.items()}
    result["complete_only"] = complete_only
    return result
