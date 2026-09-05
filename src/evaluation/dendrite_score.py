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


def _score_binary_pairs(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    # Map Ja→True for binary_metrics helper
    yt = [t == "Ja" for t in y_true]
    yp = [p == "Ja" for p in y_pred]
    metrics = binary_metrics(yt, yp)
    metrics["type"] = "binary"
    metrics["confusion"] = categorical_accuracy(y_true, y_pred)["confusion"]
    return metrics


def score_aligned(aligned: List[Dict[str, Any]]) -> Dict[str, Any]:
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
            p = pred_collapse(item["pred"].get(field))
            if g is None:
                n_missing_gold += 1
                n_excluded += 1
                continue
            if p is None:
                n_missing_pred += 1
                n_excluded += 1
                continue
            y_true.append(g)
            y_pred.append(p)
            field_pairs.append(
                {
                    "fall": item["fall"],
                    "field": field,
                    "gold_raw": gt_raw,
                    "gold": g,
                    "pred_raw": item["pred"].get(field),
                    "pred": p,
                    "match": g == p,
                }
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
    # dedupe
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


def run_dendrite_score(
    predictions_path: PathLike,
    dendrite_path: PathLike,
) -> Dict[str, Any]:
    preds = load_predictions(predictions_path)
    gold = load_dendrite_gold(dendrite_path)
    aligned = align_predictions_to_dendrite(preds, gold)
    return score_aligned(aligned)
