"""Tests for Dendrite gold parsing / pred collapse / scoring."""

import pandas as pd

from src.evaluation.dendrite_score import (
    align_predictions_to_dendrite,
    collapse_af_pred,
    collapse_cva_pred,
    collapse_pacemaker_pred,
    parse_cva_gold,
    parse_mov_gold,
    parse_reop_gold,
    parse_yn_gold,
    score_aligned,
)


def test_parse_yn_and_collapse_pacemaker():
    assert parse_yn_gold("Ja (1)") == "Ja"
    assert parse_yn_gold("Nein (0)") == "Nein"
    assert parse_yn_gold("") is None
    assert collapse_pacemaker_pred("Neu") == "Ja"
    assert collapse_pacemaker_pred("Schon vorhanden") == "Nein"
    assert collapse_pacemaker_pred("Kein") == "Nein"
    assert collapse_pacemaker_pred("Unbekannt") is None
    assert collapse_pacemaker_pred("k.A.") is None


def test_parse_cva_excludes_paraparese():
    assert parse_cva_gold("Keine (0)") == "Keine"
    assert parse_cva_gold("TIA Vorübergehender Schlaganfall (1)") == "TIA"
    assert parse_cva_gold("Dauerhafter Schlaganfall (2)") == "Schlaganfall"
    assert parse_cva_gold("Paraparese (3)") is None
    assert parse_cva_gold("Paraparese (3),Paraplegie (4)") is None
    assert collapse_cva_pred("Schlaganfall") == "Schlaganfall"
    assert collapse_cva_pred("k.A.") is None


def test_parse_reop_and_mov():
    assert parse_reop_gold("Keine erneute Operation erforderlich (0)") == "Nein"
    assert parse_reop_gold("Re-Operation wegen Blutung oder Tamponade (1)") == "Ja"
    assert (
        parse_reop_gold(
            "Re-Operation wegen Blutung oder Tamponade (1),"
            "Re-Operation for other cardiac problems (5)"
        )
        == "Ja"
    )
    assert parse_mov_gold("Yes (1)") == "Ja"
    assert parse_mov_gold("No (0)") == "Nein"
    assert parse_mov_gold("Unknown (99)") is None
    assert collapse_af_pred("Vorbestehend") == "Nein"
    assert collapse_af_pred("Neu") == "Ja"


def test_align_and_score_smoke():
    gold = pd.DataFrame(
        [
            {
                "_fall": "F100",
                "PM/ICD Implant": "Ja (1)",
                "Vorhofsarrhythmie postop": "Nein (0)",
                "Neue post-OP neurol. Funktionsstörung": "Keine (0)",
                "Reoperationen": "Keine erneute Operation erforderlich (0)",
                "Multisystem failure": "No (0)",
            },
            {
                "_fall": "F200",
                "PM/ICD Implant": "Nein (0)",
                "Vorhofsarrhythmie postop": "Ja (1)",
                "Neue post-OP neurol. Funktionsstörung": "Paraparese (3)",
                "Reoperationen": "Re-Operation wegen Blutung oder Tamponade (1)",
                "Multisystem failure": "Yes (1)",
            },
        ]
    )
    preds = pd.DataFrame(
        [
            {
                "verlegung_fallnr": "F100",
                "fall_nummers": "F100",
                "pacemaker": "Neu",
                "atrial_fibrillation": "Kein",
                "cerebrovascular_event": "Keine",
                "reoperation_required": "Nein",
                "multi_system_failure": "Nein",
            },
            {
                "verlegung_fallnr": "F200",
                "fall_nummers": "F200 | F201",
                "pacemaker": "Kein",
                "atrial_fibrillation": "Neu",
                "cerebrovascular_event": "Keine",
                "reoperation_required": "Ja",
                "multi_system_failure": "Ja",
            },
        ]
    )
    aligned = align_predictions_to_dendrite(preds, gold)
    assert len(aligned) == 2
    result = score_aligned(aligned)
    assert result["n_aligned_fall"] == 2
    assert result["per_field"]["pacemaker"]["n_scored"] == 2
    assert result["per_field"]["pacemaker"]["accuracy"] == 1.0
    # F200 CVA excluded (Paraparese); F100 scored
    assert result["per_field"]["cerebrovascular_event"]["n_scored"] == 1
    assert result["per_field"]["reoperation_required"]["accuracy"] == 1.0
