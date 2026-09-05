"""Tests for Verlegungsbericht loader and multi-source text selection."""

from pathlib import Path

import pandas as pd

from src.preprocessing.verlegung_loader import (
    DIAGNOSELISTE_TEXT_KEY,
    VERLEGUNG_TEXT_KEY,
    build_latest_verlegung_by_fall,
    pick_verlegung_for_falls,
    select_text_for_source,
)


def test_latest_verlegung_by_fall_and_berdat():
    df = pd.DataFrame(
        [
            {
                "patnr": "P1",
                "FallNummer": "F100",
                "berdat": "2024-01-01",
                "diag": "alt",
                "epikrise": "",
                "jetziges_leiden": "",
                "prozedere": "",
            },
            {
                "patnr": "P1",
                "FallNummer": "F100",
                "berdat": "2025-06-01",
                "diag": "neu SM",
                "epikrise": "Epikrise neu",
                "jetziges_leiden": "",
                "prozedere": "Prozedere",
            },
            {
                "patnr": "P2",
                "FallNummer": "F200",
                "berdat": "2024-03-01",
                "diag": "nur P2",
                "epikrise": "",
                "jetziges_leiden": "",
                "prozedere": "",
            },
        ]
    )
    by_fall = build_latest_verlegung_by_fall(df)
    assert set(by_fall) == {"F100", "F200"}
    assert "neu SM" in by_fall["F100"][VERLEGUNG_TEXT_KEY]
    assert "alt" not in by_fall["F100"][VERLEGUNG_TEXT_KEY]
    assert "[diag]" in by_fall["F100"][VERLEGUNG_TEXT_KEY]
    assert by_fall["F100"]["n_verlegung_rows"] == 2


def test_pick_verlegung_for_falls_prefers_latest_berdat():
    by_fall = {
        "F1": {
            VERLEGUNG_TEXT_KEY: "old",
            "verlegung_berdat": "2024-01-01",
            "_berdat_sort": pd.Timestamp("2024-01-01"),
        },
        "F2": {
            VERLEGUNG_TEXT_KEY: "new",
            "verlegung_berdat": "2025-06-01",
            "_berdat_sort": pd.Timestamp("2025-06-01"),
        },
    }
    chosen = pick_verlegung_for_falls(by_fall, ["F1", "F2"])
    assert chosen is not None
    assert chosen[VERLEGUNG_TEXT_KEY] == "new"
    assert chosen["n_fall_matches"] == 2
    assert "_berdat_sort" not in chosen


def test_select_text_for_source_modes():
    report = {
        DIAGNOSELISTE_TEXT_KEY: "Diagnose Text",
        VERLEGUNG_TEXT_KEY: "Verlegung Text",
        "report_text": "Diagnose Text",
    }
    assert select_text_for_source(report, "diagnoseliste") == "Diagnose Text"
    assert select_text_for_source(report, "verlegung") == "Verlegung Text"
    both = select_text_for_source(report, "both")
    assert "Diagnose Text" in both and "Verlegung Text" in both
    assert select_text_for_source(report, "report") == "Diagnose Text"


def test_latest_verlegung_accepts_ips_truncated_section_names():
    """HER_IPS exports use jetztleid / procedere instead of full names."""
    df = pd.DataFrame(
        [
            {
                "patnr": "P1",
                "FallNummer": "F100",
                "berdat": "2025-06-01",
                "diag": "Diagnose IPS",
                "epikrise": "Epikrise IPS",
                "jetztleid": "Jetztleid Text",
                "procedere": "Procedere Text",
            },
        ]
    )
    by_fall = build_latest_verlegung_by_fall(df)
    text = by_fall["F100"][VERLEGUNG_TEXT_KEY]
    assert "[diag]" in text and "Diagnose IPS" in text
    assert "[jetziges_leiden]" in text and "Jetztleid Text" in text
    assert "[prozedere]" in text and "Procedere Text" in text


def test_discover_her_ips_verlegung_paths(tmp_path: Path):
    from src.preprocessing.verlegung_loader import discover_her_verlegung_paths

    classic = tmp_path / "HER_Verlegungsbericht_old.csv"
    ips = tmp_path / "HER_IPS_Verlegungsbericht_2025.xlsx"
    classic.write_text("FallNummer;diag\nF1;x\n", encoding="utf-8")
    ips.write_bytes(b"PK\x03\x04")  # minimal zip header; discovery only checks suffix/name
    # write a real tiny excel via openpyxl/pandas if needed — discovery only needs filename
    ips.write_bytes(b"")  # empty file still matches glob + suffix
    found = discover_her_verlegung_paths(tmp_path)
    names = {p.name for p in found}
    assert "HER_Verlegungsbericht_old.csv" in names
    assert "HER_IPS_Verlegungsbericht_2025.xlsx" in names
