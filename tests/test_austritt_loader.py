"""Tests for HER Austrittsbericht loader."""

from pathlib import Path

import pandas as pd

from src.preprocessing.austritt_loader import (
    AUSTRITT_TEXT_KEY,
    build_latest_austritt_by_fall,
    discover_her_austritt_paths,
    load_austritt_by_fall,
)


def test_build_latest_austritt_prefers_newer_berdat():
    df = pd.DataFrame(
        [
            {
                "FallNummer": "F1",
                "berdat": "2025-01-01",
                "stat_ein": "alte Zirrhose Child A",
                "anamn": "alt",
            },
            {
                "FallNummer": "F1",
                "berdat": "2025-06-01",
                "stat_ein": "Keine Leberzirrhose",
                "anamn": "neu",
            },
        ]
    )
    by_fall = build_latest_austritt_by_fall(df)
    text = by_fall["F1"][AUSTRITT_TEXT_KEY]
    assert "[Austrittsbericht]" in text
    assert "[stat_ein]" in text and "Keine Leberzirrhose" in text
    assert "alte Zirrhose" not in text
    assert "[anamn]" in text and "neu" in text


def test_load_austritt_accepts_anamnese_alias(tmp_path: Path):
    path = tmp_path / "HER_Austrittsbericht_2025.csv"
    path.write_text(
        "FallNummer;berdat;stat_ein;anamnese\n"
        "F9;2025-03-01;Child-Pugh B;Alkohol\n",
        encoding="utf-8",
    )
    by_fall = load_austritt_by_fall([path])
    text = by_fall["F9"][AUSTRITT_TEXT_KEY]
    assert "Child-Pugh B" in text
    assert "Alkohol" in text


def test_discover_her_austritt_paths(tmp_path: Path):
    a = tmp_path / "HER_Austrittsbericht_2025.csv"
    a.write_text("FallNummer;stat_ein\nF1;x\n", encoding="utf-8")
    found = discover_her_austritt_paths(tmp_path)
    assert {p.name for p in found} == {"HER_Austrittsbericht_2025.csv"}
