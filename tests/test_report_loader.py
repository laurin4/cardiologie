from src.preprocessing.report_loader import (
    REPORT_ID_KEY,
    REPORT_TEXT_KEY,
    load_reports,
    load_reports_from_csv,
    load_reports_from_txt_dir,
)


def test_load_txt_dir(tmp_path):
    (tmp_path / "r1.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "r2.txt").write_text("world", encoding="utf-8")
    records = load_reports_from_txt_dir(tmp_path)
    assert len(records) == 2
    ids = {r[REPORT_ID_KEY] for r in records}
    assert ids == {"r1", "r2"}


def test_load_csv(tmp_path):
    p = tmp_path / "reports.csv"
    p.write_text(
        "report_id;report_text;dept\n1;fever noted;icu\n2;stable;ward\n", encoding="utf-8"
    )
    records = load_reports_from_csv(p, id_column="report_id", text_column="report_text")
    assert len(records) == 2
    assert records[0][REPORT_TEXT_KEY] == "fever noted"
    assert records[0]["dept"] == "icu"


def test_load_reports_dispatch(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    records = load_reports(tmp_path)
    assert len(records) == 1
