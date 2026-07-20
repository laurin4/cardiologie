from src.utils.tabular_io import read_tabular


def test_reads_semicolon_csv(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("a;b;c\n1;2;3\n", encoding="utf-8")
    df = read_tabular(p)
    assert list(df.columns) == ["a", "b", "c"]
    assert df.shape == (1, 3)


def test_reads_comma_csv(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("a,b\nx,y\n", encoding="utf-8")
    df = read_tabular(p)
    assert list(df.columns) == ["a", "b"]


def test_explicit_separator(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("a\tb\n1\t2\n", encoding="utf-8")
    df = read_tabular(p, sep="\t")
    assert list(df.columns) == ["a", "b"]
