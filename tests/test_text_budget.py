"""Tests for LLM full-text budget / truncation."""

from configs.tasks import load_task
from src.extraction.text_budget import fit_text_for_llm, max_full_text_chars


def test_short_text_unchanged():
    task = load_task("cardiology_smoke")
    text = "Kurzer Text ohne Problem."
    out, truncated = fit_text_for_llm(text, task, max_chars=1000)
    assert truncated is False
    assert out == text


def test_long_text_truncated_keeps_keyword_window():
    task = load_task("cardiology_smoke")
    filler = "lorem ipsum " * 3000  # well over 12k
    marker = "11.01.2026 Wundrevision und Secondlook mit Verschluss Thorakotomie"
    text = filler + "\n" + marker + "\n" + filler
    out, truncated = fit_text_for_llm(text, task, max_chars=4000)
    assert truncated is True
    assert len(out) < len(text)
    assert "Wundrevision" in out or "Thorakotomie" in out
    assert "HINWEIS" in out
    assert len(out) <= 4000 + 200  # notice overhead allowed


def test_default_budget_positive():
    assert max_full_text_chars() >= 2000
