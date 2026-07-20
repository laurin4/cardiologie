import pytest

from src.llm.json_parsing import parse_llm_json_output


def test_parses_plain_json():
    assert parse_llm_json_output('{"a": 1}', "ctx") == {"a": 1}


def test_parses_fenced_json():
    raw = "```json\n{\"a\": true}\n```"
    assert parse_llm_json_output(raw, "ctx") == {"a": True}


def test_parses_json_with_trailing_prose():
    raw = 'Here you go: {"a": 1, "b": "x"} hope that helps'
    assert parse_llm_json_output(raw, "ctx") == {"a": 1, "b": "x"}


def test_empty_raises():
    with pytest.raises(ValueError):
        parse_llm_json_output("", "ctx")


def test_no_json_raises():
    with pytest.raises(ValueError):
        parse_llm_json_output("no json here", "ctx")
