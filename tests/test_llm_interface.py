from src.llm.llm_interface import (
    _build_chat_url,
    _extract_system_user,
    extract_first_json_object,
)


def test_extract_first_json_object_basic():
    assert extract_first_json_object('prefix {"a": 1} suffix') == '{"a": 1}'


def test_extract_first_json_object_nested():
    text = '{"a": {"b": 1}, "c": [1, 2]}'
    assert extract_first_json_object(text) == text


def test_extract_first_json_object_strips_template_tokens():
    text = '<start_of_turn>model{"a": 1}<end_of_turn>'
    assert extract_first_json_object(text) == '{"a": 1}'


def test_extract_first_json_object_handles_braces_in_strings():
    text = '{"a": "not } a close"}'
    assert extract_first_json_object(text) == text


def test_build_chat_url():
    assert _build_chat_url("http://host:11500") == "http://host:11500/api/chat"
    assert _build_chat_url("http://host/api/generate") == "http://host/api/chat"


def test_extract_system_user():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    assert _extract_system_user(messages) == ("sys", "usr")
