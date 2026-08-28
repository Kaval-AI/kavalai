import json
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from kavalai.db import ModelCallStat
from kavalai.llm_clients.common import (
    create_model_call_stat,
    get_model_name,
    safe_parse_json,
    safe_model_validate,
)


class SimpleModel(BaseModel):
    name: str


# --- safe_parse_json tests ---


def test_safe_parse_json_valid():
    data = '{"a": "hello"}'
    assert safe_parse_json(data) == {"a": "hello"}
    assert safe_parse_json("[1, 2, 3]") == [1, 2, 3]


def test_safe_parse_json_trailing_chars():
    data = '{"a": "hello"}...'
    # ensure_json should truncate the trailing '...'
    parsed = safe_parse_json(data)
    assert parsed == {"a": "hello"}


def test_safe_parse_json_incomplete():
    data = '{"a": "hello'
    # partial_json_parser's ensure_json should close it
    parsed = safe_parse_json(data)
    assert parsed == {"a": "hello"}


def test_safe_parse_json_leading_garbage():
    data = 'xx asdf{"a": "hello"}..'
    parsed = safe_parse_json(data)
    assert parsed == {"a": "hello"}


def test_safe_parse_json_complex_trailing():
    data = '{"a": "hello"} some garbage here'
    parsed = safe_parse_json(data)
    assert parsed == {"a": "hello"}


def test_safe_parse_json_multiple_trailing_braces():
    data = '{"a": "hello"}}'
    parsed = safe_parse_json(data)
    assert parsed == {"a": "hello"}


def test_safe_parse_json_trailing_comma():
    data = '{"a": "hello",}'
    parsed = safe_parse_json(data)
    assert parsed == {"a": "hello"}
    assert safe_parse_json("[1, 2,]") == [1, 2]


def test_safe_parse_json_trailing_comma_partial():
    data = '{"a": "hello",'
    parsed = safe_parse_json(data)
    assert parsed == {"a": "hello"}


def test_safe_parse_json_no_structure():
    assert safe_parse_json("123") == 123
    assert safe_parse_json("true") is True
    assert safe_parse_json("null") is None
    assert safe_parse_json("not json at all") == {}


def test_safe_parse_json_extra_data_msg():
    # specifically trigger the "Extra data" branch
    data = '{"a": 1} {"b": 2}'
    assert safe_parse_json(data) == {"a": 1}
    with patch("json.loads") as mock_loads:
        mock_loads.side_effect = [
            json.JSONDecodeError("Extra data", '{"a": 1} {', 8),
            Exception("inner failure"),
            {"a": 1},  # ensure_json result (mocked via json.loads)
        ]
        assert safe_parse_json('{"a": 1} {') == {"a": 1}


def test_safe_parse_json_ensure_json_fails():
    with patch("kavalai.llm_clients.common.ensure_json") as mock_ensure:
        mock_ensure.side_effect = Exception("error")
        assert safe_parse_json('{"a": 1} broken') == {"a": 1}


def test_safe_parse_json_ensure_json_twice_fails():
    with patch("kavalai.llm_clients.common.ensure_json") as mock_ensure:
        mock_ensure.side_effect = [Exception("error1"), Exception("error2")]
        assert safe_parse_json('{"a": 1} broken') == {"a": 1}


def test_safe_parse_json_fallback_slicing():
    assert safe_parse_json('{"a": 1} ]') == {"a": 1}
    assert safe_parse_json("[1, 2] }") == [1, 2]


def test_safe_parse_json_fallback_fails_json_loads():
    with patch("kavalai.llm_clients.common.ensure_json") as mock_ensure:
        mock_ensure.side_effect = Exception("error")
        try:
            json.loads('{"a": 1} broken')
        except json.JSONDecodeError as e:
            print(f"DEBUG: {e.msg}")

        assert safe_parse_json('{"a": 1} broken') == {"a": 1}


def test_safe_parse_json_fallback_rfind_exception():
    # Trigger Exception in the rfind loop
    with patch("kavalai.llm_clients.common.ensure_json") as mock_ensure:
        mock_ensure.side_effect = Exception("ensure_json failed")

        # We need something where rfind finds a char, but json.loads(subset) fails.
        # And ensure_json also fails.

        # Let's mock json.loads to fail inside the fallback loop.
        with patch("json.loads") as mock_loads:
            mock_loads.side_effect = [
                json.JSONDecodeError("fail", "", 0),  # initial load
                Exception("ensure failed"),  # ensure_json -> json.loads
                Exception("fallback failed"),  # fallback loop -> json.loads
            ]
            assert safe_parse_json('{"a": 1}') == {}


def test_safe_parse_json_really_all_fails():
    with patch("kavalai.llm_clients.common.ensure_json") as mock_ensure:
        mock_ensure.side_effect = Exception("error")
        assert safe_parse_json("{ no braces here") == {}


# --- safe_model_validate tests ---


class TestModel(BaseModel):
    field1: str
    field2: int


def test_safe_model_validate_valid_dict():
    """Test that safe_model_validate works with valid JSON dict."""
    json_str = '{"field1": "test", "field2": 42}'
    result = safe_model_validate(TestModel, json_str)
    assert result.field1 == "test"
    assert result.field2 == 42


def test_safe_model_validate_empty_list():
    """Test that safe_model_validate converts empty list [] to empty dict {}."""
    json_str = "[]"
    # Should convert [] to {} and fail validation with missing field error, not type error
    with pytest.raises(Exception) as exc_info:
        safe_model_validate(TestModel, json_str)

    error_str = str(exc_info.value).lower()
    # Should be a "field required" error, not a "input should be dict" type error
    assert "required" in error_str or "missing" in error_str
    assert "input_type=list" not in str(exc_info.value)


def test_safe_model_validate_non_empty_list():
    """Test that safe_model_validate converts non-empty list to empty dict."""
    json_str = "[1, 2, 3]"
    # Lists are converted to {} before validation
    with pytest.raises(Exception) as exc_info:
        safe_model_validate(TestModel, json_str)

    error_str = str(exc_info.value).lower()
    assert "required" in error_str or "missing" in error_str
    assert "input_type=list" not in str(exc_info.value)


def test_safe_model_validate_incomplete_json():
    """Test that safe_model_validate handles incomplete JSON from streaming."""
    # Simulate what happens during streaming when LLM sends just '['
    json_str = "["
    # safe_parse_json will fix this to '[]', then safe_model_validate converts to {}
    with pytest.raises(Exception) as exc_info:
        safe_model_validate(TestModel, json_str)

    error_str = str(exc_info.value).lower()
    assert "required" in error_str or "missing" in error_str


def test_safe_model_validate_partial_dict():
    """Test that safe_model_validate handles partial dict with missing fields."""
    json_str = '{"field1": "test"}'
    with pytest.raises(Exception) as exc_info:
        safe_model_validate(TestModel, json_str)

    error_str = str(exc_info.value).lower()
    assert "field2" in error_str or "required" in error_str


def test_safe_model_validate_malformed_json():
    """Test that safe_model_validate handles malformed JSON gracefully."""
    json_str = '{"field1": "test", '
    # safe_parse_json will fix this to valid JSON
    with pytest.raises(Exception) as exc_info:
        safe_model_validate(TestModel, json_str)

    error_str = str(exc_info.value).lower()
    # Should fail on missing field2, not on JSON parsing
    assert "field2" in error_str or "required" in error_str


# --- get_model_name tests ---


def test_get_model_name():
    assert get_model_name("openai/gpt-4") == "gpt-4"
    assert get_model_name("gpt-3.5-turbo") == "gpt-3.5-turbo"


# --- create_model_call_stat tests ---


def test_create_model_call_stat_basic():
    stat = create_model_call_stat(
        call_type="llm",
        model="test-model",
        duration_seconds=1.5,
        prompt_tokens=10,
        completion_tokens=20,
        response_data={"foo": "bar"},
    )
    assert isinstance(stat, ModelCallStat)
    assert stat.call_type == "llm"
    assert stat.model == "test-model"
    assert float(stat.duration_seconds) == 1.5
    assert stat.prompt_tokens == 10
    assert stat.completion_tokens == 20
    assert stat.total_tokens == 30
    assert stat.response_data == {"foo": "bar"}
    assert stat.response_code == 200


def test_create_model_call_stat_string_response():
    stat = create_model_call_stat(
        call_type="llm",
        model="test-model",
        duration_seconds=1.0,
        response_data="just a string",
    )
    assert stat.response_data == "just a string"


def test_create_model_call_stat_none_response():
    stat = create_model_call_stat(
        call_type="llm", model="test-model", duration_seconds=1.0, response_data=None
    )
    assert stat.response_data is None


def test_create_model_call_stat_tokens():
    stat = create_model_call_stat(
        call_type="llm",
        model="test",
        duration_seconds=1.0,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=None,
    )
    assert stat.total_tokens == 30


# --- walk_json_schema tests ---


def test_walk_json_schema_visits_every_nested_object_schema():
    from kavalai.llm_clients.common import walk_json_schema

    schema = {
        "type": "object",
        "properties": {
            "child": {"type": "object", "properties": {}},
            "items": {"type": "array", "items": {"type": "object"}},
            "union": {"anyOf": [{"type": "object"}, {"type": "null"}]},
        },
        "$defs": {"Nested": {"type": "object", "allOf": [{"type": "object"}]}},
        "definitions": {"Legacy": {"oneOf": [{"type": "object"}]}},
    }
    seen = []
    walk_json_schema(schema, lambda node: seen.append(node.get("type")))

    # Root, child, the array item, the anyOf member, Nested and its allOf
    # member, and the oneOf member under definitions.
    assert seen.count("object") == 7
    assert seen.count("array") == 1
    assert seen.count("null") == 1
    assert seen.count(None) == 2  # the bare anyOf/oneOf wrappers


def test_walk_json_schema_ignores_non_dict_input():
    from kavalai.llm_clients.common import walk_json_schema

    calls = []
    walk_json_schema("not a schema", calls.append)
    walk_json_schema(None, calls.append)
    assert calls == []
