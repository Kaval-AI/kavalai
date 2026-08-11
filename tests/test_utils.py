import unicodedata
from pydantic import BaseModel
from kavalai.utils import to_plain
from kavalai.utils import clean_text


class DemoModel(BaseModel):
    name: str
    data: dict


def test_clean_text_basic():
    """Test basic functionality of clean_text."""
    # String stays string
    assert clean_text("hello") == "hello"
    # Non-string stays as is
    assert clean_text(123) == 123
    assert clean_text(None) is None
    assert clean_text(True) is True


def test_clean_text_normalization():
    """Test NFC normalization."""
    # e + combining acute accent (NFD)
    combined = "e\u0301"
    normalized = clean_text(combined)
    # Should be single character é (NFC)
    assert normalized == "\u00e9"
    assert len(normalized) == 1
    assert unicodedata.is_normalized("NFC", normalized)

    # Multi-character normalization
    nfd_string = "a\u0308o\u0308u\u0308"  # äöü in NFD
    nfc_string = "äöü"
    assert clean_text(nfd_string) == nfc_string


def test_clean_text_null_chars():
    """Test removal of null characters."""
    with_null = "Hello\u0000World"
    cleaned = clean_text(with_null)
    assert cleaned == "HelloWorld"
    assert "\u0000" not in cleaned

    # Multiple nulls
    assert clean_text("\u0000\u0000a\u0000b\u0000") == "ab"


def test_clean_text_control_chars():
    """Test removal of control characters, preserving common whitespace."""
    # \x07: Bell, \x1b: Escape, \x00: Null (handled separately but also a control char)
    # \n, \r, \t should be preserved
    text = "Line1\nLine2\r\t\x07\x1b"
    cleaned = clean_text(text)
    assert cleaned == "Line1\nLine2\r\t"
    assert "\x07" not in cleaned
    assert "\x1b" not in cleaned
    assert "\n" in cleaned
    assert "\r" in cleaned
    assert "\t" in cleaned

    # Other non-printable chars (C category)
    # \u0085 is Next Line (Cc), should be removed if not in the whitelist
    assert clean_text("a\u0085b") == "ab"


def test_to_plain_integration():
    """Test integration with to_plain which uses clean_text."""
    null_char_str = "Hello\u0000World"
    combined = "e\u0301"

    input_dict = {"key1": null_char_str, "key2": [combined, {"inner": null_char_str}]}

    result = to_plain(input_dict)

    assert result["key1"] == "HelloWorld"
    assert result["key2"][0] == "\u00e9"
    assert result["key2"][1]["inner"] == "HelloWorld"

    # Test with Pydantic model
    model = DemoModel(name=null_char_str, data={"msg": combined})
    result_model = to_plain(model)

    assert result_model["name"] == "HelloWorld"
    assert result_model["data"]["msg"] == "\u00e9"


def test_clean_text_empty_and_whitespace():
    """Test empty strings and strings with only whitespace."""
    assert clean_text("") == ""
    assert clean_text("   ") == "   "
    assert clean_text("\n\n\t") == "\n\n\t"


def test_clean_text_unicode_emojis():
    """Test that emojis and other non-control Unicode characters are preserved."""
    emoji_str = "Hello 🚀 World 🌍"
    assert clean_text(emoji_str) == emoji_str

    # Mixed with problematic chars
    mixed = "🚀\u0000🌍\x07"
    assert clean_text(mixed) == "🚀🌍"


def test_to_plain_stringifies_datetimes_and_uuids():
    from datetime import datetime
    from uuid import UUID

    moment = datetime(2026, 8, 11, 9, 30)
    identifier = UUID("11111111-2222-3333-4444-555555555555")

    assert to_plain(moment) == "2026-08-11 09:30:00"
    assert to_plain(identifier) == "11111111-2222-3333-4444-555555555555"
    assert to_plain({"at": moment, "id": identifier}) == {
        "at": "2026-08-11 09:30:00",
        "id": "11111111-2222-3333-4444-555555555555",
    }


def test_to_plain_passes_scalars_through():
    assert to_plain(5) == 5
    assert to_plain(1.5) == 1.5
    assert to_plain(None) is None
    assert to_plain(True) is True
