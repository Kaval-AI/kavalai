import unicodedata
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel


def clean_text(text: Any) -> Any:
    """Normalizes string to NFC and removes null characters and other problematic non-printable chars.

    If input is not a string, returns it as is.
    """
    if not isinstance(text, str):
        return text

    # Normalize to NFC
    text = unicodedata.normalize("NFC", text)

    # Remove null characters which PostgreSQL does not support in TEXT/VARCHAR/JSONB
    text = text.replace("\u0000", "")

    # Remove other problematic non-printable characters, but keep common whitespace
    # (newline, carriage return, tab)
    return "".join(
        ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\r\t"
    )


def to_plain(obj: Any) -> Any:
    """Recursively convert Pydantic models, dicts, and lists into plain JSON-serializable types."""
    if isinstance(obj, str):
        return clean_text(obj)
    if isinstance(obj, BaseModel):
        return to_plain(obj.model_dump())
    if isinstance(obj, (datetime, UUID)):
        return str(obj)
    if isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            k = clean_text(str(k))
            # Filters potential internal attributes that might not be serializable or cause issues.
            if not k.startswith("_"):
                res[k] = to_plain(v)
        return res
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj
