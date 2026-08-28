"""
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Any, Callable, Optional, Type

import json
from partial_json_parser import ensure_json
from pydantic import BaseModel
from loguru import logger
from kavalai.db import ModelCallStat
from kavalai.utils import to_plain


def safe_parse_json(data: str) -> Any:
    """
    Attempt to parse a JSON string that may be malformed, incomplete, or contain garbage characters.

    This function employs several strategies to recover JSON data:
    1. Trims leading characters until the first '{' or '['.
    2. Attempts standard `json.loads`.
    3. If 'Extra data' is detected, it truncates the string at the error position and retries.
    4. Uses `partial_json_parser` to fix truncated or incomplete JSON.
    5. Specifically handles common issues like trailing commas.
    6. Falls back to finding the last valid closing brace/bracket.

    Args:
        data: The input string potentially containing JSON.

    Returns:
        The parsed JSON data (dict, list, or primitive) or an empty dict if parsing fails completely.
    """
    # Find the start of JSON (first { or [)
    start_pos = -1
    for i, char in enumerate(data):
        if char in ("{", "["):
            start_pos = i
            break

    if start_pos == -1:
        # No JSON structure found, try to parse as is or return empty dict
        try:
            return json.loads(data)
        except Exception:
            return {}

    data = data[start_pos:]

    # Fast path: already valid JSON
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        # Handle extra trailing data
        if e.msg == "Extra data":
            try:
                return json.loads(data[: e.pos].strip())
            except Exception:
                pass

    # Try partial JSON parser
    try:
        fixed = ensure_json(data)
        return json.loads(fixed)
    except Exception:
        # Some partial_json_parser versions might fail on trailing commas.
        # Try a simple replacement for common case: ,} -> } and ,] -> ]
        try:
            fixed = ensure_json(data.replace(",}", "}").replace(",]", "]"))
            return json.loads(fixed)
        except Exception:
            pass

    # Fallback: find last valid closing brace/bracket
    for char in ("}", "]"):
        pos = data.rfind(char)
        if pos == -1:
            continue
        try:
            subset = data[: pos + 1]
            return json.loads(subset)
        except Exception:
            continue

    return {}


def safe_model_validate(response_model: Type[BaseModel], json_str: str) -> BaseModel:
    """
    Safely validate JSON string against a Pydantic BaseModel.

    Handles the case where LLM returns an empty/invalid list [] instead of a dict,
    which can happen during streaming failures, timeouts, or model errors.

    Args:
        response_model: The Pydantic model class to validate against
        json_str: The JSON string to parse and validate

    Returns:
        Validated instance of response_model

    Raises:
        ValidationError: If validation fails with a clear error message
    """
    parsed = safe_parse_json(json_str)

    # A list cannot validate against a model; a truncated or failed stream
    # sometimes leaves "[]" behind.
    if isinstance(parsed, list):
        logger.error(f"LLM returned invalid JSON: {json_str}")
        parsed = {}

    return response_model.model_validate(parsed)


def get_model_name(model: str) -> str:
    """
    Extract the short model name from a provider/model syntax.

    Example: 'openai/gpt-4' becomes 'gpt-4'.

    Args:
        model: The full model string.

    Returns:
        The extracted model name.
    """
    return model.rsplit("/", 1)[-1]


def walk_json_schema(schema: Any, visit: Callable[[dict], None]) -> None:
    """Call ``visit`` on ``schema`` and on every nested object schema.

    Descends through ``properties``, ``items``, ``allOf``/``anyOf``/``oneOf``
    and ``$defs``/``definitions`` — the places a Pydantic-generated schema
    keeps sub-schemas. Non-dict input is ignored.
    """
    if not isinstance(schema, dict):
        return
    visit(schema)
    for prop_schema in schema.get("properties", {}).values():
        walk_json_schema(prop_schema, visit)
    if "items" in schema:
        walk_json_schema(schema["items"], visit)
    for key in ("allOf", "anyOf", "oneOf"):
        for sub_schema in schema.get(key, ()):
            walk_json_schema(sub_schema, visit)
    for key in ("$defs", "definitions"):
        for def_schema in schema.get(key, {}).values():
            walk_json_schema(def_schema, visit)


def create_model_call_stat(
    call_type: str,
    model: str,
    duration_seconds: float,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    batch_size: Optional[int] = None,
    response_code: int = 200,
    response_data: Any = None,
) -> ModelCallStat:
    """
    Helper function to create a ModelCallStat record for database logging.

    Automatically calculates total_tokens if prompt and completion tokens are provided.
    Normalizes response_data to a plain format using `to_plain`.

    Args:
        call_type: Category of the call (e.g., 'llm', 'embedding').
        model: Name of the model used.
        duration_seconds: Time taken for the call in seconds.
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens in the output completion.
        total_tokens: Total tokens used (calculated if None).
        batch_size: Number of items in a batch call, if applicable.
        response_code: HTTP response status code.
        response_data: Raw response data from the provider.

    Returns:
        A populated ModelCallStat object.
    """
    if (
        total_tokens is None
        and prompt_tokens is not None
        and completion_tokens is not None
    ):
        total_tokens = prompt_tokens + completion_tokens

    return ModelCallStat(
        call_type=call_type,
        model=model,
        response_code=response_code,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        batch_size=batch_size,
        duration_seconds=duration_seconds,
        response_data=to_plain(response_data),
    )
