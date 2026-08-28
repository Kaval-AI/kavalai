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

from typing import Any, Dict, Type, Tuple, Optional, Literal
from pydantic import BaseModel, create_model, Field, ConfigDict


class SchemaParser:
    def __init__(self, datatypes: Dict[str, Any]):
        self.raw_schemas = datatypes
        self.models: Dict[str, Type[BaseModel]] = {}

    def _json_type_to_python(
        self, prop_def: Dict[str, Any], context: str = ""
    ) -> Tuple[Any, Any]:
        """
        Maps JSON schema types to Python types and returns
        a tuple of (type, FieldInfo).
        """
        allowed_keys = {
            "type",
            "items",
            "properties",
            "$ref",
            "enum",
            "max_length",
            "min_length",
            "required",
            "default",
        }
        for key in prop_def:
            if key not in allowed_keys:
                msg = f"Unknown attribute '{key}'"
                if context:
                    msg += f" in {context}"
                raise ValueError(msg)

        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        # Handle References
        if "$ref" in prop_def:
            python_type = self.parse_type(prop_def["$ref"])

        # A closed set of allowed values -> Literal, so the compiled model's
        # JSON schema carries the enum and structured-output backends (e.g. the
        # in-browser WebLLM grammar) constrain generation to those values.
        elif "enum" in prop_def:
            values = prop_def["enum"]
            if not values:
                raise ValueError(
                    f"'enum' must be a non-empty list in {context}"
                    if context
                    else "'enum' must be a non-empty list"
                )
            python_type = Literal[tuple(values)]

        else:
            # Determine base type
            json_type = prop_def.get("type", "string")

            if json_type == "array":
                items_def = prop_def.get("items", {})
                item_type, _ = self._json_type_to_python(
                    items_def, context=f"{context}.items"
                )
                python_type = list[item_type]
            elif json_type == "object" and "properties" in prop_def:
                # Inline object definition, create a nested model
                nested_fields = {
                    k: self._json_type_to_python(v, context=f"{context}.{k}")
                    for k, v in prop_def["properties"].items()
                }
                python_type = create_model(
                    "InlineModel",
                    __config__=ConfigDict(extra="forbid"),
                    **nested_fields,
                )
            else:
                python_type = type_map.get(json_type, Any)

        # Build Field constraints and default value
        field_kwargs = {}
        default_value = ...

        is_required = prop_def.get("required", True)
        if not is_required:
            python_type = Optional[python_type]
            default_value = prop_def.get("default", None)

        # Support maxLength -> max_length
        if "max_length" in prop_def:
            field_kwargs["max_length"] = prop_def["max_length"]

        # Support minLength -> min_length
        if "min_length" in prop_def:
            field_kwargs["min_length"] = prop_def["min_length"]

        if field_kwargs:
            return python_type, Field(default=default_value, **field_kwargs)

        return python_type, default_value

    def parse_type(self, type_name: str) -> Type[BaseModel]:
        """Recursively parses a schema definition into a Pydantic model."""
        if type_name in self.models:
            return self.models[type_name]

        schema = self.raw_schemas.get(type_name)
        if not schema:
            raise ValueError(f"Definition for '{type_name}' not found.")

        allowed_keys = {"type", "properties", "$ref"}
        for key in schema:
            if key not in allowed_keys:
                raise ValueError(f"Unknown attribute '{key}' in type '{type_name}'")

        # Handle top-level $ref
        if "$ref" in schema:
            ref_name = schema["$ref"]
            ref_model = self.parse_type(ref_name)
            # A subclass rather than an alias, so the model carries its own name.
            model = create_model(
                type_name,
                __base__=ref_model,
                __config__=ConfigDict(extra="forbid"),
            )
            self.models[type_name] = model
            return model

        fields = {
            field_name: self._json_type_to_python(
                field_def, context=f"property '{field_name}'"
            )
            for field_name, field_def in schema.get("properties", {}).items()
        }

        model = create_model(type_name, __config__=ConfigDict(extra="forbid"), **fields)
        self.models[type_name] = model
        return model

    def parse_all(self) -> Dict[str, Type[BaseModel]]:
        for type_name in self.raw_schemas:
            self.parse_type(type_name)
        return self.models
