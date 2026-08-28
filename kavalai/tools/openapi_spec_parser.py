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

import copy
from typing import Any


class OpenApiSpecParser:
    def __init__(self, spec):
        self.full_spec = copy.deepcopy(spec)
        self.resolved_spec = self._resolve_all(self.full_spec)

    def _get_referenced_data(self, ref_path: str) -> dict:
        if not ref_path.startswith("#/"):
            return {"error": f"External reference {ref_path} not supported"}

        parts = ref_path.lstrip("#/").split("/")
        current = self.full_spec

        for part in parts:
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list) and part.isdigit():
                current = current[int(part)]
            else:
                raise KeyError(f"Could not resolve pointer part: {part}")

        return current

    def _resolve_all(self, data: Any, seen_refs: frozenset = frozenset()) -> Any:
        """Inline every local ``$ref``; a circular reference is left as is."""
        if isinstance(data, dict):
            if "$ref" in data:
                ref = data["$ref"]
                if ref in seen_refs:
                    return data
                return self._resolve_all(
                    self._get_referenced_data(ref), seen_refs | {ref}
                )
            return {k: self._resolve_all(v, seen_refs) for k, v in data.items()}

        if isinstance(data, list):
            return [self._resolve_all(item, seen_refs) for item in data]

        return data

    def get_path_request_schema(self, path: str, method: str) -> dict:
        method = method.lower()
        op = self.resolved_spec.get("paths", {}).get(path, {}).get(method, {})
        return op["requestBody"]["content"]["application/json"]["schema"]

    def get_path_response_schema(self, path: str, method: str) -> dict:
        method = method.lower()
        op = self.resolved_spec.get("paths", {}).get(path, {}).get(method, {})
        return op["responses"]["200"]["content"]["application/json"]["schema"]
