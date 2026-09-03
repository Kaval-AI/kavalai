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

from typing import Optional

from kavalai.db import is_sqlite_uri, sqlite_path_from_uri
from kavalai.normalizer import Normalizer
from kavalai.rag.base import BaseRagService, RagServiceResult
from kavalai.rag.collections import CollectionInfo, CollectionRagService
from kavalai.rag.postgres import PostgresRagService
from kavalai.rag.sqllite import SqliteRagService

__all__ = [
    "BaseRagService",
    "CollectionInfo",
    "CollectionRagService",
    "PostgresRagService",
    "RagServiceResult",
    "SqliteRagService",
    "rag_service_from_uri",
]


def rag_service_from_uri(
    uri: str,
    model: Optional[str] = None,
    schema: Optional[str] = None,
    normalizer: Optional[Normalizer] = None,
) -> CollectionRagService:
    """The RAG service for a database URI, chosen by its scheme.

    ``postgresql://…`` gives a :class:`PostgresRagService` on the given
    ``schema``; ``sqlite:///path`` gives a :class:`SqliteRagService` on that
    file, ``schema`` ignored. ``model`` may be omitted to browse an index
    without embedding anything.
    """
    if is_sqlite_uri(uri):
        return SqliteRagService(
            sqlite_path_from_uri(uri), model=model, normalizer=normalizer
        )
    return PostgresRagService.from_uri(
        uri, model=model, normalizer=normalizer, schema=schema
    )
