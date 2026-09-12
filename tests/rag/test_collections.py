"""The backend contract of ``CollectionRagService``.

The behaviour of the shared storage model is exercised against both SQL
backends in ``test_conformance.py``. What remains here is the part no backend
shows: a subclass that forgets a hook is told so by name.
"""

import pytest

from kavalai.rag.collections import CollectionInfo, CollectionRagService

INFO = CollectionInfo("c", "rag_c_c", "fake/model", 4, 1)

HOOKS = [
    ("_connection", ()),
    ("_commit", (None,)),
    ("_rollback", (None,)),
    ("_registry_exists", (None,)),
    ("_create_registry", (None,)),
    ("_fetch_registry_row", (None, "c")),
    ("_list_registry", (None,)),
    ("_insert_registry_row", (None, INFO)),
    ("_set_registry_version", (None, "c", 1)),
    ("_delete_registry_row", (None, "c")),
    ("_create_collection_table", (None, INFO)),
    ("_drop_collection_table", (None, INFO)),
    ("_count_rows", (None, INFO)),
    ("_insert_rows", (None, INFO, [])),
    ("_delete_rows", (None, INFO, [])),
    ("_delete_rows_by_source_ids", (None, INFO, [])),
    ("_delete_rows_by_metadata", (None, INFO, {"k": "v"})),
    ("_iter_rows", (None, INFO, 10)),
    ("_fetch_embeddings", (None, INFO, [])),
    ("_scan", (None, INFO, [], 5, None, False)),
]


@pytest.mark.parametrize("hook, args", HOOKS, ids=[name for name, _ in HOOKS])
async def test_every_statement_hook_is_the_backend_s_to_write(hook, args):
    service = CollectionRagService(model="fake/model")

    with pytest.raises(NotImplementedError):
        result = getattr(service, hook)(*args)
        if hasattr(result, "__await__"):
            await result


async def test_prepare_collection_is_optional():
    """A backend with no per-connection set-up need not override it."""
    service = CollectionRagService(model="fake/model")
    assert await service._prepare_collection(None, INFO) is None
