---
name: kavalai-rag
description: Index and retrieve documents with Kaval.AI RAG — choosing `PostgresRagService` or `SqliteRagService`, registering a service by name, indexing text or a CSV, and the `rag_query` workflow node. Use when building retrieval, writing an embedding pipeline, or when a `rag_query` node cannot resolve its service.
---

# Kaval.AI RAG

One interface, `BaseRagService`, over two shipped backends. A workflow reaches
retrieval through a read-only `rag_query` node; indexing happens in your own
code or from a CLI.

## Pick a backend

| | `PostgresRagService` | `SqliteRagService` |
|---|---|---|
| Store | Postgres + pgvector, a table per collection with HNSW + GIN indexes | one sqlite-vector file |
| Use for | production, anything the backoffice RAG explorer should show | local development, tests, a portable file index, the browser/WASM |

```python
from kavalai import PostgresRagService, SqliteRagService
from kavalai.db import db_manager

pg = PostgresRagService(
    db_manager.get_sessionmaker(uri="postgresql://…/kavalai", schema="agents"),
    model="openai/text-embedding-3-small",
)

local = SqliteRagService(
    filename="handbook.db",          # ":memory:" for an in-memory index
    model="fastembed/BAAI/bge-small-en-v1.5",
    table_name="rag_index",
    auto_create=True,
)
```

Postgres provisions a collection lazily on first index (the embedding dimension
comes from the first batch) or explicitly via `create_collection`.

## similarity is higher-is-better

**`similarity` is cosine, reported as `1.0 - distance`, on every backend.** A
perfect match scores `1.0`; results come back ordered by *descending*
similarity. Do not invert the comparison, do not sort ascending, and do not
treat it as a distance — this is the single most common mistake carried over
from raw pgvector.

## The contract

`BaseRagService` declares three tiers, and a backend is only guaranteed the
first:

- **Required (six)**: `index`, `index_batch`, `query`, `query_batch`, `delete`,
  `delete_by_source_id`.
- **Optional (two)**: `count_entries`, `iter_entries` — guard with
  `service.supports("count_entries")` before calling.
- **Defaulted (two)**: `compute_similarity_matrix`, `learn_normalizer`.

```python
await service.index(text, source_metadata={"page": 3},
                    collection_name="handbook", source_id="handbook.pdf")

hits = await service.query("How do I book the hall?", top_k=5,
                           collection_name="handbook",
                           source_ids=None, keep_best=False,
                           include_content=True)
for hit in hits:
    print(hit.similarity, hit.source_id, hit.content)
```

`index`/`index_batch` return dicts that always carry `id`, `model`,
`collection_name`, `source_id`, `content`, `embedding_size`, `rag_metadata`,
`created_at`, `updated_at`. `query` returns `RagServiceResult` objects with
those fields plus `similarity`.

`keep_best=True` returns only the best hit per `source_id` — what you want for
a document indexed as many chunks. `include_content=False` omits the text when
the caller only needs scores.

A backend of your own implements the required tier, declares the optional one
through `supports()`, normalises the store's score to the higher-is-better
convention, and accepts a UUID id (as text if the store insists) — then runs
against the library's RAG conformance suite rather than only against its own
tests.

## Registering a service by name

A workflow names a **registration**, never a class or a connection string:

```python
from kavalai import register_rag_service, SqliteRagService

register_rag_service(
    "handbook", SqliteRagService,
    filename="handbook.db", model="fastembed/BAAI/bge-small-en-v1.5",
)
```

The target may be a class, a dotted path (imported on first use) or a callable;
`**defaults` are bound at registration, so everything the backend needs is
supplied here. Duplicates raise; `replace=True` warns.

Do this in the **setup module** (`KAVALAI_AGENT_SETUP_MODULE`) so the agent
server sees it before the workflow loads — see `kavalai-tools`.

Alternatively pass the object straight to the engine:

```python
engine = WorkflowEngine.from_yaml_path("workflow.yaml", rag_services=service)
```

A bare service is stored as `"default"`; a dict registers several by name.

## The rag_query node

```yaml
rag_service: handbook        # graph-level default
rag_collection: pages

nodes:
  - name: retrieve
    type: rag_query
    query: "{{ context.input.question }}"
    output: facts
    store: content
    top_k: 5
    next: answer
```

- **Resolution order: node `service` → graph `rag_service` → `"default"`.**
  Services passed to the engine beat registered ones. An unresolvable name
  fails when the engine is **constructed**, not when the branch first runs.
- The node is **read-only** — it reaches `query` and nothing else, so no
  workflow document can write to an index.
- `output` is the one node output that need **not** appear in `data_types`: the
  shape is Kaval.AI's, not yours.
- `store: results` (default) keeps the full hit list, so `if`/`switch` can read
  `similarity` and metadata. `store: content` stores just the hit texts joined
  by blank lines — which is what the following `llm` node's prompt usually
  wants. Choose deliberately; feeding a prompt the full result objects wastes
  tokens on ids and scores.
- `service` containing `://` is rejected at load. Register a name.

A retrieval-then-answer pair is the whole pattern:

```yaml
  - name: answer
    type: llm
    prompt: |
      Answer using only these facts. If they do not cover it, say so.

      {{ context.facts }}

      Question: {{ context.input.question }}
    output: output
    next: finish
```

## Indexing a CSV

The library ships a CLI for the common case:

```bash
python -m kavalai.tools.index_csv data.csv \
    --collection-name handbook \
    --index-fields title body \
    --source-field id \
    --metadata-fields author date \
    --model openai/text-embedding-3-small \
    --mode full --batch-size 10 --replace
```

`--index-fields` (required) are the columns embedded as content;
`--source-field` (required) becomes `source_id`; `--metadata-fields` are kept
alongside. `--mode lines` indexes each line of the field as its own entry.
`--replace` deletes matching `(collection_name, source_id)` rows first — use it
for a re-index, omit it and you will duplicate. `KAVALAI_RAG_SERVICE` names a
registered service to index into; unset means the Postgres service built from
`KAVALAI_DB_URI`.

## Embeddings

`make_embedding_client("openai/text-embedding-3-small")`, or
`fastembed/<model>` to embed locally with no API key. The model is always an
argument; only the CSV indexer falls back to `KAVALAI_DEFAULT_EMBEDDING_MODEL`
when `--model` is omitted. In a container set
`FASTEMBED_CACHE_DIR` so the model is not re-downloaded on every start, and
`FASTEMBED_THREADS` to bound CPU. For an NVIDIA GPU, install the `gpu` extra by
*replacing* `fastembed` — no code change follows.

**The embedding model is part of the index.** Changing it invalidates existing
vectors; re-index rather than mixing models in one collection.

A `Normalizer` (`KAVALAI_EMBEDDING_NORMALIZER_YAML`) can be learned from the
corpus and applied to embeddings; `learn_normalizer` is the defaulted method
that produces one.
