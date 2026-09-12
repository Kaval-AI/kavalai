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
| Store | Postgres + pgvector, a table per collection with HNSW + GIN indexes | one sqlite-vector file, same `rag_collections` registry and a table per collection |
| Use for | production | local development, tests, a portable file index |

Both share one storage model, so the backoffice RAG explorer shows either. A
SQLite file written by kavalai 1.0 (single `rag_index` table) is refused —
rebuild it. `rag_service_from_uri("postgresql://…" | "sqlite:///file.db", model)`
picks the class from the URI.

**The registry's model is authoritative.** Each collection records the model it
was created with, and querying or indexing it embeds with *that* model; the
service's `model` is only the model for collections it creates (a mismatch
logs a warning). So `model=None` queries and indexes existing collections, and
only creating one raises. One embedding client is kept per model.

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
    auto_create=True,                # False: the file and registry must exist
)
```

Everything after `model` is keyword-only (`normalizer`, `schema`, `provision`,
`stats_receiver`, `vector_type`); `PostgresRagService` has no `agent`
parameter any more.

A collection is provisioned lazily on first index (the embedding dimension
comes from the first batch) or explicitly via
`create_collection(name, dim, model=…, vector_type=…)`. Read paths
(`list_collections`, `count_entries`, `query`, deletes) never create the
registry. `provision=False` issues no DDL at all: indexing into a missing
collection raises, and an owner-privileged job calls `ensure_registry()` and
`create_collection()` (both issue DDL regardless) — the pattern for a
runtime role with data privileges only. On Postgres, `vector_type="halfvec"`
(pgvector ≥ 0.7) stores 16-bit floats and is the only way to index more than
2,000 dimensions (up to 4,000, e.g. `text-embedding-3-large`); SQLite accepts
only `"vector"`.

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
- **Optional (four)**: `count_entries`, `iter_entries`, `delete_by_metadata`,
  `replace` — guard with `service.supports("replace")` before calling.
- **Defaulted (three)**: `compute_similarity_matrix`, `learn_normalizer`,
  `delete_many`.

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

- **`source_ids=None` searches everything; `source_ids=[]` matches nothing**
  (an empty result, without embedding). Pass an empty pre-filter through as
  `[]` — never turn it into `None`, which widens it to the whole collection.
- `min_similarity=` on `query`/`query_batch` drops weaker hits *after* `top_k`,
  so fewer may come back. A useful value depends on the model and normaliser.
- `stats_receiver=` (per call, or on the constructor as the default) receives
  each embedding call's `ModelCallStat`. Without one nothing is recorded — an
  indexing job passes `StatsBridge(task_logger, agent_id)`.

Re-indexing a document is `replace`, not delete-then-index: it embeds first,
then deletes and inserts in one transaction, so a failure keeps the old rows.

```python
if service.supports("replace"):
    await service.replace("handbook", chunks, metas,
                          match={"page_id": "p-17"})  # or source_id="…"
await service.delete_by_metadata("handbook", {"page_id": "p-17"})
await service.delete_many([id1, id2], collection_name="handbook")
```

`replace` needs `match=` or `source_id=`; empty `texts` only deletes. A
`match` is equality on top-level keys with scalar values (str, int, float,
bool); an empty dict, a nested value or a key containing `"` raises
`ValueError`.

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
- `min_similarity: 0.5` drops hits below the threshold; `source_ids: []`
  matches nothing, as on the service.
- The node **records its hits** whatever `store` says — `id`, `source_id`,
  `similarity`, `metadata`, no text — on its task row and in the
  `node_completed` event's `output_data` (`{"hits": [...]}`). Cite from
  those; do not query again. The query embedding is counted in the run's
  token usage.

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

The wheel ships no indexing CLI. The repository's example
`examples/ragindex/index_csv.py` is the pattern to copy; it streams any CSV
into a collection:

```bash
python -m examples.ragindex.index_csv data.csv \
    --index postgresql://…/kavalai --schema agents \
    --collection handbook \
    --text-columns title,body \
    --metadata-columns author,date \
    --source-id-column id \
    --model openai/text-embedding-3-small \
    --batch-size 32 --replace
```

`--text-columns` are joined into the embedded text; `--source-id-column`
becomes `source_id` (an empty string numbers the rows); `--metadata-columns`
are kept alongside; `--where COLUMN=VALUE` and `--limit` select rows.
`--replace` swaps the rows with the same source ids — omit it and a re-run
duplicates; `--skip-existing` embeds only source ids not yet present.
`--index` is a database URI (a Postgres one is where the backoffice RAG
explorer looks) or a SQLite file path.

## Indexing HTML and markdown

Do not write a chunker. `kavalai.text` (standard library only, so it runs
under Pyodide) turns a page into `index_batch` input:
`parse_html(html, base_url=…)` returns the title, text blocks with their
heading paths, links and robots meta; `chunk_blocks(page.blocks,
title=page.title)` or `chunk_markdown(md, title=…)` return chunks of about
`target_chars` (1200), never over `max_chars` (2000), each prefixed with the
title and heading path. Keep `chunk.heading` and `chunk.position` in the
metadata so an answer can cite its section.

## Embeddings

`make_embedding_client("openai/text-embedding-3-small")`, or
`fastembed/<model>` to embed locally with no API key. The model is always an
argument — there is no environment default for it. In a container set
`FASTEMBED_CACHE_DIR` so the model is not re-downloaded on every start, and
`FASTEMBED_THREADS` to bound CPU. For an NVIDIA GPU, install the `gpu` extra by
*replacing* `fastembed` — no code change follows.

**The embedding model is part of the index.** The registry records it and the
service always embeds a collection with it, so a service configured with
another model cannot corrupt a query. Changing model means a new collection,
indexed from scratch.

A `Normalizer` can be learned from the corpus and applied to embeddings;
`learn_normalizer` is the defaulted method that produces one, and
`set_default_normalizer` installs it for every embedding client — the agent
server does that from `KAVALAI_EMBEDDING_NORMALIZER_YAML` at start-up.
