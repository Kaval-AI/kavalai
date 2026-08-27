"""Tests for the CSV indexing and querying example.

They never compute a real embedding: the RAG service is a stand-in that
records what it was handed. What is worth testing here is the part this
example owns --- which columns become the text, which become metadata, how
rows are filtered and batched, and which backend a given ``--index`` selects.
"""

import csv
import uuid

import pytest

from examples.ragindex.index_csv import (
    DEFAULT_MAX_CHARS,
    IndexRow,
    existing_source_ids,
    build_metadata,
    build_parser,
    build_text,
    index_rows,
    main,
    make_rag_service,
    matches,
    parse_where,
    read_rows,
    run,
    split_columns,
)
from examples.ragindex.query_index import build_parser as build_query_parser
from examples.ragindex.query_index import format_result
from examples.ragindex.query_index import main as query_main
from examples.ragindex.query_index import run as query_run
from kavalai.rag import PostgresRagService, SqliteRagService
from kavalai.rag.base import RagServiceResult

SONGS_CSV = "examples/ragindex/songs.csv"


class FakeRag:
    """A RAG service that records calls instead of embedding anything."""

    def __init__(self, results=None, existing=(), can_iter=True):
        self.batches = []
        self.deleted = []
        self.queries = []
        self._results = results or []
        self._existing = list(existing)
        self._can_iter = can_iter

    def supports(self, capability):
        return capability == "iter_entries" and self._can_iter

    async def iter_entries(self, collection_name, batch_size=500):
        for source_id in self._existing:
            yield {"source_id": source_id, "content": "x"}

    async def index_batch(self, texts, metadata_list, source_ids, collection_name):
        self.batches.append((texts, metadata_list, source_ids, collection_name))
        return [{} for _ in texts]

    async def delete_by_source_id(self, collection_name, source_id):
        self.deleted.append((collection_name, list(source_id)))

    async def query(self, text, top_k, collection_name, source_ids, keep_best):
        self.queries.append((text, top_k, collection_name, source_ids, keep_best))
        return self._results


def make_result(similarity=0.5, content="Some Song\nSome Artist\nlyrics"):
    return RagServiceResult(
        id=uuid.uuid4(),
        model="fastembed/BAAI/bge-small-en-v1.5",
        collection_name="songs",
        source_id="1",
        content=content,
        embedding_size=384,
        rag_metadata={"title": "Some Song", "artist": "Some Artist"},
        similarity=similarity,
    )


@pytest.fixture
def csv_file(tmp_path):
    """A small CSV with the same columns as the bundled songs file."""
    path = tmp_path / "rows.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["title", "artist", "lyrics", "tag", "year", "id", "language"])
        writer.writerow(["First", "Anna", "la la", "pop", "1999", "10", "en"])
        writer.writerow(["Second", "Bob", "da da", "rock", "2001", "11", "fr"])
        writer.writerow(["Third", "Cy", "na na", "pop", "2003", "12", "en"])
        writer.writerow(["", "", "", "pop", "2005", "13", "en"])
    return str(path)


# --------------------------------------------------------------------------
# Column and row helpers
# --------------------------------------------------------------------------


def test_split_columns_ignores_blanks_and_spacing():
    assert split_columns(" title , artist ,, lyrics ") == ["title", "artist", "lyrics"]
    assert split_columns("") == []


def test_parse_where_builds_a_filter():
    assert parse_where(["language=en", "tag=pop"]) == {"language": "en", "tag": "pop"}


def test_parse_where_accepts_an_empty_value():
    assert parse_where(["features="]) == {"features": ""}


@pytest.mark.parametrize("clause", ["language", "=en"])
def test_parse_where_rejects_a_clause_without_a_column_and_equals(clause):
    with pytest.raises(ValueError, match="COLUMN=VALUE"):
        parse_where([clause])


def test_build_text_joins_columns_and_drops_empty_ones():
    row = {"title": "First", "artist": "", "lyrics": "la la"}
    assert build_text(row, ["title", "artist", "lyrics"], 0) == "First\nla la"


def test_build_text_truncates_to_max_chars():
    row = {"lyrics": "abcdefghij"}
    assert build_text(row, ["lyrics"], 4) == "abcd"


def test_build_text_keeps_everything_when_max_chars_is_zero():
    row = {"lyrics": "abcdefghij"}
    assert build_text(row, ["lyrics"], 0) == "abcdefghij"


def test_build_metadata_keeps_only_populated_columns():
    row = {"title": "First", "artist": "", "year": 1999}
    assert build_metadata(row, ["title", "artist", "year", "absent"]) == {
        "title": "First",
        "year": "1999",
    }


def test_matches_requires_every_clause():
    row = {"language": "en", "tag": "pop"}
    assert matches(row, {"language": "en", "tag": "pop"})
    assert not matches(row, {"language": "en", "tag": "rock"})
    assert matches(row, {})


# --------------------------------------------------------------------------
# Reading the file
# --------------------------------------------------------------------------


def test_read_rows_maps_text_metadata_and_source_id(csv_file):
    rows = list(read_rows(csv_file, ["title", "lyrics"], ["tag"], "id"))
    assert [row.source_id for row in rows] == ["10", "11", "12"]
    assert rows[0].text == "First\nla la"
    assert rows[0].metadata == {"tag": "pop"}


def test_read_rows_skips_rows_with_no_text(csv_file):
    # The fourth row has an empty title and lyrics: there is nothing to embed.
    rows = list(read_rows(csv_file, ["title", "lyrics"], ["tag"], "id"))
    assert "13" not in [row.source_id for row in rows]


def test_read_rows_numbers_rows_when_no_source_id_column(csv_file):
    rows = list(read_rows(csv_file, ["title"], [], None))
    assert [row.source_id for row in rows] == ["0", "1", "2"]


def test_read_rows_applies_the_filter(csv_file):
    rows = list(read_rows(csv_file, ["title"], [], "id", filters={"language": "en"}))
    assert [row.source_id for row in rows] == ["10", "12"]


def test_read_rows_stops_at_the_limit(csv_file):
    rows = list(read_rows(csv_file, ["title"], [], "id", limit=2))
    assert [row.source_id for row in rows] == ["10", "11"]


def test_read_rows_rejects_a_column_the_file_does_not_have(csv_file):
    with pytest.raises(ValueError, match="no column\\(s\\) nope"):
        list(read_rows(csv_file, ["nope"], [], "id"))


def test_read_rows_reports_a_missing_filter_column(csv_file):
    with pytest.raises(ValueError, match="no column\\(s\\) missing"):
        list(read_rows(csv_file, ["title"], [], "id", filters={"missing": "x"}))


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------


def test_make_rag_service_returns_sqlite_for_a_path(tmp_path):
    service = make_rag_service(str(tmp_path / "index.db"), "fastembed/model", None)
    assert isinstance(service, SqliteRagService)
    service.close()


def test_make_rag_service_uses_the_environment_for_postgres(monkeypatch):
    captured = {}

    def fake_from_uri(uri, model, schema=None):
        captured.update(uri=uri, model=model, schema=schema)
        return "service"

    monkeypatch.setattr(PostgresRagService, "from_uri", fake_from_uri)
    monkeypatch.setenv("KAVALAI_DB_URI", "postgresql://db/x")
    monkeypatch.setenv("KAVALAI_DB_SCHEMA", "agents")

    assert make_rag_service("postgres", "fastembed/model", None) == "service"
    assert captured == {
        "uri": "postgresql://db/x",
        "model": "fastembed/model",
        "schema": "agents",
    }


def test_make_rag_service_prefers_an_explicit_schema(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        PostgresRagService,
        "from_uri",
        lambda uri, model, schema=None: captured.update(schema=schema),
    )
    monkeypatch.setenv("KAVALAI_DB_URI", "postgresql://db/x")
    monkeypatch.setenv("KAVALAI_DB_SCHEMA", "agents")

    make_rag_service("postgres", "fastembed/model", "other")
    assert captured == {"schema": "other"}


def test_make_rag_service_accepts_a_uri_directly(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        PostgresRagService,
        "from_uri",
        lambda uri, model, schema=None: captured.update(uri=uri),
    )
    make_rag_service("postgresql://elsewhere/db", "fastembed/model", None)
    assert captured == {"uri": "postgresql://elsewhere/db"}


def test_make_rag_service_needs_the_uri_to_be_set(monkeypatch):
    monkeypatch.delenv("KAVALAI_DB_URI", raising=False)
    with pytest.raises(KeyError):
        make_rag_service("postgres", "fastembed/model", None)


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------


async def test_index_rows_batches_and_counts():
    rag = FakeRag()
    rows = [IndexRow(str(i), f"text {i}", {"n": str(i)}) for i in range(5)]

    report = await index_rows(rag, iter(rows), "songs", batch_size=2)
    assert (report.indexed, report.skipped) == (5, 0)
    assert [len(batch[0]) for batch in rag.batches] == [2, 2, 1]
    assert rag.batches[0][2] == ["0", "1"]
    assert rag.batches[0][3] == "songs"
    assert rag.deleted == []


async def test_index_rows_replaces_the_same_source_ids_first():
    rag = FakeRag()
    rows = [IndexRow("7", "text", {})]

    await index_rows(rag, iter(rows), "songs", batch_size=2, replace=True)
    assert rag.deleted == [("songs", ["7"])]


async def test_index_rows_on_an_empty_file_indexes_nothing():
    rag = FakeRag()
    report = await index_rows(rag, iter([]), "songs")
    assert (report.indexed, report.skipped, report.handled) == (0, 0, 0)
    assert rag.batches == []


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def test_index_parser_defaults_describe_the_bundled_songs():
    args = build_parser().parse_args([])
    assert args.csv_path.endswith("songs.csv")
    assert args.collection == "songs"
    assert args.model == "fastembed/BAAI/bge-small-en-v1.5"
    assert args.text_columns == "title,artist,lyrics"
    assert args.source_id_column == "id"
    assert args.max_chars == DEFAULT_MAX_CHARS
    assert args.where == []


def test_index_parser_reads_repeated_where_clauses():
    args = build_parser().parse_args(["--where", "language=en", "--where", "tag=pop"])
    assert args.where == ["language=en", "tag=pop"]


async def test_run_indexes_the_file_through_the_selected_backend(csv_file, monkeypatch):
    rag = FakeRag()
    monkeypatch.setattr(
        "examples.ragindex.index_csv.make_rag_service", lambda *a, **k: rag
    )
    args = build_parser().parse_args([csv_file, "--batch-size", "10"])

    report = await run(args)
    assert report.indexed == 3
    assert rag.batches[0][3] == "songs"


def test_main_returns_zero_when_rows_were_indexed(csv_file, monkeypatch):
    monkeypatch.setattr(
        "examples.ragindex.index_csv.make_rag_service", lambda *a, **k: FakeRag()
    )
    monkeypatch.setattr("sys.argv", ["index_csv", csv_file])
    assert main() == 0


def test_main_returns_one_when_nothing_matched(csv_file, monkeypatch):
    monkeypatch.setattr(
        "examples.ragindex.index_csv.make_rag_service", lambda *a, **k: FakeRag()
    )
    monkeypatch.setattr("sys.argv", ["index_csv", csv_file, "--where", "language=xx"])
    assert main() == 1


def test_main_reports_a_bad_column_instead_of_raising(csv_file, monkeypatch):
    monkeypatch.setattr(
        "examples.ragindex.index_csv.make_rag_service", lambda *a, **k: FakeRag()
    )
    monkeypatch.setattr("sys.argv", ["index_csv", csv_file, "--text-columns", "nope"])
    assert main() == 2


# --------------------------------------------------------------------------
# Querying
# --------------------------------------------------------------------------


def test_format_result_shows_rank_similarity_and_metadata():
    block = format_result(make_result(similarity=0.8125), rank=1, content_chars=0)
    assert block.startswith(" 1. 0.8125  [1] title=Some Song, artist=Some Artist")
    assert "\n" not in block


def test_format_result_indents_and_truncates_the_content():
    block = format_result(make_result(content="a" * 50), rank=2, content_chars=10)
    assert "      aaaaaaaaaa" in block
    assert block.endswith("...")


def test_format_result_omits_a_content_that_fits():
    block = format_result(make_result(content="short"), rank=1, content_chars=100)
    assert "      short" in block
    assert not block.endswith("...")


def test_query_parser_defaults():
    args = build_query_parser().parse_args(["a song about bees"])
    assert args.text == "a song about bees"
    assert args.collection == "songs"
    assert args.top_k == 5
    assert args.keep_best is False


async def test_query_run_passes_the_arguments_through(monkeypatch, capsys):
    rag = FakeRag(results=[make_result()])
    monkeypatch.setattr(
        "examples.ragindex.query_index.make_rag_service", lambda *a, **k: rag
    )
    args = build_query_parser().parse_args(
        ["bees", "--top-k", "3", "--source-ids", "1,2", "--keep-best"]
    )

    results = await query_run(args)
    assert len(results) == 1
    assert rag.queries == [("bees", 3, "songs", ["1", "2"], True)]
    assert "Top 1 for 'bees'" in capsys.readouterr().out


async def test_query_run_says_so_when_there_are_no_hits(monkeypatch, capsys):
    monkeypatch.setattr(
        "examples.ragindex.query_index.make_rag_service", lambda *a, **k: FakeRag()
    )
    args = build_query_parser().parse_args(["bees"])

    assert await query_run(args) == []
    assert "No hits" in capsys.readouterr().out


def test_query_main_returns_one_when_nothing_was_found(monkeypatch):
    monkeypatch.setattr(
        "examples.ragindex.query_index.make_rag_service", lambda *a, **k: FakeRag()
    )
    monkeypatch.setattr("sys.argv", ["query_index", "bees"])
    assert query_main() == 1


def test_query_main_returns_zero_on_a_hit(monkeypatch):
    monkeypatch.setattr(
        "examples.ragindex.query_index.make_rag_service",
        lambda *a, **k: FakeRag(results=[make_result()]),
    )
    monkeypatch.setattr("sys.argv", ["query_index", "bees"])
    assert query_main() == 0


def test_query_main_reports_a_missing_environment_variable(monkeypatch):
    monkeypatch.delenv("KAVALAI_DB_URI", raising=False)
    monkeypatch.setattr("sys.argv", ["query_index", "bees", "--index", "postgres"])
    assert query_main() == 2


# --------------------------------------------------------------------------
# The bundled CSV
# --------------------------------------------------------------------------


def test_bundled_csv_has_a_hundred_invented_songs():
    rows = list(csv.DictReader(open(SONGS_CSV, encoding="utf-8")))
    assert len(rows) == 100
    assert len({row["title"] for row in rows}) == 100
    assert all(row["lyrics"].startswith("[Verse 1]\n") for row in rows)


def test_bundled_csv_columns_match_the_defaults():
    with open(SONGS_CSV, encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    for column in ["title", "artist", "lyrics", "tag", "year", "id"]:
        assert column in header


def test_bundled_csv_reads_with_the_default_settings():
    rows = list(
        read_rows(SONGS_CSV, ["title", "artist", "lyrics"], ["title", "tag"], "id")
    )
    assert len(rows) == 100
    assert rows[0].text.startswith("My Stapler Has Tenure\nThe Beige Alarmists\n")
    assert rows[0].metadata["tag"] == "rock"


# --------------------------------------------------------------------------
# Skipping what is already indexed
# --------------------------------------------------------------------------


async def test_existing_source_ids_reads_the_collection():
    rag = FakeRag(existing=["1", "2", "2"])
    assert await existing_source_ids(rag, "songs") == {"1", "2"}


async def test_existing_source_ids_needs_iter_entries():
    with pytest.raises(ValueError, match="--skip-existing is unavailable"):
        await existing_source_ids(FakeRag(can_iter=False), "songs")


async def test_index_rows_never_embeds_a_skipped_row():
    rag = FakeRag()
    rows = [IndexRow(str(i), f"text {i}", {}) for i in range(4)]

    report = await index_rows(
        rag, iter(rows), "songs", batch_size=10, skip_source_ids=frozenset({"1", "3"})
    )
    assert (report.indexed, report.skipped, report.handled) == (2, 2, 4)
    # The skipped rows never reached the embedding call at all.
    assert rag.batches[0][2] == ["0", "2"]


async def test_run_skips_rows_already_in_the_collection(csv_file, monkeypatch):
    rag = FakeRag(existing=["10", "11"])
    monkeypatch.setattr(
        "examples.ragindex.index_csv.make_rag_service", lambda *a, **k: rag
    )
    args = build_parser().parse_args([csv_file, "--skip-existing"])

    report = await run(args)
    assert (report.indexed, report.skipped) == (1, 2)
    assert rag.batches[0][2] == ["12"]


def test_skip_existing_and_replace_are_mutually_exclusive(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--replace", "--skip-existing"])
    assert "not allowed with" in capsys.readouterr().err


def test_an_up_to_date_skip_existing_rerun_succeeds(csv_file, monkeypatch):
    """Nothing left to index is a success, not a failure exit code."""
    rag = FakeRag(existing=["10", "11", "12"])
    monkeypatch.setattr(
        "examples.ragindex.index_csv.make_rag_service", lambda *a, **k: rag
    )
    monkeypatch.setattr("sys.argv", ["index_csv", csv_file, "--skip-existing"])

    assert main() == 0
    assert rag.batches == []
