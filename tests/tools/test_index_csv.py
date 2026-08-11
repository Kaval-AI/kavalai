import pytest

from kavalai.rag.base import BaseRagService
from kavalai.rag.postgres import PostgresRagService
from kavalai.tools.index_csv import (
    content_splitter_generator,
    csv_row_generator,
    index_csv,
    main,
    parse_args,
    rag_service_from_env,
)


class RecordingRagService(BaseRagService):
    """Collects what would have been indexed, instead of touching a database."""

    def __init__(self):
        self.batches = []
        self.deleted = []

    async def index(self, *args, **kwargs):
        raise NotImplementedError

    async def query(self, *args, **kwargs):
        raise NotImplementedError

    async def query_batch(self, *args, **kwargs):
        raise NotImplementedError

    async def delete(self, *args, **kwargs):
        raise NotImplementedError

    async def index_batch(self, texts, metadata_list, collection_name, source_ids):
        self.batches.append(
            {
                "texts": list(texts),
                "metadata": list(metadata_list),
                "collection_name": collection_name,
                "source_ids": list(source_ids),
            }
        )

    async def delete_by_source_id(self, collection_name, source_id):
        self.deleted.append((collection_name, sorted(source_id)))


@pytest.fixture
def songs_csv(tmp_path):
    path = tmp_path / "songs.csv"
    # Two rows; the first row's lyrics field spans two lines with a blank between.
    path.write_text(
        'id,title,lyrics,notes\n1,First,"line one\n\nline two",n1\n2,Second,solo,n2\n',
        encoding="utf-8",
    )
    return path


def test_csv_row_generator_respects_the_limit(songs_csv):
    assert len(list(csv_row_generator(str(songs_csv)))) == 2
    rows = list(csv_row_generator(str(songs_csv), limit=1))
    assert [row["id"] for row in rows] == ["1"]


def test_content_splitter_full_mode_keeps_the_field_whole(songs_csv):
    entries = list(
        content_splitter_generator(
            csv_row_generator(str(songs_csv)),
            index_fields=["lyrics"],
            metadata_fields=["title"],
            source_field="id",
            mode="full",
        )
    )

    assert [e["source_id"] for e in entries] == ["1", "2"]
    assert entries[0]["meta"] == {"title": "First"}
    assert "line one" in entries[0]["text"] and "line two" in entries[0]["text"]


def test_content_splitter_lines_mode_skips_blank_lines(songs_csv):
    entries = list(
        content_splitter_generator(
            csv_row_generator(str(songs_csv)),
            index_fields=["lyrics"],
            metadata_fields=[],
            source_field="id",
            mode="lines",
        )
    )

    assert [e["text"] for e in entries] == ["line one", "line two", "solo"]


def test_content_splitter_ignores_missing_fields(songs_csv):
    entries = list(
        content_splitter_generator(
            csv_row_generator(str(songs_csv)),
            index_fields=["lyrics", "not_a_column"],
            metadata_fields=[],
            source_field="id",
            mode="full",
        )
    )

    assert len(entries) == 2


def test_content_splitter_falls_back_to_a_default_source_id(songs_csv):
    entries = list(
        content_splitter_generator(
            csv_row_generator(str(songs_csv)),
            index_fields=["lyrics"],
            metadata_fields=[],
            source_field="missing_column",
            mode="full",
        )
    )

    assert {e["source_id"] for e in entries} == {"default"}


async def test_index_csv_indexes_in_batches(songs_csv):
    service = RecordingRagService()

    await index_csv(
        csv_path=str(songs_csv),
        collection_name="songs",
        metadata_fields=["title"],
        index_fields=["lyrics"],
        source_field="id",
        mode="lines",
        limit=None,
        batch_size=1,
        rag_service=service,
    )

    # One batch per row, because batch_size is 1.
    assert [b["source_ids"] for b in service.batches] == [["1", "1"], ["2"]]
    assert service.batches[0]["collection_name"] == "songs"
    assert service.deleted == []


async def test_index_csv_replaces_existing_source_ids(songs_csv):
    service = RecordingRagService()

    await index_csv(
        csv_path=str(songs_csv),
        collection_name="songs",
        metadata_fields=[],
        index_fields=["lyrics"],
        source_field="id",
        mode="full",
        limit=None,
        replace=True,
        batch_size=10,
        rag_service=service,
    )

    assert service.deleted == [("songs", ["1", "2"])]
    assert len(service.batches) == 1


async def test_index_csv_skips_batches_without_indexable_text(songs_csv):
    service = RecordingRagService()

    await index_csv(
        csv_path=str(songs_csv),
        collection_name="songs",
        metadata_fields=[],
        index_fields=["not_a_column"],
        source_field="id",
        mode="full",
        limit=None,
        rag_service=service,
    )

    assert service.batches == []


def test_parse_args_reads_the_documented_flags(songs_csv):
    args = parse_args(
        [
            str(songs_csv),
            "--collection-name",
            "songs",
            "--index-fields",
            "lyrics",
            "--source-field",
            "id",
            "--mode",
            "lines",
            "--limit",
            "5",
            "--replace",
            "--batch-size",
            "3",
        ]
    )

    assert args.collection_name == "songs"
    assert args.index_fields == ["lyrics"]
    assert (args.mode, args.limit, args.replace, args.batch_size) == (
        "lines",
        5,
        True,
        3,
    )


def test_main_indexes_the_file(songs_csv):
    service = RecordingRagService()

    main(
        [
            str(songs_csv),
            "--collection-name",
            "songs",
            "--index-fields",
            "lyrics",
            "--source-field",
            "id",
        ],
        rag_service=service,
    )

    assert len(service.batches) == 1


def test_main_exits_when_the_csv_is_missing(tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                str(tmp_path / "nope.csv"),
                "--collection-name",
                "songs",
                "--index-fields",
                "lyrics",
                "--source-field",
                "id",
            ]
        )

    assert exit_info.value.code == 1


def test_rag_service_from_env_uses_the_configured_database(monkeypatch):
    monkeypatch.setenv("KAVALAI_DB_URI", "postgresql+asyncpg://u:p@localhost:5432/db")
    monkeypatch.setenv("KAVALAI_DB_SCHEMA", "agents")
    monkeypatch.setenv(
        "KAVALAI_DEFAULT_EMBEDDING_MODEL", "openai/text-embedding-3-small"
    )

    service = rag_service_from_env()

    assert isinstance(service, PostgresRagService)
    # An explicit model overrides the environment default.
    assert isinstance(
        rag_service_from_env("openai/text-embedding-3-large"), PostgresRagService
    )


async def test_index_csv_builds_the_service_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("KAVALAI_DB_URI", "postgresql+asyncpg://u:p@localhost:5432/db")
    monkeypatch.setenv(
        "KAVALAI_DEFAULT_EMBEDDING_MODEL", "openai/text-embedding-3-small"
    )
    empty = tmp_path / "empty.csv"
    empty.write_text("id,lyrics\n", encoding="utf-8")

    # No rows, so the run finishes without the service ever being called.
    await index_csv(
        csv_path=str(empty),
        collection_name="songs",
        metadata_fields=[],
        index_fields=["lyrics"],
        source_field="id",
        mode="full",
        limit=None,
    )
