import os
import pytest
import pytest_asyncio
from sqlalchemy import text
from kavalai.migrate_db import migrate
from kavalai.db import build_db_uri


# The backoffice application reads these at call time (AsyncBackofficeSession
# defaults); the library itself never reads them at import time anymore.
os.environ["KAVALAI_BO_DB_SCHEMA"] = "test_backoffice"
# The application refuses to start without these; the values never reach
# Google because no test completes a sign-in.
for _name, _value in {
    "KAVALAI_BO_GOOGLE_CLIENT_ID": "test-client-id",
    "KAVALAI_BO_GOOGLE_CLIENT_SECRET": "test-client-secret",
    "KAVALAI_BO_SESSION_SECRET_KEY": "test-session-secret",
    "KAVALAI_BO_FRONTEND_URL": "http://localhost:4200",
}.items():
    if not os.environ.get(_name):
        os.environ[_name] = _value


@pytest.fixture(scope="session")
def backoffice_db_config(postgres_container):
    config = dict(
        uri=build_db_uri(
            user=postgres_container.username,
            password=postgres_container.password,
            host=postgres_container.get_container_host_ip(),
            port=int(postgres_container.get_exposed_port(5432)),
            db_name=postgres_container.dbname,
        ),
        schema="test_backoffice",
    )
    # Set environment variables BEFORE importing kavalai.backoffice.db
    os.environ["KAVALAI_BO_DB_URI"] = config["uri"]
    os.environ["KAVALAI_BO_DB_SCHEMA"] = "test_backoffice"
    return config


@pytest.fixture(scope="session")
def migrated_backoffice_db(backoffice_db_config):
    migrate(
        "backoffice",
        uri=backoffice_db_config["uri"],
        schema=backoffice_db_config["schema"],
    )


@pytest.fixture(scope="session")
def sqlite_backoffice_uri(tmp_path_factory):
    """A migrated SQLite backoffice database, one per test session."""
    path = tmp_path_factory.mktemp("backoffice") / "backoffice.db"
    uri = f"sqlite:///{path}"
    migrate("backoffice", uri=uri)
    return uri


@pytest_asyncio.fixture(scope="function", params=["postgres", "sqlite"])
async def backoffice_db(request, monkeypatch):
    """An empty backoffice database, on each backend the backoffice supports.

    The application reads ``KAVALAI_BO_DB_URI`` at call time, so pointing the
    variable at the backend under test is enough for the endpoints to follow.
    """
    if request.param == "postgres":
        config = request.getfixturevalue("backoffice_db_config")
        request.getfixturevalue("migrated_backoffice_db")
        monkeypatch.setenv("KAVALAI_BO_DB_URI", config["uri"])
        monkeypatch.setenv("KAVALAI_BO_DB_SCHEMA", config["schema"])
    else:
        monkeypatch.setenv(
            "KAVALAI_BO_DB_URI", request.getfixturevalue("sqlite_backoffice_uri")
        )
        monkeypatch.delenv("KAVALAI_BO_DB_SCHEMA", raising=False)

    from kavalai.backoffice.db import AsyncBackofficeSession, Base

    async with AsyncBackofficeSession() as session:
        tables = list(reversed(Base.metadata.sorted_tables))
        if request.param == "postgres":
            names = ", ".join(f"test_backoffice.{table.name}" for table in tables)
            await session.execute(text(f"TRUNCATE {names} RESTART IDENTITY CASCADE;"))
        else:
            for table in tables:
                await session.execute(text(f"DELETE FROM {table.name}"))
        await session.commit()

        yield session
        await session.rollback()
