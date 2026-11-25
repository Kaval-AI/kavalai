import io
import uuid

import pytest
from sqlalchemy import select

from kavalai.backoffice import db, users
from kavalai.backoffice.users import (
    MAX_PICTURE_URL_LENGTH,
    add_user,
    hash_password,
    main,
    normalise_host,
    read_password,
    set_password,
    validate_password,
    validate_picture_url,
    verify_password,
)


def test_a_hash_verifies_only_its_own_password():
    stored = hash_password("correct horse")

    assert stored.startswith("scrypt$16384$8$5$")
    assert "correct horse" not in stored
    assert verify_password("correct horse", stored)
    assert not verify_password("wrong horse", stored)


def test_each_hash_has_its_own_salt():
    assert hash_password("correct horse") != hash_password("correct horse")


def test_a_hash_is_verified_with_the_parameters_it_records():
    """Raising the module's cost parameters must not lock existing users out."""
    import base64
    import hashlib

    salt = b"0123456789abcdef"
    digest = hashlib.scrypt(b"correct horse", salt=salt, n=2**10, r=8, p=1, dklen=32)
    stored = "$".join(
        [
            "scrypt",
            "1024",
            "8",
            "1",
            base64.b64encode(salt).decode(),
            base64.b64encode(digest).decode(),
        ]
    )

    assert verify_password("correct horse", stored)
    assert not verify_password("wrong horse", stored)


@pytest.mark.parametrize(
    "stored",
    [None, "", "plaintext", "md5$1$2$3$4$5", "scrypt$x$8$5$AAAA$AAAA", "scrypt$1$2"],
)
def test_no_hash_or_a_malformed_one_verifies_nothing(stored):
    assert not verify_password("anything", stored)


def test_a_short_password_is_refused():
    with pytest.raises(ValueError, match="at least 8 characters"):
        validate_password("short")
    validate_password("longenough")


OWN_HOSTS = ["backoffice.example.com", "127.0.0.1", "::1"]


@pytest.mark.parametrize(
    "value, stored",
    [
        (None, None),
        ("", None),
        ("   ", None),
        (" https://cdn.example.org/a.png ", "https://cdn.example.org/a.png"),
        ("HTTPS://CDN.example.org/a.png", "HTTPS://CDN.example.org/a.png"),
        (
            "https://lh3.googleusercontent.com/a/x=s96-c",
            "https://lh3.googleusercontent.com/a/x=s96-c",
        ),
        ("https://203.0.113.7:8443/a.png", "https://203.0.113.7:8443/a.png"),
        ("https://[2001:db8::1]/a.png", "https://[2001:db8::1]/a.png"),
        ("https://other.example.com./a.png", "https://other.example.com./a.png"),
    ],
)
def test_a_picture_url_is_stored_trimmed_and_an_empty_one_clears(value, stored):
    assert validate_picture_url(value, OWN_HOSTS) == stored


@pytest.mark.parametrize(
    "url",
    [
        "/api/logout",
        "api/logout",
        "//evil.example.com/a.png",
        "http://cdn.example.org/a.png",
        "javascript:alert(1)",
        "data:image/png;base64,AAAA",
        "https://",
        "https:///a.png",
        "https:\\\\backoffice.example.com\\api\\logout",
        "https://evil.example.com\\@backoffice.example.com/",
        "https://backoffice.example.com\\@evil.example.com/",
        "https://user@evil.example.com/a.png",
        "https://back office.example.com/",
        "https://backoffice.example.com/a\tb",
        "https://%62ackoffice.example.com/",
        "https://ｂackoffice.example.com/",
        "https://2130706433/",
        "https://0x7f.0.0.1/",
        "https://127.000.000.001/",
        "https://[::1%25lo]/",
        "https://[not:an:address]/",
        "https://evil.example.com:99999/",
        "https://under_score.example.com/",
        "https://./",
    ],
)
def test_a_picture_url_a_browser_could_read_otherwise_is_refused(url):
    with pytest.raises(ValueError, match="absolute https:// URL"):
        validate_picture_url(url, OWN_HOSTS)


@pytest.mark.parametrize(
    "url",
    [
        "https://backoffice.example.com/api/logout",
        "https://BACKOFFICE.example.com/a.png",
        "https://backoffice.example.com.:8443/a.png",
        "https://127.0.0.1/a.png",
        "https://[0:0::1]/a.png",
    ],
)
def test_a_picture_on_the_backoffice_host_is_refused_whatever_the_port(url):
    """A cookie is not scoped to a port, so any port of the host is refused."""
    with pytest.raises(ValueError, match="backoffice's own host"):
        validate_picture_url(url, OWN_HOSTS)


def test_a_picture_url_longer_than_the_cookie_can_carry_is_refused():
    url = "https://cdn.example.org/" + "a" * MAX_PICTURE_URL_LENGTH
    with pytest.raises(ValueError, match=f"at most {MAX_PICTURE_URL_LENGTH}"):
        validate_picture_url(url, OWN_HOSTS)
    fits = url[:MAX_PICTURE_URL_LENGTH]
    assert validate_picture_url(fits, OWN_HOSTS) == fits


def test_hosts_are_compared_as_a_browser_compares_them():
    assert normalise_host("Example.COM.") == "example.com"
    assert normalise_host("0:0::1") == "::1"
    assert normalise_host("127.0.0.1") == "127.0.0.1"


@pytest.mark.asyncio
async def test_add_user_stores_a_hash_not_the_password(backoffice_db):
    user = await add_user("pw@example.com", "correct horse", name="Pw", is_admin=True)

    row = await backoffice_db.get(db.User, user.id)
    assert row.email == "pw@example.com"
    assert row.name == "Pw"
    assert row.is_admin is True
    assert row.password_hash != "correct horse"
    assert verify_password("correct horse", row.password_hash)


@pytest.mark.asyncio
async def test_add_user_without_a_password_is_oauth_only(backoffice_db):
    user = await add_user("oauth@example.com")

    assert user.password_hash is None
    assert user.name == "oauth@example.com"
    assert user.is_admin is False


@pytest.mark.asyncio
async def test_add_user_refuses_a_duplicate_and_a_short_password(backoffice_db):
    await add_user("dup@example.com", "correct horse")

    with pytest.raises(ValueError, match="already exists"):
        await add_user("dup@example.com", "another horse")
    with pytest.raises(ValueError, match="at least 8"):
        await add_user("short@example.com", "short")

    rows = (await backoffice_db.execute(select(db.User))).scalars().all()
    assert [row.email for row in rows] == ["dup@example.com"]


@pytest.mark.asyncio
async def test_set_password_replaces_the_hash(backoffice_db):
    user = await add_user("change@example.com", "correct horse")

    await set_password("change@example.com", "battery staple")

    backoffice_db.expire_all()
    row = await backoffice_db.get(db.User, user.id)
    assert verify_password("battery staple", row.password_hash)
    assert not verify_password("correct horse", row.password_hash)

    with pytest.raises(ValueError, match="No user"):
        await set_password("nobody@example.com", "battery staple")
    with pytest.raises(ValueError, match="at least 8"):
        await set_password("change@example.com", "short")


def test_read_password_from_stdin_and_from_the_prompt(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("from stdin\n"))
    assert read_password(from_stdin=True) == "from stdin"

    answers = iter(["typed twice", "typed twice"])
    monkeypatch.setattr(users.getpass, "getpass", lambda prompt: next(answers))
    assert read_password(from_stdin=False) == "typed twice"

    answers = iter(["typed once", "typed wrong"])
    with pytest.raises(ValueError, match="do not match"):
        read_password(from_stdin=False)


def test_the_script_adds_a_user_and_sets_a_password(tmp_path, monkeypatch, capsys):
    """End to end on a SQLite file named on the command line, so the script
    is shown to need no environment."""
    from kavalai.migrate_db import migrate

    path = tmp_path / f"{uuid.uuid4()}.db"
    uri = f"sqlite:///{path}"
    migrate("backoffice", uri=uri)

    monkeypatch.setattr("sys.stdin", io.StringIO("correct horse\n"))
    assert (
        main(
            ["add", "cli@example.com", "--name", "Cli", "--admin"]
            + ["--password-stdin", "--db-uri", uri]
        )
        == 0
    )
    assert "Created administrator cli@example.com." in capsys.readouterr().out

    monkeypatch.setattr("sys.stdin", io.StringIO("correct horse\n"))
    assert main(["add", "cli@example.com", "--password-stdin", "--db-uri", uri]) == 1
    assert "already exists" in capsys.readouterr().err

    assert main(["add", "google@example.com", "--no-password", "--db-uri", uri]) == 0
    assert "Created user google@example.com." in capsys.readouterr().out

    monkeypatch.setattr("sys.stdin", io.StringIO("battery staple\n"))
    assert (
        main(["set-password", "cli@example.com", "--password-stdin", "--db-uri", uri])
        == 0
    )
    assert "Password set for cli@example.com." in capsys.readouterr().out

    import sqlite3

    rows = dict(sqlite3.connect(path).execute("SELECT email, password_hash FROM users"))
    assert verify_password("battery staple", rows["cli@example.com"])
    assert rows["google@example.com"] is None
