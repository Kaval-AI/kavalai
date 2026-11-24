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

Backoffice user accounts and their passwords.

A password is never stored: ``users.password_hash`` holds a salted scrypt
hash in the form ``scrypt$n$r$p$salt$hash`` (salt and hash base64), so the
cost parameters travel with each hash and can be raised without invalidating
the existing ones. scrypt comes from the standard library, which keeps the
backoffice free of a password-hashing dependency. The parameters
(``n=2**14, r=8, p=5``) are one of the OWASP-recommended scrypt settings and
fit OpenSSL's default memory limit.

The module is also the script that creates accounts::

    python -m kavalai.backoffice.users add you@example.com --name "You" --admin
    python -m kavalai.backoffice.users set-password you@example.com

The password is prompted for, or read from standard input with
``--password-stdin``; it is never a command-line argument, which would leave
it in the shell history and the process list.
"""

import argparse
import asyncio
import base64
import getpass
import hashlib
import hmac
import secrets
import sys

from sqlalchemy import select

from kavalai.backoffice.db import AsyncBackofficeSession, User

SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 5
SALT_BYTES = 16
HASH_BYTES = 32
MIN_PASSWORD_LENGTH = 8


def validate_password(password: str) -> None:
    """Raise ``ValueError`` for a password too short to accept."""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(
            f"The password must be at least {MIN_PASSWORD_LENGTH} characters long."
        )


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int, length: int) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=length
    )


def hash_password(password: str) -> str:
    """Hash ``password`` with a fresh salt, in the stored format."""
    salt = secrets.token_bytes(SALT_BYTES)
    digest = _scrypt(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P, HASH_BYTES)
    return "$".join(
        [
            "scrypt",
            str(SCRYPT_N),
            str(SCRYPT_R),
            str(SCRYPT_P),
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(digest).decode("ascii"),
        ]
    )


def verify_password(password: str, stored: str | None) -> bool:
    """Whether ``password`` matches the stored hash.

    With no stored hash (an unknown user, or one that signs in through OAuth
    only) a hash is computed anyway and discarded, so the response time does
    not tell a caller which addresses have a password.
    """
    if not stored:
        _scrypt(password, b"\0" * SALT_BYTES, SCRYPT_N, SCRYPT_R, SCRYPT_P, HASH_BYTES)
        return False
    try:
        scheme, n, r, p, salt_b64, digest_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        salt = base64.b64decode(salt_b64, validate=True)
        expected = base64.b64decode(digest_b64, validate=True)
        actual = _scrypt(password, salt, int(n), int(r), int(p), len(expected))
    except ValueError:
        return False
    return hmac.compare_digest(actual, expected)


async def add_user(
    email: str,
    password: str | None = None,
    *,
    name: str | None = None,
    is_admin: bool = False,
    uri: str | None = None,
    schema: str | None = None,
) -> User:
    """Create a backoffice user; ``password=None`` creates one without a password.

    A user without a password can sign in through OAuth only. Raises
    ``ValueError`` when the address is already registered or the password is
    too short. ``uri`` / ``schema`` default to the backoffice's own database
    settings (see :func:`~kavalai.backoffice.db.AsyncBackofficeSession`).
    """
    if password is not None:
        validate_password(password)
    async with AsyncBackofficeSession(uri, schema) as session:
        existing = await session.execute(select(User).where(User.email == email))
        if existing.scalars().first() is not None:
            raise ValueError(f"A user with the address {email} already exists.")
        user = User(
            email=email,
            name=name or email,
            is_admin=is_admin,
            password_hash=hash_password(password) if password is not None else None,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


async def set_password(
    email: str,
    password: str,
    *,
    uri: str | None = None,
    schema: str | None = None,
) -> User:
    """Replace the password of an existing user.

    Raises ``ValueError`` when no user has the address or the password is too
    short.
    """
    validate_password(password)
    async with AsyncBackofficeSession(uri, schema) as session:
        result = await session.execute(select(User).where(User.email == email))
        user = result.scalars().first()
        if user is None:
            raise ValueError(f"No user with the address {email}.")
        user.password_hash = hash_password(password)
        await session.commit()
        await session.refresh(user)
        return user


def read_password(from_stdin: bool) -> str:
    """The new password: one line of standard input, or two matching prompts."""
    if from_stdin:
        return sys.stdin.readline().rstrip("\r\n")
    password = getpass.getpass("Password: ")
    if getpass.getpass("Repeat password: ") != password:
        raise ValueError("The passwords do not match.")
    return password


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m kavalai.backoffice.users",
        description="Create backoffice users and set their passwords.",
    )
    database = argparse.ArgumentParser(add_help=False)
    database.add_argument(
        "--db-uri", help="Backoffice database URI. Default: KAVALAI_BO_DB_URI."
    )
    database.add_argument(
        "--db-schema", help="Backoffice schema. Default: KAVALAI_BO_DB_SCHEMA."
    )
    database.add_argument(
        "--password-stdin",
        action="store_true",
        help="Read the password from standard input instead of prompting.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    add = commands.add_parser("add", parents=[database], help="Create a user.")
    add.add_argument("email", help="The address the user signs in with.")
    add.add_argument("--name", help="Display name. Default: the address.")
    add.add_argument("--admin", action="store_true", help="Make an administrator.")
    add.add_argument(
        "--no-password",
        action="store_true",
        help="Create a user that signs in through OAuth only.",
    )

    change = commands.add_parser(
        "set-password", parents=[database], help="Replace a user's password."
    )
    change.add_argument("email", help="The address of an existing user.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "add":
            password = None if args.no_password else read_password(args.password_stdin)
            user = asyncio.run(
                add_user(
                    args.email,
                    password,
                    name=args.name,
                    is_admin=args.admin,
                    uri=args.db_uri,
                    schema=args.db_schema,
                )
            )
            role = "administrator" if user.is_admin else "user"
            print(f"Created {role} {user.email}.")
        else:
            password = read_password(args.password_stdin)
            user = asyncio.run(
                set_password(
                    args.email, password, uri=args.db_uri, schema=args.db_schema
                )
            )
            print(f"Password set for {user.email}.")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
