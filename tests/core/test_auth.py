from pathlib import Path

import pytest

from finnie.core.auth import AuthError, AuthService
from finnie.core.store import SqliteKeyValueStore


@pytest.fixture
def auth(tmp_path: Path) -> AuthService:
    return AuthService(SqliteKeyValueStore(db_path=tmp_path / "auth.sqlite3"))


def test_register_then_authenticate_succeeds(auth):
    auth.register("janedoe", "correct-password")
    assert auth.authenticate("janedoe", "correct-password") == "janedoe"


def test_authenticate_is_case_insensitive_on_username_but_preserves_canonical_case(auth):
    auth.register("JaneDoe", "correct-password")
    assert auth.authenticate("janedoe", "correct-password") == "JaneDoe"
    assert auth.authenticate("JANEDOE", "correct-password") == "JaneDoe"


def test_authenticate_rejects_wrong_password(auth):
    auth.register("janedoe", "correct-password")
    with pytest.raises(AuthError):
        auth.authenticate("janedoe", "wrong-password")


def test_authenticate_rejects_unknown_username(auth):
    with pytest.raises(AuthError):
        auth.authenticate("nobody", "whatever123")


def test_register_rejects_duplicate_username(auth):
    auth.register("janedoe", "correct-password")
    with pytest.raises(AuthError, match="already taken"):
        auth.register("janedoe", "a-different-password")


def test_register_rejects_duplicate_username_case_insensitively(auth):
    auth.register("janedoe", "correct-password")
    with pytest.raises(AuthError, match="already taken"):
        auth.register("JaneDoe", "a-different-password")


def test_register_rejects_short_username(auth):
    with pytest.raises(AuthError, match="Username"):
        auth.register("jd", "correct-password")


def test_register_rejects_username_with_invalid_characters(auth):
    with pytest.raises(AuthError, match="Username"):
        auth.register("jane doe!", "correct-password")


def test_register_rejects_short_password(auth):
    with pytest.raises(AuthError, match="Password"):
        auth.register("janedoe", "short")


def test_passwords_are_never_stored_in_plaintext(tmp_path):
    db_path = tmp_path / "auth.sqlite3"
    auth = AuthService(SqliteKeyValueStore(db_path=db_path))
    auth.register("janedoe", "a-very-specific-password-123")

    raw_bytes = db_path.read_bytes()
    assert b"a-very-specific-password-123" not in raw_bytes
