"""Minimal local username/password authentication for the Streamlit UI.

Deliberately basic, and the limits are worth being explicit about: PBKDF2-
HMAC-SHA256 password hashing with a random per-user salt (stdlib only, no
new dependency), constant-time comparison on verify, a minimum
length-based password policy (length matters more than forced character
classes -- current NIST guidance), and that's it. There's no email
verification, no password-reset flow, no login-attempt rate limiting /
lockout, and no session-cookie hardening beyond what Streamlit itself
provides. This is "real enough" for a locally-run capstone demo -- it is
not a production-grade auth system, and shouldn't be presented as one.

Replaces the earlier fully-anonymous random-UUID session identifier
(web_app/app.py used to generate one per browser tab) with a durable
identity: once signed in, `session_id` becomes the username itself, so
`ProfileStore`-backed persistence (risk tolerance, goals) now follows the
same person across browser sessions and devices, not just within one tab.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from datetime import UTC, datetime

from finnie.core.store import SqliteKeyValueStore

_NAMESPACE = "auth_users"
_PBKDF2_ITERATIONS = 260_000  # OWASP's current minimum recommendation for PBKDF2-SHA256
_MIN_USERNAME_LENGTH = 3
_MIN_PASSWORD_LENGTH = 8
_USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class AuthError(Exception):
    """A user-facing registration/login failure message."""


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS).hex()


class AuthService:
    def __init__(self, store: SqliteKeyValueStore | None = None):
        self._store = store or SqliteKeyValueStore()

    def register(self, username: str, password: str) -> None:
        username = username.strip()
        if len(username) < _MIN_USERNAME_LENGTH or not _USERNAME_PATTERN.match(username):
            raise AuthError(
                f"Username must be at least {_MIN_USERNAME_LENGTH} characters, using only "
                "letters, numbers, '.', '_', or '-'."
            )
        if len(password) < _MIN_PASSWORD_LENGTH:
            raise AuthError(f"Password must be at least {_MIN_PASSWORD_LENGTH} characters.")

        key = username.lower()  # sign-in is case-insensitive; storage key reflects that
        if self._store.get(_NAMESPACE, key):
            raise AuthError("That username is already taken.")

        salt = secrets.token_bytes(16)
        record = {
            "username": username,  # canonical, as-registered casing
            "salt": salt.hex(),
            "password_hash": _hash_password(password, salt),
            "created_at": datetime.now(UTC).isoformat(),
        }
        self._store.set(_NAMESPACE, key, record)

    def authenticate(self, username: str, password: str) -> str:
        """Returns the canonical (as-registered) username on success."""
        record = self._store.get(_NAMESPACE, username.strip().lower())
        if not record:
            # Same message as a wrong password -- a login form shouldn't reveal
            # whether a username exists.
            raise AuthError("Incorrect username or password.")
        salt = bytes.fromhex(record["salt"])
        candidate_hash = _hash_password(password, salt)
        if not hmac.compare_digest(candidate_hash, record["password_hash"]):
            raise AuthError("Incorrect username or password.")
        return record["username"]
