"""Persists UserProfile across sessions, per docs/design/langgraph-workflow.md
§3: this is deliberately separate from the LangGraph conversation checkpointer
so a user's risk tolerance/goals survive a cleared chat or a new session_id.
"""

from __future__ import annotations

from finnie.core.state import UserProfile
from finnie.core.store import SqliteKeyValueStore

NAMESPACE = "user_profile"


class ProfileStore:
    def __init__(self, kv: SqliteKeyValueStore | None = None):
        self._kv = kv or SqliteKeyValueStore()

    def load(self, session_id: str) -> UserProfile:
        raw = self._kv.get(NAMESPACE, session_id)
        if raw:
            return UserProfile.model_validate(raw)
        return UserProfile(session_id=session_id)

    def save(self, profile: UserProfile) -> None:
        self._kv.set(NAMESPACE, profile.session_id, profile.model_dump(mode="json"))
