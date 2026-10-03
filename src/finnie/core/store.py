"""A small generic SQLite key-value store.

Backs two separate uses that happen to need the same shape (namespaced,
JSON-valued, survives a restart):
- market-data stale-cache fallback, docs/design/market-data-integration.md §4
- user-profile persistence, docs/design/langgraph-workflow.md §3

Kept as one small piece of shared infra rather than two bespoke persistence
layers.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock

DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "finnie.sqlite3"


class SqliteKeyValueStore:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH, table: str = "kv_store"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.table = table
        self._lock = Lock()
        with self._connect() as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} "
                "(namespace TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, "
                "PRIMARY KEY (namespace, key))"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def get(self, namespace: str, key: str) -> dict | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                f"SELECT value FROM {self.table} WHERE namespace = ? AND key = ?",
                (namespace, key),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, namespace: str, key: str, value: dict) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                f"INSERT INTO {self.table} (namespace, key, value) VALUES (?, ?, ?) "
                "ON CONFLICT(namespace, key) DO UPDATE SET value = excluded.value",
                (namespace, key, json.dumps(value)),
            )

    def get_all(self, namespace: str) -> dict[str, dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"SELECT key, value FROM {self.table} WHERE namespace = ?",
                (namespace,),
            ).fetchall()
        return {key: json.loads(value) for key, value in rows}
