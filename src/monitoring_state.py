"""SQLite-backed persistence for V7 regulatory monitoring snapshots."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List


class MonitoringStateStore:
    """Persist the latest source snapshot for deterministic change detection."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS monitoring_state (
                    regulation_key TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    regulation_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def load_snapshot(self) -> List[Dict[str, Any]]:
        """Load the latest snapshot in deterministic key order."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT regulation_json FROM monitoring_state ORDER BY regulation_key"
            ).fetchall()
        return [json.loads(row["regulation_json"]) for row in rows]

    def save_snapshot(self, snapshot: Iterable[Dict[str, Any]]) -> None:
        """Atomically replace the stored snapshot."""
        from src.monitoring import build_regulation_fingerprint, build_regulation_key, normalize_regulation

        normalized_snapshot = []
        seen = set()
        for raw in snapshot:
            regulation = normalize_regulation(raw)
            key = build_regulation_key(regulation["type"], regulation["number"], regulation["year"])
            if key in seen:
                raise ValueError(f"Duplicate regulation_key in snapshot: {key}")
            seen.add(key)
            normalized_snapshot.append((
                key,
                build_regulation_fingerprint(regulation),
                json.dumps(regulation, ensure_ascii=False, sort_keys=True),
            ))

        with self._connect() as connection:
            try:
                connection.execute("BEGIN")
                connection.execute("DELETE FROM monitoring_state")
                connection.executemany(
                    """
                    INSERT INTO monitoring_state
                        (regulation_key, fingerprint, regulation_json)
                    VALUES (?, ?, ?)
                    """,
                    normalized_snapshot,
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
