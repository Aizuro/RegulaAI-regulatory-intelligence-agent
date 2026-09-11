"""Read-only SQL tools for structured regulatory data analysis.

V4 adds SQL/data-analysis capability without changing the existing legal RAG
or comparison engine.

The database is intentionally a synthetic portfolio dataset. Its metadata must
not be treated as authoritative legal facts.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_DB_PATH = Path(__file__).resolve().parent / "regulatory_data.db"
MAX_ROWS = 100


def _connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create the synthetic regulatory dataset if it does not exist."""

    with _connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS regulations (
                id INTEGER PRIMARY KEY,
                document_type TEXT NOT NULL,
                regulation_number TEXT,
                year INTEGER,
                title TEXT NOT NULL,
                sector TEXT NOT NULL,
                issuing_body TEXT NOT NULL,
                status TEXT NOT NULL,
                effective_year INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_regulations_year
                ON regulations(year);

            CREATE INDEX IF NOT EXISTS idx_regulations_sector
                ON regulations(sector);

            CREATE INDEX IF NOT EXISTS idx_regulations_status
                ON regulations(status);
            """
        )

        count = connection.execute(
            "SELECT COUNT(*) FROM regulations"
        ).fetchone()[0]

        if count:
            return

        rows = [
            ("UU", "1", 2024, "Synthetic Digital Governance Act A", "Digital", "DPR RI", "active", 2024),
            ("UU", "3", 2024, "Synthetic Electronic Systems Act B", "Digital", "DPR RI", "active", 2024),
            ("UU", "7", 2024, "Synthetic Data Protection Act C", "Digital", "DPR RI", "active", 2025),
            ("UU", "10", 2023, "Synthetic Financial Technology Act D", "Finance", "DPR RI", "active", 2023),
            ("UU", "4", 2023, "Synthetic Public Service Act E", "Public Service", "DPR RI", "amended", 2023),
            ("UU", "8", 2022, "Synthetic Environmental Governance Act F", "Environment", "DPR RI", "active", 2022),
            ("UU", "2", 2022, "Synthetic Industrial Data Act G", "Industry", "DPR RI", "repealed", 2022),
            ("Perpres", "4", 2025, "Synthetic National Digital Strategy Regulation H", "Digital", "Presiden RI", "active", 2025),
            ("Perpres", "9", 2025, "Synthetic Public Data Regulation I", "Public Service", "Presiden RI", "active", 2025),
            ("Perpres", "12", 2024, "Synthetic Green Industry Regulation J", "Environment", "Presiden RI", "active", 2024),
            ("Perpres", "18", 2023, "Synthetic Financial Innovation Regulation K", "Finance", "Presiden RI", "amended", 2023),
            ("Perpres", "21", 2022, "Synthetic Industrial Policy Regulation L", "Industry", "Presiden RI", "active", 2022),
            ("PP", "5", 2025, "Synthetic Digital Compliance Regulation M", "Digital", "Pemerintah RI", "active", 2025),
            ("PP", "8", 2024, "Synthetic Financial Reporting Regulation N", "Finance", "Pemerintah RI", "active", 2024),
            ("PP", "11", 2023, "Synthetic Environmental Reporting Regulation O", "Environment", "Pemerintah RI", "active", 2023),
            ("PP", "14", 2022, "Synthetic Industrial Standards Regulation P", "Industry", "Pemerintah RI", "repealed", 2022),
        ]

        connection.executemany(
            """
            INSERT INTO regulations (
                document_type,
                regulation_number,
                year,
                title,
                sector,
                issuing_body,
                status,
                effective_year
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def _validate_select_query(query: str) -> str:
    """Allow one read-only SELECT/WITH statement and reject mutations."""

    cleaned = query.strip()

    if not cleaned:
        raise ValueError("SQL query tidak boleh kosong.")

    # Remove one optional trailing semicolon.
    cleaned = cleaned.rstrip(";").strip()

    if not re.match(r"^(SELECT|WITH)\b", cleaned, flags=re.IGNORECASE):
        raise ValueError("SQL tool hanya mengizinkan query SELECT atau WITH.")

    forbidden = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|"
        r"VACUUM|PRAGMA|REINDEX|TRUNCATE)\b",
        flags=re.IGNORECASE,
    )

    if forbidden.search(cleaned):
        raise ValueError("SQL query mengandung operasi yang tidak diizinkan.")

    if ";" in cleaned:
        raise ValueError("SQL tool hanya menerima satu statement.")

    return cleaned


def run_sql_query(
    query: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    max_rows: int = MAX_ROWS,
) -> Dict[str, Any]:
    """Execute a bounded, read-only SQL query."""

    if max_rows < 1:
        raise ValueError("max_rows harus >= 1.")

    sql = _validate_select_query(query)

    initialize_database(db_path)

    # Bound result size at the application layer as an additional guardrail.
    limited_sql = f"SELECT * FROM ({sql}) AS result LIMIT {int(max_rows)}"

    with _connect(db_path) as connection:
        cursor = connection.execute(limited_sql)
        columns = [column[0] for column in cursor.description or []]
        rows = [dict(row) for row in cursor.fetchall()]

    return {
        "type": "sql_result",
        "query": sql,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "max_rows": max_rows,
        "database": "synthetic_regulatory_dataset",
        "status": "ok",
    }
