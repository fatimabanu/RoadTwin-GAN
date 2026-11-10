"""SQLite storage helper for civic infrastructure road-damage reports."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class CivicReportStore:
    """Lightweight SQLite store for civic road damage inspection reports."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS civic_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    location TEXT NOT NULL,
                    damage_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    affected_area_pct REAL NOT NULL,
                    notes TEXT DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'Open',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def create_report(
        self,
        location: str,
        damage_type: str,
        severity: str,
        confidence: float,
        affected_area_pct: float = 0.0,
        notes: str = "",
        status: str = "Open",
    ) -> dict[str, object]:
        """Insert a new road inspection report and return the stored record."""
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO civic_reports (
                    location, damage_type, severity, confidence, affected_area_pct, notes, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    location.strip(),
                    damage_type.strip().lower(),
                    severity.strip(),
                    float(confidence),
                    float(affected_area_pct),
                    notes.strip(),
                    status.strip(),
                    created_at,
                ),
            )
            conn.commit()
            report_id = int(cursor.lastrowid)
        return self.get_report(report_id)

    def get_report(self, report_id: int) -> dict[str, object]:
        """Fetch a single report by primary key."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM civic_reports WHERE id = ?", (report_id,)).fetchone()
        if row is None:
            raise KeyError(f"Civic report {report_id} not found.")
        return dict(row)

    def list_reports(self, limit: int = 50) -> list[dict[str, object]]:
        """List recent civic inspection reports ordered newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM civic_reports ORDER BY id DESC LIMIT ?",
                (max(1, min(limit, 200)),),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_status(self, report_id: int, status: str) -> dict[str, object]:
        """Update maintenance workflow status for an existing report."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE civic_reports SET status = ? WHERE id = ?",
                (status.strip(), report_id),
            )
            conn.commit()
            if cursor.rowcount == 0:
                raise KeyError(f"Civic report {report_id} not found.")
        return self.get_report(report_id)
