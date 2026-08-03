"""
database.py — Lightweight analytics store for every inspection event.

SQLite is used here for portability/demo purposes (zero setup). For a real
multi-line deployment, point this at Postgres/TimescaleDB instead (swap the
connection in `Database.__init__`; the SQL below is close to portable, and
Timescale in particular is a good fit since inspection events are a
time series you'll want to downsample/aggregate over shifts, days, weeks).

Every row is one part inspected: what was found, how bad, on which line,
during which shift, from which raw-material batch — the four dimensions
the dashboard slices by to find the ROOT CAUSE of a quality problem
(e.g. "batch B-2291 has 4x the scratch rate of every other batch this
month" -> points at incoming material, not the machine or the operator).
"""

import sqlite3
import time
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    part_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    shift TEXT NOT NULL,             -- 'Morning' | 'Afternoon' | 'Night'
    machine_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    camera_id TEXT,
    defect_type TEXT NOT NULL,
    severity REAL NOT NULL,
    confidence REAL NOT NULL,
    band TEXT NOT NULL,              -- none | minor | major | critical
    action TEXT NOT NULL,            -- pass | reject | human_review
    inference_ms REAL
);

CREATE INDEX IF NOT EXISTS idx_insp_ts ON inspections(timestamp);
CREATE INDEX IF NOT EXISTS idx_insp_machine ON inspections(machine_id);
CREATE INDEX IF NOT EXISTS idx_insp_batch ON inspections(batch_id);
CREATE INDEX IF NOT EXISTS idx_insp_shift ON inspections(shift);
"""


class Database:
    def __init__(self, path: str = "qc_inspections.db"):
        self.path = path
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def log_inspection(self, part_id, shift, machine_id, batch_id, camera_id,
                        defect_type, severity, confidence, band, action,
                        inference_ms=None, timestamp=None):
        with self._conn() as c:
            c.execute(
                """INSERT INTO inspections
                   (part_id, timestamp, shift, machine_id, batch_id, camera_id,
                    defect_type, severity, confidence, band, action, inference_ms)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (part_id, timestamp or time.time(), shift, machine_id, batch_id,
                 camera_id, defect_type, severity, confidence, band, action, inference_ms),
            )

    def defect_rate_by(self, dimension: str):
        """dimension in {'shift', 'machine_id', 'batch_id', 'defect_type'}"""
        assert dimension in {"shift", "machine_id", "batch_id", "defect_type"}
        with self._conn() as c:
            rows = c.execute(f"""
                SELECT {dimension} AS key,
                       COUNT(*) AS total,
                       SUM(CASE WHEN defect_type != 'ok' THEN 1 ELSE 0 END) AS defects,
                       SUM(CASE WHEN action = 'reject' THEN 1 ELSE 0 END) AS rejects,
                       AVG(severity) AS avg_severity
                FROM inspections
                GROUP BY {dimension}
                ORDER BY defects DESC
            """).fetchall()
            return [dict(r) for r in rows]

    def export_json(self):
        with self._conn() as c:
            rows = c.execute("SELECT * FROM inspections ORDER BY timestamp").fetchall()
            return [dict(r) for r in rows]
