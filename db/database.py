"""
db/database.py — SQLite schema and helper functions.
All tables live in data/app.db.
"""

import sqlite3
import os
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "app.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def db():
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            email         TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name          TEXT NOT NULL DEFAULT '',
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS applications (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id             INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            job_title           TEXT NOT NULL DEFAULT '',
            company             TEXT NOT NULL DEFAULT '',
            location            TEXT DEFAULT '',
            status              TEXT NOT NULL DEFAULT 'Applied'
                                    CHECK(status IN ('Applied','Screening','Interview','Offer','Rejected')),
            source              TEXT DEFAULT 'manual',
            greenhouse_board    TEXT DEFAULT '',
            greenhouse_job_id   TEXT DEFAULT '',
            fit_score           INTEGER DEFAULT NULL,
            job_description     TEXT DEFAULT '',
            notes               TEXT DEFAULT '',
            applied_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TRIGGER IF NOT EXISTS applications_updated
            AFTER UPDATE ON applications
            FOR EACH ROW
            BEGIN
                UPDATE applications SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
            END;
        """)


# ── User helpers ───────────────────────────────────────────────────────────────
def create_user(email: str, password_hash: str, name: str) -> int:
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO users (email, password_hash, name) VALUES (?,?,?)",
            (email.lower().strip(), password_hash, name.strip()),
        )
        return cur.lastrowid


def get_user_by_email(email: str) -> dict | None:
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
        ).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    with db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


# ── Application tracker helpers ────────────────────────────────────────────────
def add_application(
    user_id: int,
    job_title: str,
    company: str,
    location: str = "",
    source: str = "manual",
    greenhouse_board: str = "",
    greenhouse_job_id: str = "",
    fit_score: int | None = None,
    job_description: str = "",
    notes: str = "",
) -> int:
    with db() as conn:
        cur = conn.execute(
            """INSERT INTO applications
               (user_id, job_title, company, location, source,
                greenhouse_board, greenhouse_job_id, fit_score,
                job_description, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (user_id, job_title, company, location, source,
             greenhouse_board, greenhouse_job_id, fit_score,
             job_description, notes),
        )
        return cur.lastrowid


def get_applications(user_id: int) -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM applications WHERE user_id = ? ORDER BY applied_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def update_application_status(app_id: int, user_id: int, status: str, notes: str = None):
    with db() as conn:
        if notes is not None:
            conn.execute(
                "UPDATE applications SET status=?, notes=? WHERE id=? AND user_id=?",
                (status, notes, app_id, user_id),
            )
        else:
            conn.execute(
                "UPDATE applications SET status=? WHERE id=? AND user_id=?",
                (status, app_id, user_id),
            )


def delete_application(app_id: int, user_id: int):
    with db() as conn:
        conn.execute(
            "DELETE FROM applications WHERE id=? AND user_id=?", (app_id, user_id)
        )
