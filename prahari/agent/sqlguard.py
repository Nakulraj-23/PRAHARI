"""Read-only SQL tool: parsed and rejected unless a single allow-listed SELECT."""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from sqlalchemy.engine import Engine

ALLOWED_TABLES = {
    "nodes",
    "links",
    "kpi_samples",
    "alarms",
    "predictions",
    "actions",
    "audit_log",
    "drift_reports",
}
FORBIDDEN_TABLES = {"incidents"}  # ground truth
FORBIDDEN_COLUMN_PREFIX = "label_"
FORBIDDEN_WORDS = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "attach",
    "detach",
    "pragma",
    "replace",
    "vacuum",
    "reindex",
    "truncate",
    "grant",
    "load_extension",
}
MAX_ROWS = 200


class SqlRejected(ValueError):
    """Raised when a query violates the read-only policy."""


def _strip_strings_and_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"--[^\n]*", " ", sql)
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def validate_sql(sql: str) -> str:
    """Return the cleaned single SELECT statement or raise `SqlRejected`."""
    if not sql or not sql.strip():
        raise SqlRejected("empty query")
    body = _strip_strings_and_comments(sql).strip()
    body = body.rstrip(";").strip()
    if ";" in body:
        raise SqlRejected("multiple statements are not allowed")
    low = body.lower()
    if not low.startswith("select"):
        raise SqlRejected("only SELECT statements are allowed")
    words = set(re.findall(r"[a-z_][a-z0-9_]*", low))
    bad = words & FORBIDDEN_WORDS
    if bad:
        raise SqlRejected(f"forbidden keyword: {sorted(bad)[0]}")
    if any(w.startswith(FORBIDDEN_COLUMN_PREFIX) for w in words):
        raise SqlRejected("ground-truth columns (label_*) are not accessible")
    if words & FORBIDDEN_TABLES:
        raise SqlRejected("ground-truth table is not accessible")
    tables = set(re.findall(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", low))
    tables |= {
        t.strip()
        for m in re.findall(
            r"\bfrom\s+[a-z_][a-z0-9_]*(?:\s+(?:as\s+)?[a-z_]\w*)?"
            r"((?:\s*,\s*[a-z_]\w*(?:\s+(?:as\s+)?[a-z_]\w*)?)+)",
            low,
        )
        for t in re.findall(r",\s*([a-z_]\w*)", m)
    }
    if not tables:
        raise SqlRejected("a FROM clause on an allow-listed table is required")
    unknown = tables - ALLOWED_TABLES
    if unknown:
        raise SqlRejected(f"table not allowed: {sorted(unknown)[0]}")
    if "alarms" in tables and re.search(r"(?<!count\()\*", low.replace("count(*)", "")):
        raise SqlRejected("SELECT * on alarms is not allowed; name the columns")
    return sql.strip().rstrip(";")


def _authorizer(action: int, arg1: Any, arg2: Any, *_: Any) -> int:
    if action == sqlite3.SQLITE_READ:
        if arg1 in FORBIDDEN_TABLES or str(arg2).startswith(FORBIDDEN_COLUMN_PREFIX):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION):
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def run_sql(engine: Engine, sql: str, limit: int = MAX_ROWS) -> dict:
    """Validate and execute a query; returns columns and at most `limit` rows."""
    clean = validate_sql(sql)
    limit = min(int(limit), MAX_ROWS)
    raw = engine.raw_connection()
    try:
        conn = raw.driver_connection
        if isinstance(conn, sqlite3.Connection):
            conn.execute("PRAGMA query_only = ON")
            conn.set_authorizer(_authorizer)  # defence in depth: parser is the first layer
        cur = raw.cursor()
        try:
            cur.execute(clean)
            cols = [d[0] for d in cur.description or []]
            rows = cur.fetchmany(limit + 1)
        finally:
            cur.close()
            if isinstance(conn, sqlite3.Connection):
                conn.set_authorizer(None)
                conn.execute("PRAGMA query_only = OFF")
    except sqlite3.DatabaseError as e:
        raise SqlRejected(f"query rejected by database: {e}") from e
    finally:
        raw.close()
    truncated = len(rows) > limit
    return {"columns": cols, "rows": [list(r) for r in rows[:limit]], "truncated": truncated}
