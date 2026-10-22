"""The review queue: proposed trace links wait for a person to accept or reject them.

An agent (the built-in one, or any MCP client) can only propose. A link enters the trace matrix only after a
reviewer accepts it, and the record keeps who decided, when, and why, which is what an auditor of a regulated
product (for example, software in a medical device under IEC 62304) would ask to see.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS links (
  id           INTEGER PRIMARY KEY,
  requirement  TEXT NOT NULL,
  class_name   TEXT NOT NULL,
  status       TEXT NOT NULL DEFAULT 'proposed' CHECK (status IN ('proposed', 'accepted', 'rejected')),
  reason       TEXT NOT NULL DEFAULT '',
  proposed_by  TEXT NOT NULL,
  proposed_at  TEXT NOT NULL,
  reviewed_by  TEXT,
  reviewed_at  TEXT,
  note         TEXT,
  UNIQUE (requirement, class_name)
);
"""


class ReviewError(Exception):
    pass


def clean(text: str, limit: int = 500) -> str:
    """Reasons and notes come from a model or an MCP client: keep printable text only, so a reason can't carry
    terminal escape codes that change what a reviewer sees in the queue."""
    return " ".join("".join(ch if ch.isprintable() else " " for ch in str(text)).split())[:limit]


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path):
        self.con = sqlite3.connect(str(path), check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(SCHEMA)

    def propose(self, requirement: str, class_name: str, reason: str, by: str) -> tuple[int, bool]:
        """Add a proposal; returns (id, created). A link that already exists, in any status, is left alone, so a
        rejected link is not re-proposed by the next agent run."""
        with self.con:
            cur = self.con.execute(
                """INSERT INTO links (requirement, class_name, reason, proposed_by, proposed_at)
                   VALUES (?, ?, ?, ?, ?) ON CONFLICT (requirement, class_name) DO NOTHING""",
                (requirement, class_name, clean(reason), clean(by, 80), now()),
            )
            if cur.rowcount:
                return int(cur.lastrowid), True
        row = self.con.execute(
            "SELECT id FROM links WHERE requirement = ? AND class_name = ?", (requirement, class_name)
        ).fetchone()
        return int(row["id"]), False

    def review(self, link_id: int, decision: str, reviewer: str, note: str = "") -> dict:
        if decision not in ("accepted", "rejected"):
            raise ReviewError("decision must be 'accepted' or 'rejected'")
        if not reviewer.strip():
            raise ReviewError("a reviewer name is required")
        with self.con:
            cur = self.con.execute(
                """UPDATE links SET status = ?, reviewed_by = ?, reviewed_at = ?, note = ?
                   WHERE id = ? AND status = 'proposed'""",
                (decision, clean(reviewer, 80), now(), clean(note), link_id),
            )
        row = self.get(link_id)
        if row is None:
            raise ReviewError(f"no link {link_id}")
        if not cur.rowcount:
            raise ReviewError(f"link {link_id} was already {row['status']} by {row['reviewed_by']}")
        return row

    def get(self, link_id: int) -> dict | None:
        row = self.con.execute("SELECT * FROM links WHERE id = ?", (link_id,)).fetchone()
        return dict(row) if row else None

    def links(self, status: str | None = None, requirement: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM links WHERE 1 = 1", []
        if status:
            q, args = q + " AND status = ?", [*args, status]
        if requirement:
            q, args = q + " AND requirement = ?", [*args, requirement]
        return [dict(r) for r in self.con.execute(q + " ORDER BY requirement, class_name", args)]

    def matrix(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for r in self.links("accepted"):
            out.setdefault(r["requirement"], []).append(r["class_name"])
        return out

    def gaps(self, requirements: list[str]) -> list[str]:
        """Requirements with no accepted link: untraced, so a review must confirm nothing implements them."""
        traced = set(self.matrix())
        return [r for r in requirements if r not in traced]
