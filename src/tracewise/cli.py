"""tracewise: propose trace links with the agent, review them, export the trace matrix.

tracewise propose UC10S1 UC10S2        # the agent proposes links (or --all); they wait for review
tracewise import results/runs/test-agent.jsonl
tracewise queue                        # proposed links with the agent's reason for each
tracewise accept 12 --by "A. Pallath" --note "checked EditPHRAction.updateInformation"
tracewise reject 13 --by "A. Pallath" --note "bean, not behavior"
tracewise matrix --out trace_matrix.csv
tracewise gaps                         # requirements with no accepted link
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from .corpus import DATA, load
from .store import ReviewError, Store

DB = Path(os.environ.get("TRACEWISE_DB", DATA / "review.sqlite"))


def cell(value: object) -> str:
    """A CSV cell that a spreadsheet won't run as a formula (model-written reasons are untrusted text)."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def write_matrix(f, rows: list[dict]) -> None:
    w = csv.writer(f)
    cols = ("requirement", "class_name", "reason", "proposed_by", "reviewed_by", "reviewed_at", "note")
    w.writerow(["requirement", "class", *cols[2:]])
    w.writerows([cell(r[k]) for k in cols] for r in rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tracewise")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("propose")
    pr.add_argument("requirements", nargs="*")
    pr.add_argument("--all", action="store_true")
    im = sub.add_parser("import")
    im.add_argument("run", type=Path)
    sub.add_parser("queue")
    for name in ("accept", "reject"):
        r = sub.add_parser(name)
        r.add_argument("id", type=int)
        r.add_argument("--by", required=True)
        r.add_argument("--note", default="")
    mx = sub.add_parser("matrix")
    mx.add_argument("--out", type=Path)
    sub.add_parser("gaps")
    a = p.parse_args(argv)
    store = Store(DB)

    if a.cmd == "propose":
        from . import trace
        from .search import Index

        corpus = load()
        index = Index(corpus, cache=DATA / "embeddings.json")
        ids = list(corpus.requirements) if a.all else a.requirements
        for rid in ids:
            if rid not in corpus.requirements:
                print(f"unknown requirement {rid}", file=sys.stderr)
                return 2
            prop = trace.agent(corpus, index, rid)
            for cls, reason in prop.links.items():
                link_id, created = store.propose(rid, cls, reason, "agent")
                print(f"{'+' if created else '='} #{link_id} {rid} -> {cls}: {reason}")
            if not prop.links:
                print(f"  {rid}: the agent found no implementing class")
    elif a.cmd == "import":
        n = 0
        for line in a.run.read_text().splitlines():
            row = json.loads(line)
            for cls, reason in row["links"].items():
                n += store.propose(row["id"], cls, reason, f"agent ({a.run.name})")[1]
        print(f"imported {n} new proposals")
    elif a.cmd == "queue":
        for r in store.links("proposed"):
            print(f"#{r['id']:<5} {r['requirement']:<8} -> {r['class_name']:<36} {r['reason']}")
    elif a.cmd in ("accept", "reject"):
        try:
            r = store.review(a.id, a.cmd + "ed", a.by, a.note)
        except ReviewError as e:
            print(e, file=sys.stderr)
            return 1
        print(f"#{r['id']} {r['requirement']} -> {r['class_name']}: {r['status']} by {r['reviewed_by']}")
    elif a.cmd == "matrix":
        rows = store.links("accepted")
        if a.out:
            with open(a.out, "w", newline="") as f:
                write_matrix(f, rows)
            print(f"{len(rows)} accepted links -> {a.out}")
        else:
            write_matrix(sys.stdout, rows)
    else:
        for rid in store.gaps(list(load().requirements)):
            print(rid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
