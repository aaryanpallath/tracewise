"""Evaluate every method against the gold trace links.

    uv run python -m tracewise.evaluate --split test     # results/eval.json, results/runs/<split>-<method>.jsonl

Rankers are scored by MAP and recall@k, and as link sets by taking their top 3 (the median requirement has 2-3
gold links). LLM methods propose sets, scored by precision, recall, F1 and F2. LLM runs are saved one requirement
per line, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import asdict
from pathlib import Path

from . import metrics as M
from . import trace
from .corpus import DATA, ROOT, Corpus, load, split
from .search import Index

RESULTS = ROOT / "results"


def run_llm(corpus: Corpus, index: Index, method: str, ids: list[str], out: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if out.exists():
        for line in out.read_text().splitlines():
            row = json.loads(line)
            done[row["id"]] = row
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for i, rid in enumerate(ids):
            if rid in done:
                continue
            p = trace.judge(corpus, index, rid) if method == "judge" else trace.agent(corpus, index, rid)
            row = {"id": rid, **asdict(p), "gold": sorted(corpus.gold[rid])}
            done[rid] = row
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{method} {i + 1}/{len(ids)} {rid} {sorted(p.links)} gold={sorted(corpus.gold[rid])}", flush=True)
    return {r: done[r] for r in ids}


def evaluate(which: str, methods: list[str], sample: int = 0) -> dict:
    corpus = load()
    index = Index(corpus, cache=DATA / "embeddings.json")
    ids = [r for r in corpus.requirements if split(r) == which]
    if sample:  # a fixed random sample, so the LLM runs fit in a short session
        ids = sorted(random.Random(0).sample(ids, sample))
    gold = {r: corpus.gold[r] for r in ids}
    out: dict = {
        "split": which + (f" (random sample of {sample})" if sample else ""),
        "requirements": len(ids),
        "gold_links": sum(len(g) for g in gold.values()),
        "rankers": {},
        "link_sets": {},
    }
    for m in ("vsm", "embed", "hybrid"):
        ranked = {r: index.search(corpus.requirements[r].text, len(corpus.classes), m) for r in ids}
        out["rankers"][m] = {
            "map": M.mean_ap(ranked, gold),
            **{f"recall@{k}": M.recall_at(ranked, gold, k) for k in (3, 5, 10, 20)},
        }
        out["link_sets"][f"{m} top 3"] = M.set_metrics({r: set(v[:3]) for r, v in ranked.items()}, gold)
    for m in methods:
        rows = run_llm(corpus, index, m, ids, RESULTS / "runs" / f"{which}-{m}.jsonl")
        pred = {r: set(row["links"]) for r, row in rows.items()}
        res = M.set_metrics(pred, gold)
        res["mean_seconds"] = round(statistics.mean(row["seconds"] for row in rows.values()), 1)
        res["mean_prompt_tokens"] = round(statistics.mean(row["prompt_tokens"] for row in rows.values()))
        res["mean_steps"] = round(statistics.mean(row["steps"] for row in rows.values()), 1)
        res["invented_class_names"] = sum(len(row["invalid"]) for row in rows.values())
        out["link_sets"][m] = res
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"eval-{which}{f'-sample{sample}' if sample else ''}.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--methods", default="judge,agent")
    p.add_argument("--sample", type=int, default=0)
    a = p.parse_args()
    print(json.dumps(evaluate(a.split, [m for m in a.methods.split(",") if m], a.sample), indent=2))


if __name__ == "__main__":
    main()
