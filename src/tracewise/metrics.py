"""Trace-link metrics, pooled over requirements (micro-averaged, as in the traceability literature).

precision  share of proposed links that are in the gold set
recall     share of gold links proposed
F2         recall-weighted F score; traceability studies favor recall, since a missed link is costlier for a
           safety review than an extra one a reviewer rejects
MAP        mean average precision of a ranking, over requirements that have gold links
"""

from __future__ import annotations


def set_metrics(pred: dict[str, set[str]], gold: dict[str, set[str]]) -> dict[str, float]:
    tp = sum(len(pred.get(r, set()) & g) for r, g in gold.items())
    proposed = sum(len(pred.get(r, set())) for r in gold)
    actual = sum(len(g) for g in gold.values())
    p = tp / proposed if proposed else 0.0
    r = tp / actual if actual else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    f2 = 5 * p * r / (4 * p + r) if p + r else 0.0
    unlinked = [q for q, g in gold.items() if not g]
    return {
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(f1, 3),
        "f2": round(f2, 3),
        "proposed": proposed,
        "correct": tp,
        "gold": actual,
        # requirements the gold set links to nothing: did the method keep them empty?
        "empty_kept_empty": f"{sum(not pred.get(q) for q in unlinked)}/{len(unlinked)}",
    }


def average_precision(ranked: list[str], gold: set[str]) -> float:
    hits, total = 0, 0.0
    for i, name in enumerate(ranked, 1):
        if name in gold:
            hits += 1
            total += hits / i
    return total / len(gold) if gold else 0.0


def mean_ap(rankings: dict[str, list[str]], gold: dict[str, set[str]]) -> float:
    aps = [average_precision(rankings[r], g) for r, g in gold.items() if g]
    return round(sum(aps) / len(aps), 3) if aps else 0.0


def recall_at(rankings: dict[str, list[str]], gold: dict[str, set[str]], k: int) -> float:
    found = sum(len(set(rankings[r][:k]) & g) for r, g in gold.items())
    total = sum(len(g) for g in gold.values())
    return round(found / total, 3) if total else 0.0
