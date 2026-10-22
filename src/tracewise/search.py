"""Ranking classes for a requirement: the classic information-retrieval baseline, embeddings, and both combined.

vsm     TF-IDF cosine over the class's whole source: identifiers split at camelCase, comments, string literals.
        The standard baseline in traceability research.
embed   cosine similarity of nomic-embed-text embeddings of the requirement and of the class summary
        (package, class comment, method signatures and their comments)
hybrid  reciprocal-rank fusion of the two
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .corpus import Corpus

STOP = set(
    """a an and are as at be by for from has have if in into is it its of on or that the their them then there these
    this to was were will with which who can may must not no all any each other than so such only own same also
    system user users page shall should would""".split()
)
JAVA = set(
    """public private protected static final void class interface extends implements import package return new null
    true false throws throw try catch finally if else for while do switch case break continue this super int long
    boolean double float char byte short string object list arraylist map hashmap set override exception java util
    sql edu ncsu csc itrust get set""".split()
)


def stem(w: str) -> str:
    """A light suffix stripper: enough to match 'visits'/'visit' and 'editing'/'edit' without a dependency."""
    for suf, rep in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("ed", ""), ("es", "e"), ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + rep
    return w


def tokens(text: str, code: bool = False) -> list[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)  # officeVisit -> office Visit
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)  # HCPAction -> HCP Action
    words = re.findall(r"[a-z]+", text.lower())
    drop = STOP | JAVA if code else STOP
    return [stem(w) for w in words if len(w) > 1 and w not in drop]


class Index:
    def __init__(self, corpus: Corpus, cache: Path | None = None, use_embeddings: bool = True):
        self.corpus = corpus
        self.names = list(corpus.classes)
        docs = [Counter(tokens(c.source, code=True) + tokens(c.name, code=True) * 3) for c in corpus.classes.values()]
        df = Counter(t for d in docs for t in d)
        n = len(docs)
        self.idf = {t: math.log((1 + n) / (1 + k)) + 1 for t, k in df.items()}
        self.doc_vecs = [self._weights(d) for d in docs]
        self.emb: np.ndarray | None = None
        if use_embeddings:
            self.emb = self._embeddings(cache)

    def _weights(self, counts: Counter) -> dict[str, float]:
        w = {t: (1 + math.log(k)) * self.idf.get(t, 0.0) for t, k in counts.items()}
        norm = math.sqrt(sum(v * v for v in w.values())) or 1.0
        return {t: v / norm for t, v in w.items()}

    def _embeddings(self, cache: Path | None) -> np.ndarray:
        from .llm import EMBED_MODEL, embed

        texts = [c.summary()[:2000] for c in self.corpus.classes.values()]
        key = {"model": EMBED_MODEL, "names": self.names, "chars": [len(t) for t in texts]}
        if cache and cache.exists():
            stored = json.loads(cache.read_text())
            if stored.get("key") == key:
                return np.array(stored["vectors"])
        vecs = embed([f"search_document: {t}" for t in texts])
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"key": key, "vectors": vecs}))
        return np.array(vecs)

    def vsm(self, query: str) -> np.ndarray:
        q = self._weights(Counter(tokens(query)))
        return np.array([sum(v * d.get(t, 0.0) for t, v in q.items()) for d in self.doc_vecs])

    def embed_scores(self, query: str) -> np.ndarray:
        from .llm import embed

        if self.emb is None:
            raise RuntimeError("index built without embeddings")
        q = np.array(embed([f"search_query: {query}"])[0])
        e = self.emb
        return e @ q / (np.linalg.norm(e, axis=1) * np.linalg.norm(q) + 1e-9)

    def rank(self, query: str, method: str = "hybrid") -> list[tuple[str, float]]:
        if method == "vsm" or (method == "hybrid" and self.emb is None):
            s = self.vsm(query)
        elif method == "embed":
            s = self.embed_scores(query)
        elif method == "hybrid":
            s = rrf([self.vsm(query), self.embed_scores(query)])
        else:
            raise ValueError(method)
        order = np.argsort(-s, kind="stable")
        return [(self.names[i], float(s[i])) for i in order]

    def search(self, query: str, k: int = 8, method: str = "hybrid") -> list[str]:
        return [n for n, _ in self.rank(query, method)[:k]]


def rrf(score_lists: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal-rank fusion: each ranking contributes 1 / (k + rank)."""
    out = np.zeros(len(score_lists[0]))
    for s in score_lists:
        ranks = np.empty(len(s))
        ranks[np.argsort(-s, kind="stable")] = np.arange(1, len(s) + 1)
        out += 1.0 / (k + ranks)
    return out
