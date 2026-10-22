"""The iTrust traceability benchmark: requirements, code, the gold trace links and the call graph.

iTrust is an open-source electronic health records system from NC State, used for years as a benchmark for
requirements-to-code traceability. The benchmark (from CoEST, the Center of Excellence for Software & Systems
Traceability) has:
  requirements  131 use-case flows, e.g. UC10S1 (use case 10, main flow 1) and UC10E1 (its first error flow)
  code          226 Java classes (23.6K lines)
  gold links    286 requirement -> class links set by hand (26 requirements link to no class)

`fetch` downloads it from github.com/tobhey/finegrained-traceability (GPL-3.0) into data/; it is not committed.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO = "https://github.com/tobhey/finegrained-traceability.git"
SUBDIR = "datasets/iTrust"
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


@dataclass
class Requirement:
    id: str
    text: str

    @property
    def use_case(self) -> str:
        return re.match(r"UC\d+", self.id).group(0)


@dataclass
class Method:
    signature: str
    doc: str = ""


@dataclass
class JavaClass:
    name: str
    package: str  # relative to edu.ncsu.csc.itrust, e.g. "dao.mysql"; "" for the root package
    source: str
    doc: str = ""
    methods: list[Method] = field(default_factory=list)

    def summary(self, max_methods: int = 25) -> str:
        """What a reader (or a model) needs to judge what the class does: package, class comment, methods."""
        lines = [f"{self.name} (package {self.package or 'itrust'})"]
        if self.doc:
            lines.append(self.doc)
        for m in self.methods[:max_methods]:
            lines.append(f"- {m.signature}" + (f": {m.doc}" if m.doc else ""))
        if len(self.methods) > max_methods:
            lines.append(f"- ... {len(self.methods) - max_methods} more methods")
        return "\n".join(lines)


@dataclass
class Corpus:
    requirements: dict[str, Requirement]
    classes: dict[str, JavaClass]
    gold: dict[str, set[str]]  # requirement id -> class names; requirements with no links map to an empty set
    calls: dict[str, set[str]]  # class -> project classes it uses

    def related(self, name: str) -> dict[str, list[str]]:
        return {
            "uses": sorted(self.calls.get(name, set()) - {name}),
            "used_by": sorted(c for c, used in self.calls.items() if name in used and c != name),
        }

    def flows(self, use_case: str) -> list[Requirement]:
        return [r for r in self.requirements.values() if r.use_case == use_case]


def fetch(dest: Path = DATA) -> Path:
    """Sparse-clone just the iTrust dataset (about 5 MB)."""
    out = dest / "itrust"
    if out.exists():
        return out
    tmp = dest / "_clone"
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", REPO, str(tmp)], check=True)
    subprocess.run(["git", "-C", str(tmp), "sparse-checkout", "set", SUBDIR], check=True)
    (tmp / SUBDIR).rename(out)
    subprocess.run(["rm", "-rf", str(tmp)], check=True)
    return out


_EDGE = re.compile(r"C:edu\.ncsu\.csc\.itrust\.(?:[\w.]+\.)?(\w+) edu\.ncsu\.csc\.itrust\.(?:[\w.]+\.)?(\w+)$")
_DOC = re.compile(r"/\*\*(.*?)\*/", re.S)
_CLASS = re.compile(r"\b(?:class|interface|enum)\s+(\w+)")
# a method declaration: modifiers, a return type, a name and a parameter list, then a body or a semicolon
_METHOD = re.compile(
    r"(?:/\*\*(?P<doc>(?:(?!\*/).)*?)\*/\s*)?"
    r"^[ \t]*(?:@\w+\s*)*(?P<sig>(?:public|protected|private)\s+(?:static\s+|final\s+|abstract\s+|synchronized\s+)*"
    r"[\w<>\[\], ?]+\s+\w+\s*\([^)]*\))\s*(?:throws [\w., ]+)?\s*[{;]",
    re.S | re.M,
)


def clean_doc(raw: str, sentences: int = 1) -> str:
    """Javadoc text without stars, tags or HTML, cut to the first sentence(s)."""
    text = re.sub(r"^\s*\*", " ", raw, flags=re.M)
    text = re.split(r"\s@\w+", " " + text)[0]
    text = re.sub(r"<[^>]+>|\{@\w+\s+([^}]*)\}", r"\1", text)
    text = " ".join(text.split())
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:sentences]).strip()


def parse_java(name: str, package: str, source: str) -> JavaClass:
    head = source[: _CLASS.search(source).start()] if _CLASS.search(source) else ""
    docs = _DOC.findall(head)
    methods = [
        Method(" ".join(m.group("sig").split()), clean_doc(m.group("doc") or ""))
        for m in _METHOD.finditer(source)
        if not re.match(r"(?:public|protected|private)\s+(?:static\s+)?(?:class|interface|enum)\b", m.group("sig"))
    ]
    return JavaClass(name, package, source, clean_doc(docs[-1], 2) if docs else "", methods)


def load(path: Path | None = None) -> Corpus:
    path = path or fetch()
    reqs = {
        p.stem: Requirement(p.stem, " ".join(p.read_text(errors="replace").split()))
        for p in sorted((path / "req").glob("*.txt"))
    }
    classes = {}
    for p in sorted((path / "code").rglob("*.java")):
        package = ".".join(p.parent.relative_to(path / "code").parts)
        classes[p.stem] = parse_java(p.stem, package, p.read_text(errors="replace"))
    gold: dict[str, set[str]] = {r: set() for r in reqs}
    for line in (path / "itrust_solution_links.txt").read_text().splitlines():
        if ": " in line:
            r, c = (s.strip() for s in line.split(": ", 1))
            gold[r.removesuffix(".txt")].add(c.removesuffix(".java"))
    calls: dict[str, set[str]] = {}
    graph = path / "itrust_raw_callgraph.txt"
    if graph.exists():
        for line in graph.read_text().splitlines():
            m = _EDGE.match(line)
            if m and m.group(1) in classes and m.group(2) in classes:
                calls.setdefault(m.group(1), set()).add(m.group(2))
    return Corpus(reqs, classes, gold, calls)


def split(req_id: str) -> str:
    """Prompts and settings are chosen on the dev split (use cases numbered 4, 8, 12, ...: 26 flows); everything
    reported is measured on the other 105 flows."""
    return "dev" if int(re.match(r"UC(\d+)", req_id).group(1)) % 4 == 0 else "test"
