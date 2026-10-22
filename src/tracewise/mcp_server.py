"""An MCP server, so any MCP client (Claude Code, Claude Desktop, an IDE agent) can trace requirements:

    claude mcp add tracewise -- uv --directory /path/to/tracewise run tracewise-mcp

Tools: get_requirement, search_code, read_class, related_classes, propose_link, trace_matrix, coverage_gaps.
A client can propose links but not accept them: acceptance stays with a person, through `tracewise accept`.
"""

from __future__ import annotations

import os
from functools import cache

from mcp.server.mcpserver import MCPServer

from . import trace
from .cli import DB
from .corpus import DATA, Corpus, load
from .search import Index
from .store import Store

mcp = MCPServer(
    "tracewise",
    instructions="Requirements and source code of the iTrust health records system. Propose trace links with "
    "evidence; a person reviews every proposal before it enters the trace matrix.",
)


@cache
def state() -> tuple[Corpus, Index, Store]:
    corpus = load()
    use_embeddings = os.environ.get("TRACEWISE_EMBEDDINGS", "1") == "1"
    return corpus, Index(corpus, cache=DATA / "embeddings.json", use_embeddings=use_embeddings), Store(DB)


@mcp.tool()
def get_requirement(requirement_id: str) -> dict:
    """A requirement (use-case flow, e.g. UC10S1) with the other flows of its use case."""
    corpus, _, _ = state()
    if requirement_id not in corpus.requirements:
        return {"error": f"no requirement {requirement_id}"}
    return {"id": requirement_id, "text": trace.requirement_context(corpus, requirement_id)}


@mcp.tool()
def search_code(query: str) -> str:
    """Find classes by what they do. Returns class names, packages and purposes."""
    corpus, index, _ = state()
    return trace.run_tool(corpus, index, "search_code", {"query": query})


@mcp.tool()
def read_class(name: str) -> str:
    """A class's package, comment and method signatures."""
    corpus, index, _ = state()
    return trace.run_tool(corpus, index, "read_class", {"name": name})


@mcp.tool()
def related_classes(name: str) -> str:
    """Project classes a class uses and the classes that use it, from the call graph."""
    corpus, index, _ = state()
    return trace.run_tool(corpus, index, "related_classes", {"name": name})


@mcp.tool()
def propose_link(requirement_id: str, class_name: str, reason: str) -> dict:
    """Propose that a class implements a requirement. It goes to the review queue, not the trace matrix."""
    corpus, _, store = state()
    if requirement_id not in corpus.requirements:
        return {"error": f"no requirement {requirement_id}"}
    if class_name not in corpus.classes:
        return {"error": f"no class {class_name}; use search_code to find class names"}
    if not reason.strip():
        return {"error": "give the evidence for the link"}
    link_id, created = store.propose(requirement_id, class_name, reason, "mcp client")
    return {"id": link_id, "status": "proposed" if created else store.get(link_id)["status"]}


@mcp.tool()
def trace_matrix() -> dict[str, list[str]]:
    """Reviewed and accepted links: requirement id -> classes."""
    return state()[2].matrix()


@mcp.tool()
def coverage_gaps() -> list[str]:
    """Requirements with no accepted link yet."""
    corpus, _, store = state()
    return store.gaps(list(corpus.requirements))


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
