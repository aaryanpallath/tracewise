"""Ways to propose trace links for one requirement, compared in the evaluation.

top_k     the first k classes of a ranking (vsm, embed or hybrid): no model involved
judge     retrieve candidates, then the LLM picks the ones that implement the requirement (one call)
agent     the LLM investigates with tools (search, read a class, follow the call graph) and submits links with
          a reason for each, up to a step limit

All LLM methods see the requirement together with the other flows of its use case, as a developer reading the
use case document would.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .corpus import Corpus
from .llm import chat
from .search import Index

# The project's trace convention, read off the dev split's gold links (not the test split's): a flow links to the
# action class that handles it and the DAO classes that read or write its data, plus validators or risk classes
# where the flow is about them. Data beans, bean loaders, enums and exceptions are almost never linked.
CONVENTION = (
    "iTrust is a Java electronic health records system. Packages: action (one class per page or user task, the "
    "business logic), dao.mysql (database access, one DAO per table), beans and beans.loaders (plain data holders), "
    "validate (input validators), risk (patient risk calculators), enums, exception. A trace link connects a "
    "requirement to the classes that implement it: the action class(es) that perform the flow and the DAO classes "
    "those actions use to read or write the data the flow describes, plus a validator or risk class when the flow "
    "is specifically about validation or risk, and EmailUtil when the flow sends an email. Do not link data beans, "
    "bean loaders, enums, exceptions or general utilities. Some flows (for example, 'the patient chosen is not the "
    "desired patient') describe something no class implements: link nothing for those."
)


@dataclass
class Proposal:
    links: dict[str, str]  # class name -> reason
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    steps: int = 1
    invalid: list[str] = field(default_factory=list)  # names the model gave that are not classes in the code
    error: str = ""


def requirement_context(corpus: Corpus, req_id: str) -> str:
    r = corpus.requirements[req_id]
    others = [f"  {o.id}: {o.text}" for o in corpus.flows(r.use_case) if o.id != req_id]
    ctx = f"Requirement {r.id}: {r.text}"
    if others:
        ctx += "\n\nOther flows of the same use case, for context only:\n" + "\n".join(others)
    return ctx


def top_k(index: Index, text: str, method: str, k: int) -> Proposal:
    return Proposal({n: f"{method} rank {i + 1}" for i, n in enumerate(index.search(text, k, method))}, seconds=0.0)


def _parse_links(text: str, corpus: Corpus) -> tuple[dict[str, str], list[str]]:
    """Read {"links": [{"class": ..., "reason": ...}]} from a reply, tolerating code fences and bare names."""
    m = re.search(r"\{.*\}", text, re.S)
    items: list = []
    if m:
        try:
            items = json.loads(m.group(0)).get("links", [])
        except (json.JSONDecodeError, AttributeError):
            items = []
    return _validate(items, corpus)


def _validate(items: list, corpus: Corpus) -> tuple[dict[str, str], list[str]]:
    links, invalid = {}, []
    for it in items:
        if isinstance(it, dict):  # models sometimes name the key "name" or "action" instead of "class"
            name = it.get("class") or it.get("name") or next((v for k, v in it.items() if k != "reason"), "")
            reason = it.get("reason", "")
        else:
            name, reason = str(it), ""
        name = str(name).strip().removesuffix(".java").split(".")[-1]
        if name in corpus.classes:
            links[name] = str(reason)
        elif name:
            invalid.append(name)
    return links, invalid


JUDGE = (
    CONVENTION
    + "\n\nYou are given a requirement and candidate classes found by search. Choose the candidates that implement "
    'the requirement. Reply with JSON only: {"links": [{"class": "ClassName", "reason": "one short sentence"}]}. '
    "Use an empty list when no candidate implements it."
)


def judge(corpus: Corpus, index: Index, req_id: str, k: int = 12) -> Proposal:
    text = corpus.requirements[req_id].text
    cands = index.search(text, k)
    listing = "\n\n".join(corpus.classes[n].summary(max_methods=10)[:900] for n in cands)
    prompt = f"{requirement_context(corpus, req_id)}\n\nCandidate classes:\n\n{listing}"
    r = chat([{"role": "system", "content": JUDGE}, {"role": "user", "content": prompt}])
    links, invalid = _parse_links(r.content, corpus)
    return Proposal(links, r.seconds, r.prompt_tokens, r.completion_tokens, 1, invalid)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search the iTrust classes. Returns the best matches with their package and purpose.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string", "description": "What the code would do, in plain words"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_class",
            "description": "A class's package, comment and method signatures.",
            "parameters": {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "related_classes",
            "description": "The project classes a class uses, and the classes that use it (from the call graph). "
            "Use it to find the DAOs an action class reads and writes.",
            "parameters": {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_links",
            "description": "Submit the final trace links for the requirement. Call once, at the end. An empty "
            "list means no class implements the requirement.",
            "parameters": {
                "type": "object",
                "required": ["links"],
                "properties": {
                    "links": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["class", "reason"],
                            "properties": {"class": {"type": "string"}, "reason": {"type": "string"}},
                        },
                    }
                },
            },
        },
    },
]

AGENT = (
    CONVENTION
    + "\n\nFind the classes that implement the requirement. Search results for the requirement are already below. "
    "Steps: 1) pick the action class(es) that perform this flow, searching again with other words if none fits; "
    "2) call related_classes on each one; 3) link the DAOs it uses for the data this flow reads or writes (for "
    "example PatientDAO for patient demographics, AuthDAO for passwords and security questions), reading a DAO "
    "if unsure. Then call submit_links with each class and a one-sentence reason. Link only classes you have "
    "evidence for."
)


FINAL = 'Step limit reached. Reply with JSON only: {"links": [{"class": "ClassName", "reason": "one sentence"}]}'


def _summary_line(corpus: Corpus, name: str) -> dict:
    c = corpus.classes[name]
    return {"class": name, "package": c.package or "itrust", "purpose": c.doc[:200]}


def run_tool(corpus: Corpus, index: Index, name: str, args: dict) -> str:
    if name == "search_code":
        hits = index.search(str(args.get("query", "")), 8)
        return json.dumps([_summary_line(corpus, n) for n in hits])
    target = str(args.get("name", "")).strip().removesuffix(".java").split(".")[-1]
    if name in ("read_class", "related_classes") and target not in corpus.classes:
        return json.dumps({"error": f"no class named {target!r}; use search_code to find class names"})
    if name == "read_class":
        return corpus.classes[target].summary()[:2500]
    if name == "related_classes":
        rel = corpus.related(target)
        return json.dumps(
            {k: [f"{n} ({corpus.classes[n].package or 'itrust'})" for n in v][:25] for k, v in rel.items()}
        )
    return json.dumps({"error": f"unknown tool {name}"})


def _args(call: dict) -> dict:
    a = call["function"].get("arguments") or {}
    if isinstance(a, str):
        try:
            a = json.loads(a or "{}")
        except json.JSONDecodeError:
            a = {}
    return a if isinstance(a, dict) else {}


def agent(corpus: Corpus, index: Index, req_id: str, max_steps: int = 12) -> Proposal:
    text = corpus.requirements[req_id].text
    seed = run_tool(corpus, index, "search_code", {"query": text})
    messages = [
        {"role": "system", "content": AGENT},
        {"role": "user", "content": requirement_context(corpus, req_id)},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "search_code", "arguments": {"query": text}}}],
        },
        {"role": "tool", "content": seed, "tool_name": "search_code"},
    ]
    p = Proposal({})
    for step in range(1, max_steps + 1):
        r = chat(messages, tools=TOOLS)
        p.seconds += r.seconds
        p.prompt_tokens += r.prompt_tokens
        p.completion_tokens += r.completion_tokens
        p.steps = step
        if not r.tool_calls:  # answered in text instead of submitting: read links from the text if any
            p.links, p.invalid = _parse_links(r.content, corpus)
            return p
        messages.append(r.message)
        for call in r.tool_calls:
            if call["function"]["name"] == "submit_links":
                links = _args(call).get("links", [])
                p.links, p.invalid = _validate(links if isinstance(links, list) else [], corpus)
                return p
            result = run_tool(corpus, index, call["function"]["name"], _args(call))
            messages.append({"role": "tool", "content": result, "tool_name": call["function"]["name"]})
    final = chat(messages + [{"role": "user", "content": FINAL}])
    p.seconds += final.seconds
    p.prompt_tokens += final.prompt_tokens
    p.completion_tokens += final.completion_tokens
    p.links, p.invalid = _parse_links(final.content, corpus)
    p.error = "step limit"
    return p
