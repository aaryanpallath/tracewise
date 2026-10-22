# tracewise: AI Agents for Requirements-to-Code Traceability

Software in regulated products such as medical devices has to show, for every requirement, which code implements
it (IEC 62304 traceability). Keeping that trace matrix by hand is slow, so this project asks: **how well can AI
agents propose trace links, and which design works best?** It also builds the workflow around them:
- **Baselines:** keyword search (TF-IDF) and embeddings.
- **An LLM verifier:** it picks implementing classes from search results in one call.
- **A tool-calling agent:** it searches the code, reads classes and follows the call graph from an action class
  to the database classes it uses.
- **A review queue:** a person accepts or rejects every proposal before it enters the trace matrix.
- **An MCP server:** Claude Code or any other MCP client can do the tracing.

All models run locally through Ollama (Gemma 4, 8B, and nomic-embed-text), so evaluation costs nothing.

## Data

The **iTrust** benchmark from CoEST (Center of Excellence for Software & Systems Traceability). iTrust is an
open-source electronic health records system from NC State. The benchmark has:
- 131 requirements (use-case flows such as "the HCP enters a patient's office visit");
- 226 Java classes (23.6K lines);
- 286 trace links made by hand, which serve as the gold set.

`tracewise.corpus.fetch` downloads it (GPL-3.0) from
[tobhey/finegrained-traceability](https://github.com/tobhey/finegrained-traceability); it is not committed.

**Honest splits.** Prompts and the trace convention were chosen on a dev split (use cases 4, 8, 12, …: 26 flows).
Results below are on the other 105 flows, which were not used for tuning. The LLM methods ran on a fixed random
sample of 40 of them, to keep the run short on a laptop (`--sample 40`, seed 0).

## Results

Test sample: 40 requirements, 85 gold links (`results/eval-test-sample40.json`):

| Method | Precision | Recall | F1 | Time per requirement | Prompt tokens |
|---|---|---|---|---|---|
| Keyword search (TF-IDF), top 3 | 0.18 | 0.25 | 0.21 | <0.1 s | - |
| Embeddings, top 3 | 0.14 | 0.20 | 0.17 | <0.1 s | - |
| Hybrid (both, rank fusion), top 3 | 0.23 | 0.32 | 0.26 | <0.1 s | - |
| **LLM verifier** (hybrid top 12 → model picks) | **0.54** | 0.38 | **0.44** | 6.3 s | 2.3K |
| Tool-calling agent (search, read, call graph) | 0.42 | **0.40** | 0.41 | 12.0 s | 12.2K |

Ranking quality on all 105 test requirements (`results/rankers-test-full.json`):

| Ranker | MAP | Recall@5 | Recall@10 | Recall@20 |
|---|---|---|---|---|
| TF-IDF | 0.292 | 0.356 | 0.479 | 0.593 |
| Embeddings | 0.237 | 0.258 | 0.335 | 0.470 |
| Hybrid | **0.370** | **0.411** | **0.487** | **0.593** |

- **Both LLM methods roughly double the precision of search alone.** At about the same recall as the hybrid
  ranker's top 3, the verifier is right on 54% of its links against 23%.
- **The simpler design won.** One verifier call matched the agent's recall within 2 points, with higher precision,
  half the time and a fifth of the tokens. The agent's advantage is reach: through the call graph it can find a
  DAO that search never ranked. That helped on some flows and also produced plausible but unlinked DAOs on others.
- **Neither method invented a class name** (0 of 140 proposed links), because every proposal is checked against
  the code and tools return only real names.
- **Search sets the ceiling.** Even the top 20 search results contain only 59% of the gold links, so a verifier
  can't find the rest. That is the argument for an agent that can explore, and it is where the agent should improve.

**The gold set is incomplete, so these scores understate both methods.** 16 of the agent's 47 "wrong" links are on
requirements the gold set links to nothing, including flows with an obvious implementation:
- **UC22S1**, "the administrator can add a new appointment type or modify its duration": `EditApptTypeAction` has
  `addApptType` and `editApptType` methods, yet the gold set has no link.
- **Use case 40** (the appointment calendar, dev split): its seven flows have no gold links although
  `GenerateCalendarAction` implements them. These features look newer than the trace set.

This is exactly why proposals go to a reviewer instead of straight into the matrix.

**What changed during development (dev split only).**
- **A parsing bug, not the prompt, was the biggest problem.** When the agent hit its step limit, it wrote its
  answer as `{"action": "SetSecurityQuestionAction"}` instead of `{"class": ...}`. The parser read that as no
  links, so investigations that had found the right classes scored zero. The parser now accepts either key, and
  the fallback prompt spells out the format.
- **Prompt revisions mostly traded precision for recall.** Telling the agent to follow the call graph to the DAOs,
  and to link `EmailUtil` when a flow sends email, moved dev F1 between 0.42 and 0.44.

## The workflow

```bash
uv sync
uv run python -c "from tracewise.corpus import fetch; fetch()"   # data/itrust
ollama pull gemma4 && ollama pull nomic-embed-text

uv run tracewise propose UC22S1 UC22S3       # the agent proposes links with a reason for each
uv run tracewise import results/runs/test-agent.jsonl   # or load a saved evaluation run
uv run tracewise queue
#   #22  UC22S1 -> EditApptTypeAction  This action class provides methods to add and edit appointment types, ...
#   #23  UC22S1 -> ApptTypeDAO         This DAO is used by EditApptTypeAction to read and write appointment ...
uv run tracewise accept 22 --by "A. Pallath" --note "addApptType and editApptType implement both parts"
uv run tracewise accept 22 --by "someone else"   # -> link 22 was already accepted by A. Pallath
uv run tracewise matrix --out trace_matrix.csv   # accepted links with who reviewed them and when
uv run tracewise gaps                            # requirements nothing has been traced to yet

uv run python -m tracewise.evaluate --split test --sample 40   # the evaluation above
```

**MCP server.** `claude mcp add tracewise -- uv --directory /path/to/tracewise run tracewise-mcp` gives any MCP
client these tools: `get_requirement`, `search_code`, `read_class`, `related_classes`, `propose_link`,
`trace_matrix` and `coverage_gaps`. A client can only *propose*; acceptance stays with a person. Proposals are
validated: no unknown requirements, no invented classes, no empty reasons.

**Design notes**
- **A rejected link stays rejected.** The next agent run cannot re-propose it, and a link can't be reviewed twice.
  The database enforces both, and records the reviewer, time and note for an audit.
- **The agent starts with a search already run.** Before the first model call it already has the hybrid search
  results for the requirement. In an earlier project (rulebook-ai), agents that had to decide to search on their
  own often skipped it.
- **LLM runs are saved per requirement,** so an interrupted evaluation resumes where it stopped.

## Limitations

- One benchmark and one 8B local model. The 40-requirement sample makes the verifier-vs-agent gap (0.44 vs 0.41
  F1) too small to call a winner; the gap to search alone is not.
- The trace convention given to the models ("link the action and the DAOs it uses, not data beans") was read off
  the dev split's gold links. Another project's convention would need its own.
- Classes are the unit of tracing, as in the benchmark; a real trace matrix often goes down to methods.

## Tests

`uv run pytest`: a synthetic three-class codebase checks Java parsing, the gold and call-graph loading, ranking and
the metrics against hand-computed values, and link parsing with odd keys and invented names. A scripted fake model
drives the agent through tool calls and the step-limit fallback. The review queue, the CLI and the MCP tools are
tested too. CI runs lint and tests on every push; no model is needed.

Built with AI assistance (Claude Code). I reviewed the code and checked the results; the parsing bug and the gold
set gaps above came from reading the agent's runs one by one.
