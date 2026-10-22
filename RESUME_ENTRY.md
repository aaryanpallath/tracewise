# tracewise — AI Agents for Requirements-to-Code Traceability
Tech: Python, Ollama, Gemma, embeddings, RAG, MCP, SQLite, pytest, GitHub Actions
- Built AI agents that trace requirements to code for a health records system (131 requirements, 226 Java classes), with tool calling over search and the call graph, an MCP server and a human review queue for every link.
- Evaluated 5 methods against 286 hand-made trace links: an LLM verifier reached 54% precision vs 23% for keyword and embedding search, with zero invented class names across 140 proposed links.
- Found a parser bug that zeroed out correct agent answers and showed the benchmark's gold set misses real links, which is why every link goes through human review.
