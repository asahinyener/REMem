# ReMem Fork Diff Notes

This fork is intentionally narrow. It adds enough plumbing to run an exploratory recursive RLM trace over a temporal knowledge graph built from TimelineKGQA.

## Changes In ReMem

These changes modify existing ReMem behavior or fix integration issues needed by the demo.

### Temporal Graph Bugfixes

File:

```text
src/remem/rag_strategies/temporal_strategy.py
```

Changes:

- Hashes temporal facts with the same formatted representation used by retrieval.
- Adds graph edges from verbatim nodes to fact nodes.
- Adds graph edges from subject entities to fact nodes.
- Adds graph edges from fact nodes to object entities.
- Uses the available embedding stores: `verbatim`, `entity`, and `facts`.
- Fixes igraph node attribute access by checking `node.attributes()`.

Why:

Without these changes, temporal facts can be extracted but are harder to traverse as a graph. The demo needs entity-to-fact and fact-to-entity traversal for structured branch retrieval.

### Temporal Extraction Control

File:

```text
src/remem/information_extraction/temporal_extraction_openai.py
```

Change:

- Adds `TEMPORAL_EXTRACTION_WORKERS`.

Why:

This makes extraction easier to run against API-backed models without hard-coding worker count.

### OpenAI-Compatible Config

Files:

```text
src/remem/utils/config_utils.py
src/remem/remem.py
src/remem/embedding_model/
src/remem/llm/
```

Changes:

- Adds explicit LLM and embedding API-key resolution.
- Adds embedding base URL resolution.
- Avoids logging raw API keys.
- Adds `agent_type` to config so temporal retrieval can select the graph controller.

Why:

The example uses OpenAI-compatible chat and embedding APIs. These changes make that path explicit and easier to configure.

## Additions On Top Of ReMem

These additions are new wrapper/controller layers. They are not meant to replace ReMem's existing retrieval logic.

### DSPy ReAct Over Graph Tools

File:

```text
src/remem/agent/dspy_react_agent.py
```

Adds a DSPy ReAct controller over existing ReMem graph tools:

- `semantic_retrieve`
- `find_gist_contexts`
- `find_entity_contexts`

The wrapper logs tool calls through ReMem's existing agent logging path and formats retrieved graph evidence for DSPy.

### DSPy RLM Over Graph Tools

File:

```text
src/remem/agent/dspy_rlm_agent.py
```

Adds a DSPy `RLM` controller over the same graph tools. When the installed DSPy implementation supports `max_depth`, the controller enables recursive child RLM calls so `llm_query` and `llm_query_batched` can spawn branch agents with graph-tool access.

This is the main agent behavior used by the TimelineKGQA demo.

## Demo Files

```text
examples/timelinekgqa/run_timelinekgqa_temporal_rlm_trace.py
docs/artifacts/traces/timelinekgqa_q25107_recursive_rlm_trace.json
docs/artifacts/visualizations/timelinekgqa_q25107_recursive_rlm_trace.html
```

The example:

1. Loads one TimelineKGQA CRON question.
2. Converts relevant KG facts into temporal fact sentences.
3. Builds or loads a ReMem temporal graph.
4. Runs the DSPy RLM graph agent.
5. Saves the trace.
6. Displays the branch/gather/reduce trace through the HTML visualization.

## Non-Goals

This is not a new benchmark result or a claim of novelty. It is a worked example showing how a recursive RLM can decompose a temporal KGQA question, gather graph evidence in branches, and aggregate intervals in a Python REPL.

## Known Limitations

- Recursive branch RLMs require a DSPy `RLM` implementation with `max_depth`. In this workspace, that came from `external/dspy_recursive`.
- The trace is a teaching artifact. The parent RLM trajectory and child graph-tool logs are both available, but the raw log schema is not normalized into a single nested trace format.
- TimelineKGQA is expected to live outside the repo under `external/timelinekgqa/`.
