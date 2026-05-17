# TimelineKGQA Recursive RLM Demo

This example is a small, inspectable temporal KGQA run. It answers one TimelineKGQA CRON question by building a temporal ReMem graph, running a DSPy RLM agent over graph tools, and saving the full trace.

The goal is reproducibility and inspection, not a benchmark claim.

## Question

```text
During which period did Andre Hahn serve as a member of the German Bundestag
while Jason Shackell was a member of Burnley F.C.?
```

The answer is the overlap between two temporal facts:

```text
Andre Hahn -> member of the German Bundestag:
2013-01-01 to 2017-01-01

Jason Shackell -> Burnley F.C.:
2012-01-01 to 2015-01-01
```

Expected interval:

```text
(2013-01-01, 2015-01-01)
```

## What The Runner Does

```text
TimelineKGQA dataset files
-> select CRON question 25107
-> create 10 short temporal fact documents
-> run temporal extraction
-> write OpenIE/extraction cache
-> embed chunks, facts, entities, and verbatim text
-> write vector-store .pkl caches
-> build graph.pkl
-> run a DSPy RLM graph agent
-> save timelinekgqa_cron_q25107_dspy_rlm_trace.json
```

On the first run, the script pays the extraction and embedding cost. On later runs with the same model names and `FORCE_INDEX=false FORCE_OPENIE=false`, it loads the cached graph and vector stores.

## Requirements

Required local dataset files:

```text
external/timelinekgqa/Datasets/unified_kg_cron.json
external/timelinekgqa/Datasets/unified_kg_cron_questions_all.json
```

Required runtime pieces:

- Python dependencies from ReMem
- Deno on `PATH`, because DSPy RLM executes Python through its Deno-backed interpreter
- an OpenAI-compatible chat endpoint
- an embedding endpoint for `text-embedding-3-small`
- recursive DSPy RLM support if you want child RLMs that can call graph tools

The original artifact used a local recursive DSPy checkout:

```text
../external/dspy_recursive
```

If that directory exists, `run_cached_trace.sh` adds it to `PYTHONPATH` automatically.

## Environment

Create a local `.env` in the repository root. Do not commit it.

```bash
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_KEY=...

OPENAI_EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BASE_URL=
EMBEDDING_API_KEY=
```

For GPT-5-family models, keep:

```bash
LITELLM_DROP_PARAMS=true
```

Those models reject some legacy chat parameters such as `temperature=0`; the runner enables LiteLLM parameter dropping by default.

## Preflight

Run this before spending tokens:

```bash
PYTHONPATH=../external/dspy_recursive:src \
TIMELINEKGQA_DATASET_DIR=../external/timelinekgqa/Datasets \
python examples/timelinekgqa/check_reproduction.py
```

The preflight checks:

- dataset files
- packaged reference trace and visualization
- whether `dspy.RLM` supports recursive `max_depth`
- whether `deno` is on `PATH`
- visible provider/model settings
- whether credentials are present

It does not call the network.

## Recommended Run

From the repository root:

```bash
bash examples/timelinekgqa/run_cached_trace.sh
```

Useful overrides:

```bash
TIMELINEKGQA_DATASET_DIR=../external/timelinekgqa/Datasets \
TIMELINEKGQA_QUESTION_ID=25107 \
OPENAI_MODEL=gpt-5.4-nano \
AGENT_MAX_STEPS=8 \
RLM_MAX_TOKENS=8192 \
FORCE_INDEX=false \
FORCE_OPENIE=false \
bash examples/timelinekgqa/run_cached_trace.sh
```

If Deno is installed but not on `PATH`, set:

```bash
DENO_BIN_DIR=/root/.deno/bin bash examples/timelinekgqa/run_cached_trace.sh
```

## Direct Python Run

```bash
export PATH=/root/.deno/bin:$PATH

PYTHONPATH=../external/dspy_recursive:src \
TIMELINEKGQA_DATASET_DIR=../external/timelinekgqa/Datasets \
TIMELINEKGQA_QUESTION_ID=25107 \
TIMELINEKGQA_RLM_SUBLM_DEMO=true \
AGENT_MAX_STEPS=8 \
RLM_MAX_TOKENS=8192 \
FORCE_INDEX=false \
FORCE_OPENIE=false \
LITELLM_DROP_PARAMS=true \
python examples/timelinekgqa/run_timelinekgqa_temporal_rlm_trace.py
```

## Cache Layout

For question `25107`, model `gpt-5.4-nano`, and embedding model `text-embedding-3-small`, the run writes:

```text
outputs/timelinekgqa_one_question_temporal_graph/
  cron_q25107_gpt-5.4-nano_text-embedding-3-small/
    graph.pkl
    chunk_embeddings/vdb_chunk.pkl
    verbatim_embeddings/vdb_verbatim.pkl
    facts_embeddings/vdb_facts.pkl
    entity_embeddings/vdb_entity.pkl
    timelinekgqa_cron_q25107_dspy_rlm_trace.json
```

Extraction cache is stored separately:

```text
outputs/timelinekgqa_cron_q25107/
  openie_results_ner_<model>.json
```

To force a rebuild:

```bash
FORCE_INDEX=true FORCE_OPENIE=true bash examples/timelinekgqa/run_cached_trace.sh
```

## Reading The Trace

The trace JSON has:

```text
question
gold_answer
events
docs
answer
retrieval_metrics
qa_metrics
agent_session_logs
```

The most useful fields are:

```text
agent_session_logs.rlm_trajectory
agent_session_logs.llm_interactions
```

`rlm_trajectory` shows parent REPL turns. `llm_interactions` shows graph tool calls, including `semantic_retrieve` and `find_entity_contexts`.

The intended trace shape is:

```text
parent RLM
  llm_query_batched([
    child branch 1: gather Andre Hahn -> German Bundestag evidence from graph tools
    child branch 2: gather Jason Shackell -> Burnley F.C. evidence from graph tools
  ])

parent REPL
  parse branch intervals
  compute max(start dates)
  compute min(end dates)
  submit final overlap
```

## Included Reference Artifacts

The saved reference trace and visualization are in:

```text
docs/artifacts/traces/timelinekgqa_q25107_recursive_rlm_trace.json
docs/artifacts/visualizations/timelinekgqa_q25107_recursive_rlm_trace.html
```
