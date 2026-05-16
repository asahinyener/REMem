# TimelineKGQA Recursive RLM Demo

This example is a small, inspectable temporal KGQA run. It uses a single TimelineKGQA CRON question where the answer is the intersection of two temporal facts:

```text
During which period did Andre Hahn serve as a member of the German Bundestag
while Jason Shackell was a member of Burnley F.C.?
```

The intended trace shape is:

```text
parent RLM
  llm_query_batched([
    child branch 1: gather Andre Hahn -> German Bundestag evidence from graph tools
    child branch 2: gather Jason Shackell -> Burnley F.C. evidence from graph tools
  ])

parent REPL
  store returned branch evidence
  compute interval intersection in Python
  submit final answer
```

## Inputs

The runner expects the TimelineKGQA dataset directory:

```text
external/timelinekgqa/Datasets
```

Required files:

```text
unified_kg_cron.json
unified_kg_cron_questions_all.json
```

## Run

From the repository root:

```bash
PYTHONPATH=src python examples/timelinekgqa/check_reproduction.py
```

```bash
PYTHONPATH=src \
TIMELINEKGQA_DATASET_DIR=../external/timelinekgqa/Datasets \
TIMELINEKGQA_QUESTION_ID=25107 \
TIMELINEKGQA_RLM_SUBLM_DEMO=true \
AGENT_MAX_STEPS=8 \
RLM_MAX_TOKENS=8192 \
python examples/timelinekgqa/run_timelinekgqa_temporal_rlm_trace.py
```

Recursive branch RLMs require a DSPy `RLM` implementation that supports `max_depth`. In the original workspace, the recursive run used a local DSPy source checkout:

```bash
PYTHONPATH=../external/dspy_recursive:src \
TIMELINEKGQA_DATASET_DIR=../external/timelinekgqa/Datasets \
TIMELINEKGQA_QUESTION_ID=25107 \
TIMELINEKGQA_RLM_SUBLM_DEMO=true \
AGENT_MAX_STEPS=8 \
RLM_MAX_TOKENS=8192 \
python examples/timelinekgqa/run_timelinekgqa_temporal_rlm_trace.py
```

The runner still works with non-recursive DSPy RLM, but `llm_query_batched` will behave as plain sub-LM calls rather than child RLMs with graph tools.

## Included Artifacts

The saved trace and visualization for the reference run are in:

```text
docs/artifacts/traces/timelinekgqa_q25107_recursive_rlm_trace.json
docs/artifacts/visualizations/timelinekgqa_q25107_recursive_rlm_trace.html
```
