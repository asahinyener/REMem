# Recursive RLMs over a Temporal Knowledge Graph

This is an exploratory walkthrough built on top of ReMem. The goal is to make an agent trace easy to inspect, not to claim a new method or benchmark result.

The demo uses TimelineKGQA CRON question `25107`:

```text
During which period did Andre Hahn serve as a member of the German Bundestag
while Jason Shackell was a member of Burnley F.C.?
```

The answer requires two independent temporal facts:

```text
Andre Hahn -> German Bundestag:
2013-01-01 to 2017-01-01

Jason Shackell -> Burnley F.C.:
2012-01-01 to 2015-01-01
```

The parent RLM asks child branches to gather graph evidence, stores the returned branch strings in the REPL, and computes:

```python
overlap_start = max(hahn_start, shackell_start)
overlap_end = min(hahn_end, shackell_end)
```

Final answer:

```text
(2013-01-01, 2015-01-01)
```

## Trace Visualization

Open the HTML visualization:

```text
docs/artifacts/visualizations/timelinekgqa_q25107_recursive_rlm_trace.html
```

It shows:

- the parent `llm_query_batched` call
- child branch graph retrieval
- parent REPL state
- Python interval aggregation
- final submission

## Reproduce

See:

```text
examples/timelinekgqa/README.md
```
