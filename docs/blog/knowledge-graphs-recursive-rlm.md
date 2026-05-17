# From Knowledge Graphs to Recursive RLMs

This post explains the small TimelineKGQA artifact in this repository. It starts with knowledge graphs, adds time, and ends with a recursive language-model trace that answers one temporal question by searching a graph and computing an interval overlap.

The point is pedagogical. The trace is not a new benchmark result. It is a compact example of how symbolic graph structure and language-model reasoning can work together.

## Knowledge Graphs

A knowledge graph stores facts as entities and relationships.

```text
Andre Hahn --position held--> member of the German Bundestag
Jason Shackell --member of sports team--> Burnley F.C.
```

This representation is old in spirit. Semantic networks, expert systems, RDF triples, ontologies, linked data, and knowledge bases all tried to make facts machine-readable. The phrase "knowledge graph" became familiar to many people through web search.

Google's 2012 Knowledge Graph announcement framed the shift as search over things, not only strings. Instead of treating a query as a bag of words, a search engine can recognize real entities, connect them to known facts, and show a card with structured information. A card for a person, city, film, company, or sports team works because the system has some model of entities and their relationships.

That public search-card experience is a useful way to explain the idea: a knowledge graph gives software handles for real-world things.

## Extracting Text into a Graph

A graph can start from simple extraction. Given a sentence:

```text
Andre Hahn served as a member of the German Bundestag from 2013 to 2017.
```

an extractor can produce a triple:

```text
subject: Andre Hahn
relation: position held
object: member of the German Bundestag
```

The triple becomes edges and nodes:

```text
(Andre Hahn) --[position held]--> (member of the German Bundestag)
```

This is useful because later we can search by entity, relation, or nearby facts. The graph gives the retrieval system structure. It is no longer only matching text chunks; it can follow relationships.

## Adding Time

Many facts are only true during an interval. A temporal knowledge graph annotates facts with dates:

```text
Andre Hahn --position held--> member of the German Bundestag
valid from: 2013-01-01
valid to:   2017-01-01

Jason Shackell --member of sports team--> Burnley F.C.
valid from: 2012-01-01
valid to:   2015-01-01
```

Time turns lookup into reasoning. The question is not only "what relation exists?" It may be "when were two relations true at the same time?"

For the TimelineKGQA question in this artifact, the answer is the intersection of two intervals:

```text
max(2013-01-01, 2012-01-01) = 2013-01-01
min(2017-01-01, 2015-01-01) = 2015-01-01
```

So the overlap is:

```text
(2013-01-01, 2015-01-01)
```

## Tools over the Graph

ReMem exposes graph operations as tools. A language-model controller can call those tools to retrieve evidence.

Examples include:

- semantic retrieval over indexed facts
- entity-context lookup
- gist-context lookup

The important detail is the division of labor. The graph stores structured evidence. The tools retrieve parts of it. The language model decides which tool to call, reads the result, and decides what to do next.

That makes the system neuro-symbolic in a practical sense. The neural part interprets the question and plans. The symbolic part stores entities, relations, and dates in a form that can be inspected and computed over.

## From ReAct Loops to Recursive Loops

A standard ReAct-style agent alternates between reasoning and action:

```text
think -> call tool -> observe result -> think -> call tool -> observe result
```

This is enough for many tasks. But some questions contain independent subproblems. The TimelineKGQA example has two branches:

```text
1. Find Andre Hahn's Bundestag interval.
2. Find Jason Shackell's Burnley F.C. interval.
```

A recursive loop lets a parent model delegate those branches to child models. Each child can run its own observe-act loop over the same graph tools. The parent then gathers the branch results and performs the final aggregation.

That changes the shape of the trace:

```text
parent RLM
  -> child RLM: find interval A
       -> graph tool calls
       -> evidence and interval
  -> child RLM: find interval B
       -> graph tool calls
       -> evidence and interval
  -> parent REPL computes overlap
  -> final answer
```

## What an RLM Does

In DSPy, an RLM is a language model with a Python REPL in its action space. It can write and run small pieces of Python while solving a task.

That matters because not every step should be done in prose. Once the model has two date intervals, Python is the right place to compute their intersection:

```python
hahn = ("2013-01-01", "2017-01-01")
shackell = ("2012-01-01", "2015-01-01")

overlap_start = max(hahn[0], shackell[0])
overlap_end = min(hahn[1], shackell[1])
answer = (overlap_start, overlap_end)
```

Recursive RLMs add another operation: the parent can call sub-LMs. In this demo, the parent uses `llm_query_batched` to ask child RLMs to solve the independent temporal lookups. Those child RLMs can use graph tools themselves.

The result is a small branch-and-gather program:

```text
branch: ask child RLMs to gather evidence
gather: store their evidence in the parent REPL
compute: intersect dates in Python
submit: return the final answer
```

## The TimelineKGQA Trace

The question is:

```text
During which period did Andre Hahn serve as a member of the German Bundestag
while Jason Shackell was a member of Burnley F.C.?
```

The parent RLM creates two branch prompts:

```text
Find the exact period when Andre Hahn served as a member of the German Bundestag.
Find the exact period when Jason Shackell was a member of Burnley F.C.
```

Each branch searches the graph. The parent receives the two intervals:

```text
Andre Hahn:      2013-01-01 to 2017-01-01
Jason Shackell: 2012-01-01 to 2015-01-01
```

The parent computes the overlap in the REPL and submits:

```text
(2013-01-01, 2015-01-01)
```

The packaged trace records:

```text
parent RLM trajectory steps: 3
graph tool interactions: 10
contains llm_query_batched: yes
ExactMatch: 1.0
```

## Why This Example Is Useful

This example is small enough to inspect by hand, but it contains the core pattern:

- extract facts into a graph
- attach temporal qualifiers
- expose graph search as tools
- let a language model choose tool calls
- let recursive child models solve independent branches
- use a REPL for exact computation

That is the value of the artifact. It makes the control flow visible.

## Links

- [Interactive trace visualization](../artifacts/visualizations/timelinekgqa_q25107_recursive_rlm_trace.html)
- [Trace JSON](../artifacts/traces/timelinekgqa_q25107_recursive_rlm_trace.json)
- [Reproduction notes](../../examples/timelinekgqa/README.md)
- [Google's 2012 Knowledge Graph announcement](https://blog.google/products/search/introducing-knowledge-graph-things-not/)
