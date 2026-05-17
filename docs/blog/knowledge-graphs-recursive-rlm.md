# From Knowledge Graphs to Recursive RLMs

This post explains the small TimelineKGQA artifact in this repository. It starts with knowledge graphs, adds time, and ends with a recursive language-model trace that answers one temporal question by searching a graph and computing an interval overlap.

The point is pedagogical. The trace is not a new benchmark result. It is a compact example of how symbolic graph structure and language-model reasoning can work together.

## Abstract

Information retrieval from sources that change over time require structuring  and saving certain attributes of a fact across time. [[TEQUILA](#ref-tequila); [EXAQT](#ref-exaqt); [RTQA](#ref-rtqa); [Prog-TQA](#ref-progtqa); [MultiTQ/MultiQA](#ref-multitq)] Human endevours and science belong to this category of sources, where the state of the art is always on the move and the application of any fact is time sensitive.

The ontology of time sensitive facts and the method to save them are wide and varied however[[Allen 1983](#ref-allen); [Event Calculus](#ref-eventcalculus); [TEILP](#ref-teilp); [TILP](#ref-tilp); [Prog-TQA](#ref-progtqa)] a small interface common to all of the approaches can be used for analysis. This implementation and demonstration uses a knowledge graph based method to extract named entities and predicates from the raw data giving a processed triplet structure. Nodes in the graph are named entities and the predicate is encoded into the edges of the graph representation.

A certain set of questions over this time graph  that explicitly require comparisons of non intersecting triplet entries are part of the set of the questions that require multi-hop reasoning. # While classical approaches of symbolic operations on string based time format is enough to produce answers when supplied the key, finding the key is the main focus of here    # Decomposing the queries into its components 
and using paralel approaches of information fetching keep increasing the depth of search without significant cost to attention based context[[RTQA](#ref-rtqa)] and yield higher accuracy at lower cost. Tool calling with python functions is a strong
fit for modern day LLMs due to abundance and variance of such data in the pretraining corpus[[TReMu](#ref-tremu)], #further integration of auto regressive generation and programming enviroments to produce harnesses for code execution with tool calling LLMs#,
modern day harnesses enable REPL like stateful interaction for code execution environments[dspy; [RLM](#ref-rlm)]. Work of [[RLM](#ref-rlm)] treats each turn to be composed of emitted code to be executed against the REPL, and supplies the REPL state and the produced output back to the LLM. Function that is of interest is calling sub-llms with their own scoped REPL environment. This enables the model to write the execution and retrieval stages as executed python code and have stateful returns of context from the sub-llm calls.

RLM harness provides the ability to delegate the execution and aggregation to symbolic processing , and the knowledge graph enables these processes to statefuly interact with the raw knowledge through functions with semantic queries and graph operations. The following demo is based on the REMEM architecture, with dspy enabled RLM harness and Deno based python execution through the bundled docker image. It explores the intermediate stages and representation of the algorithm and aims to introduce these concepts of information retrieval to a wider audience.

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

## References

<a id="ref-allen"></a>Allen, J. F. (1983). Maintaining knowledge about temporal intervals. *Communications of the ACM, 26*(11), 832-843. https://doi.org/10.1145/182.358434

<a id="ref-eventcalculus"></a>Kowalski, R., & Sergot, M. (1986). A logic-based calculus of events. *New Generation Computing, 4*(1), 67-95. https://doi.org/10.1007/BF03037383

<a id="ref-tequila"></a>Jia, Z., Abujabal, A., Saha Roy, R., Stroetgen, J., & Weikum, G. (2018). TEQUILA: Temporal Question Answering over Knowledge Bases. In *Proceedings of the 27th ACM International Conference on Information and Knowledge Management (CIKM '18)*. https://arxiv.org/abs/1908.03650

<a id="ref-exaqt"></a>Jia, Z., Pramanik, S., Saha Roy, R., & Weikum, G. (2021). Complex temporal question answering on knowledge graphs. In *Proceedings of the 30th ACM International Conference on Information & Knowledge Management (CIKM '21)*. https://doi.org/10.1145/3459637.3482416

<a id="ref-multitq"></a>Chen, Z., Liao, J., & Zhao, X. (2023). Multi-granularity temporal question answering over knowledge graphs. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 11378-11392. https://aclanthology.org/2023.acl-long.637/

<a id="ref-tilp"></a>Xiong, S., Yang, Y., Fekri, F., & Kerce, J. C. (2023). TILP: Differentiable learning of temporal logical rules on knowledge graphs. In *The Eleventh International Conference on Learning Representations (ICLR 2023)*. https://openreview.net/forum?id=_X12NmQKvX

<a id="ref-teilp"></a>Xiong, S., Yang, Y., Payani, A., Kerce, J. C., & Fekri, F. (2024). TEILP: Time prediction over knowledge graphs via logical reasoning. In *Proceedings of the AAAI Conference on Artificial Intelligence, 38*(14), 16112-16120. https://doi.org/10.1609/aaai.v38i14.29544

<a id="ref-progtqa"></a>Chen, Z., Zhang, Z., Li, Z., Wang, F., Zeng, Y., Jin, X., & Xu, Y. (2024). Self-improvement programming for temporal knowledge graph question answering. In *Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)*. https://arxiv.org/abs/2404.01720

<a id="ref-tremu"></a>Ge, Y., Romeo, S., Cai, J., Shu, R., Benajiba, Y., Sunkara, M., & Zhang, Y. (2025). TReMu: Towards neuro-symbolic temporal reasoning for LLM-agents with memory in multi-session dialogues. In *Findings of the Association for Computational Linguistics: ACL 2025*, 18974-18988. https://aclanthology.org/2025.findings-acl.972/

<a id="ref-rtqa"></a>Gong, Z., Li, J., Liu, Z., Liang, L., Chen, H., & Zhang, W. (2025). RTQA: Recursive thinking for complex temporal knowledge graph question answering with large language models. In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)*. https://aclanthology.org/2025.emnlp-main.499/

<a id="ref-rlm"></a>Zhang, A. L., Kraska, T., & Khattab, O. (2025). Recursive Language Models. *arXiv preprint arXiv:2512.24601*. https://arxiv.org/abs/2512.24601
