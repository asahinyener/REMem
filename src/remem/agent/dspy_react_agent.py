from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import dspy

from remem.agent.answer_generator import FinalAnswerGenerator
from remem.agent.logging_utils import AgentLogger
from remem.agent.node_utils import NodeUtils
from remem.agent.strategies.tool_strategy import RetrievalResult
from remem.agent.tools import ToolContext, ToolType
from remem.agent.tools.find_entity_contexts_tool import FindEntityContextsTool
from remem.agent.tools.find_gist_contexts_tool import FindGistContextsTool
from remem.agent.tools.semantic_retrieve_tool import SemanticRetrieveTool
from remem.utils.config_utils import resolve_llm_api_key


class DSPyReactSignature(dspy.Signature):
    """Use the supplied tools to gather memory graph evidence before finishing."""

    query: str = dspy.InputField()
    question_date: str = dspy.InputField(desc="Question date if available, otherwise empty string.")
    answer: str = dspy.OutputField(
        desc="Final short answer only. Prefer the exact answer span, not a full sentence."
    )


@dataclass
class DSPyReactState:
    query: str
    max_steps: int
    question_metadata: Optional[Dict[str, Any]] = None
    current_step: int = 0
    visited_nodes: set = None
    retrieved_passages: List[Dict[str, Any]] = None
    reasoning_chain: List[str] = None
    final_answer: str = ""
    final_answer_generated: bool = False

    def __post_init__(self):
        if self.visited_nodes is None:
            self.visited_nodes = set()
        if self.retrieved_passages is None:
            self.retrieved_passages = []
        if self.reasoning_chain is None:
            self.reasoning_chain = []


class DSPyReactGraphAgent:
    """DSPy ReAct controller that runs on top of the existing ReMem graph tools."""

    def __init__(self, llm_model, remem_instance, node_chunks_dict: Dict[str, str], logger=None):
        self.llm_client = llm_model
        self.remem = remem_instance
        self.node_chunks_dict = node_chunks_dict
        self.logger = logger

        from remem.agent import AgentConfig

        self.agent_config = AgentConfig(
            fixed_tools=False,
            max_steps=getattr(remem_instance.global_config, "agent_max_steps", 3),
            fixed_retrieval_tool=getattr(remem_instance.global_config, "agent_fixed_retrieval_tool", "semantic_retrieve"),
        )
        self.session_logs = {"num_steps": 0, "final_results": {}, "llm_interactions": []}
        self.agent_logger = AgentLogger(self.logger, self.session_logs)
        self.node_utils = NodeUtils()
        self.answer_generator = FinalAnswerGenerator(self.llm_client, self.logger)

        self.semantic_retrieve_tool = SemanticRetrieveTool(remem_instance)
        self.find_gist_contexts_tool = FindGistContextsTool(remem_instance)
        self.find_entity_contexts_tool = FindEntityContextsTool(remem_instance)

        self._state: Optional[DSPyReactState] = None
        self._beam_size = 10

    def _build_tools(self) -> List[dspy.Tool]:
        tool_cls = getattr(dspy, "Tool", None)
        if tool_cls is None:
            return [
                self.semantic_retrieve,
                self.find_gist_contexts,
                self.find_entity_contexts,
            ]

        return [
            tool_cls(
                self.semantic_retrieve,
                name="semantic_retrieve",
                desc=(
                    "Default first-step memory search. Use this to semantically search the memory graph for relevant "
                    "gists and facts from a natural-language question. Start here to discover candidate entities, "
                    "relations, or a bridge time, then switch to structured lookups when those fields are known."
                ),
                arg_desc={
                    "query": "Natural-language search query for the memory you want to find.",
                    "start_time": "Optional lower time bound if the question refers to a date or period.",
                    "end_time": "Optional upper time bound if the question refers to a date or period.",
                    "start_operator": "Comparison operator for start_time, usually 'ge'.",
                    "end_operator": "Comparison operator for end_time, usually 'le'.",
                },
            ),
            tool_cls(
                self.find_gist_contexts,
                name="find_gist_contexts",
                desc=(
                    "Follow-up lookup from a known gist node id to nearby supporting context. Use only after you "
                    "already have a gist id from prior retrieval."
                ),
                arg_desc={
                    "gist_id": "Numeric gist node id returned or referenced from prior retrieval context.",
                    "start_time": "Optional lower time bound.",
                    "end_time": "Optional upper time bound.",
                    "start_operator": "Comparison operator for start_time, usually 'ge'.",
                    "end_operator": "Comparison operator for end_time, usually 'le'.",
                },
            ),
            tool_cls(
                self.find_entity_contexts,
                name="find_entity_contexts",
                desc=(
                    "Structured fact lookup over subject, object, predicate, and optional time filters. Use this only "
                    "when the question is best expressed as an explicit relation query. For decomposable temporal "
                    "questions, use it after deriving a bridge time or relation to gather the final target evidence."
                ),
                arg_desc={
                    "subject": "Entity or subject string to match exactly or closely.",
                    "object": "Object string to match when looking for facts.",
                    "predicate": "Relation or predicate to match.",
                    "start_time": "Optional lower time bound.",
                    "end_time": "Optional upper time bound.",
                    "start_operator": "Comparison operator for start_time. Use exactly one of 'ge', 'gt', 'le', 'lt', 'eq'.",
                    "end_operator": "Comparison operator for end_time. Use exactly one of 'ge', 'gt', 'le', 'lt', 'eq'.",
                    "limit": "Optional maximum number of facts to return.",
                    "ordering": "Optional sort mode if needed.",
                    "offset": "Optional pagination offset.",
                    "aggregation": "Optional aggregation mode.",
                },
            ),
        ]

    def retrieve_with_agent(
        self,
        query: str,
        max_steps: Optional[int] = None,
        return_chunk: Optional[str] = None,
        beam_size: int = 10,
        gold_answer: Optional[str] = None,
        question_metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        max_steps = max_steps or self.agent_config.max_steps
        self._beam_size = beam_size
        self._state = DSPyReactState(query=query, max_steps=max_steps, question_metadata=question_metadata)
        self.agent_logger.log_session_start(query, max_steps, gold_answer)

        lm = self._build_dspy_lm()
        previous_lm = getattr(dspy.settings, "lm", None)
        try:
            dspy.settings.configure(lm=lm)
            react = dspy.ReAct(
                signature=DSPyReactSignature,
                tools=self._build_tools(),
                max_iters=max_steps,
            )
            prediction = react(
                query=query,
                question_date=(question_metadata or {}).get("date", "") if question_metadata else "",
                max_iters=max_steps,
            )
        finally:
            dspy.settings.configure(lm=previous_lm)

        self._hydrate_reasoning_from_trajectory(getattr(prediction, "trajectory", {}))
        prediction_answer = getattr(prediction, "answer", "") or ""
        if prediction_answer.strip():
            self._state.final_answer = prediction_answer.strip()
            self._state.final_answer_generated = True

        if not self._state.final_answer_generated:
            self._state.final_answer = self.answer_generator.generate_answer(
                query=self._state.query,
                visited_nodes=self._state.visited_nodes,
                reasoning_step=self._state.current_step,
                previous_steps=self.agent_logger.session_logs.get("llm_interactions", []),
                question_metadata=self._state.question_metadata,
                interaction_type="final_answer_generation",
                remem_instance=self.remem,
            )
            self._state.final_answer_generated = True
            self._state.reasoning_chain.append(f"Tool: Final answer generated: {self._state.final_answer}")

        result = self._format_result(return_chunk)
        return result.chunk_ids, result.chunk_scores, result.logs

    def _build_dspy_lm(self):
        llm_name = self.remem.global_config.llm_name
        llm_base_url = self.remem.global_config.llm_base_url
        api_key = resolve_llm_api_key(self.remem.global_config)
        if api_key is None and (llm_base_url is None or "localhost" in llm_base_url):
            api_key = "sk-"
        return dspy.LM(
            model=f"openai/{llm_name}",
            api_base=llm_base_url,
            api_key=api_key,
            temperature=self.remem.global_config.temperature,
            max_tokens=self.remem.global_config.max_new_tokens or 1024,
            cache=False,
        )

    def _build_context(self) -> ToolContext:
        return ToolContext(
            query=self._state.query,
            visited_nodes=self._state.visited_nodes,
            reasoning_step=self._state.current_step,
            max_steps=self._state.max_steps,
            available_entries=self.node_utils.get_available_entries(self.remem, self.node_chunks_dict),
            content_map=self.node_utils.get_content_map_for_nodes(self._state.visited_nodes, self.node_chunks_dict),
            previous_steps=self.agent_logger.session_logs.get("llm_interactions", []),
            question_metadata=self._state.question_metadata,
        )

    def semantic_retrieve(
        self,
        query: str,
        start_time: str = "",
        end_time: str = "",
        start_operator: str = "ge",
        end_operator: str = "le",
    ) -> str:
        """Default first-step search over the memory graph using a natural-language query."""
        return self._execute_tool(
            self.semantic_retrieve_tool,
            {
                "query": query,
                "start_time": start_time,
                "end_time": end_time,
                "start_operator": start_operator,
                "end_operator": end_operator,
            },
        )

    def find_gist_contexts(
        self,
        gist_id: int,
        start_time: str = "",
        end_time: str = "",
        start_operator: str = "ge",
        end_operator: str = "le",
    ) -> str:
        """Use only after prior retrieval when you already have a gist node id and need nearby context."""
        return self._execute_tool(
            self.find_gist_contexts_tool,
            {
                "gist_id": gist_id,
                "start_time": start_time,
                "end_time": end_time,
                "start_operator": start_operator,
                "end_operator": end_operator,
            },
        )

    def find_entity_contexts(
        self,
        subject: str = "",
        object: str = "",
        predicate: str = "",
        start_time: str = "",
        end_time: str = "",
        start_operator: str = "ge",
        end_operator: str = "le",
        limit: int = 0,
        ordering: str = "",
        offset: int = 0,
        aggregation: str = "",
    ) -> str:
        """Structured fact lookup for explicit subject/object/predicate queries, not the default first search."""
        return self._execute_tool(
            self.find_entity_contexts_tool,
            {
                "subject": subject,
                "object": object,
                "predicate": predicate,
                "start_time": start_time,
                "end_time": end_time,
                "start_operator": start_operator,
                "end_operator": end_operator,
                "limit": limit or None,
                "ordering": ordering,
                "offset": offset,
                "aggregation": aggregation,
            },
        )

    def _execute_tool(self, tool, parameters: Dict[str, Any]) -> str:
        context = self._build_context()
        clean_parameters = {k: v for k, v in parameters.items() if v not in (None, "", [])}
        tool_result = tool.execute(context, self._beam_size, **clean_parameters)
        tool_result.parameters = clean_parameters

        if tool_result.tool_type != ToolType.OUTPUT_ANSWER:
            self._state.visited_nodes.update(tool_result.nodes_found)
            if getattr(tool_result, "node_contents", None):
                new_passages = self._create_passages_from_tool_result(tool_result, self._state.current_step + 1)
            else:
                new_passages = self.node_utils.collect_node_content_with_scores(
                    tool_result.nodes_found,
                    tool_result.scores,
                    self._state.current_step + 1,
                    self.node_chunks_dict,
                    self.remem,
                )
            self._state.retrieved_passages.extend(new_passages)
            self.agent_logger.log_step(self._state.current_step + 1, tool_result, new_passages, context)
        else:
            self.agent_logger.log_step(self._state.current_step + 1, tool_result, [], context)

        self._state.current_step += 1
        self._state.reasoning_chain.append(f"Tool: {tool_result.observation}")
        return self._format_observation_for_dspy(tool_result)

    def _format_observation_for_dspy(self, tool_result) -> str:
        lines = [tool_result.observation]
        if getattr(tool_result, "node_contents", None):
            for idx, content in enumerate(tool_result.node_contents[:10], 1):
                display = content[:1000] + "..." if len(content) > 1000 else content
                lines.append(f"{idx}. {display}")
        return "\n".join(lines)

    def _create_passages_from_tool_result(self, tool_result, step: int):
        passages = []
        for node_key, score, content in zip(tool_result.nodes_found, tool_result.scores, tool_result.node_contents):
            node_type = self.node_utils.get_node_type(node_key, self.remem)
            passages.append(
                {
                    "node_name": node_key,
                    "content": content,
                    "embedding_score": score,
                    "step": step,
                    "node_type": node_type,
                }
            )
        return passages

    def _hydrate_reasoning_from_trajectory(self, trajectory: Dict[str, Any]) -> None:
        interactions = self.agent_logger.session_logs.get("llm_interactions", [])
        step_idx = 0
        while f"thought_{step_idx}" in trajectory:
            if step_idx < len(interactions):
                interactions[step_idx]["reasoning"] = trajectory.get(f"thought_{step_idx}", "")
                interactions[step_idx]["parameters"] = trajectory.get(f"tool_args_{step_idx}", {}) or {}
            step_idx += 1

    def _format_result(self, return_chunk: Optional[str]) -> RetrievalResult:
        chunk_ids, chunk_scores = self._format_agent_results(return_chunk)
        logs = {
            "agent_observation": self._state.reasoning_chain,
            "agent_answer": self._state.final_answer,
            "agent_chunks": [p["content"] for p in self._state.retrieved_passages if p.get("content")],
            "agent_session_logs": self.agent_logger._get_session_logs(),
        }
        return RetrievalResult(
            chunk_ids=chunk_ids,
            chunk_scores=chunk_scores,
            logs=logs,
            final_answer=self._state.final_answer,
        )

    def _format_agent_results(self, return_chunk: Optional[str] = None):
        if not self._state.retrieved_passages:
            return [], []

        chunk_id_to_max_score = {}
        for passage in self._state.retrieved_passages:
            node_name = passage.get("node_name")
            if not node_name:
                continue

            node_type = self.node_utils.get_node_type(node_name, self.remem)
            if return_chunk == "verbatim" and node_type != "verbatim":
                continue
            elif return_chunk == "gists" and node_type != "gists":
                continue

            embedding_score = passage.get("embedding_score", 0.0)
            chunk_id_to_max_score[node_name] = max(chunk_id_to_max_score.get(node_name, 0.0), embedding_score)

        if not chunk_id_to_max_score:
            return [], []

        sorted_items = sorted(chunk_id_to_max_score.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items], [item[1] for item in sorted_items]
