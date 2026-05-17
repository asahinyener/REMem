from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

import dspy

from remem.agent.dspy_react_agent import DSPyReactGraphAgent, DSPyReactState


class DSPyRLMSignature(dspy.Signature):
    """Use the supplied memory-graph tools to answer with grounded evidence.

    For temporal or multi-hop questions, decompose before answering:
    1. Identify the bridge fact, entity, relation, ordinal, or time constraint asked for by the first clause.
    2. Run a separate targeted lookup for the second clause using the derived constraint.
    3. Answer only after the trace contains evidence for both the bridge constraint and the final target fact.

    Prefer structured graph lookups over broad semantic retrieval once entity or relation names are known. For temporal
    filters, use operator names exactly as 'ge', 'gt', 'le', 'lt', or 'eq'.

    The REPL includes DSPy's built-in sub-LM calls: llm_query(prompt) and llm_query_batched(prompts). For decomposable
    questions, use these calls to run independent branches before the final answer. When recursive RLM is available,
    each branch can use the same memory-graph tools in its own REPL and return grounded evidence to the parent. Do not
    skip sub-LM calls on overlap, union, intersection, comparison, before/after, count, min/max, or duration questions.
    A good pattern is:
    1. Call llm_query_batched with one prompt per branch asking each branch to use graph tools and return evidence.
    2. Store the branch return strings in Python variables.
    3. Parse exact dates, entities, or intervals from those variables.

    For questions that ask for overlap, union, intersection, comparison, count, min/max, before/after, or duration:
    gather each branch separately through sub-LM calls, store the returned evidence in Python variables, then write
    explicit Python code to aggregate the branches before answering.
    """

    query: str = dspy.InputField()
    question_date: str = dspy.InputField(desc="Question date if available, otherwise empty string.")
    answer: str = dspy.OutputField(
        desc="Final short answer only. Prefer the exact answer span, not a full sentence."
    )


class DSPyRLMGraphAgent(DSPyReactGraphAgent):
    """DSPy RLM controller that runs on top of the existing ReMem graph tools."""

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
            rlm_kwargs = {
                "signature": DSPyRLMSignature,
                "tools": self._build_tools(),
                "max_iterations": max_steps,
                "max_llm_calls": max(max_steps + 1, 4),
                "sub_lm": lm,
                "interpreter": dspy.PythonInterpreter(),
                "verbose": False,
            }
            if "max_depth" in inspect.signature(dspy.RLM.__init__).parameters:
                rlm_kwargs["max_depth"] = 2
            rlm = dspy.RLM(**rlm_kwargs)
            prediction = rlm(
                query=query,
                question_date=(question_metadata or {}).get("date", "") if question_metadata else "",
            )
        finally:
            dspy.settings.configure(lm=previous_lm)

        trajectory = getattr(prediction, "trajectory", [])
        self.session_logs["rlm_trajectory"] = trajectory
        self._hydrate_reasoning_from_trajectory(trajectory)

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

    def _hydrate_reasoning_from_trajectory(self, trajectory: Any) -> None:
        interactions = self.agent_logger.session_logs.get("llm_interactions", [])
        if not isinstance(trajectory, list):
            return

        for step_idx, step in enumerate(trajectory):
            if step_idx >= len(interactions) or not isinstance(step, dict):
                continue

            reasoning = step.get("reasoning", "")
            if reasoning:
                interactions[step_idx]["reasoning"] = reasoning

            code = step.get("code", "")
            if code:
                parameters = interactions[step_idx].get("parameters")
                if not isinstance(parameters, dict):
                    parameters = {}
                parameters["_rlm_code"] = code
                interactions[step_idx]["parameters"] = parameters
