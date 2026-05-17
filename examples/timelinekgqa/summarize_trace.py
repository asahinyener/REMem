import argparse
import json
from pathlib import Path


def shorten(value, limit=220):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main():
    parser = argparse.ArgumentParser(description="Summarize a TimelineKGQA recursive RLM trace.")
    parser.add_argument("trace_path", type=Path)
    args = parser.parse_args()

    trace = json.loads(args.trace_path.read_text())
    logs = trace.get("agent_session_logs") or {}
    trajectory = logs.get("rlm_trajectory") or []
    interactions = logs.get("llm_interactions") or []

    print(f"question_id: {trace.get('question_id')}")
    print(f"gold_answer: {trace.get('gold_answer')}")
    print(f"predicted_answer: {trace.get('answer')}")
    print(f"qa_metrics: {trace.get('qa_metrics')}")
    print(f"parent_rlm_turns: {len(trajectory)}")
    print(f"graph_tool_interactions: {len(interactions)}")
    print(f"contains_llm_query_batched: {'llm_query_batched' in json.dumps(trajectory)}")

    print("\nParent RLM trajectory")
    for idx, turn in enumerate(trajectory, 1):
        print(f"\n[{idx}] reasoning: {shorten(turn.get('reasoning', ''), 500)}")
        print(f"[{idx}] code: {shorten(turn.get('code', ''), 500)}")
        print(f"[{idx}] output: {shorten(turn.get('output', ''), 500)}")

    print("\nGraph tool calls")
    for idx, item in enumerate(interactions, 1):
        function = item.get("function")
        params = item.get("parameters") or {}
        params = {key: value for key, value in params.items() if key != "_rlm_code"}
        print(f"\n[{idx}] {function}")
        print(f"parameters: {shorten(params, 360)}")
        print(f"observation: {shorten(item.get('observation', ''), 360)}")


if __name__ == "__main__":
    main()
