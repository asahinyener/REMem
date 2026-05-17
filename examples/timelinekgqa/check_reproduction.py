import inspect
import json
import os
import shutil
from pathlib import Path

import dspy


def main():
    repo_root = Path(__file__).resolve().parents[2]
    workspace_root = repo_root.parent
    dataset_dir = Path(
        os.environ.get(
            "TIMELINEKGQA_DATASET_DIR",
            workspace_root / "external/timelinekgqa/Datasets",
        )
    )
    trace_path = repo_root / "docs/artifacts/traces/timelinekgqa_q25107_recursive_rlm_trace.json"
    viz_path = repo_root / "docs/artifacts/visualizations/timelinekgqa_q25107_recursive_rlm_trace.html"

    checks = {
        "timelinekgqa_kg": dataset_dir / "unified_kg_cron.json",
        "timelinekgqa_questions": dataset_dir / "unified_kg_cron_questions_all.json",
        "reference_trace": trace_path,
        "trace_visualization": viz_path,
    }

    ok = True
    for name, path in checks.items():
        exists = path.exists()
        ok = ok and exists
        print(f"{name}: {'ok' if exists else 'missing'} ({path})")

    has_max_depth = "max_depth" in inspect.signature(dspy.RLM.__init__).parameters
    ok = ok and has_max_depth
    print(f"dspy_rlm_file: {inspect.getsourcefile(dspy.RLM)}")
    print(f"dspy_rlm_recursive_max_depth: {'ok' if has_max_depth else 'missing'}")

    deno_path = shutil.which("deno")
    print(f"deno: {'ok' if deno_path else 'missing'} ({deno_path or 'not on PATH'})")

    llm_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    llm_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    has_llm_key = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY"))
    has_embedding_key = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("EMBEDDING_API_KEY"))
    print(f"llm_base_url: {llm_base_url}")
    print(f"llm_model: {llm_model}")
    print(f"embedding_model: {embedding_model}")
    print(f"llm_credential: {'ok' if has_llm_key else 'missing'}")
    print(f"embedding_credential: {'ok' if has_embedding_key else 'missing'}")

    if trace_path.exists():
        trace = json.loads(trace_path.read_text())
        logs = trace.get("agent_session_logs") or {}
        trajectory = logs.get("rlm_trajectory") or []
        has_batched = "llm_query_batched" in json.dumps(trajectory)
        ok = ok and has_batched
        print(f"reference_question_id: {trace.get('question_id')}")
        print(f"reference_answer: {trace.get('answer')}")
        print(f"reference_exact_match: {trace.get('qa_metrics', {}).get('ExactMatch')}")
        print(f"reference_parent_steps: {len(trajectory)}")
        print(f"reference_has_llm_query_batched: {'ok' if has_batched else 'missing'}")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
