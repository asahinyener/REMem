import json
import os
from pathlib import Path

from remem.embedding_model import _get_embedding_client
from remem.llm import CacheOpenAI
from remem.remem import ReMem
from remem.utils.config_utils import BaseConfig


def event_to_sentence(event):
    subject, predicate, obj, start_time, end_time = event.split("|", 4)
    return f"{subject} {predicate} {obj} from {start_time} to {end_time}."


def load_question(dataset_dir, question_id):
    questions = json.load(open(dataset_dir / "unified_kg_cron_questions_all.json"))
    for question in questions:
        if int(question["id"]) == int(question_id):
            return question
    raise ValueError(f"Question id {question_id} not found")


def add_light_distractors(dataset_dir, events, max_distractors=8):
    kg = json.load(open(dataset_dir / "unified_kg_cron.json"))
    event_keys = set(events)
    subjects = {event.split("|", 1)[0] for event in events}
    predicates = {event.split("|", 2)[1] for event in events}
    docs = list(events)
    for row in kg:
        event = "|".join(
            [
                str(row["subject"]),
                str(row["predicate"]),
                str(row["object"]),
                str(row["start_time"]),
                str(row["end_time"]),
            ]
        )
        if event in event_keys:
            continue
        if row["subject"] in subjects or row["predicate"] in predicates:
            docs.append(event)
        if len(docs) >= len(events) + max_distractors:
            break
    return docs


def main():
    dataset_dir = Path(os.environ.get("TIMELINEKGQA_DATASET_DIR", "../external/timelinekgqa/Datasets"))
    question_id = int(os.environ.get("TIMELINEKGQA_QUESTION_ID", "25107"))
    sample = load_question(dataset_dir, question_id)

    question = os.environ.get("TIMELINEKGQA_USE_CANONICAL", "").lower() == "true"
    question = sample["question"] if question else sample.get("paraphrased_question") or sample["question"]
    question = os.environ.get("TIMELINEKGQA_CUSTOM_QUERY", question)
    if os.environ.get("TIMELINEKGQA_RLM_SUBLM_DEMO", "").lower() == "true":
        question = (
            question
            + "\n\nTrace requirement: demonstrate DSPy RLM's built-in sub-LM calls. In the Python REPL, first call "
            + "llm_query_batched with one prompt for each independent temporal branch. Each branch prompt should ask "
            + "that child RLM to use the memory-graph tools itself and return the grounded fact/evidence and interval. "
            + "Store the returned branch evidence strings as Python variables, then use Python to aggregate the branch "
            + "intervals before SUBMIT."
        )
    gold_answer = str(sample["answer"])
    events = sample["events"]
    if isinstance(events, str):
        events = events.split("|")

    event_docs = add_light_distractors(dataset_dir, events)
    docs = [event_to_sentence(event) for event in event_docs]

    llm_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    llm_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    embedding_name = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    llm_label = llm_name.replace("/", "_")
    embedding_label = embedding_name.replace("/", "_")
    working_dir = (
        Path("outputs/timelinekgqa_one_question_temporal_graph")
        / f"cron_q{question_id}_{llm_label}_{embedding_label}"
    )

    print("TimelineKGQA question id:", question_id)
    print("Level:", sample.get("question_level"))
    print("Question type:", sample.get("question_type"))
    print("Answer type:", sample.get("answer_type"))
    print("Temporal relation:", sample.get("temporal_relation"))
    print("Question:", question)
    print("Gold answer:", gold_answer)
    print("Event docs:", len(docs))
    print("Working dir:", working_dir)

    config = BaseConfig(
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        dataset=f"timelinekgqa_cron_q{question_id}",
        embedding_model_name=embedding_name,
        force_index_from_scratch=os.environ.get("FORCE_INDEX", "false").lower() == "true",
        force_openie_from_scratch=os.environ.get("FORCE_OPENIE", "false").lower() == "true",
        retrieval_top_k=60,
        linking_top_k=5,
        qa_top_k=10,
        do_eval_retrieval=True,
        do_eval_qa=True,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=int(os.environ.get("RLM_MAX_TOKENS", "4096")),
        corpus_len=None,
        llm_infer_mode="online",
        preprocess_chunk_func="none",
        extract_method="temporal",
        qa_passage_prefix="Based on the temporal facts: ",
        qa_prompt_template="rag_qa_unified",
        agent_max_steps=int(os.environ.get("AGENT_MAX_STEPS", "8")),
        agent_type="dspy_rlm",
    )
    config.extract_llm_label = llm_label
    config.qa_llm_label = llm_label

    extract_llm = CacheOpenAI(
        "outputs/timelinekgqa_one_question_temporal_graph/extract_llm_cache",
        llm_name=llm_name,
        llm_base_url=llm_base_url,
    )
    qa_llm = CacheOpenAI(
        "outputs/timelinekgqa_one_question_temporal_graph/qa_llm_cache",
        llm_name=llm_name,
        llm_base_url=llm_base_url,
    )

    rag = ReMem(global_config=config, working_dir=str(working_dir), extract_llm=extract_llm, qa_llm=qa_llm)
    rag.set_embedding_model(_get_embedding_client(global_config=config, embedding_model_name=embedding_name))
    rag.index(docs)

    query_solutions, _, _, retrieval_metrics, qa_metrics = rag.rag_for_qa(
        queries=[question],
        gold_docs=[docs],
        gold_answers=[[gold_answer]],
        metrics=("retrieval_recall", "qa_em"),
        question_metadata=[
            {
                "question_id": question_id,
                "question_level": sample.get("question_level"),
                "question_type": sample.get("question_type"),
                "answer_type": sample.get("answer_type"),
                "temporal_relation": sample.get("temporal_relation"),
            }
        ],
    )

    solution = query_solutions[0]
    trace = {
        "question_id": question_id,
        "question": question,
        "gold_answer": gold_answer,
        "events": events,
        "docs": docs,
        "answer": getattr(solution, "answer", ""),
        "retrieval_metrics": retrieval_metrics,
        "qa_metrics": qa_metrics,
        "agent_session_logs": getattr(solution, "agent_session_logs", None),
    }

    output_path = working_dir / f"timelinekgqa_cron_q{question_id}_dspy_rlm_trace.json"
    with open(output_path, "w") as f:
        json.dump(trace, f, indent=2, default=str)

    print("Predicted answer:", trace["answer"])
    print("Retrieval metrics:", retrieval_metrics)
    print("QA metrics:", qa_metrics)
    print("Trace path:", output_path)


if __name__ == "__main__":
    main()
