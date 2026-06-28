# ============================================================
# src/evaluate.py
# Phase 4: Evaluation with RAGAS
# ============================================================
# WHY THIS FILE EXISTS:
# Most student RAG projects skip evaluation entirely — they just
# show it "working" on a demo question. RAGAS gives you objective
# METRICS, which is what separates "I built a chatbot" from
# "I built and validated a retrieval system" in an interview.
#
# Metrics explained:
# - Faithfulness: Does the answer avoid hallucinating facts not
#   in the retrieved context? (catches made-up answers)
# - Answer Relevancy: Does the answer actually address the
#   question asked? (catches off-topic answers)
# - Context Precision: Of the retrieved chunks, how many were
#   actually relevant? (measures retriever quality, not LLM)
# - Context Recall: Did retrieval pull in everything needed to
#   answer correctly? (measures retriever completeness)
# ============================================================

import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from rag_chain import ask

load_dotenv()

# ============================================================
# STEP 1 — BUILD A TEST SET
# WHY: You need ground-truth questions + reference answers to
# measure accuracy. In a real job, this might come from domain
# experts; here, we write a small hand-crafted set based on the
# PDFs we ingested.
#
# >>> EDIT THIS to match the actual content of YOUR PDFs <<<
# ============================================================
test_questions = [
    {
        "question": "What machine learning model achieved the best accuracy in the study?",
        "ground_truth": "Replace this with the actual answer from your paper.",
    },
    {
        "question": "What dataset was used for training?",
        "ground_truth": "Replace this with the actual answer from your paper.",
    },
    # Add 5-10 more for a meaningful evaluation sample size
]


def build_eval_dataset(test_questions):
    """
    WHY: RAGAS needs a specific format — question, the model's
    answer, the retrieved contexts, and the ground truth — all
    aligned per row, packaged as a HuggingFace Dataset.
    """
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in test_questions:
        answer, sources = ask(item["question"])
        questions.append(item["question"])
        answers.append(answer)
        contexts.append([s["snippet"] for s in sources])
        ground_truths.append(item["ground_truth"])

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def run_evaluation():
    print("[1/2] Generating answers for test questions...")
    dataset = build_eval_dataset(test_questions)

    print("[2/2] Running RAGAS metrics (this calls the LLM as a judge)...")
    # WHY ChatGroq AS THE JUDGE: RAGAS needs an LLM to score
    # faithfulness/relevancy. Reusing Groq keeps the project
    # free end-to-end instead of needing an OpenAI key just for eval.
    judge_llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    judge_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    print("\n" + "=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    df = results.to_pandas()
    print(df[["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]])

    print("\nAverage Scores:")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        print(f"  {metric}: {df[metric].mean():.3f}")

    df.to_csv("evaluation_results.csv", index=False)
    print("\n✅ Saved detailed results to evaluation_results.csv")


if __name__ == "__main__":
    run_evaluation()
