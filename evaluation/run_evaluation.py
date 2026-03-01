import pandas as pd
import time
import numpy as np
import contextlib
import os

from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, f1_score
from app.crag.graph import build_crag_graph
from app.config import ABSTENTION_MSG
from evaluation.judge import evaluate_with_llm

load_dotenv()


# ---------------------------------------------------------------------------
# GOLD SET — CRAG Evaluation Dataset
# Repository: transformers/docs/source/en
# ---------------------------------------------------------------------------
test_dataset = [
    # 1. INTEGRITY — Base Facts & API Usage
    {
        "query": "How do I load a pre-trained BERT model using AutoModel?",
        "expected_behavior": "answer",
        "gold_source": "bert.md",
    },
    {
        "query": "Which class should be used for sequence classification with BERT?",
        "expected_behavior": "answer",
        "gold_source": "bert.md",
    },
    {
        "query": "What is the purpose of the attention_mask returned by the tokenizer?",
        "expected_behavior": "answer",
        "gold_source": "tokenizer.md",
    },
    {
        "query": "How do I save a fine-tuned model locally?",
        "expected_behavior": "answer",
        "gold_source": "training.md",
    },

    # 2. REASONING & AMBIGUITY — Similar Concepts
    {
        "query": "What is the difference between BertModel and BertForMaskedLM?",
        "expected_behavior": "answer",
        "gold_source": "bert.md",
    },
    {
        "query": "When should I use AutoModel instead of a task-specific model?",
        "expected_behavior": "answer",
        "gold_source": "auto.md",
    },
    {
        "query": "Why is gradient accumulation useful during training?",
        "expected_behavior": "answer",
        "gold_source": "training.md",
    },
    {
        "query": "How can I reduce GPU memory usage during training without reducing batch size?",
        "expected_behavior": "answer",
        "gold_source": "training.md",
    },

    # 3. SAFETY / HALLUCINATION — System MUST Abstain
    {
        "query": "How do I initialize the GalaxyTransformer model?",
        "expected_behavior": "abstain",
        "gold_source": None,
    },
    {
        "query": "What does the force_gpu_burn flag do in TrainingArguments?",
        "expected_behavior": "abstain",
        "gold_source": None,
    },
    {
        "query": "How do I enable quantum attention in BERT?",
        "expected_behavior": "abstain",
        "gold_source": None,
    },
    {
        "query": "What does the auto_delete_dataset parameter do in Trainer?",
        "expected_behavior": "abstain",
        "gold_source": None,
    },
    {
        "query": "How do I use BertForEntityLinking for named entity disambiguation?",
        "expected_behavior": "abstain",
        "gold_source": None,
    },
    {
        "query": "How does Trainer.auto_shard_model() distribute layers across GPUs?",
        "expected_behavior": "abstain",
        "gold_source": None,
    },
    {
        "query": "How do I configure the neural_cache parameter in GenerationConfig?",
        "expected_behavior": "abstain",
        "gold_source": None,
    },

    # 4. CORRECTIVE / EDGE CASES — Hard Retrieval
    {
        "query": "What deprecation warning is shown for the old Adam optimizer?",
        "expected_behavior": "answer",
        "gold_source": "optimizers.md",
    },
    {
        "query": "What optimizer is recommended instead of the deprecated Adam implementation?",
        "expected_behavior": "answer",
        "gold_source": "optimizers.md",
    },
    {
        "query": "How do I use BitsAndBytesConfig for 4-bit quantization?",
        "expected_behavior": "answer",
        "gold_source": "overview.md",
    },
    {
        "query": "How do I enable mixed precision training with the Trainer API?",
        "expected_behavior": "answer",
        "gold_source": "trainer.md",
    },

    # 5. COMPLETENESS / SYNTHESIS — Multi-Document Answers
    {
        "query": "Summarize the steps required to fine-tune a model using the Trainer API.",
        "expected_behavior": "answer",
        "gold_source": None,
    },
    {
        "query": "Explain the full preprocessing pipeline before model training.",
        "expected_behavior": "answer",
        "gold_source": None,
    },
    {
        "query": "Describe how to load, fine-tune, and save a transformer model.",
        "expected_behavior": "answer",
        "gold_source": None,
    },
]


def calculate_recall_at_k(docs: list, gold_source: str) -> float:
    """Recall@k — il gold source è tra i documenti forniti?"""
    if not gold_source or not docs:
        return 0.0
    target = gold_source.lower()
    for doc in docs:
        path_str  = str(doc.metadata.get("source", "")).lower()
        fname_str = str(doc.metadata.get("file_name", "")).lower()
        if target in path_str or target in fname_str:
            return 1.0
    return 0.0


def compute_decision_metrics(y_expected: list, y_actual: list) -> dict:
    """Metriche di routing separate per classe"""
    answer_indices  = [i for i, e in enumerate(y_expected) if e == 1]
    abstain_indices = [i for i, e in enumerate(y_expected) if e == 0]

    answer_acc = (
        sum(1 for i in answer_indices  if y_actual[i] == 1) / len(answer_indices)
        if answer_indices else 0.0
    )
    abstain_acc = (
        sum(1 for i in abstain_indices if y_actual[i] == 0) / len(abstain_indices)
        if abstain_indices else 0.0
    )

    return {
        "answer_acc": answer_acc,
        "abstain_acc": abstain_acc,
        "balanced_acc": (answer_acc + abstain_acc) / 2,
        "precision": precision_score(y_expected, y_actual, zero_division = 0),
        "recall": recall_score(y_expected, y_actual, zero_division = 0),
        "f1": f1_score(y_expected, y_actual, zero_division = 0),
    }


def run_benchmark():
    print("=" * 55)
    print(" AVVIO VALUTAZIONE CRAG")
    print(f" Test set: {len(test_dataset)} query")
    print("=" * 55)

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        app = build_crag_graph()

    results = []
    y_expected = []
    y_actual = []
    total = len(test_dataset)

    # Nodi da tracciare per il timing
    timing_keys = ["retrieve", "grade_documents", "transform_query",
                   "corrective_retriever", "discard_knowledge", "generate"]

    for i, item in enumerate(test_dataset[:1]):
        q = item["query"]
        behavior_type = item["expected_behavior"]
        gold_src = item.get("gold_source")

        print(f" [{i+1:02d}/{total}] {q[:60]}...")

        # Esecuzione + latenza
        start = time.perf_counter()
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            final_state = app.invoke({"question": q})
        latency = time.perf_counter() - start

        gen_ans = final_state.get("generation", "")
        k_in = final_state.get("k_in", [])
        k_ex = final_state.get("k_ex", [])
        valid_docs = k_in + k_ex
        all_docs = final_state.get("documents", [])
        crag_action = final_state.get("crag_action", "unknown")
        node_timings = final_state.get("node_timings", {})

        # Decisione
        did_abstain   = (ABSTENTION_MSG in gen_ans) or (not valid_docs)
        did_answer    = not did_abstain
        should_answer = (behavior_type == "answer")

        y_expected.append(1 if should_answer else 0)
        y_actual.append(1 if did_answer else 0)

        status = "✅" if (did_answer == should_answer) else "❌"
        print(f"        {status} Expected [{behavior_type}] -> Got [{'answer' if did_answer else 'abstain'}]"
              f" | Action: {crag_action} | {latency:.1f}s")

        # Recall@k pre-grading (retriever puro) e post-grading (dopo grading+refinement)
        rec_pre  = calculate_recall_at_k(all_docs,   gold_src) if gold_src else np.nan
        rec_post = calculate_recall_at_k(valid_docs, gold_src) if gold_src else np.nan

        # LLM Judge — solo se ha risposto
        faith_val = None
        rel_val   = None
        reasoning = None

        if did_answer:
            context_text = "\n\n".join(d.page_content for d in valid_docs)[:6000]
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                scores = evaluate_with_llm(q, context_text, gen_ans)
            faith_val = scores.faithfulness
            rel_val   = scores.answer_relevance
            reasoning = scores.reasoning

        row = {
            "Query": q,
            "Expected": behavior_type,
            "Actual": "answer" if did_answer else "abstain",
            "Correct": did_answer == should_answer,
            "CRAG_Action": crag_action,
            "Latency_Seconds": round(latency, 2),
            "Recall@k_Pre": rec_pre,
            "Recall@k_Post": rec_post,
            "Faithfulness": faith_val,
            "Relevance": rel_val,
            "Judge_Reasoning": reasoning,
            "Answer": gen_ans,
        }

        # Aggiungi colonne timing per ogni nodo
        for key in timing_keys:
            row[f"t_{key}"] = round(node_timings.get(key, 0.0), 2)

        results.append(row)

    print("\nElaborazione completata.")

    # Report finale
    df      = pd.DataFrame(results)
    metrics = compute_decision_metrics(y_expected, y_actual)

    avg_faith    = df["Faithfulness"].mean()
    avg_rel      = df["Relevance"].mean()
    avg_rec_pre  = df["Recall@k_Pre"].mean()
    avg_rec_post = df["Recall@k_Post"].mean()
    avg_latency  = df["Latency_Seconds"].mean()

    print("\n" + "=" * 55)
    print(" REPORT FINALE")
    print("=" * 55)
    print("--- QUALITÀ DELLA DECISIONE (CRAG Routing) ---")
    print(f"  Answer  Accuracy  : {metrics['answer_acc']:.2%}")
    print(f"  Abstain Accuracy  : {metrics['abstain_acc']:.2%}")
    print(f"  Balanced Accuracy : {metrics['balanced_acc']:.2%}")
    print(f"  Precision         : {metrics['precision']:.2f}")
    print(f"  Recall            : {metrics['recall']:.2f}")
    print(f"  F1                : {metrics['f1']:.2f}")
    print("--- QUALITÀ DEL TESTO GENERATO (LLM Judge) ---")
    print(f"  Avg Faithfulness  : {avg_faith:.2f}/5  (anti-allucinazione)")
    print(f"  Avg Relevance     : {avg_rel:.2f}/5  (utilità)")
    print("--- QUALITÀ DEL RETRIEVAL ---")
    print(f"  Recall@k Pre      : {avg_rec_pre:.2f}  (retriever puro)")
    print(f"  Recall@k Post     : {avg_rec_post:.2f}  (dopo grading+refinement)")

    print("--- LATENZA ---")
    print(f"  Avg Totale        : {avg_latency:.2f}s")
    for key in timing_keys:
        col = f"t_{key}"
        if col in df.columns:
            avg_t = df[col].mean()
            if avg_t > 0.01:  # Mostra solo nodi che hanno girato
                pct = (avg_t / avg_latency * 100) if avg_latency > 0 else 0
                print(f"  Avg {key:<22s}: {avg_t:.2f}s  ({pct:.0f}%)")
    print("=" * 55)

    df.to_csv("crag_metrics.csv", index = False)
    print("\nRisultati salvati in: crag_metrics.csv")

if __name__ == "__main__":
    run_benchmark()