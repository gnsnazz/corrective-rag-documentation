import os
import contextlib
import time
import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score
from app.crag.graph import build_crag_graph
from app.config import ABSTENTION_MSG
from evaluation.judge import evaluate_with_llm
from evaluation.datasets import monai_dataset


# Nodi da tracciare per il timing — allineati con i nodi del grafo
TIMING_KEYS = [
    "retrieve", "grade_documents", "transform_query",
    "corrective_retriever", "discard_knowledge", "generate"
]


def calculate_recall_at_k(docs: list, gold_source: str) -> float:
    """
    Recall@k — il gold source è tra i documenti forniti?
    Controlla sia il path completo che il nome file.
    """
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
    """Metriche di routing separate per classe."""
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
        "f1": f1_score(y_expected, y_actual, zero_division = 0)
    }


def run_benchmark():
    print("=" * 55)
    print(" AVVIO VALUTAZIONE CRAG")
    print(f" Test set: {len(monai_dataset)} query")
    print("=" * 55)

    # Build del grafo senza stampare i log di caricamento
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        app = build_crag_graph()

    results = []
    y_expected = []
    y_actual = []
    total = len(monai_dataset)

    for i, item in enumerate(monai_dataset):
        q = item["query"]
        behavior_type = item["expected_behavior"]
        gold_src = item.get("gold_source")

        print(f" [{i+1:02d}/{total}] {q[:60]}...")

        # --- Esecuzione CRAG ---
        start = time.perf_counter()
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            try:
                final_state = app.invoke({"question": q})
            except Exception as e:
                print(f"    Errore CRAG: {e}")
                final_state = {}
        latency = time.perf_counter() - start

        gen_ans = final_state.get("generation", "")
        k_in = final_state.get("k_in", [])
        k_ex = final_state.get("k_ex", [])
        valid_docs = k_in + k_ex
        # final_documents = accumulatore di tutti i doc visti (pre-grading)
        all_docs = final_state.get("final_documents", [])
        crag_action = final_state.get("crag_action", "unknown")
        node_timings = final_state.get("node_timings", {})

        # Decisione
        did_abstain   = (ABSTENTION_MSG in gen_ans) or (not valid_docs)
        did_answer    = not did_abstain
        should_answer = (behavior_type == "answer")

        y_expected.append(1 if should_answer else 0)
        y_actual.append(1 if did_answer   else 0)

        status = "✅" if (did_answer == should_answer) else "❌"
        print(f"        {status} Expected [{behavior_type}] -> " f"Got [{'answer' if did_answer else 'abstain'}]"
              f" | Action: {crag_action} | {latency:.1f}s")

        # Recall@k
        # Pre-grading:  tutti i doc recuperati dal retriever (final_documents)
        # Post-grading: solo i doc validati dal grader (k_in + k_ex)
        rec_pre  = calculate_recall_at_k(all_docs,   gold_src) if gold_src else np.nan
        rec_post = calculate_recall_at_k(valid_docs, gold_src) if gold_src else np.nan

        # --- LLM Judge (solo se ha risposto) ---
        faith_val = None
        rel_val = None
        reasoning = None

        if did_answer:
            context_text = "\n\n".join(d.page_content for d in valid_docs)[:6000]
            scores    = evaluate_with_llm(q, context_text, gen_ans)
            faith_val = scores.faithfulness
            rel_val   = scores.answer_relevance
            reasoning = scores.reasoning

        # --- Costruzione riga risultato ---
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
            "Answer": gen_ans
        }

        for key in TIMING_KEYS:
            row[f"t_{key}"] = round(node_timings.get(key, 0.0), 2)

        results.append(row)

    print("\nElaborazione completata.")

    # Report finale
    df = pd.DataFrame(results)
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
    print(f"  Recall@k Pre      : {avg_rec_pre:.2f}  (retriever puro, su final_documents)")
    print(f"  Recall@k Post     : {avg_rec_post:.2f}  (dopo grading+refinement, su k_in+k_ex)")
    print("--- LATENZA ---")
    print(f"  Avg Totale        : {avg_latency:.2f}s")

    for key in TIMING_KEYS:
        col   = f"t_{key}"
        avg_t = df[col].mean() if col in df.columns else 0.0
        if avg_t > 0.01:
            pct = (avg_t / avg_latency * 100) if avg_latency > 0 else 0
            print(f"  Avg {key:<22s}: {avg_t:.2f}s  ({pct:.0f}%)")
    print("=" * 55)

    os.makedirs("evaluation", exist_ok = True)
    output_path = "evaluation/crag_metrics.csv"
    df.to_csv(output_path, index = False)
    print(f"\nRisultati salvati in: {output_path}")


if __name__ == "__main__":
    run_benchmark()