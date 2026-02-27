import os
import contextlib
from app.crag.graph import build_crag_graph
from app.config import ABSTENTION_MSG
from validation_set import validation_queries


def test_threshold(app, threshold_value):
    """ Testa un singolo threshold e calcola metriche separate per categoria."""

    # Contatori globali
    total_expected_answer = sum(1 for q in validation_queries if q["expected"] == "answer")
    total_expected_abstain = sum(1 for q in validation_queries if q["expected"] == "abstain")

    correct_answers = 0
    correct_abstentions = 0

    print(f"\nTest in corso per Threshold = {threshold_value:.2f}")

    for i, item in enumerate(validation_queries, 1):
        query = item["query"]
        expected = item["expected"]

        # Esecuzione silenziosa
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            result = app.invoke({
                "question": query,
                "confidence_threshold": threshold_value
            })

        generation = result.get("generation", "")
        did_abstain = ABSTENTION_MSG in generation
        actual = "abstain" if did_abstain else "answer"

        # Logica di conteggio
        if expected == "answer" and actual == "answer":
            correct_answers += 1
            print(f" [ANSWER]  Corretta: {query[:40]}...")
        elif expected == "abstain" and actual == "abstain":
            correct_abstentions += 1
            print(f" [ABSTAIN] Corretta: {query[:40]}...")
        else:
            print(f" [ERRORE]  Atteso {expected}, ottenuto {actual}: {query[:30]}...")

    # Calcolo Metriche Separate
    answer_acc = correct_answers / total_expected_answer if total_expected_answer > 0 else 0
    abstain_acc = correct_abstentions / total_expected_abstain if total_expected_abstain > 0 else 0

    # Metrica per l'ottimizzazione
    balanced_acc = (answer_acc + abstain_acc) / 2

    print(f"\n Risultati per Threshold {threshold_value:.2f}:")
    print(f"  - Answer Accuracy:  {answer_acc:.1%} ({correct_answers}/{total_expected_answer})")
    print(f"  - Abstain Accuracy: {abstain_acc:.1%} ({correct_abstentions}/{total_expected_abstain})")
    print(f"  -> BALANCED ACCURACY: {balanced_acc:.1%}")

    return {
        "threshold": threshold_value,
        "answer_acc": answer_acc,
        "abstain_acc": abstain_acc,
        "balanced_acc": balanced_acc
    }


def find_best_threshold():
    print("=" * 65)
    print(" AVVIO OTTIMIZZAZIONE THRESHOLD (Balanced Accuracy)")
    print("=" * 65)

    app = build_crag_graph()
    thresholds_to_test = [0.2, 0.4, 0.5, 0.6, 0.8]
    results = []

    for t in thresholds_to_test:
        metrics = test_threshold(app, t)
        results.append(metrics)

    # Sceglie il migliore basato sulla BALANCED ACCURACY
    best_result = max(results, key=lambda x: x["balanced_acc"])

    print("\n" + "=" * 65)
    print(" SUMMARY FINALE")
    print("=" * 65)
    print(f"{'Threshold':<12} | {'Answer Acc.':<15} | {'Abstain Acc.':<15} | {'Balanced Acc.':<15}")
    print("-" * 65)

    for res in sorted(results, key=lambda x: x["threshold"]):
        t = res["threshold"]
        ans = f"{res['answer_acc']:.1%}"
        abs_acc = f"{res['abstain_acc']:.1%}"
        bal = f"{res['balanced_acc']:.1%}"

        marker = "  <-- BEST!" if t == best_result["threshold"] else ""
        print(f"{t:<12.2f} | {ans:<15} | {abs_acc:<15} | {bal:<15}{marker}")

    print("\n CONCLUSIONE:")
    print(f"Threshold ottimale: {best_result['threshold']:.2f}")


if __name__ == "__main__":
    find_best_threshold()