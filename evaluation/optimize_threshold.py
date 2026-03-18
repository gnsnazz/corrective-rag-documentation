import os
import contextlib
from itertools import product

from app.crag.graph import build_crag_graph
from app.config import ABSTENTION_MSG
from evaluation.datasets import monai_validation

# Griglia di soglie da testare
LOWER_VALUES = [0.15, 0.25, 0.35]
UPPER_VALUES = [0.45, 0.55, 0.65, 0.75]


def test_threshold_pair(app, lower: float, upper: float) -> dict:
    """
    Testa una coppia (lower, upper) sul validation set e restituisce le metriche.
    Le soglie vengono passate come input iniziale al grafo — GraphState le accetta
    come campi Pydantic e sovrascrivono i default di config.
    """
    total_answer  = sum(1 for q in monai_validation if q["expected"] == "answer")
    total_abstain = sum(1 for q in monai_validation if q["expected"] == "abstain")

    correct_answers     = 0
    correct_abstentions = 0

    print(f"\n  Testing: lower={lower:.2f}, upper={upper:.2f}")

    for item in monai_validation:
        query    = item["query"]
        expected = item["expected"]

        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            try:
                result = app.invoke({
                    "question": query,
                    "lower_threshold": lower,
                    "upper_threshold": upper
                })
            except Exception as e:
                print(f"    Errore su '{query[:30]}...': {e}")
                continue

        generation  = result.get("generation", "")
        did_abstain = ABSTENTION_MSG in generation
        actual      = "abstain" if did_abstain else "answer"

        if expected == "answer" and actual == "answer":
            correct_answers += 1
            print(f"    [ANSWER]  OK: {query[:45]}...")
        elif expected == "abstain" and actual == "abstain":
            correct_abstentions += 1
            print(f"    [ABSTAIN] OK: {query[:45]}...")
        else:
            print(f"    [ERRORE]  Atteso {expected}, ottenuto {actual}: {query[:35]}...")

    answer_acc = correct_answers / total_answer  if total_answer  > 0 else 0.0
    abstain_acc = correct_abstentions / total_abstain if total_abstain > 0 else 0.0
    balanced = (answer_acc + abstain_acc) / 2

    print(f"   Answer Acc: {answer_acc:.1%} ({correct_answers}/{total_answer})"
          f" | Abstain Acc: {abstain_acc:.1%} ({correct_abstentions}/{total_abstain})"
          f" | Balanced: {balanced:.1%}")

    return {
        "lower": lower,
        "upper": upper,
        "answer_acc": answer_acc,
        "abstain_acc": abstain_acc,
        "balanced_acc": balanced
    }


def find_best_thresholds():
    print("=" * 65)
    print(" OTTIMIZZAZIONE SOGLIE CRAG (lower, upper)")
    print(f" Validation set: {len(monai_validation)} query")
    print("=" * 65)

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        app = build_crag_graph()

    results = []
    pairs   = [(lo, up) for lo, up in product(LOWER_VALUES, UPPER_VALUES) if lo < up]
    total   = len(pairs)

    for idx, (lo, up) in enumerate(pairs, 1):
        print(f"\n[{idx}/{total}]", end = "")
        metrics = test_threshold_pair(app, lo, up)
        results.append(metrics)

    best = max(results, key = lambda x: x["balanced_acc"])

    print("\n" + "=" * 65)
    print(" SUMMARY")
    print("=" * 65)
    print(f"{'Lower':<8} | {'Upper':<8} | {'Answer':<10} | {'Abstain':<10} | {'Balanced':<10}")
    print("-" * 65)

    for r in sorted(results, key=lambda x: (x["lower"], x["upper"])):
        marker = " <-- BEST" if (r["lower"] == best["lower"] and r["upper"] == best["upper"]) else ""
        print(
            f"{r['lower']:<8.2f} | {r['upper']:<8.2f} | "
            f"{r['answer_acc']:<10.1%} | {r['abstain_acc']:<10.1%} | "
            f"{r['balanced_acc']:<10.1%}{marker}"
        )

    print(f"\n  Soglie ottimali: lower={best['lower']:.2f}, upper={best['upper']:.2f}")
    print(f"  Balanced Accuracy: {best['balanced_acc']:.1%}")
    print("=" * 65)


if __name__ == "__main__":
    find_best_thresholds()