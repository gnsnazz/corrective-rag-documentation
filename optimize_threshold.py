import os
import contextlib
from itertools import product
from app.crag.graph import build_crag_graph
from app.config import ABSTENTION_MSG
from validation_set import validation_queries


def test_threshold_pair(app, lower, upper):
    """Testa una coppia di soglie (lower, upper) e calcola la balanced accuracy."""

    total_expected_answer = sum(1 for q in validation_queries if q["expected"] == "answer")
    total_expected_abstain = sum(1 for q in validation_queries if q["expected"] == "abstain")

    correct_answers = 0
    correct_abstentions = 0

    print(f"\n Testing: lower={lower:.2f}, upper={upper:.2f}")

    for i, item in enumerate(validation_queries, 1):
        query = item["query"]
        expected = item["expected"]

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            result = app.invoke({
                "question": query,
                "lower_threshold": lower,
                "upper_threshold": upper
            })

        generation = result.get("generation", "")
        did_abstain = ABSTENTION_MSG in generation
        actual = "abstain" if did_abstain else "answer"

        if expected == "answer" and actual == "answer":
            correct_answers += 1
            print(f" [ANSWER]  OK: {query[:40]}...")
        elif expected == "abstain" and actual == "abstain":
            correct_abstentions += 1
            print(f" [ABSTAIN] OK: {query[:40]}...")
        else:
            print(f" [ERRORE]  Atteso {expected}, ottenuto {actual}: {query[:30]}...")

    answer_acc = correct_answers / total_expected_answer if total_expected_answer > 0 else 0
    abstain_acc = correct_abstentions / total_expected_abstain if total_expected_abstain > 0 else 0
    balanced_acc = (answer_acc + abstain_acc) / 2

    print(f"\n Risultati per lower={lower:.2f}, upper={upper:.2f}:")
    print(f"  Answer Acc:   {answer_acc:.1%} ({correct_answers}/{total_expected_answer})")
    print(f"  Abstain Acc:  {abstain_acc:.1%} ({correct_abstentions}/{total_expected_abstain})")
    print(f"  Balanced Acc: {balanced_acc:.1%}")

    return {
        "lower": lower,
        "upper": upper,
        "answer_acc": answer_acc,
        "abstain_acc": abstain_acc,
        "balanced_acc": balanced_acc
    }


def find_best_thresholds():
    print("=" * 65)
    print(" OTTIMIZZAZIONE SOGLIE CRAG (lower, upper)")
    print("=" * 65)

    app = build_crag_graph()

    # Griglia di coppie (lower, upper) dove lower < upper
    lowers = [0.15, 0.25, 0.35]
    uppers = [0.45, 0.55, 0.65, 0.75]

    results = []
    for lo, up in product(lowers, uppers):
        if lo >= up:
            continue  # lower deve essere < upper
        metrics = test_threshold_pair(app, lo, up)
        results.append(metrics)

    best = max(results, key=lambda x: x["balanced_acc"])

    print("\n" + "=" * 65)
    print(" SUMMARY")
    print("=" * 65)
    print(f"{'Lower':<8} | {'Upper':<8} | {'Answer':<10} | {'Abstain':<10} | {'Balanced':<10}")
    print("-" * 65)

    for r in sorted(results, key=lambda x: (x["lower"], x["upper"])):
        marker = " <-- BEST" if (r["lower"] == best["lower"] and r["upper"] == best["upper"]) else ""
        print(f"{r['lower']:<8.2f} | {r['upper']:<8.2f} | {r['answer_acc']:<10.1%} | "
              f"{r['abstain_acc']:<10.1%} | {r['balanced_acc']:<10.1%}{marker}")

    print(f"\n Soglie ottimali: lower={best['lower']:.2f}, upper={best['upper']:.2f}")


if __name__ == "__main__":
    find_best_thresholds()