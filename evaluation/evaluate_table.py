import os
import contextlib
import time
import pandas as pd

from app.config import REQUIREMENTS_TEMPLATE, REPO_NAME
from app.template_parser import parse_template, extract_template_fields
from app.compiler.crag_compiler import process_crag_compliance
from evaluation.judge import evaluate_table


def run_table_benchmark():
    print("=" * 55)
    print(" AVVIO VALUTAZIONE TABELLA")
    print("=" * 55)

    template = parse_template(REQUIREMENTS_TEMPLATE)
    template_fields = extract_template_fields(template)
    print(f"Template: {template.title}")
    print(f"Campi: {template_fields}")

    print("\nGenerazione documento in corso...")
    start = time.perf_counter()

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        try:
            compiled = process_crag_compliance(template, template_fields, repo_name = REPO_NAME)
        except Exception as e:
            print(f"Errore durante la compilazione CRAG: {e}")
            return

    latency = time.perf_counter() - start

    if not compiled.sections:
        print("Nessuna sezione generata.")
        return

    section = compiled.sections[0]

    if section.is_abstention or not section.generated_content:
        print("ABSTENTION: Nessuna tabella generata.")
        return

    generated_table = section.generated_content
    context = section.context or "No context available"

    print(f"  Tabella generata in {latency:.1f}s")
    print(f"  Righe stimate: {generated_table.count(chr(10))}")
    print(f"  Context disponibile: {'Sì' if section.context else 'No (fallback)'}")

    print("\nValutazione LLM Judge in corso...")
    scores = evaluate_table(template_fields, context, generated_table)

    print("\n" + "=" * 55)
    print(" REPORT VALUTAZIONE TABELLA")
    print("=" * 55)
    print(f"  Completeness  : {scores.completeness}/5")
    print(f"  Correctness   : {scores.correctness}/5")
    print(f"  Hallucination : {scores.hallucination}/5  (5=nessuna allucinazione)")
    print(f"  Latency       : {latency:.2f}s")
    print(f"\n  Reasoning: {scores.reasoning}")
    print("=" * 55)

    os.makedirs("evaluation", exist_ok = True)
    df = pd.DataFrame([{
        "Template":         template.title,
        "Completeness":     scores.completeness,
        "Correctness":      scores.correctness,
        "Hallucination":    scores.hallucination,
        "Latency_Seconds":  round(latency, 2),
        "Reasoning":        scores.reasoning,
        "Generated_Table":  generated_table
    }])

    output_path = "evaluation/table_metrics.csv"
    df.to_csv(output_path, index = False)
    print(f"\nRisultati salvati in: {output_path}")


if __name__ == "__main__":
    run_table_benchmark()