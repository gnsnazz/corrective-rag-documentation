import pandas as pd
import time
import numpy as np
import contextlib
import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, precision_score, recall_score
from app.crag.graph import build_crag_graph
from app.config import ABSTENTION_MSG

load_dotenv()

class LLMScore(BaseModel):
    faithfulness: int = Field(description = "Score 1-5: Is the answer grounded in the context? (1 = Hallucination, 5 = Fully Supported)")
    answer_relevance: int = Field(description = "Score 1-5: Does the answer address the user query? (1 = Irrelevant, 5 = Perfect)")
    reasoning: str = Field(description = "Short Reasoning")


llm_judge = ChatOllama(
    model = "llama3.1",
    temperature = 0,
    format = "json"
)
parser = JsonOutputParser(pydantic_object = LLMScore)

# GOLD SET – CRAG EVALUATION DATASET
# Repository: transformers/docs/source/en

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
        "query": "Which parameter enables automatic dataset cleaning in Trainer?",
        "expected_behavior": "abstain",
        "gold_source": None,
    },

    # 4. CORRECTIVE / EDGE CASES — Hard Retrieval
    {
        "query": "What deprecation warning is shown for the old Adam optimizer?",
        "expected_behavior": "answer",
        "gold_source": "optimization.md",
    },
    {
        "query": "What optimizer is recommended instead of the deprecated Adam implementation?",
        "expected_behavior": "answer",
        "gold_source": "optimization.md",
    },
    {
        "query": "How do I use BitsAndBytesConfig for 4-bit quantization?",
        "expected_behavior": "answer",
        "gold_source": "overview.md",
    },
    {
        "query": "How do I enable mixed precision training with the Trainer API?",
        "expected_behavior": "answer",
        "gold_source": "training_args.md",
    },

    # 5. COMPLETENESS / SYNTHESIS — Multi-Document Answers
    # (Recall@k not applicable → gold_source = None)
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

def evaluate_with_llm(query, context, answer):
    prompt = PromptTemplate(
        template = """You are an evaluator. Rate the RAG system output.

            QUERY: {query}
            RETRIEVED CONTEXT: {context}
            SYSTEM ANSWER: {answer}

            Provide scores (1-5) for:
            1. Faithfulness: Is the answer derived ONLY from the context?
            2. Answer Relevance: Is the answer useful for the query?

            {format_instructions}""",
        input_variables = ["query", "context", "answer"],
        partial_variables = {"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | llm_judge | parser
    try:
        res = chain.invoke({"query": query, "context": context, "answer": answer})
        return LLMScore(**res)
    except Exception as e:
        print(f"\n Errore Judge: {e}")
        return LLMScore(faithfulness = 1, answer_relevance = 1, reasoning = "Error")


# Funzione helper per Recall@k
def calculate_recall_at_k(retrieved_docs, gold_source):
    if not gold_source or not retrieved_docs: return 0.0
    for doc in retrieved_docs:
        path_str = str(doc.metadata.get("source", "")).lower()
        fname_str = str(doc.metadata.get("file_name", "")).lower()
        target = gold_source.lower()

        if target in path_str or target in fname_str:
            return 1.0
    return 0.0

# BENCHMARK LOOP
def run_benchmark():
    print("AVVIO VALUTAZIONE CRAG...")
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        app = build_crag_graph()

    results = []

    # Liste per metriche di decisione (Globali)
    y_expected = []
    y_actual = []

    total = len(test_dataset)

    for i, item in enumerate(test_dataset[:1]):
        q = item["query"]
        behavior_type = item["expected_behavior"]  # "answer" o "abstain"
        gold_src = item.get("gold_source")

        print(f" Elaborazione {i + 1}/{total}: {q}")

        # --- ESECUZIONE ---
        # misurazione latenza ed esecuzione
        start_time = time.perf_counter()  # Start timer

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            final_state = app.invoke({"question": q})

        end_time = time.perf_counter()  # Stop timer
        latency = end_time - start_time  # Secondi totali

        gen_ans = final_state["generation"]
        retrieved_docs = final_state.get("documents", [])

        # --- STATO SISTEMA ---
        did_abstain = ABSTENTION_MSG in gen_ans or not retrieved_docs
        did_answer = not did_abstain
        should_answer = (behavior_type == "answer")

        # Liste decisione (1 = Answer, 0 = Abstain)
        y_expected.append(1 if should_answer else 0)
        y_actual.append(1 if did_answer else 0)

        # --- CALCOLO METRICHE (Condizionale) ---
        # 1. Recall@k (Retrieval puro) -> gold source
        rec_at_k = calculate_recall_at_k(retrieved_docs, gold_src) if gold_src else np.nan

        # 2. Inizializzazione metriche di Generazione
        faith_val = None
        rel_val = None

        # Calcolo della qualità solo se il sistema ha effettivamente risposto
        if did_answer:
            # LLM Metrics
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                context_text = "\n".join([d.page_content[:2000] for d in retrieved_docs])
                llm_scores = evaluate_with_llm(q, context_text, gen_ans)
            faith_val = llm_scores.faithfulness
            rel_val = llm_scores.answer_relevance

        results.append({
            "Query": q,
            "Expected": behavior_type,
            "Actual": "abstain" if did_abstain else "answer",
            "Latency_Seconds": latency,
            "Recall@k": rec_at_k,
            "Faithfulness": faith_val,
            "Relevance": rel_val,
            "Answer": gen_ans
        })

    print(" " * 80, end="\r")
    print("Elaborazione Completata.")

    # REPORT FINALE
    df = pd.DataFrame(results)

    # Metriche di Decisione (Classificazione)
    acc = accuracy_score(y_expected, y_actual)
    prec = precision_score(y_expected, y_actual, zero_division = 0)
    rec_dec = recall_score(y_expected, y_actual, zero_division = 0)

    # Metriche di Generazione (Media escludendo i NaN)
    avg_faith = df["Faithfulness"].mean()
    avg_rel = df["Relevance"].mean()

    # Metriche di Retrieval
    avg_rec_at_k = df["Recall@k"].mean()

    # Latency
    avg_latency = df["Latency_Seconds"].mean()  # Media tempo

    print("\n" + "=" * 50)
    print(" REPORT ")
    print("=" * 50)
    print("--- 1. QUALITÀ DELLA DECISIONE (CRAG Logic) ---")
    print(f"Avg Latency:         {avg_latency:.2f}")
    print(f"Abstention Accuracy: {acc:.2%}")
    print(f"Answering Precision: {prec:.2f}")
    print(f"Answering Recall:    {rec_dec:.2f}")
    print("-" * 50)
    print("--- 2. QUALITÀ DEL TESTO (Solo sulle Risposte) ---")
    print(f"Avg Faithfulness:    {avg_faith:.2f}/5 (Safety)")
    print(f"Avg Relevance:       {avg_rel:.2f}/5 (Utility)")
    print("-" * 50)
    print("--- 3. QUALITÀ DEL RETRIEVAL ---")
    print(f"Avg Recall@k:        {avg_rec_at_k:.2f}")
    print("=" * 50)

    df.to_csv("crag_metrics.csv", index = False)

if __name__ == "__main__":
    run_benchmark()