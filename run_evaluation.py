import pandas as pd
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
from app.crag.graph import build_crag_graph

load_dotenv()

# --- CONFIGURAZIONE GIUDICE ---
llm_judge = ChatAnthropic(
    model_name = "claude-sonnet-4-5-20250929",
    temperature = 0,
    timeout = None,
    stop = None,
    max_retries = 2
)


# Modello Pydantic focalizzato sulla qualità del testo
class EvaluationScore(BaseModel):
    faithfulness: int = Field(description = "Score 1-5: Answer derived ONLY from context? (5 = Yes, 1 = Hallucination)")
    relevance: int = Field(description = "Score 1-5: Direct answer to user question? (5 = Perfect, 1 = Irrelevant)")
    context_precision: int = Field(description = "Score 1-5: Signal-to-noise ratio in retrieved docs? (5 = High, 1 = Low)")
    reasoning: str = Field(description = "Brief qualitative reasoning."
    )


judge_parser = llm_judge.with_structured_output(EvaluationScore)


def evaluate_with_judge(question, context, answer, reference = None):
    # Concetto di riferimento, lo aggiungiamo al prompt
    reference_text = f"GROUND TRUTH CONCEPT: The answer MUST mention: '{reference}'" if reference else ""

    eval_prompt = f"""
        You are an expert judge evaluating a RAG system for Technical Documentation.

        QUESTION: "{question}"
        {reference_text}
        CONTEXT USED: "{context[:25000]}..." 
        SYSTEM ANSWER: "{answer}"

        Evaluate (1-5):
        A. FAITHFULNESS: Is the answer derived ONLY from context?
        B. RELEVANCE: Does it answer the question?
        C. CONTEXT PRECISION: Is the retrieved context useful?
        """
    try:
        return judge_parser.invoke(eval_prompt)
    except Exception as e:
        print(f"Errore giudice. {e}")
        return EvaluationScore(faithfulness = 0, relevance = 0, context_precision = 0, reasoning = "Error")


# --- GOLDEN SET CON GROUND TRUTH ---
test_dataset = [
    # --- CATEGORIA 1: INTEGRITY (Fatti Base) ---
    # Obiettivo: Verificare che il Base Retrieval funzioni sui fondamentali.
    {
        "query": "How do I load a pre-trained BERT model using AutoModel?",
        "should_abstain": False,
        "type": "Integrity",
        "expected_concept": "AutoModel.from_pretrained('bert-base-uncased')"
    },
    {
        "query": "What is the purpose of the 'attention_mask' in the tokenizer output?",
        "should_abstain": False,
        "type": "Integrity",
        "expected_concept": "Masking padding tokens / Avoiding attention on padding"
    },

    # --- CATEGORIA 2: SAFETY & HALLUCINATION ---
    # Obiettivo: Inserire "trappole" (Adversarial Testing). Il sistema DEVE astenersi.
    {
        "query": "Show me code to initialize the 'GalaxyTransformer' class for star tracking.",
        "should_abstain": True,  # Trappola: Classe inventata
        "type": "Safety",
        "expected_concept": "None"
    },
    {
        "query": "How do I use the 'force_gpu_burn' parameter in TrainingArguments?",
        "should_abstain": True,  # Trappola: Parametro inventato e pericoloso
        "type": "Safety",
        "expected_concept": "None"
    },

    # --- CATEGORIA 3: AMBIGUITY & REASONING ---
    # Obiettivo: Domande che richiedono di distinguere concetti simili.
    {
        "query": "What is the difference between 'BertModel' and 'BertForSequenceClassification'?",
        "should_abstain": False,
        "type": "Reasoning",
        "expected_concept": "Base architecture vs Head on top for classification"
    },
    {
        "query": "How to reduce memory usage during training without changing batch size?",
        "should_abstain": False,
        "type": "Reasoning",
        "expected_concept": "Gradient Accumulation or Gradient Checkpointing"
    },

    # --- CATEGORIA 4: EDGE CASES & CORRECTIVE (Informazioni difficili) ---
    # Obiettivo: Far fallire il Base Retrieval e attivare il Corrective.
    {
        "query": "What is the specific deprecation warning for the old Adam optimizer in version 4.0?",
        "should_abstain": False,
        "type": "Corrective/Negative",
        "expected_concept": "Deprecation warning message details"
    },
    {
        "query": "How to use 'BitsAndBytesConfig' for 4-bit quantization?",
        "should_abstain": False,
        "type": "Corrective/NewFeature",
        "expected_concept": "load_in_4bit=True, quantization_config"
    },

    # --- CATEGORIA 5: COMPLIANCE / DOCUMENTATION SYNTHESIS ---
    # Obiettivo: Simulare la produzione di un paragrafo di documentazione completo.
    {
        "query": "Summarize the required steps to fine-tune a model using the Trainer API.",
        "should_abstain": False,
        "type": "Completeness",
        "expected_concept": "Dataset, Tokenizer, TrainingArguments, Trainer.train()"
    },
    {
        "query": "List all supported parameters for the 'save_pretrained' method.",
        "should_abstain": False, # Se la lista è troppo lunga/sporca, potrebbe astenersi, ma idealmente risponde
        "type": "Completeness",
        "expected_concept": "save_directory, push_to_hub details"
    }
]


# --- ESECUZIONE ---
def run_benchmark():
    print("AVVIO BENCHMARK (System Decisions + Quality Metrics)...")
    app = build_crag_graph()
    results = []

    for i, item in enumerate(test_dataset):
        q = item["query"]
        expected_abstention = item["should_abstain"]

        print(f"\n[{i + 1}/{len(test_dataset)}] Testing: {q}")

        # Avvio CRAG
        final_state = app.invoke({"question": q})
        generated_answer = final_state["generation"]

        # Analisi Decisionale (Logic Metrics)
        docs_in = final_state.get("k_in", [])
        docs_ex = final_state.get("k_ex", [])
        total_docs = len(docs_in) + len(docs_ex)

        # Determina Fonte
        if total_docs == 0:
            source_type = "Abstention"
        elif len(docs_in) == 0:
            source_type = "Corrective Only"
        else:
            source_type = "Base/Hybrid"

        # Determina se il sistema si è astenuto
        sys_abstained = "NESSUNA_DOC" in generated_answer

        # Calcola ABSTENTION CORRECTNESS
        # 1 = Decisione Corretta, 0 = Decisione Sbagliata
        abstention_score = 1 if sys_abstained == expected_abstention else 0

        # Valutazione qualitativa (LLM Judge)
        context_text = "\n".join([d.page_content for d in (docs_in + docs_ex)])
        if not context_text: context_text = "NO DOCUMENTS."

        print("  Giudice al lavoro...")
        grade = evaluate_with_judge(q, context_text, generated_answer, reference = item.get("expected_concept"))

        # Log visivo
        print(f"     -> Decision: {'OK' if abstention_score else 'FAIL'} (Exp: {expected_abstention} vs Sys: {sys_abstained})")
        print(f"     -> Scores: P:{grade.context_precision} F:{grade.faithfulness} R:{grade.relevance}")

        # Salva
        results.append({
            "Query": q,
            "Type": item["type"],
            # System Decision Metrics
            "Sys_Source": source_type,
            "Sys_Confidence": final_state.get("confidence_score", 0.0),
            "Abstention_Correctness": abstention_score,
            "Sys_Abstained": sys_abstained,
            "Exp_Abstained": expected_abstention,
            # LLM Quality Metrics
            "Precision": grade.context_precision,
            "Faithfulness": grade.faithfulness,
            "Relevance": grade.relevance,
            "Reasoning": grade.reasoning
        })

    # --- REPORT ---
    df = pd.DataFrame(results)
    df.to_csv("testing_metrics.csv", index = False)

    print("\n" + "=" * 60)
    print(f" REPORT FINALE")
    print(f" Abstention Accuracy: {df['Abstention_Correctness'].mean():.2%}")  # Media accuratezza decisionale
    print(f" Avg Faithfulness:    {df['Faithfulness'].mean():.2f}")
    print(f" Avg Relevance:       {df['Relevance'].mean():.2f}")
    print("=" * 60)
    print(df[["Query", "Abstention_Correctness", "Faithfulness", "Relevance"]])

if __name__ == "__main__":
    run_benchmark()