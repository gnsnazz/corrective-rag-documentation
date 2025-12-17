import os
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal

from app.embeddings import get_embedding_model
from app.config import DB_DIR
from app.crag.state import GraphState, CragDocument

from app.crag.prompts import (
    GRADER_SYSTEM_MSG,
    refine_prompt,
    rewrite_prompt,
    generate_prompt
)

# Modello locale
llm = ChatAnthropic(
    model_name = "claude-haiku-4-5-20251001", #claude-sonnet-4-5-20250929
    temperature = 0,
    timeout = None,
    stop = None,
    max_retries = 2
)

# (dopo) usare sonnet-4-5 per la generazione

class Grade(BaseModel):
    """Score for relevance check."""
    score: Literal["correct", "ambiguous", "incorrect"] = Field(
        description = """Relevance classification: 'correct' (explicit answer), 'ambiguous' (needs refinement),
         or 'incorrect' (irrelevant)."""
    )

# Grader strutturato
llm_grader = llm.with_structured_output(Grade)

# Vector Store
if not os.path.exists(DB_DIR):
    raise FileNotFoundError(f"DB non trovato in {DB_DIR}")

embeddings = get_embedding_model()
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Parametri CRAG
CONFIDENCE_THRESHOLD = 0.5  # Se < 50% dei docs sono validi -> Corrective Search
MAX_RETRIES = 1             # Numero di cicli correttivi
K_CORRECTIVE = 10           # Documenti giro correttivo

# --- NODI ---
def retrieve(state: GraphState):
    print("\n   [1] BASE RETRIEVER")
    # Recupero iniziale
    raw_docs = retriever.invoke(state.question)

    # Wrapping in CragDocument con metadata
    crag_docs = [
        CragDocument(
            page_content = d.page_content,
            metadata = d.metadata,
            relevance_score = "unknown",
            retrieval_source = "base"
        )
        for d in raw_docs
    ]

    return {
        "documents": crag_docs,
        "k_in": [],     # Reset liste
        "k_ex": [],
        "confidence_score": 0.0,
        "previous_queries": [state.question]  # Inizializza memoria
    }


def grade_documents(state: GraphState):
    """
    Implementazione Corrective-RAG:
    Classifica in Correct, Ambiguous, Incorrect.
    Knowledge Refinement applicato a tutti i documenti rilevanti (Correct & Ambiguous).
    Knowledge Searching applicato ai documenti irrilevanti (Ambiguous & Incorrect).
    Nessuna deduplicazione per fonte: accetta chunk multipli dallo stesso file.
    """
    print("\n   [2] EVIDENCE SCORER")

    current_valid_docs = []
    total_docs_in_batch = len(state.documents)
    valid_count = 0  # Conta sia 'correct' che 'refined'

    refine_chain = refine_prompt | llm | StrOutputParser()

    for doc in state.documents:
        # 1. EVALUATION (LLM Grader)
        grade = llm_grader.invoke([
            ("system", GRADER_SYSTEM_MSG),
            ("user", f"Question: {state.question}\nDoc Snippet: {doc.page_content}")
        ])
        score = grade.score.lower()

        # 2. LOGICA
        if score == "incorrect":
            print(f"  Incorrect: {doc.metadata.get('source')}")
            doc.relevance_score = "incorrect"
            continue  # Passa al prossimo documento

        # 3. KNOWLEDGE REFINEMENT
        # Il documento è 'correct' o 'ambiguous', applichiamo il Refinement per estrarre strip precisi.
        print(f"  {score.capitalize()} -> Refining... {doc.metadata.get('source')}")

        try:
            refined_text = refine_chain.invoke({
                "question": state.question,
                "document": doc.page_content
            })
        except Exception as e:
            print(f"    Error during refinement: {e}")
            continue

        # 4. VALIDAZIONE POST-REFINEMENT
        # Accettiamo il documento solo se il Refiner ha estratto contenuto utile
        if "IRRELEVANT" not in refined_text and len(refined_text) > 20:
            print(f"    Refined Success")

            # Sovrascrive il contenuto grezzo con quello pulito (Strip)
            doc.page_content = refined_text
            doc.relevance_score = "refined" # Contrassegnato come refined

            current_valid_docs.append(doc)
            valid_count += 1                # Conta per la confidence
        else:
            print(f"    Refinement Failed (Irrelevant)")

    # Carica lo stato attuale
    new_k_in = list(state.k_in)
    new_k_ex = list(state.k_ex)

    for d in current_valid_docs:
        source_type = getattr(d, "retrieval_source", None)
        if source_type == "base":
            # Internal Knowledge
            new_k_in.append(d)
        elif source_type == "corrective":
            # External/Extended Knowledge
            new_k_ex.append(d)

    # Calcola confidenza sul batch corrente
    confidence = valid_count / total_docs_in_batch if total_docs_in_batch > 0 else 0.0
    print(f"  Batch Confidence: {confidence:.2f} (Threshold: {CONFIDENCE_THRESHOLD})")
    print(f"  Accumulated -> k_in: {len(new_k_in)} | k_ex: {len(new_k_ex)}")

    return {"k_in": new_k_in, "k_ex": new_k_ex, "confidence_score": confidence}

def transform_query(state: GraphState):
    print("\n   [3] QUERY REWRITE")

    # Verifica anti-loop: conosce le query precedenti
    history_str = "\n- ".join(state.previous_queries)

    chain = rewrite_prompt | llm | StrOutputParser()
    new_query = chain.invoke({"question": state.question, "history": history_str})

    print(f"  Rewritten: {new_query}")

    return {
        "question": new_query,
        "retry_count": state.retry_count + 1,
        "previous_queries": state.previous_queries + [new_query]  # Aggiorna memoria
    }

def corrective_retriever(state: GraphState):
    print("\n   [4] CORRECTIVE RETRIEVER")

    # k leggermente più alto per cercare più a fondo
    corrective = vectorstore.as_retriever(search_kwargs={"k": K_CORRECTIVE})

    raw_docs = corrective.invoke(state.question)

    crag_docs = [
        CragDocument(
            page_content = d.page_content,
            metadata = d.metadata,
            relevance_score = "unknown",
            retrieval_source = "corrective"  # Traccia che viene dal fix
        )
        for d in raw_docs
    ]

    print(f"  Retrieved {len(crag_docs)} new docs.")
    for d in crag_docs:
        print(f"  New Doc: {d.metadata.get('source', 'unknown')}")  # test

    # Lo scorer analizzerà SOLO quelli nuovi
    return {"documents": crag_docs}


def generate(state: GraphState):
    """"
    Genera la risposta finale (documento)
    """
    print("\n   [5] ANSWER GENERATOR")

    # Unione delle conoscenze per il generatore
    k_in_docs = state.k_in
    k_ex_docs = state.k_ex
    all_docs = k_in_docs + k_ex_docs

    # Controllo finale
    if not all_docs:
        return {"generation": "NESSUNA_DOC: Fallimento completo del retrieval (Base + Corrective)."}

    context_parts = []

    if k_in_docs:
        context_parts.append("--- INTERNAL KNOWLEDGE ---")
        context_parts.extend([d.page_content for d in k_in_docs])

    if k_ex_docs:
        context_parts.append("\n--- EXTENDED KNOWLEDGE ---")
        context_parts.extend([d.page_content for d in k_ex_docs])

    context = "\n\n".join(context_parts)

    chain = generate_prompt | llm | StrOutputParser()   # (dopo) usare modello più potente (sonnet-4-5)
    response = chain.invoke({
        "context": context,
        "question": state.question,
        "len_docs": len(all_docs)
    })

    return {"generation": str(response)}