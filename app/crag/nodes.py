import os
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from app.embeddings import get_embedding_model
from app.config import DB_DIR, ABSTENTION_MSG, K_CORRECTIVE, STRIP_SIMILARITY_THRESHOLD, format_source
from app.crag.state import GraphState, CragDocument

from app.crag.prompts import (
    GRADER_SYSTEM_MSG,
    refine_prompt,
    rewrite_prompt,
    generate_prompt
)

# Modello locale
llm = ChatAnthropic(
    model_name = "claude-sonnet-4-5-20250929", #claude-haiku-4-5-20251001
    temperature = 0,
    timeout = None,
    stop = None,
    max_retries = 2
)


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
vectorstore = Chroma(persist_directory = DB_DIR, embedding_function = embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

strip_splitter = RecursiveCharacterTextSplitter(
    chunk_size      = 200,
    chunk_overlap   = 20,
    length_function = len,
)

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
        "final_documents": crag_docs,
        "k_in": [],     # Reset liste
        "k_ex": [],
        "confidence_score": 0.0,
        "previous_queries": [state.question]  # Inizializza memoria
    }


def decompose_then_recompose(doc: CragDocument, question: str) -> str | None:
    """
    Knowledge Refinement:
    1. DECOMPOSE — split algoritmico in strip
    2. FILTER — cosine similarity tra query e strip
    3. RECOMPOSE — riconcatena gli strip rilevanti in ordine originale
    """

    # --- 1. DECOMPOSE ---
    strips = strip_splitter.split_text(doc.page_content)
    if not strips:
        return None

    # Strip troppo corti non portano informazione
    strips = [s for s in strips if len(s.strip()) > 30]
    if not strips:
        return None

    # --- 2. FILTER (similarità semantica) ---
    query_emb  = np.array(embeddings.embed_query(question))
    strip_embs = np.array(embeddings.embed_documents(strips))

    # Cosine similarity (i vettori sono già normalizzati con normalize_embeddings=True)
    similarities = strip_embs @ query_emb

    relevant_strips = [
        strip for strip, sim in zip(strips, similarities)
        if sim >= STRIP_SIMILARITY_THRESHOLD
    ]

    print(f"    Strips: {len(strips)} totali -> {len(relevant_strips)} rilevanti "
          f"(soglia: {STRIP_SIMILARITY_THRESHOLD})")

    if not relevant_strips:
        return None

    # --- 3. RECOMPOSE ---
    return "\n".join(relevant_strips)


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
        src = format_source(doc.metadata.get('source', ''))

        # 2. LOGICA
        if score == "incorrect":
            print(f"  Incorrect: {src}")
            doc.relevance_score = "incorrect"
            continue  # Passa al prossimo documento

        # 3. KNOWLEDGE REFINEMENT
        # Il documento è 'correct' o 'ambiguous', applichiamo il Refinement per estrarre strip precisi.
        print(f"  {score.capitalize()} -> Refining... {src}")

        try:
            refined_text = decompose_then_recompose(doc, state.question)
        except Exception as e:
            print(f"    Error during refinement: {e}")
            continue

        # 4. VALIDAZIONE POST-REFINEMENT
        # Accettiamo il documento solo se il Refiner ha estratto contenuto utile
        if refined_text:
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

    # Calcola confidenza cumulativa
    confidence = valid_count / total_docs_in_batch if total_docs_in_batch > 0 else 0.0
    current_threshold = getattr(state, "confidence_threshold", 0.5)

    print(f"  Batch valid: {valid_count}/{total_docs_in_batch}")
    print(f"  Batch confidence {confidence:.2f} (Current Threshold: {current_threshold:.2f})")
    print(f"  Accumulated total -> k_in: {len(new_k_in)} | k_ex: {len(new_k_ex)}")

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
    corrective = vectorstore.as_retriever(search_kwargs = {"k": K_CORRECTIVE})
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

    all_docs = state.final_documents + crag_docs

    # Lo scorer analizzerà solo quelli nuovi, vengono accumulati tutti gli altri
    return {"documents": crag_docs, "final_documents": all_docs}


def generate(state: GraphState):
    """"
    Genera la risposta finale (documento)
    """
    print("\n   [5] ANSWER GENERATOR")

    # Unione delle conoscenze per il generatore
    k_in_docs = getattr(state, "k_in", []) or []
    k_ex_docs = getattr(state, "k_ex", []) or []
    all_docs = k_in_docs + k_ex_docs

    # Controllo finale
    # Se il Grader (Evaluator) ha scartato tutto, non si delega all'LLM.
    if not all_docs:
        print("   HARD STOP: Nessuna evidenza valida trovata -> Astensione.")
        return {"generation": ABSTENTION_MSG}

    context_parts = []

    if k_in_docs:
        context_parts.append("--- INTERNAL KNOWLEDGE ---")
        for d in k_in_docs:
            safe_content = d.page_content.replace("<context>", "").replace("</context>", "")
            context_parts.append(safe_content)

    if k_ex_docs:
        context_parts.append("\n--- EXTENDED KNOWLEDGE ---")
        for d in k_ex_docs:
            safe_content = d.page_content.replace("<context>", "").replace("</context>", "")
            context_parts.append(safe_content)

    context = "\n\n".join(context_parts)

    chain = generate_prompt | llm | StrOutputParser()
    response = chain.invoke({
        "context": context,
        "question": state.question
    })

    return {"generation": str(response)}