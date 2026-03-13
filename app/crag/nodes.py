import numpy as np
from langchain_core.output_parsers import StrOutputParser

from app.config import  ABSTENTION_MSG, K_CORRECTIVE, STRIP_SIMILARITY_THRESHOLD,format_source
from app.crag.state import GraphState, CragDocument
from app.crag.models import llm, llm_grader, embeddings, vectorstore, retriever, strip_splitter
from app.crag.prompts import GRADER_SYSTEM_MSG, rewrite_prompt, generate_prompt, requirements_generate_prompt

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
        "total_docs_examined": 0,
        "crag_action" : "pending",
        "previous_queries": [state.question]
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


    for doc in state.documents:
        # 1. EVALUATION (LLM Grader)
        grade = llm_grader.invoke([
            ("system", GRADER_SYSTEM_MSG),
            ("user", f"Question: {state.question}\nDoc Snippet: {doc.page_content}")
        ])
        score = grade.score.lower()
        src = format_source(doc.metadata.get('source', ''))

        # 2. INCORRECT — scarta
        if score == "incorrect":
            print(f"  Incorrect: {src}")
            doc.relevance_score = "incorrect"
            continue  # Passa al prossimo documento

        # 3. CORRECT — accetta as-is
        if score == "correct":
            print(f"  Correct -> Accepted as-is: {src}")
            doc.relevance_score = "correct"
            current_valid_docs.append(doc)
            valid_count += 1
            continue

        # 4. AMBIGUOUS -> Knowledge Refinement
        # Il documento è 'ambiguous', applichiamo il Refinement per estrarre strip precisi.
        print(f"  Ambiguous -> Refining... {src}")

        try:
            refined_text = decompose_then_recompose(doc, state.question)
        except Exception as e:
            print(f"    Error during refinement: {e}")
            continue

        if refined_text:
            print(f"    Refined Success")
            doc.page_content = refined_text
            doc.relevance_score = "refined"
            current_valid_docs.append(doc)
            valid_count += 1
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
    new_total_examined = state.total_docs_examined + total_docs_in_batch
    cumulative_valid = len(new_k_in) + len(new_k_ex)
    batch_confidence = valid_count / total_docs_in_batch if total_docs_in_batch > 0 else 0.0
    confidence = cumulative_valid / new_total_examined if new_total_examined > 0 else 0.0

    upper = state.upper_threshold
    lower = state.lower_threshold

    if confidence >= upper:
        action = "correct"
    elif confidence >= lower:
        action = "ambiguous"
    else:
        action = "incorrect"

    print(f"\n  Batch: {valid_count}/{total_docs_in_batch} validi (batch confidence: {batch_confidence:.2f})")
    print(f"  Cumulative: {cumulative_valid}/{new_total_examined} validi -> confidence {confidence:.2f}")
    print(f"  CRAG Action: {action.upper()} (thresholds: {lower:.2f} / {upper:.2f})")
    print(f"  Accumulated -> k_in: {len(new_k_in)} | k_ex: {len(new_k_ex)}")

    return {
        "k_in": new_k_in,
        "k_ex": new_k_ex,
        "confidence_score": confidence,
        "total_docs_examined": new_total_examined,
        "crag_action": action
    }

def discard_knowledge(state: GraphState):
    """
    Nodo INCORRECT: scarta la knowledge interna (k_in) raccolta finora.

    Quando il retrieval evaluator determina che i documenti recuperati sono
    complessivamente irrilevanti (INCORRECT), mantenere quei documenti
    inquinerebbe la generazione. Si azzerano i k_in e si procede
    solo con il corrective retrieval.
    """
    print("\n   [2b] DISCARD KNOWLEDGE (INCORRECT action)")
    discarded = len(state.k_in)
    print(f"  Discarding {discarded} internal docs (k_in) — retrieval deemed unreliable")

    return {
        "k_in": [],  # Azzera la knowledge interna
        # k_ex viene preservato se presente (da giri correttivi precedenti)
    }


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
    corrective = vectorstore.as_retriever(
        search_type = "mmr",
        search_kwargs = {
            "k": K_CORRECTIVE,
            "fetch_k": K_CORRECTIVE * 3,  # pool da cui MMR sceglie
            "lambda_mult": 0.7  # 0 = max diversità, 1 = max similarità
        }
    )
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
    all_docs = state.k_in + state.k_ex

    # Controllo finale
    # Se il Grader (Evaluator) ha scartato tutto, non si delega all'LLM.
    if not all_docs:
        print("   HARD STOP: Nessuna evidenza valida trovata -> Astensione.")
        return {"generation": ABSTENTION_MSG, "context": ""}

    context_parts = []

    if state.k_in:
        context_parts.append("--- INTERNAL KNOWLEDGE ---")
        for d in state.k_in:
            safe_content = d.page_content.replace("<context>", "").replace("</context>", "")
            context_parts.append(safe_content)

    if state.k_ex:
        context_parts.append("\n--- EXTENDED KNOWLEDGE ---")
        for d in state.k_ex:
            safe_content = d.page_content.replace("<context>", "").replace("</context>", "")
            context_parts.append(safe_content)

    context = "\n\n".join(context_parts)

    if state.template_fields:
        fields_list_str = "\n".join([f"- {f}" for f in state.template_fields])
        chain = requirements_generate_prompt | llm | StrOutputParser()
        response = chain.invoke({
            "template_fields": fields_list_str,
            "context": context
        })
    else:
        chain = generate_prompt | llm | StrOutputParser()
        response = chain.invoke({
            "context": context,
            "question": state.question
        })

    return {"generation": str(response), "context": context}