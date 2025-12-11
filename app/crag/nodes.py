import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.embeddings import get_embedding_model
from app.config import OLLAMA_MODEL, DB_DIR
from app.crag.state import GraphState

# Modello locale
local_llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0)

# Vector Store
if not os.path.exists(DB_DIR):
    raise FileNotFoundError(f"DB non trovato in {DB_DIR}")

embeddings = get_embedding_model()
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- NODI ---

def retrieve(state: GraphState):
    print("\n--- 1. RETRIEVE ---")
    question = state["question"]
    documents = retriever.invoke(question)
    print(f"   Recuperati {len(documents)} documenti.")
    return {"documents": documents, "question": question}


def grade_documents(state: GraphState):
    """
    Implementazione Corrective-RAG:
    Classifica in Correct, Ambiguous, Incorrect.
    Esegue Knowledge Refinement sui documenti ambigui.
    """
    print("--- 2. EVALUATOR & REFINEMENT ---")
    question = state["question"]
    documents = state["documents"]

    # prompt valutazione
    eval_prompt = PromptTemplate(
        template="""You are a hyper-strict Evaluator.
        
            Your Goal: Determine if the document contains TECHNICAL information to answer the question.
        
            Rules for classification:
            1. 'correct': ONLY if the document explicitly defines the specific concept in the question.
            2. 'incorrect': If the document is unrelated.
            3. 'ambiguous': If the document mentions the keywords (e.g. in code or lists) but does NOT provide a full textual explanation.
        
            MOST IMPORTANT: If you are unsure, choose 'ambiguous'.
        
            Document: {document}
            Question: {question}
        
            Return ONLY one word: 'correct', 'incorrect', or 'ambiguous'.""",
        input_variables=["question", "document"],
    )

    # prompt raffinamento - caso ambiguous
    refine_prompt = PromptTemplate(
        template="""You are performing Knowledge Refinement for a RAG system.
        The following document was retrieved for the question but is ambiguous.
        
        Task: Extract ONLY the sentences from the document that directly answer the question.
        
        STRICT RULES:
        1. Do NOT add any information from your own knowledge.
        2. Use ONLY the text provided below.
        3. If the document does not contain the specific answer, return EXACTLY the word "IRRELEVANT".
        
        Document: {document}
        Question: {question}
        
        Refined Content:""",
        input_variables=["question", "document"]
    )

    eval_chain = eval_prompt | local_llm | StrOutputParser()
    refine_chain = refine_prompt | local_llm | StrOutputParser()

    filtered_docs = []
    for d in documents:
        try:
            score = eval_chain.invoke({"question": question, "document": d.page_content})
            score = score.strip().lower()

            if "correct" in score:
                print(f"   CORRECT: {d.metadata.get('source', 'unknown')}")
                filtered_docs.append(d)
            elif "ambiguous" in score:
                print(f"  AMBIGUOUS: {d.metadata.get('source', 'unknown')} -> Avvio Refinement...")

                # B. Knowledge Refinement
                refined_content = refine_chain.invoke({"question": question, "document": d.page_content})

                if "IRRELEVANT" in refined_content:
                    print(f"   Refinement fallito: Info non trovata nel testo. Scartato.")
                else:
                    # Aggiorna il contenuto del documento con la versione pulita
                    d.page_content = refined_content
                    # Aggiunge un tag metadata per tracciare che è stato modificato
                    d.metadata["is_refined"] = True
                    print(f"    Refined (Knowledge Strip done)")
                    filtered_docs.append(d)
            else:
                # Incorrect
                print(f"   INCORRECT (Scartato)")
        except Exception as e:
            print(f"    Errore grading: {e}")
            continue

    search_needed = len(filtered_docs) == 0
    return {"documents": filtered_docs, "question": question, "search_needed": search_needed}


def generate(state: GraphState):
    print("--- 3. GENERATE ---")
    question = state["question"]
    documents = state["documents"]

    if state["search_needed"] or not documents:
        return {"generation": "NESSUNA_DOC: Informazioni insufficienti (Tutti i documenti scartati come 'Incorrect')."}

    context = "\n\n".join([d.page_content for d in documents])

    prompt = PromptTemplate(
        template="""You are a Technical Writer. Write documentation based on the context.
        If the context contains 'refined' knowledge, integrate it smoothly.
        
        Topic: {question}
        Context:
        {context}
        
        Documentation (Markdown):""",
        input_variables=["question", "context"]
    )

    chain = prompt | local_llm

    response = chain.invoke({"question": question, "context": context})

    if isinstance(response, dict):
        response = str(response)

    return {"generation": str(response)}