import os
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from app.embeddings import get_embedding_model
from app.config import DB_DIR
from app.crag.state import GraphState

# Modello locale
llm = ChatAnthropic(
    model_name = "claude-3-haiku-20240307",
    temperature = 0,
    timeout=None,
    stop=None,
    max_retries=2
)

class Grade(BaseModel):
    """Binary score for relevance check."""
    score: str = Field(description="Must be 'yes' if the document is technically relevant, 'no' if irrelevant.")

# Grader strutturato
llm_grader = llm.with_structured_output(Grade)

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
    loop_step = state.get("loop_step", 0)
    print(f"   Recuperati {len(documents)} documenti.")
    return {"documents": documents, "question": question, "loop_step": loop_step}


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
    system_msg = """You are a strict technical evaluator. 
        Check if the document contains TECHNICAL instructions to answer the user question.

        Rules:
        1. Score 'yes' ONLY if the document explains the technical concept.
        2. Score 'no' if the document mentions keywords only in linguistic examples.
        3. Score 'no' if the document is unrelated.
        
        Output MUST match the format description ('yes' or 'no')."""

    filtered_docs = []
    for d in documents:
        try:
            grade_result = llm_grader.invoke([
                ("system", system_msg),
                ("user", f"Question: {question}\nDocument Snippet: {d.page_content}")
            ])

            if grade_result.score.lower() == "yes":
                print(f"   Rilevante: {d.metadata.get('source', 'unknown')}")
                filtered_docs.append(d)
            else:
                print(f"  SCARTATO (Irrilevante)")

        except Exception as e:
            print(f"   Errore API: {e}")
            continue

    search_needed = len(filtered_docs) == 0

    return {
        "documents": filtered_docs,
        "question": question,
        "search_needed": search_needed,
        "loop_step": state.get("loop_step", 0)
    }

def transform_query(state: GraphState):
    """
    Trasforma la query.
    Usato quando il Retrieval iniziale fallisce.
    """
    print("--- TRANSFORM QUERY (Rewriter) ---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    # Prompt per riscrivere la domanda
    prompt = PromptTemplate(
        template="""You are an AI assistant optimizing queries for documentation retrieval.
        The previous attempt failed.
        
        User Question: {question}
        
        Task: Re-phrase the question to be more technical and specific.
        Return ONLY the rewritten question string.
        New Question:""",
        input_variables=["question"]
    )

    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question})

    print(f"   Original: {question}")
    print(f"   Rewritten: {better_question}")

    # Aggiorna lo stato con la nuova domanda
    return {"question": better_question, "documents": documents, "loop_step": loop_step + 1}

def generate(state: GraphState):
    print("--- 3. GENERATE ---")
    question = state["question"]
    documents = state["documents"]

    if state["search_needed"] or not documents:
        return {"generation": "NESSUNA_DOC: Informazioni insufficienti (Tutti i documenti scartati come 'Incorrect')."}

    context = "\n\n".join([d.page_content for d in documents])

    prompt = PromptTemplate(
        template="""You are a Technical Writer. 
        Write a Markdown documentation based ONLY on the context provided.
        
        User Question: {question}
        
        Context Data (Documentation snippets):
        {context}
        
        Instructions:
        1. Focus on HOW to use the feature requested.
        2. EXTRACT CODE SNIPPETS from the context if available.
        3. Synthesize the information found in the documents.
        5. Only return "NESSUNA_DOC" if the context is completely empty or unrelated.
        
        Documentation (Markdown):""",
        input_variables=["question", "context"]
    )

    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({"question": question, "context": context})
        return {"generation": str(response)}
    except Exception as e:
        return {"generation": f"Errore generazione: {e}"}