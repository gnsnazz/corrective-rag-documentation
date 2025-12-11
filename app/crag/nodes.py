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
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# --- NODI ---

def retrieve(state: GraphState):
    print("\n--- 1. RETRIEVE ---")
    question = state["question"]
    documents = retriever.invoke(question)
    print(f"   Recuperati {len(documents)} documenti.")
    return {"documents": documents, "question": question}


def grade_documents(state: GraphState):
    print("--- 2. GRADE DOCUMENTS (Evaluator) ---")
    question = state["question"]
    documents = state["documents"]

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question.
        Document: {document}
        User Question: {question}

        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["question", "document"],
    )

    chain = prompt | local_llm | StrOutputParser()

    filtered_docs = []
    for d in documents:
        try:
            score_data = chain.invoke({"question": question, "document": d.page_content})
            if "yes" in score_data.lower():
                print(f"   Rilevante: {d.metadata.get('source')}")
                filtered_docs.append(d)
            else:
                print(f"   Irrilevante (Score: {score_data})")
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
        return {"generation": "Mi dispiace, non ho trovato informazioni rilevanti nei documenti locali."}

    context = "\n\n".join([d.page_content for d in documents])

    prompt = PromptTemplate(
        template="""Answer the question based only on the following context:
        {context}

        Question: {question}
        Answer:""",
        input_variables=["question", "context"]
    )

    chain = prompt | local_llm

    response = chain.invoke({"question": question, "context": context})

    if isinstance(response, dict):
        response = str(response)

    return {"generation": response}