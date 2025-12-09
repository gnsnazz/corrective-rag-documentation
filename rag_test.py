import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from app.embeddings import get_embedding_model
from app.config import OLLAMA_MODEL, DB_DIR

# Configurazione Path

TEST_QUERY = "Where is the latest contributing guide available online according to the context?"

def test_retrieval():
    print("\n--- TEST RETRIEVAL ---")

    if not os.path.exists(DB_DIR):
        print(f"Errore: Database non trovato in {DB_DIR}.")
        return

    print("Carico vectorstore...")
    embeddings = get_embedding_model()

    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    # recupera i 3 chunk più simili
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    print(f"Query: {TEST_QUERY}")

    docs = retriever.invoke(TEST_QUERY)

    print(f"Trovati {len(docs)} documenti.")

    for i, d in enumerate(docs):
        print(f"\n[Doc {i + 1}] Fonte: {d.metadata.get('source', 'unknown')}")
        print(f"Contenuto: {d.page_content[:200]}...")


def test_generation():
    print("\n--- TEST GENERAZIONE (RAG) ---")

    embeddings = get_embedding_model()
    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # Inizializza Llama
    print(f"Caricamento modello: {OLLAMA_MODEL}...")
    llm = OllamaLLM(model=OLLAMA_MODEL)

    # 1. Recupero
    print("Recupero contesto...")
    context_docs = retriever.invoke(TEST_QUERY)

    if not context_docs:
        print("Nessun documento trovato. Impossibile rispondere.")
        return

    # 2. Preparazione Contesto
    context_text = "\n\n".join([d.page_content for d in context_docs])

    # 3. Prompt
    prompt = f"""You are a helpful AI assistant specialized in coding.
    Use the following pieces of context to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context:
    {context_text}

    Question: {TEST_QUERY}

    Answer:"""

    # 4. Generazione
    print("Generazione risposta in corso...")
    response = llm.invoke(prompt)

    print("\nRISPOSTA:")
    print("-" * 50)
    print(response)
    print("-" * 50)


if __name__ == "__main__":
    # test retrieval
    test_retrieval()

    # test generation
    test_generation()