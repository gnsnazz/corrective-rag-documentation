import os
import time
from pathlib import Path

from langchain_text_splitters import MarkdownTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from embeddings import get_embedding_model
from config import REPO_PATH, DB_DIR

FILE_EXTENSIONS = [".md"]   # solo file markdown per ora

def load_markdown_files(repo_path: str):
    """
    Carica tutti i file markdown dalla repository.
    """
    documents = []

    for root, _, files in os.walk(repo_path):
        for file in files:
            ext = Path(file).suffix.lower()

            if ext in FILE_EXTENSIONS:
                full_path = os.path.join(root, file)
                #print(f"Carico: {full_path}")
                try:
                    loader = TextLoader(full_path, encoding="utf-8", autodetect_encoding=True)
                    loaded_docs = loader.load()

                    for doc in loaded_docs:
                        doc.metadata["source"] = full_path
                        doc.metadata["filename"] = file

                    documents.extend(loaded_docs)
                except Exception as e:
                    print(f" Errore nel file: {file}: {e}")
                    continue

    print(f"\nTotale file caricati: {len(documents)}")
    return documents


def split_documents(documents):
    """
    Divide i documenti in chunk ottimizzati per RAG.
    """
    splitter = MarkdownTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    return splitter.split_documents(documents)


def ingest_repository(repo_path: str = REPO_PATH, db_dir: str = DB_DIR):
    """
    Pipeline completa:
    - carica markdown
    - splitta
    - crea embeddings
    - salva vectorstore
    """
    print("\nInizio ingest della repository...\n")

    # 1. Carica file markdown
    documents = load_markdown_files(repo_path)

    if not documents:
        print("Nessun file trovato. Controlla REPO_PATH.")
        return

    # 2. Splitta in chunk
    print("\nSplit dei documenti...")
    chunks = split_documents(documents)
    total_chunks = len(chunks)
    print(f"   → {total_chunks} chunk generati\n")

    # 3. Embeddings MiniLM
    print("Creo embeddings (MiniLM)...")
    embeddings = get_embedding_model()

    # 4. Salva in vectorstore
    print(f"\nConnessione al DB: {db_dir}")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=db_dir
    )

    batch_size = 100  # Numero di chunk per volta
    print(f"🔄 Inizio inserimento a batch (Dimensione batch: {batch_size})...")

    start_time = time.time()

    for i in range(0, total_chunks, batch_size):
        # Prendi una fetta di chunk
        batch = chunks[i: i + batch_size]

        # Aggiunge al DB
        vectorstore.add_documents(batch)

        # Feedback visivo
        percent = ((i + len(batch)) / total_chunks) * 100
        print(f"   Batch salvato: {i + len(batch)}/{total_chunks} ({percent:.1f}%)")

    end_time = time.time()
    duration = end_time - start_time

    print(f"\n Ingest completato in {duration:.2f} secondi!")
    print(f"\n Vector DB in {db_dir}\n")

if __name__ == "__main__":
    ingest_repository()