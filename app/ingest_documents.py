import os
from pathlib import Path

from langchain_text_splitters import MarkdownTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from embeddings import get_embedding_model

REPO_PATH = "data/sklearn_repo"        # directory della repository
DB_DIR = "data/vectorstore/sklearn_md" # vector store di output
FILE_EXTENSIONS = [".md"]              # markdown

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
                print(f"Carico: {full_path}")

                loader = TextLoader(full_path, encoding="utf-8")
                loaded_docs = loader.load()
                documents.extend(loaded_docs)

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
    print("\n✂Split dei documenti...")
    chunks = split_documents(documents)
    print(f"   → {len(chunks)} chunk generati\n")

    # 3. Embeddings MiniLM
    print("Creo embeddings (MiniLM)...")
    embeddings = get_embedding_model()

    # 4. Salva in vectorstore
    print(f"\nSalvo nel vectorstore: {db_dir}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir
    )

    print("\nIngest completato! Repository indicizzata correttamente.\n")

if __name__ == "__main__":
    ingest_repository()