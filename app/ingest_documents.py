import os
import shutil

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from embeddings import get_embedding_model
from config import DB_DIR, REPO_PATH

def load_documents():
    """Carica i documenti Markdown dalla directory."""
    loader = DirectoryLoader(
        REPO_PATH,
        glob = "**/*.md",
        loader_cls = TextLoader,
        loader_kwargs = {"encoding": "utf-8", "autodetect_encoding": True},
        silent_errors = True,
        show_progress = True
    )
    docs = loader.load()
    print(f"--- Caricati {len(docs)} documenti grezzi ---")
    return docs


def split_documents(documents):
    """
    Divide i documenti in chunk.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100,
        length_function = len,
        is_separator_regex = False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"--- Generati {len(chunks)} chunks di testo ---")
    return chunks


def save_to_chroma(chunks):
    """Crea e salva il Vector Store."""

    # 1. Pulizia iniziale
    if os.path.exists(DB_DIR):
        print("--- Cancellazione vecchio DB ---")
        shutil.rmtree(DB_DIR)

    # 2. Creazione nuovo DB
    model = get_embedding_model()

    print("--- Inizio indicizzazione ---")

    db = Chroma.from_documents(
        chunks,
        model,
        persist_directory = DB_DIR
    )

    print(f"--- Salvataggio completato in {DB_DIR} ---")


def main():
    print("--- INIZIO INGESTION ---")
    documents = load_documents()
    chunks = split_documents(documents)
    save_to_chroma(chunks)
    print("--- FINE INGESTION ---")

if __name__ == "__main__":
    main()