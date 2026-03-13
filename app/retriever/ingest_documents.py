import os
import shutil

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from app.embeddings import get_embedding_model
from app.config import DB_DIR, PROJECT_PATH


def load_documents():
    """Carica i documenti Markdown e RST dalla directory."""
    docs = []

    for glob_pattern in ["**/*.md", "**/*.rst"]:
        loader = DirectoryLoader(
            PROJECT_PATH,
            glob = glob_pattern,
            loader_cls = TextLoader,
            loader_kwargs = {"encoding": "utf-8", "autodetect_encoding": True},
            silent_errors = True,
            show_progress = True
        )
        loaded = loader.load()
        docs.extend(loaded)
        print(f"--- Caricati {len(loaded)} documenti ({glob_pattern}) ---")

    print(f"--- Totale documenti grezzi: {len(docs)} ---")
    return docs


def split_documents(documents):
    """Divide i documenti in chunk."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 150,
        length_function = len,
        is_separator_regex = False
    )
    chunks = text_splitter.split_documents(documents)
    print(f"--- Generati {len(chunks)} chunks di testo ---")
    return chunks


def save_to_chroma(chunks):
    """Crea e salva il Vector Store."""

    if os.path.exists(DB_DIR):
        print("--- Cancellazione vecchio DB ---")
        shutil.rmtree(DB_DIR)

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