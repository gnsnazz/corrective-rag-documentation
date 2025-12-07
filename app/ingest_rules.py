import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def load_and_chunk_pdf(pdf_path: str, filename: str) -> list[Document]:
    reader = PdfReader(pdf_path)
    raw_docs = []

    # Estrae testo per pagina
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            # Salviamo il nome file pulito e la pagina
            meta = {"source": filename, "page": i + 1}
            raw_docs.append(Document(page_content=text, metadata=meta))

    # Splitta ulteriormente i chunk gestendo l'overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    # split_documents preserva e propaga i metadati (pagina) ai chunk figli
    return splitter.split_documents(raw_docs)


def ingest_rules(
        rules_dir: str = "data/rules",
        db_dir: str = "data/vectorstore/rules_db"
):
    print("Avvio ingest delle regole…")

    all_docs = []

    if not os.path.exists(rules_dir):
        os.makedirs(rules_dir)
        print(f"Cartella {rules_dir} creata. Inserisci i PDF e riavvia.")
        return

    for filename in os.listdir(rules_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(rules_dir, filename)
            print(f" Leggo: {filename}")
            docs = load_and_chunk_pdf(pdf_path, filename)
            print(f"   → {len(docs)} chunk creati")
            all_docs.extend(docs)

    if not all_docs:
        print("Nessuna regola trovata o PDF vuoti.")
        return

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creo/aggiorno il vector store…")

    vectordb = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=db_dir
    )

    print("Ingest completato!")
    print(f"Vector DB salvato in: {db_dir}")
if __name__ == "__main__":
    ingest_rules()