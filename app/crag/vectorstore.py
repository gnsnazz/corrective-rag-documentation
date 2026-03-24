import os
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.embeddings import get_embedding_model
from app.config import DB_DIR, K_BASE


# --- VECTOR STORE ---
embeddings = get_embedding_model()
vectorstore = Chroma(persist_directory = DB_DIR, embedding_function = embeddings) if os.path.exists(DB_DIR) else None
retriever = vectorstore.as_retriever(search_kwargs = {"k": K_BASE}) if vectorstore else None

# --- STRIP SPLITTER (Knowledge Refinement) ---
strip_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 20,
    length_function = len
)
