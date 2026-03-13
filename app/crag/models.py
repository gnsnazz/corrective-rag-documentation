import os
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import Literal

from app.embeddings import get_embedding_model
from app.config import DB_DIR

# --- LLM ---
llm = ChatAnthropic(
    model_name = "claude-haiku-4-5-20251001", #claude-sonnet-4-5-20250929
    temperature = 0,
    timeout = None,
    stop = None,
    max_retries = 2
)

# --- GRADER ---
class Grade(BaseModel):
    """Score for relevance check."""
    score: Literal["correct", "ambiguous", "incorrect"] = Field(
        description = """Relevance classification: 'correct' (explicit answer), 'ambiguous' (needs refinement),
         or 'incorrect' (irrelevant)."""
    )

llm_grader = llm.with_structured_output(Grade)

# --- VECTOR STORE ---
if os.path.exists(DB_DIR):
    embeddings = get_embedding_model()
    vectorstore = Chroma(persist_directory = DB_DIR, embedding_function = embeddings)
    retriever = vectorstore.as_retriever(search_kwargs = {"k": 8})
else:
    embeddings = None
    vectorstore = None
    retriever = None

# --- STRIP SPLITTER (Knowledge Refinement) ---
strip_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 20,
    length_function = len
)