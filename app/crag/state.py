from typing import List, TypedDict
from langchain_core.documents import Document

class GraphState(TypedDict):
    """
    Rappresenta lo stato del grafo CRAG.
    """
    question: str
    generation: str
    documents: List[Document]
    search_needed: bool # after, if needed