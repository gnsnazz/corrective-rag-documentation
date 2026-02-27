from typing import List, Literal, Optional
from pydantic import BaseModel, Field

RelevanceType = Literal["correct", "ambiguous", "incorrect", "refined", "unknown"]
RetrievalType = Literal["base", "corrective"]

class CragDocument(BaseModel):
    """
    Rappresentazione unificata del documento nel grafo.
    """
    page_content: str
    metadata: dict
    relevance_score: RelevanceType = "unknown"
    retrieval_source: RetrievalType = "base"

class GraphState(BaseModel):
    """
    Rappresenta lo stato del grafo CRAG.
    """
    question: str
    generation: Optional[str] = None
    # Documenti attualmente sotto esame (batch corrente)
    documents: List[CragDocument] = Field(default_factory = list)
    # Accumulatore di documenti validi (Correct + Refined)
    final_documents: List[CragDocument] = Field(default_factory = list)
    k_in: List[CragDocument] = Field(default_factory = list)  # correct + refined (Knowledge Interna)
    k_ex: List[CragDocument] = Field(default_factory = list)  # corrective research (Knowledge Esterna/Correttiva)

    # Metriche di controllo
    confidence_threshold: float = 0.5
    confidence_score: float = 0.0
    retry_count: int = 0

    # Memoria per evitare loop di riscrittura identici
    previous_queries: List[str] = Field(default_factory = list)