from typing import List, Literal, Optional
from pydantic import BaseModel, Field

RelevanceType = Literal["correct", "ambiguous", "incorrect", "refined", "unknown"]
RetrievalType = Literal["base", "corrective"]
CragAction = Literal["correct", "ambiguous", "incorrect", "pending"]

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
    upper_threshold: float = 0.60
    lower_threshold: float = 0.30
    confidence_score: float = 0.0
    retry_count: int = 0
    total_docs_examined: int = 0  # Contatore cumulativo di tutti i doc esaminati

    # Azione CRAG corrente a livello di sistema (per logging/debug)
    crag_action: CragAction = "pending"

    # Memoria per evitare loop di riscrittura identici
    previous_queries: List[str] = Field(default_factory = list)

    # Timing per nodo
    node_timings: dict = Field(default_factory = dict)