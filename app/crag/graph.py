from langgraph.graph import END, StateGraph
from app.crag.state import GraphState
from app.config import MAX_RETRIES
from app.crag.nodes import (
    retrieve,
    grade_documents,
    transform_query,
    corrective_retriever,
    discard_knowledge,
    generate
)

def decide_next_node(state: GraphState):
    """
    Logica di Routing CRAG:
    - High confidence -> answer
    - Low confidence -> rewrite ONLY if ci sono documenti
    - Max retries -> answer (best-effort)
    """
    confidence = state.confidence_score
    retries = state.retry_count
    upper = state.upper_threshold
    lower = state.lower_threshold

    # Fallback: retry esauriti -> genera best-effort
    if retries >= MAX_RETRIES:
        print(f"  Max Retries ({retries}/{MAX_RETRIES}) -> Best-Effort Generation")
        return "generate"

    # 1. CORRECT: abbastanza evidenza, genera
    if confidence >= upper:
        print(f"  CORRECT ({confidence:.2f} >= {upper:.2f}) -> Generate")
        return "generate"

    # 2. AMBIGUOUS: qualcosa c'è ma non basta, mantieni k_in + correttivo
    if confidence >= lower:
        print(f"  AMBIGUOUS ({lower:.2f} <= {confidence:.2f} < {upper:.2f}) -> Refine + Corrective")
        return "corrective_ambiguous"

    # 3. INCORRECT: retrieval fallito, scarta e ricerca da zero
    print(f"  INCORRECT ({confidence:.2f} < {lower:.2f}) -> Discard + Corrective")
    return "corrective_incorrect"


def build_crag_graph():
    """
    Costruisce e compila il grafo CRAG.
    """
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("corrective_retriever", corrective_retriever)
    workflow.add_node("discard_knowledge", discard_knowledge)
    workflow.add_node("generate", generate)

    # Define Flow
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # Conditional Edge
    workflow.add_conditional_edges(
        "grade_documents",
        decide_next_node,
        {
            "generate": "generate",
            "corrective_ambiguous": "transform_query",
            "corrective_incorrect": "discard_knowledge"
        }
    )

    # Loop Correttivo
    workflow.add_edge("transform_query", "corrective_retriever")
    # I nuovi documenti correttivi devono essere valutati!
    workflow.add_edge("corrective_retriever", "grade_documents")

    # INCORRECT path: discard -> transform -> (si riaggancia al flusso sopra)
    workflow.add_edge("discard_knowledge", "transform_query")

    # Exit
    workflow.add_edge("generate", END)

    return workflow.compile()