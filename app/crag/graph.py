from langgraph.graph import END, StateGraph
from app.crag.state import GraphState
from app.config import MAX_RETRIES, CONFIDENCE_THRESHOLD
from app.crag.nodes import (
    retrieve,
    grade_documents,
    transform_query,
    corrective_retriever,
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
    #threshold = state.confidence_threshold
    threshold = getattr(state, "confidence_threshold", 0.5)

    # 1: Confidenza alta (abbastanza doc Correct/Refined)
    if confidence >= threshold:
        print(f"  Confidence High ({confidence:.2f} >= {confidence:.2f}) -> Generating Answer")
        return "generate"

    # 2: Confidenza bassa, retry
    if retries < MAX_RETRIES:
        print(f"  Confidence Low ({confidence:.2f} < {threshold:.2f}) -> Corrective Search needed")
        return "transform_query"

    # 3: Confidenza bassa, tentativi finiti -> Genera con quello che abbiamo
    print(f"  Max Retries Reached ({retries} / {MAX_RETRIES}) -> Generating Best-Effort Answer")
    return "generate"


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
            "transform_query": "transform_query"
        }
    )

    # Loop Correttivo
    workflow.add_edge("transform_query", "corrective_retriever")
    # I nuovi documenti correttivi devono essere valutati!
    workflow.add_edge("corrective_retriever", "grade_documents")

    # Exit
    workflow.add_edge("generate", END)

    return workflow.compile()