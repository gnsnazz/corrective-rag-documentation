from langgraph.graph import END, StateGraph
from app.crag.state import GraphState
from app.crag.nodes import retrieve, grade_documents, generate

def build_crag_graph():
    """
    Costruisce e compila il grafo CRAG.
    """
    workflow = StateGraph(GraphState)

    # Aggiunge i nodi
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)

    # Definisce le connessioni (Edges)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("grade_documents", "generate")
    workflow.add_edge("generate", END)

    # Compila
    app = workflow.compile()
    return app