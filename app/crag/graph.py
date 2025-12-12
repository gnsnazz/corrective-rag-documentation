from langgraph.graph import END, StateGraph
from app.crag.state import GraphState
from app.crag.nodes import retrieve, grade_documents, generate, transform_query

def decide_next_node(state):
    """
    Decide se generare o riscrivere, con un limite di tentativi.
    """
    loop_step = state.get("loop_step", 0)
    max_retries = 1

    if state["search_needed"]:
        if loop_step > max_retries:
            print(f"  Max retries ({max_retries}) raggiunti.")
            # genera
            return "generate"

        # altrimenti riscrivi e riprova
        else:
            print(f"  Search needed (Tentativo {loop_step + 1}/{max_retries + 1}) -> Rewrite")
            return "transform_query"
    else:
        print("  Documents found -> Generate")
        return "generate"

def build_crag_graph():
    """
    Costruisce e compila il grafo CRAG.
    """
    workflow = StateGraph(GraphState)

    # Aggiunge i nodi
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    # Definisce le connessioni (Edges)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_next_node,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        }
    )

    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", END)

    # Compila
    app = workflow.compile()
    return app