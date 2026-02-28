from app.crag.graph import build_crag_graph
from app.utils import save_documentation
from app.config import ABSTENTION_MSG, format_source

def main():
    print("Avvio CRAG...")

    app = build_crag_graph()
    query = "How do I enable quantum attention in BERT?"

    print(f"\n Generazione per: '{query}'...")

    result = app.invoke({"question": query})
    content = result.get("generation", "")
    action = result.get("crag_action", "unknown")

    print(f"\n  CRAG Action: {action.upper()}")

    if not content or ABSTENTION_MSG in content:
        print("  Skip: Informazioni non trovate.")
    else:
        path = save_documentation(content, query)
        print(f"\n Documentazione salvata in: {format_source(path)}")

        print("\n   ANTEPRIMA CONTENUTO GENERATO:")
        print("." * 40)
        print(content)
        print("." * 40)

if __name__ == "__main__":
    main()