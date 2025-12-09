from app.crag.graph import build_crag_graph

def main():
    print("Avvio CRAG...")

    app = build_crag_graph()

    # testing query
    query = "What are the brand guidelines for the logo?"

    print(f"\nDomanda Utente: {query}")
    print("-" * 40)

    inputs = {"question": query}

    result = app.invoke(inputs)

    # risultato
    print("\n" + "=" * 40)
    print("🤖 RISPOSTA GENERATA:")
    print("=" * 40)
    print(result.get("generation", "Nessuna risposta generata."))

if __name__ == "__main__":
    main()