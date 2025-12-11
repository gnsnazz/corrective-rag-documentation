from app.crag.graph import build_crag_graph

def main():
    print("Avvio CRAG...")

    app = build_crag_graph()

    # testing query
    # sklearn_repo
    #query = "How do I implement a Support Vector Machine (SVM)?"

    # transformers_repo - positive test
    #query = "How do i use the pipeline function for text classification?"
    #query = "What is the BERT model and how is it pre-trained?"

    # transformers_repo - negative test
    query = "How do I cook a carbonara with guanciale?"

    print(f"\nDomanda Utente: {query}")
    print("-" * 40)

    inputs = {"question": query}

    result = app.invoke(inputs)

    # risultato
    print("\n" + "=" * 40)
    print("RISPOSTA GENERATA:")
    print("=" * 40)
    print(result.get("generation", "Nessuna risposta generata."))

if __name__ == "__main__":
    main()