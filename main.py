from app.crag.graph import build_crag_graph
from app.utils import save_documentation

def main():
    print("Avvio CRAG...")

    app = build_crag_graph()
    print("-" * 30)
    print("   TEST GENERAZIONE DOC    ")
    print("-" * 30)

    topics = [
        #"How do I use the pipeline function for text classification?"
        #"How to use the Trainer API for fine-tuning",
        #"Explain the Tokenizer architecture",
        #"What is the pipeline abstraction?",
        "What is the default dropout rate in BertConfig?",
        #"How to use the pipeline for sentiment analysis?"
        #"How do I cook a carbonara with guanciale?"
    ]

    for topic in topics:
        print(f"\n Generazione per: '{topic}'...")

        result = app.invoke({"question": topic})
        content = result["generation"]

        if "NESSUNA_DOC" in content:
            print("  Skip: Informazioni non trovate.")
        else:
            path = save_documentation(content, topic)
            print(f" Documentazione salvata in: {path}")

            # Anteprima
            print("\n   ANTEPRIMA CONTENUTO GENERATO:")
            print("." * 40)
            print(content)
            print("." * 40)

if __name__ == "__main__":
    main()