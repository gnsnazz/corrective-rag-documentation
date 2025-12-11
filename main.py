from app.crag.graph import build_crag_graph
from app.utils import save_documentation

def main():
    print("Avvio CRAG...")

    app = build_crag_graph()
    print("-" * 30)
    print("   TEST GENERAZIONE DOC    ")
    print("-" * 30)

    # testing query - sklearn_repo
    #query = "How do I implement a Support Vector Machine (SVM)?"

    # transformers_repo - positive test
    #query = "How do i use the pipeline function for text classification?"
    #query = "What is the BERT model and how is it pre-trained?"

    # transformers_repo - negative test
    #query = "How do I cook a carbonara with guanciale?"
    #query = "How do I use a PreTrainedTokenizer to prepare text for a model?"

    topics = [
        #"How to use the Trainer API for fine-tuning",
        #"What are the supported optimizers?",
        "Explain the Tokenizer architecture"
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
            print("\n💡 ANTEPRIMA CONTENUTO GENERATO:")
            print("." * 40)
            print(content)
            print("." * 40)

if __name__ == "__main__":
    main()