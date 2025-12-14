from app.crag.graph import build_crag_graph
from app.utils import save_documentation

def main():
    print("Avvio CRAG...")

    app = build_crag_graph()
    print("-" * 30)
    print("   TEST GENERAZIONE DOC    ")
    print("-" * 30)

    # query tecnica corretta/valida
    query = "How to initialize a pipeline for text classification in Transformers?"
    #query = "How to load a pre-trained BERT model using from_pretrained?"

    # query ambigua
    #query = "Explain BERT model architecture"
    #query = "What is the default dropout rate in ALBERT?"
    #query = "What is the architecture of BERT?"
    #query = "How does T5 handle text-to-text tasks?"

    # query non corretta
    #query = "asdasd qweqwe transformers banana"
    #query = "t5 model training arguments banana config"

    # query breve
    #query = "What is Trainer?"

    # query errata
    #query = "How does the moon affect deep learning?"
    #query = "How to use React Hooks?
    #query = "How to cook carbonara with Transformers?"

    print(f"\n Generazione per: '{query}'...")

    # invoca il workflow CRAG
    result = app.invoke({"question": query})
    content = result.get("generation", "")

    if not content or "NESSUNA_DOC" in content:
        print("  Skip: Informazioni non trovate.")
    else:
        # Salva documentazione generata
        path = save_documentation(content, query)
        print(f"\n Documentazione salvata in: {path}")

        # Anteprima contenuto
        print("\n   ANTEPRIMA CONTENUTO GENERATO:")
        print("." * 40)
        print(content)
        print("." * 40)

if __name__ == "__main__":
    main()