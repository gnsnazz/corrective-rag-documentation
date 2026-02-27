from app.crag.graph import build_crag_graph
from app.utils import save_documentation
from app.config import ABSTENTION_MSG, format_source

def main():
    print("Avvio CRAG...")

    app = build_crag_graph()
    print("-" * 30)
    print("   TEST GENERAZIONE DOC    ")
    print("-" * 30)

    # query tecnica corretta/valida
    #query = "How do I use the VideoEditor class to trim an mp4 file?"
    #query = "How to load a pre-trained BERT model using from_pretrained?"
    query = "How do I save a model using save_pretrained?"

    # query ambigua
    #query = "Explain BERT model architecture"
    #query = "How does T5 handle text-to-text tasks?"

    # query non corretta
    #query = "asdasd qweqwe transformers banana"

    # query errata
    #query = "How does the moon affect deep learning?"

    print(f"\n Generazione per: '{query}'...")

    # invoca il workflow CRAG
    result = app.invoke({"question": query})
    content = result.get("generation", "")

    if not content or ABSTENTION_MSG in content:
        print("  Skip: Informazioni non trovate.")
    else:
        # Salva documentazione generata
        path = save_documentation(content, query)
        print(f"\n Documentazione salvata in: {format_source(path)}")

        # Anteprima contenuto
        print("\n   ANTEPRIMA CONTENUTO GENERATO:")
        print("." * 40)
        print(content)
        print("." * 40)

if __name__ == "__main__":
    main()