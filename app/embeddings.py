from sentence_transformers import SentenceTransformer

# Modello di embeddings locale
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def embed_text(texts: list[str]):
    """Restituisce gli embeddings per una lista di testi."""
    return embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )
