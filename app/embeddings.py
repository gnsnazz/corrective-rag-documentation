from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    return HuggingFaceEmbeddings(model_name=model_name)
