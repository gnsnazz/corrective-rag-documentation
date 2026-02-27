from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name = "BAAI/bge-m3",
        model_kwargs = {'device': 'cuda'},
        encode_kwargs = {'normalize_embeddings': True}
    )