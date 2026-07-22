from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model():
    """
    Load the embedding model.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("✅ Embedding model loaded successfully.")

    return embedding_model

