import os
from langchain_community.vectorstores import FAISS
from utils.config import VECTOR_DB_PATH


def create_vector_store(chunks, embedding_model):
    """
    Create and save the FAISS vector database.
    """

    print("\nCreating FAISS Vector Store...")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    vector_store.save_local(VECTOR_DB_PATH)

    print(f"✅ Vector Store saved at '{VECTOR_DB_PATH}'")

    return vector_store


def load_vector_store(embedding_model):
    """
    Load an existing FAISS vector database.
    """

    if not os.path.exists(VECTOR_DB_PATH):
        raise FileNotFoundError("Vector database not found.")

    print("Loading existing FAISS Vector Store...")

    vector_store = FAISS.load_local(
        VECTOR_DB_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    print("✅ Vector Store loaded successfully.")

    return vector_store