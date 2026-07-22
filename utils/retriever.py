def get_retriever(vector_store):
    """
    Create a retriever using Maximum Marginal Relevance (MMR)
    to retrieve diverse but relevant chunks.
    """

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

    print("✅ Retriever created successfully.")

    return retriever