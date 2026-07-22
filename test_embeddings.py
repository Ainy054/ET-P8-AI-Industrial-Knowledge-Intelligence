from utils.embeddings import get_embedding_model

embedding_model = get_embedding_model()

vector = embedding_model.embed_query(
    "Why is the pump overheating?"
)

print(f"Embedding Dimension: {len(vector)}")
print(vector[:10])